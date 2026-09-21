"""Build the canonical ingested datasets from aligned extraction outputs.

Historical pipeline:

    raw PCAP / auxiliary data
        -> src/extraction/
        -> data/extracted/
        -> src/ingestion/
        -> data/ingested/

The final ML submission starts from ``data/ingested``. This module is retained
for provenance and for users who also possess the intermediate extraction
outputs.

Ingestion performs only dataset assembly and metadata validation. It does NOT
fit preprocessing statistics, normalize features, select features, split
train/validation/test data, or reshape sequences.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    # Package-style import.
    from .ingestion_config import (
        DATASET1_BENIGN_DIR,
        DATASET1_INGESTED_CSV,
        DATASET1_MALICIOUS_DIR,
        DATASET2_BENIGN_DIR,
        DATASET2_INGESTED_CSV,
        DATASET2_MALICIOUS_DIR,
        DATASET3_BENIGN_DIR,
        DATASET3_INGESTED_CSV,
        DATASET3_MALICIOUS_DIR,
        INGESTED_ROOT,
        LABEL_COL,
        REQUIRED_COLUMNS,
        SOURCE_FILE_COL,
        TIMESTAMP_COL,
    )
except ImportError:
    # Direct-script execution:
    #   python src/ingestion/data_ingestion.py
    from ingestion_config import (
        DATASET1_BENIGN_DIR,
        DATASET1_INGESTED_CSV,
        DATASET1_MALICIOUS_DIR,
        DATASET2_BENIGN_DIR,
        DATASET2_INGESTED_CSV,
        DATASET2_MALICIOUS_DIR,
        DATASET3_BENIGN_DIR,
        DATASET3_INGESTED_CSV,
        DATASET3_MALICIOUS_DIR,
        INGESTED_ROOT,
        LABEL_COL,
        REQUIRED_COLUMNS,
        SOURCE_FILE_COL,
        TIMESTAMP_COL,
    )


# =============================================================================
# INGESTION CONTRACT
# =============================================================================
#
# This module intentionally does NOT:
#   - replace inf / -inf
#   - impute missing numeric values
#   - fit or apply scaling / normalization
#   - remove constant or redundant features
#   - apply correlation / RMI feature selection
#   - drop identifiers such as Flow ID, IPs, ports, Protocol, or Timestamp
#   - construct train / validation / test splits
#   - construct LSTM / LITEMV sequences
#
# Those responsibilities belong to later pipeline stages.
# =============================================================================


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace/BOM markers from CSV column names."""
    result = df.copy()
    result.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in result.columns
    ]

    duplicated = result.columns[result.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"Duplicate columns after header cleaning: {duplicated}"
        )

    return result


def validate_required_columns(
    columns: list[str],
    source: Path,
) -> None:
    """Require the canonical identifiers and extraction Label field."""
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {source}:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def validate_exact_schema(
    actual_columns: list[str],
    expected_columns: list[str],
    source: Path,
) -> None:
    """Reject missing, extra, or reordered columns across aligned CSVs."""
    if actual_columns == expected_columns:
        return

    missing = [
        column
        for column in expected_columns
        if column not in actual_columns
    ]
    extra = [
        column
        for column in actual_columns
        if column not in expected_columns
    ]
    order_only_difference = (
        not missing
        and not extra
        and actual_columns != expected_columns
    )

    message = [
        f"Schema mismatch in: {source}",
        f"Expected columns: {len(expected_columns)}",
        f"Actual columns:   {len(actual_columns)}",
    ]

    if missing:
        message.append(f"Missing columns: {missing}")
    if extra:
        message.append(f"Extra columns: {extra}")
    if order_only_difference:
        message.append(
            "The same columns exist, but their order is different."
        )

    raise ValueError("\n".join(message))


def parse_and_validate_timestamp(
    df: pd.DataFrame,
    source: Path,
) -> pd.DataFrame:
    """Parse every Timestamp and fail rather than silently discarding bad rows."""
    result = df.copy()
    original_timestamp = result[TIMESTAMP_COL].copy()

    try:
        parsed = pd.to_datetime(
            original_timestamp,
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError):
        # Compatibility fallback for pandas versions without format="mixed".
        parsed = pd.to_datetime(
            original_timestamp,
            errors="coerce",
        )

    invalid_mask = parsed.isna()
    if invalid_mask.any():
        examples = (
            original_timestamp[invalid_mask]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"Failed to parse {int(invalid_mask.sum()):,} Timestamp "
            f"value(s) in {source}.\n"
            f"Examples: {examples}"
        )

    result[TIMESTAMP_COL] = parsed
    return result


def load_single_extracted_csv(
    csv_path: Path,
    label_value: int,
    source_class: str,
    expected_schema: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load one aligned CICFlowMeter CSV and assign authoritative metadata."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = clean_column_names(df)

    columns = list(df.columns)
    validate_required_columns(columns, csv_path)

    if expected_schema is None:
        expected_schema = columns
    else:
        validate_exact_schema(
            actual_columns=columns,
            expected_columns=expected_schema,
            source=csv_path,
        )

    df = parse_and_validate_timestamp(df, csv_path)

    # The containing benign/malicious directory is authoritative. Do not trust
    # any pre-existing label spelling/capitalization from the extraction CSV.
    df[LABEL_COL] = int(label_value)

    source_id = f"{source_class}/{csv_path.name}"
    insert_position = df.columns.get_loc(TIMESTAMP_COL) + 1

    if SOURCE_FILE_COL in df.columns:
        raise ValueError(
            f"{csv_path} already contains {SOURCE_FILE_COL!r}; "
            "ingestion owns this provenance field."
        )

    df.insert(
        insert_position,
        SOURCE_FILE_COL,
        source_id,
    )

    # Stable sorting preserves original row order when timestamps tie.
    return (
        df.sort_values(
            by=TIMESTAMP_COL,
            kind="mergesort",
        )
        .reset_index(drop=True),
        expected_schema,
    )


def load_class_directory(
    input_dir: Path,
    label_value: int,
    source_class: str,
    expected_schema: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load every aligned CSV in one benign or malicious directory."""
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Extraction directory not found:\n{input_dir}"
        )

    csv_files = sorted(
        path
        for path in input_dir.glob("*.csv")
        if path.is_file()
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{input_dir}"
        )

    frames: list[pd.DataFrame] = []

    for csv_path in csv_files:
        frame, expected_schema = load_single_extracted_csv(
            csv_path=csv_path,
            label_value=label_value,
            source_class=source_class,
            expected_schema=expected_schema,
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values(
            by=[SOURCE_FILE_COL, TIMESTAMP_COL],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return combined, expected_schema


def load_dataset(
    dataset_name: str,
    benign_dir: Path,
    malicious_dir: Path,
    expected_schema: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Combine the benign and malicious aligned flows for one dataset."""
    print("\n" + "=" * 80)
    print(f"INGESTING {dataset_name.upper()}")
    print("=" * 80)

    benign_df, expected_schema = load_class_directory(
        input_dir=benign_dir,
        label_value=0,
        source_class="benign",
        expected_schema=expected_schema,
    )
    malicious_df, expected_schema = load_class_directory(
        input_dir=malicious_dir,
        label_value=1,
        source_class="malicious",
        expected_schema=expected_schema,
    )

    dataset = pd.concat(
        [benign_df, malicious_df],
        ignore_index=True,
    )
    dataset = (
        dataset.sort_values(
            by=[SOURCE_FILE_COL, TIMESTAMP_COL],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    print(f"Benign rows:    {len(benign_df):,}")
    print(f"Malicious rows: {len(malicious_df):,}")
    print(f"Total rows:     {len(dataset):,}")
    print(f"Source files:   {dataset[SOURCE_FILE_COL].nunique():,}")
    print(
        f"Timestamp range: "
        f"{dataset[TIMESTAMP_COL].min()} -> "
        f"{dataset[TIMESTAMP_COL].max()}"
    )

    return dataset, expected_schema


def load_dataset1(
    expected_schema: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load Dataset 1, including the aligned Friday benign HTTPS CSV."""
    return load_dataset(
        dataset_name="dataset1",
        benign_dir=DATASET1_BENIGN_DIR,
        malicious_dir=DATASET1_MALICIOUS_DIR,
        expected_schema=expected_schema,
    )


def load_dataset2(
    expected_schema: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load Dataset 2."""
    return load_dataset(
        dataset_name="dataset2",
        benign_dir=DATASET2_BENIGN_DIR,
        malicious_dir=DATASET2_MALICIOUS_DIR,
        expected_schema=expected_schema,
    )


def load_dataset3(
    expected_schema: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load Dataset 3."""
    return load_dataset(
        dataset_name="dataset3",
        benign_dir=DATASET3_BENIGN_DIR,
        malicious_dir=DATASET3_MALICIOUS_DIR,
        expected_schema=expected_schema,
    )


def verify_ingested_datasets(
    datasets: dict[str, pd.DataFrame],
) -> None:
    """Validate the complete D1/D2/D3 ingested-dataset contract."""
    required_dataset_names = {"dataset1", "dataset2", "dataset3"}
    actual_dataset_names = set(datasets)

    if actual_dataset_names != required_dataset_names:
        raise ValueError(
            "Expected exactly dataset1, dataset2, and dataset3; "
            f"found {sorted(actual_dataset_names)}."
        )

    reference_name = "dataset1"
    reference_columns = list(datasets[reference_name].columns)

    for name, df in datasets.items():
        if df.empty:
            raise ValueError(f"{name} is empty.")

        if list(df.columns) != reference_columns:
            raise ValueError(
                f"Ingested schema mismatch: {name} differs "
                f"from {reference_name}."
            )

        labels = set(
            df[LABEL_COL]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if labels != {0, 1}:
            raise ValueError(
                f"{name} must contain both binary classes {{0, 1}}; "
                f"found {sorted(labels)}."
            )

        if df[LABEL_COL].isna().any():
            raise ValueError(f"{name} contains missing Label values.")

        if df[TIMESTAMP_COL].isna().any():
            raise ValueError(
                f"{name} contains missing parsed timestamps."
            )

        if not pd.api.types.is_datetime64_any_dtype(
            df[TIMESTAMP_COL]
        ):
            raise TypeError(
                f"{name}.{TIMESTAMP_COL} is not a datetime dtype."
            )

        if df[SOURCE_FILE_COL].isna().any():
            raise ValueError(
                f"{name} contains missing SourceFile values."
            )

        # SourceFile is expected to be class-pure because it is generated from
        # either a benign or malicious extraction directory.
        source_label_counts = (
            df.groupby(SOURCE_FILE_COL, sort=False)[LABEL_COL]
            .nunique()
        )
        mixed_sources = source_label_counts[source_label_counts != 1]
        if not mixed_sources.empty:
            raise ValueError(
                f"{name} contains mixed-label SourceFile values: "
                f"{mixed_sources.index.tolist()[:10]}"
            )

        for source_file, group in df.groupby(
            SOURCE_FILE_COL,
            sort=False,
        ):
            if not group[TIMESTAMP_COL].is_monotonic_increasing:
                raise ValueError(
                    f"{name}: timestamps are not chronological "
                    f"inside {source_file}"
                )

    friday_source = "benign/friday_benign_https_aligned.csv"
    if friday_source not in set(
        datasets["dataset1"][SOURCE_FILE_COL].unique()
    ):
        raise ValueError(
            "Dataset 1 does not contain "
            "friday_benign_https_aligned.csv. "
            "Run prepare_cicids2017_friday_benign_https.py first "
            "when rebuilding from extraction outputs."
        )

    print(
        "\nVerified: all three ingested datasets have the same "
        "schema, both binary classes, valid datetime timestamps, "
        "class-pure SourceFile provenance, chronological ordering "
        "within each source file, and Friday benign HTTPS traffic "
        "is included in Dataset 1."
    )


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """Load all three datasets while enforcing one common extraction schema."""
    expected_schema: list[str] | None = None

    dataset1, expected_schema = load_dataset1(expected_schema)
    dataset2, expected_schema = load_dataset2(expected_schema)
    dataset3, expected_schema = load_dataset3(expected_schema)

    datasets = {
        "dataset1": dataset1,
        "dataset2": dataset2,
        "dataset3": dataset3,
    }

    verify_ingested_datasets(datasets)
    return datasets


def save_ingested_datasets(
    datasets: dict[str, pd.DataFrame],
) -> None:
    """Write the canonical CSVs consumed by the final ML pipeline."""
    INGESTED_ROOT.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "dataset1": DATASET1_INGESTED_CSV,
        "dataset2": DATASET2_INGESTED_CSV,
        "dataset3": DATASET3_INGESTED_CSV,
    }

    for name, df in datasets.items():
        output_path = output_paths[name]
        df.to_csv(
            output_path,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S.%f",
        )
        print(
            f"Saved {name}: {output_path} "
            f"| shape={df.shape}"
        )


def main() -> None:
    """Rebuild all canonical ingested datasets from data/extracted."""
    datasets = load_all_datasets()
    save_ingested_datasets(datasets)

    print("\n" + "=" * 80)
    print("DATA INGESTION COMPLETE")
    print("=" * 80)
    print(
        "No train/validation/test splitting, preprocessing, "
        "normalization, feature selection, or sequence construction "
        "has been applied."
    )


if __name__ == "__main__":
    main()
