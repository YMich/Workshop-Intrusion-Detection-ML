from __future__ import annotations

import pandas as pd

from feature_config import (
    CORRELATION_RMI_DROP_COLUMNS,
    EXPECTED_FINAL_FEATURE_COUNT,
    LABEL_COL,
    METADATA_COLUMNS,
    RAW_NON_PREDICTIVE_COLUMNS,
    REPORT_CONSTANT_COLUMNS,
    SOURCE_FILE_COL,
    TIMESTAMP_COL,
)


# ============================================================
# SCHEMA HELPERS
# ============================================================

def clean_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Defensive cleanup only.
    """
    df = df.copy()

    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()
        for column in df.columns
    ]

    duplicated = (
        df.columns[
            df.columns.duplicated()
        ]
        .tolist()
    )

    if duplicated:
        raise ValueError(
            "Duplicate columns after header cleanup: "
            f"{duplicated}"
        )

    return df


def parse_timestamp_if_needed(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restore Timestamp datetime after loading a CSV snapshot.
    """
    df = df.copy()

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Missing required Timestamp column: "
            f"{TIMESTAMP_COL}"
        )

    if pd.api.types.is_datetime64_any_dtype(
        df[TIMESTAMP_COL]
    ):
        return df

    try:
        parsed = pd.to_datetime(
            df[TIMESTAMP_COL],
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(
            df[TIMESTAMP_COL],
            errors="coerce",
        )

    invalid_mask = (
        parsed.isna()
    )

    if invalid_mask.any():
        examples = (
            df.loc[
                invalid_mask,
                TIMESTAMP_COL,
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Failed to parse "
            f"{int(invalid_mask.sum()):,} "
            f"Timestamp value(s). "
            f"Examples: {examples}"
        )

    df[TIMESTAMP_COL] = (
        parsed
    )

    return df


def validate_required_columns(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Metadata/identifiers are retained for every downstream model.
    """
    required = (
        METADATA_COLUMNS
        + [LABEL_COL]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name}: "
            f"missing required columns: "
            f"{missing}"
        )


def validate_dataset_schemas(
    datasets: dict[str, pd.DataFrame],
) -> None:
    """
    All preprocessed datasets must have exactly the same schema/order.
    """
    if not datasets:
        raise ValueError(
            "No preprocessed datasets were provided."
        )

    names = list(
        datasets.keys()
    )

    reference_name = (
        names[0]
    )

    reference_columns = list(
        datasets[
            reference_name
        ].columns
    )

    for (
        dataset_name,
        df,
    ) in datasets.items():
        actual_columns = list(
            df.columns
        )

        if (
            actual_columns
            != reference_columns
        ):
            missing = [
                column
                for column in reference_columns
                if column not in actual_columns
            ]

            extra = [
                column
                for column in actual_columns
                if column not in reference_columns
            ]

            raise ValueError(
                f"{dataset_name}: schema differs "
                f"from {reference_name}. "
                f"Missing={missing}; Extra={extra}"
            )


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_candidate_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Candidate predictive features are every column except:
      - retained metadata/identifiers
      - Label
    """
    excluded = set(
        METADATA_COLUMNS
        + [LABEL_COL]
    )

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


def validate_numeric_candidate_features(
    df: pd.DataFrame,
    candidate_columns: list[str],
    dataset_name: str,
) -> None:
    """
    Step 2 should have produced finite numeric predictive features.
    """
    non_numeric = (
        df[
            candidate_columns
        ]
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric:
        raise ValueError(
            f"{dataset_name}: "
            "non-numeric candidate features remain after Step 2: "
            f"{non_numeric}"
        )

    if (
        df[
            candidate_columns
        ]
        .isna()
        .any()
        .any()
    ):
        columns = (
            df[
                candidate_columns
            ]
            .columns[
                df[
                    candidate_columns
                ]
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            f"{dataset_name}: "
            f"NaN remains in predictive features: {columns}"
        )


def build_drop_manifest(
    available_columns: list[str],
) -> pd.DataFrame:
    """
    Reproducible feature-removal manifest.
    """
    rows = []

    groups = [
        (
            RAW_NON_PREDICTIVE_COLUMNS,
            "raw_non_predictive",
            "Raw/environment-specific field excluded from predictive features",
        ),
        (
            REPORT_CONSTANT_COLUMNS,
            "constant",
            "Reported constant/uninformative feature",
        ),
        (
            CORRELATION_RMI_DROP_COLUMNS,
            "correlation_rmi",
            "Removed by final universal correlation + RMI reduction",
        ),
    ]

    seen = set()

    for (
        columns,
        category,
        reason,
    ) in groups:
        for column in columns:
            if column in seen:
                continue

            rows.append({
                "Feature": column,
                "Category": category,
                "Reason": reason,
                "Present_In_Current_Schema": (
                    column in available_columns
                ),
            })

            seen.add(
                column
            )

    return pd.DataFrame(
        rows
    )


def select_feature_columns(
    df: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
    pd.DataFrame,
]:
    """
    Return:
        selected_features,
        dropped_present_features,
        drop_manifest
    """
    candidate_columns = (
        get_candidate_feature_columns(
            df
        )
    )

    candidate_set = set(
        candidate_columns
    )

    fixed_drop_columns = (
        RAW_NON_PREDICTIVE_COLUMNS
        + REPORT_CONSTANT_COLUMNS
        + CORRELATION_RMI_DROP_COLUMNS
    )

    dropped_present = []

    seen = set()

    for column in fixed_drop_columns:
        if (
            column in candidate_set
            and column not in seen
        ):
            dropped_present.append(
                column
            )

            seen.add(
                column
            )

    drop_set = set(
        dropped_present
    )

    selected_features = [
        column
        for column in candidate_columns
        if column not in drop_set
    ]

    manifest = (
        build_drop_manifest(
            candidate_columns
        )
    )

    return (
        selected_features,
        dropped_present,
        manifest,
    )


# ============================================================
# MODEL-AGNOSTIC FEATURE VIEW
# ============================================================

def create_feature_selected_view(
    df: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Create ONE universal output for all downstream models.

    It retains:
      - metadata / identifiers
      - selected predictive features
      - Label

    No model-specific representation is created here.
    """
    columns = (
        METADATA_COLUMNS
        + selected_features
        + [LABEL_COL]
    )

    result = (
        df[
            columns
        ]
        .copy()
    )

    # Preserve stable source/capture and chronological order for all files.
    result = (
        result.sort_values(
            by=[
                SOURCE_FILE_COL,
                TIMESTAMP_COL,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return result


def select_and_engineer_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    expected_selected_features: list[str] | None = None,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    pd.DataFrame,
]:
    """
    Apply the fixed Step 3 feature-selection rules.

    Step 3 remains model-agnostic.
    """
    df = clean_column_names(
        df
    )

    df = parse_timestamp_if_needed(
        df
    )

    validate_required_columns(
        df,
        dataset_name,
    )

    candidate_features = (
        get_candidate_feature_columns(
            df
        )
    )

    validate_numeric_candidate_features(
        df,
        candidate_features,
        dataset_name,
    )

    (
        selected_features,
        dropped_present,
        manifest,
    ) = select_feature_columns(
        df
    )

    if expected_selected_features is not None:
        if (
            selected_features
            != expected_selected_features
        ):
            missing = [
                column
                for column in expected_selected_features
                if column not in selected_features
            ]

            extra = [
                column
                for column in selected_features
                if column not in expected_selected_features
            ]

            raise ValueError(
                f"{dataset_name}: selected feature schema differs "
                f"from the other datasets. "
                f"Missing={missing}; Extra={extra}"
            )

    feature_selected = (
        create_feature_selected_view(
            df,
            selected_features,
        )
    )

    return (
        feature_selected,
        selected_features,
        dropped_present,
        manifest,
    )


def select_and_engineer_all_datasets(
    datasets: dict[str, pd.DataFrame],
) -> tuple[
    dict[str, pd.DataFrame],
    list[str],
    pd.DataFrame,
]:
    """
    Apply the exact same selected-feature schema to all datasets.

    Returns one model-agnostic DataFrame per dataset.
    """
    validate_dataset_schemas(
        datasets
    )

    outputs = {}

    expected_selected_features = None
    common_manifest = None

    for (
        dataset_name,
        df,
    ) in datasets.items():
        (
            feature_selected,
            selected_features,
            _dropped_present,
            manifest,
        ) = select_and_engineer_dataset(
            dataset_name=dataset_name,
            df=df,
            expected_selected_features=(
                expected_selected_features
            ),
        )

        if expected_selected_features is None:
            expected_selected_features = list(
                selected_features
            )

            common_manifest = (
                manifest.copy()
            )

        outputs[
            dataset_name
        ] = feature_selected

    if (
        len(
            expected_selected_features
        )
        != EXPECTED_FINAL_FEATURE_COUNT
    ):
        raise ValueError(
            "Final selected feature count does not match the "
            "Milestone 2 final schema. "
            f"Expected {EXPECTED_FINAL_FEATURE_COUNT}, "
            f"got {len(expected_selected_features)}.\n"
            "This usually means an input schema changed."
        )

    return (
        outputs,
        expected_selected_features,
        common_manifest,
    )
