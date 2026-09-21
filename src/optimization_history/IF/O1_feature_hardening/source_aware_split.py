from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FILE_COL = "SourceFile"
LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"
ORIGINAL_ROW_COL = "__OriginalRowIndex"
SPLIT_COL = "__Split"

SPLIT_NAMES = ("train", "validation", "test")
TARGET_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

# A whole-file assignment is considered usable when every split remains within
# +/- 5 percentage points of the requested global row ratio and both classes
# are present in each split.
MAX_TOTAL_RATIO_DEVIATION = 0.05

# A source is considered oversized only after whole-SourceFile assignment has
# already failed.  The smallest target partitions are validation/test at 15%.
# With a +/-5 percentage-point whole-file tolerance, any single source larger
# than 20% cannot fit wholly inside either holdout partition without making that
# partition unreasonable.  Such a source is therefore eligible for the one and
# only exception: chronological 70/15/15 splitting of that SourceFile.
#
# This is NOT permission for generic row-level splitting.  All other SourceFiles
# remain intact.
OVERSIZED_SOURCE_MIN_FRACTION = (
    TARGET_RATIOS["validation"] + MAX_TOTAL_RATIO_DEVIATION
)

# The hybrid split should be very close to the target because the oversized
# source itself is divided exactly 70/15/15. A slightly wider guard allows the
# remaining intact SourceFiles to be coarse-grained.
MAX_HYBRID_RATIO_DEVIATION = 0.07


class SplitConstructionError(RuntimeError):
    pass


def _validate_input(df: pd.DataFrame) -> None:
    required = {SOURCE_FILE_COL, LABEL_COL, TIMESTAMP_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns required for source-aware splitting: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Cannot split an empty dataset.")


def _source_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(SOURCE_FILE_COL, sort=True)
        .agg(
            Label=(LABEL_COL, "first"),
            LabelNunique=(LABEL_COL, "nunique"),
            Rows=(LABEL_COL, "size"),
        )
        .reset_index()
    )

    mixed = table[table["LabelNunique"] != 1]
    if not mixed.empty:
        raise ValueError(
            "Every SourceFile must have one ground-truth label. Mixed-label "
            "sources found:\n"
            + mixed.to_string(index=False)
        )

    table = table.drop(columns=["LabelNunique"])
    table["Label"] = table["Label"].astype(int)
    table["Rows"] = table["Rows"].astype(int)

    labels = set(table["Label"].unique())
    if labels != {0, 1}:
        raise ValueError(
            "Both benign (0) and malicious (1) SourceFiles are required. "
            f"Found labels: {sorted(labels)}"
        )

    return table


def _objective(
    counts: dict[str, float],
    targets: dict[str, float],
) -> float:
    value = 0.0
    for split_name in SPLIT_NAMES:
        target = max(float(targets[split_name]), 1.0)
        relative_error = (float(counts[split_name]) - target) / target
        value += relative_error * relative_error
    return value


def _assign_whole_sources(
    class_sources: pd.DataFrame,
    targets: dict[str, float],
    require_each_split: bool,
) -> dict[str, str]:
    """
    Deterministic row-aware assignment of whole SourceFiles for one class.

    The algorithm uses a greedy initialization followed by local single-source
    moves and pairwise swaps. It never splits rows.
    """
    if class_sources.empty:
        return {}

    sources = class_sources.sort_values(
        ["Rows", SOURCE_FILE_COL],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    if require_each_split and len(sources) < 3:
        raise SplitConstructionError(
            "At least three SourceFiles are required for this class to keep "
            "train/validation/test class representation while preserving whole "
            "files."
        )

    assignment: dict[str, str] = {}
    counts = {name: 0.0 for name in SPLIT_NAMES}
    members = {name: [] for name in SPLIT_NAMES}

    # Seed all three bins deterministically when class representation is needed.
    start_index = 0
    if require_each_split and len(sources) >= 3:
        seed_order = ("train", "validation", "test")
        for idx, split_name in enumerate(seed_order):
            source = str(sources.loc[idx, SOURCE_FILE_COL])
            rows = float(sources.loc[idx, "Rows"])
            assignment[source] = split_name
            counts[split_name] += rows
            members[split_name].append(source)
        start_index = 3

    row_lookup = {
        str(row[SOURCE_FILE_COL]): float(row["Rows"])
        for _, row in sources.iterrows()
    }

    # Greedy placement of remaining whole sources.
    for idx in range(start_index, len(sources)):
        source = str(sources.loc[idx, SOURCE_FILE_COL])
        rows = float(sources.loc[idx, "Rows"])

        best_split = None
        best_score = None

        for split_name in SPLIT_NAMES:
            trial = dict(counts)
            trial[split_name] += rows
            score = _objective(trial, targets)

            # Stable tie-break: train, then validation, then test.
            tie_rank = SPLIT_NAMES.index(split_name)
            candidate = (score, tie_rank)

            if best_score is None or candidate < best_score:
                best_score = candidate
                best_split = split_name

        assignment[source] = str(best_split)
        counts[str(best_split)] += rows
        members[str(best_split)].append(source)

    # Local improvement with whole-source moves and swaps only.
    for _ in range(100):
        current_score = _objective(counts, targets)
        best_action = None
        best_score = current_score

        # Single-source moves.
        for source, current_split in list(assignment.items()):
            if require_each_split and len(members[current_split]) <= 1:
                continue

            rows = row_lookup[source]
            for new_split in SPLIT_NAMES:
                if new_split == current_split:
                    continue

                trial = dict(counts)
                trial[current_split] -= rows
                trial[new_split] += rows
                score = _objective(trial, targets)

                if score + 1e-12 < best_score:
                    best_score = score
                    best_action = ("move", source, current_split, new_split)

        # Pairwise swaps can fix coarse greedy placements.
        source_items = list(assignment.items())
        for i in range(len(source_items)):
            source_a, split_a = source_items[i]
            rows_a = row_lookup[source_a]

            for j in range(i + 1, len(source_items)):
                source_b, split_b = source_items[j]
                if split_a == split_b:
                    continue

                rows_b = row_lookup[source_b]
                trial = dict(counts)
                trial[split_a] += rows_b - rows_a
                trial[split_b] += rows_a - rows_b
                score = _objective(trial, targets)

                if score + 1e-12 < best_score:
                    best_score = score
                    best_action = (
                        "swap",
                        source_a,
                        split_a,
                        source_b,
                        split_b,
                    )

        if best_action is None:
            break

        if best_action[0] == "move":
            _, source, old_split, new_split = best_action
            rows = row_lookup[source]
            assignment[source] = new_split
            members[old_split].remove(source)
            members[new_split].append(source)
            counts[old_split] -= rows
            counts[new_split] += rows
        else:
            _, source_a, split_a, source_b, split_b = best_action
            rows_a = row_lookup[source_a]
            rows_b = row_lookup[source_b]

            assignment[source_a] = split_b
            assignment[source_b] = split_a

            members[split_a].remove(source_a)
            members[split_a].append(source_b)
            members[split_b].remove(source_b)
            members[split_b].append(source_a)

            counts[split_a] += rows_b - rows_a
            counts[split_b] += rows_a - rows_b

    return assignment


def _whole_source_assignment(source_table: pd.DataFrame) -> dict[str, str]:
    assignment: dict[str, str] = {}

    for label in (0, 1):
        class_sources = source_table[source_table["Label"] == label].copy()
        total_rows = float(class_sources["Rows"].sum())
        targets = {
            split_name: TARGET_RATIOS[split_name] * total_rows
            for split_name in SPLIT_NAMES
        }

        class_assignment = _assign_whole_sources(
            class_sources=class_sources,
            targets=targets,
            require_each_split=True,
        )
        assignment.update(class_assignment)

    return assignment


def _assignment_row_counts(
    df: pd.DataFrame,
    assignment: dict[str, str],
) -> dict[str, int]:
    split_series = df[SOURCE_FILE_COL].map(assignment)
    if split_series.isna().any():
        missing_sources = sorted(
            df.loc[split_series.isna(), SOURCE_FILE_COL].astype(str).unique()
        )
        raise SplitConstructionError(
            f"Missing SourceFile assignments: {missing_sources}"
        )

    return {
        split_name: int((split_series == split_name).sum())
        for split_name in SPLIT_NAMES
    }


def _class_presence_from_assignment(
    source_table: pd.DataFrame,
    assignment: dict[str, str],
) -> bool:
    for split_name in SPLIT_NAMES:
        labels = set(
            source_table.loc[
                source_table[SOURCE_FILE_COL].map(assignment) == split_name,
                "Label",
            ].astype(int)
        )
        if labels != {0, 1}:
            return False
    return True


def _ratios_reasonable(
    row_counts: dict[str, int],
    total_rows: int,
    max_deviation: float,
) -> bool:
    for split_name in SPLIT_NAMES:
        observed = float(row_counts[split_name]) / float(total_rows)
        target = TARGET_RATIOS[split_name]
        if abs(observed - target) > max_deviation:
            return False
    return True


def _split_oversized_source_chronologically(
    oversized_df: pd.DataFrame,
) -> dict[int, str]:
    """
    Split ONLY the oversized SourceFile:
      earliest 70% -> train
      next 15%     -> validation
      latest 15%   -> test
    """
    parsed = pd.to_datetime(
        oversized_df[TIMESTAMP_COL],
        errors="coerce",
        utc=True,
    )

    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise SplitConstructionError(
            "Cannot chronologically split the oversized SourceFile because "
            f"{bad_count} Timestamp values could not be parsed."
        )

    ordered = oversized_df.assign(__ParsedTimestamp=parsed).sort_values(
        ["__ParsedTimestamp", ORIGINAL_ROW_COL],
        kind="mergesort",
    )

    n = len(ordered)
    train_end = int(np.floor(TARGET_RATIOS["train"] * n))
    validation_end = int(
        np.floor((TARGET_RATIOS["train"] + TARGET_RATIOS["validation"]) * n)
    )

    # Defensive minimums for very small sources. In normal use the source is
    # oversized, so this is only a guardrail.
    train_end = min(max(train_end, 1), n - 2)
    validation_end = min(max(validation_end, train_end + 1), n - 1)

    result: dict[int, str] = {}

    for row_index in ordered.iloc[:train_end][ORIGINAL_ROW_COL].astype(int):
        result[int(row_index)] = "train"

    for row_index in ordered.iloc[train_end:validation_end][ORIGINAL_ROW_COL].astype(int):
        result[int(row_index)] = "validation"

    for row_index in ordered.iloc[validation_end:][ORIGINAL_ROW_COL].astype(int):
        result[int(row_index)] = "test"

    return result


def _hybrid_assignment(
    working_df: pd.DataFrame,
    source_table: pd.DataFrame,
    oversized_source: str,
) -> tuple[pd.Series, dict]:
    oversized_mask = working_df[SOURCE_FILE_COL].astype(str) == str(oversized_source)
    oversized_df = working_df.loc[oversized_mask].copy()
    remaining_df = working_df.loc[~oversized_mask].copy()
    remaining_sources = source_table[
        source_table[SOURCE_FILE_COL].astype(str) != str(oversized_source)
    ].copy()

    row_split_map = _split_oversized_source_chronologically(oversized_df)

    oversized_label = int(oversized_df[LABEL_COL].iloc[0])

    whole_assignment: dict[str, str] = {}

    for label in (0, 1):
        class_sources = remaining_sources[remaining_sources["Label"] == label].copy()

        if class_sources.empty:
            if label != oversized_label:
                raise SplitConstructionError(
                    "The non-oversized class has no remaining SourceFiles. "
                    "Keeping all smaller SourceFiles intact cannot provide both "
                    "classes in train/validation/test."
                )
            continue

        total_rows = float(class_sources["Rows"].sum())
        targets = {
            split_name: TARGET_RATIOS[split_name] * total_rows
            for split_name in SPLIT_NAMES
        }

        # The oversized source itself already contributes its class to all three
        # splits. Other classes still need at least one whole source per split.
        require_each_split = label != oversized_label

        class_assignment = _assign_whole_sources(
            class_sources=class_sources,
            targets=targets,
            require_each_split=require_each_split,
        )
        whole_assignment.update(class_assignment)

    split_series = pd.Series(index=working_df.index, dtype="object")

    for source, split_name in whole_assignment.items():
        mask = working_df[SOURCE_FILE_COL].astype(str) == str(source)
        split_series.loc[mask] = split_name

    for original_row, split_name in row_split_map.items():
        split_series.loc[
            working_df[ORIGINAL_ROW_COL].astype(int) == int(original_row)
        ] = split_name

    if split_series.isna().any():
        missing_count = int(split_series.isna().sum())
        raise SplitConstructionError(
            f"Hybrid splitter left {missing_count} rows unassigned."
        )

    metadata = {
        "mode": "hybrid_oversized_source_chronological",
        "oversized_source": str(oversized_source),
        "oversized_rows": int(len(oversized_df)),
        "oversized_fraction": float(len(oversized_df) / len(working_df)),
        "oversized_label": int(oversized_label),
    }

    return split_series, metadata


def construct_source_aware_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Build one deterministic source-aware 70/15/15 holdout.

    Priority:
      1. whole SourceFile assignment;
      2. if unreasonable because one source dominates, split only that source
         chronologically 70/15/15;
      3. all other SourceFiles remain intact;
      4. never random row-level splitting.
    """
    _validate_input(df)

    working_df = df.copy().reset_index(drop=True)
    working_df[ORIGINAL_ROW_COL] = np.arange(len(working_df), dtype=np.int64)

    source_table = _source_table(working_df)

    try:
        whole_assignment = _whole_source_assignment(source_table)
        whole_counts = _assignment_row_counts(working_df, whole_assignment)
        whole_reasonable = (
            _ratios_reasonable(
                row_counts=whole_counts,
                total_rows=len(working_df),
                max_deviation=MAX_TOTAL_RATIO_DEVIATION,
            )
            and _class_presence_from_assignment(source_table, whole_assignment)
        )
    except SplitConstructionError:
        whole_assignment = {}
        whole_counts = {split_name: 0 for split_name in SPLIT_NAMES}
        whole_reasonable = False

    if whole_reasonable:
        split_series = working_df[SOURCE_FILE_COL].map(whole_assignment)
        metadata = {
            "mode": "whole_sourcefile",
            "oversized_source": None,
            "oversized_rows": 0,
            "oversized_fraction": 0.0,
        }
    else:
        largest = source_table.sort_values(
            ["Rows", SOURCE_FILE_COL],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]

        oversized_source = str(largest[SOURCE_FILE_COL])
        oversized_rows = int(largest["Rows"])
        oversized_fraction = oversized_rows / len(working_df)

        if oversized_fraction < OVERSIZED_SOURCE_MIN_FRACTION:
            raise SplitConstructionError(
                "Whole-SourceFile assignment is outside the allowed 70/15/15 "
                "tolerance, but no SourceFile is large enough to justify the "
                "chronological exception. A source must exceed "
                f"{OVERSIZED_SOURCE_MIN_FRACTION:.2%} of dataset rows "
                "(15% holdout target + 5 percentage-point tolerance). "
                "Largest source is "
                f"{oversized_source!r} with {oversized_rows:,} rows "
                f"({oversized_fraction:.2%})."
            )

        split_series, metadata = _hybrid_assignment(
            working_df=working_df,
            source_table=source_table,
            oversized_source=oversized_source,
        )
        metadata["oversized_trigger_fraction"] = float(
            OVERSIZED_SOURCE_MIN_FRACTION
        )
        metadata["oversized_trigger_reason"] = (
            "whole-SourceFile assignment outside tolerance and largest source "
            "exceeds validation/test target plus allowed tolerance"
        )

    working_df[SPLIT_COL] = split_series.astype(str)

    split_counts = {
        split_name: int((working_df[SPLIT_COL] == split_name).sum())
        for split_name in SPLIT_NAMES
    }

    if not _ratios_reasonable(
        row_counts=split_counts,
        total_rows=len(working_df),
        max_deviation=(
            MAX_TOTAL_RATIO_DEVIATION
            if metadata["mode"] == "whole_sourcefile"
            else MAX_HYBRID_RATIO_DEVIATION
        ),
    ):
        raise SplitConstructionError(
            "Final source-aware split is still too far from 70/15/15. "
            f"Row counts: {split_counts}."
        )

    # Both classes must be present in every split.
    for split_name in SPLIT_NAMES:
        labels = set(
            working_df.loc[
                working_df[SPLIT_COL] == split_name,
                LABEL_COL,
            ].astype(int)
        )
        if labels != {0, 1}:
            raise SplitConstructionError(
                f"Final {split_name} split does not contain both classes."
            )

    # Verify that no smaller source crosses partitions.
    oversized_source = metadata.get("oversized_source")
    for source, group in working_df.groupby(SOURCE_FILE_COL, sort=False):
        split_count = int(group[SPLIT_COL].nunique())
        if oversized_source is not None and str(source) == str(oversized_source):
            if split_count != 3:
                raise SplitConstructionError(
                    "Oversized SourceFile must contribute chronologically to all "
                    "three splits."
                )
        elif split_count != 1:
            raise SplitConstructionError(
                f"Non-oversized SourceFile {source!r} crosses split boundaries."
            )

    # Verify chronological oversized-source boundaries.
    if oversized_source is not None:
        oversized = working_df[
            working_df[SOURCE_FILE_COL].astype(str) == str(oversized_source)
        ].copy()
        oversized["__ParsedTimestamp"] = pd.to_datetime(
            oversized[TIMESTAMP_COL],
            errors="raise",
            utc=True,
        )
        ordered = oversized.sort_values(
            ["__ParsedTimestamp", ORIGINAL_ROW_COL],
            kind="mergesort",
        )
        split_order = ordered[SPLIT_COL].map(
            {"train": 0, "validation": 1, "test": 2}
        ).to_numpy()
        if np.any(np.diff(split_order) < 0):
            raise SplitConstructionError(
                "Oversized SourceFile chronological split ordering is invalid."
            )

    metadata.update(
        {
            "target_ratios": TARGET_RATIOS,
            "row_counts": split_counts,
            "row_ratios": {
                split_name: float(split_counts[split_name] / len(working_df))
                for split_name in SPLIT_NAMES
            },
            "total_rows": int(len(working_df)),
            "source_files": int(source_table[SOURCE_FILE_COL].nunique()),
            "whole_source_attempt_row_counts": whole_counts,
        }
    )

    manifest_columns = [
        ORIGINAL_ROW_COL,
        SOURCE_FILE_COL,
        LABEL_COL,
        TIMESTAMP_COL,
        SPLIT_COL,
    ]

    manifest = working_df[manifest_columns].copy()
    manifest["AssignmentMode"] = metadata["mode"]
    manifest["OversizedSource"] = metadata.get("oversized_source")

    return manifest, metadata


def apply_split_manifest(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Apply a previously saved row-level manifest to an unchanged ingested CSV.
    """
    if len(df) != len(manifest):
        raise SplitConstructionError(
            "Split manifest row count does not match the ingested dataset. "
            f"Dataset={len(df):,}; manifest={len(manifest):,}."
        )

    required = {
        ORIGINAL_ROW_COL,
        SOURCE_FILE_COL,
        LABEL_COL,
        SPLIT_COL,
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Split manifest is missing columns: {sorted(missing)}"
        )

    ordered_manifest = manifest.sort_values(ORIGINAL_ROW_COL).reset_index(drop=True)
    expected_indices = np.arange(len(df), dtype=np.int64)

    if not np.array_equal(
        ordered_manifest[ORIGINAL_ROW_COL].astype(np.int64).to_numpy(),
        expected_indices,
    ):
        raise SplitConstructionError(
            "Split manifest OriginalRowIndex is not a complete 0..N-1 sequence."
        )

    # Validate identity-defining columns before trusting the manifest.
    source_matches = (
        df[SOURCE_FILE_COL].astype(str).reset_index(drop=True)
        == ordered_manifest[SOURCE_FILE_COL].astype(str)
    )
    label_matches = (
        df[LABEL_COL].astype(int).reset_index(drop=True)
        == ordered_manifest[LABEL_COL].astype(int)
    )

    if not bool(source_matches.all()) or not bool(label_matches.all()):
        raise SplitConstructionError(
            "The ingested dataset differs from the data used to create the "
            "saved split manifest. Refusing to apply a stale split."
        )

    working = df.copy().reset_index(drop=True)
    working[ORIGINAL_ROW_COL] = expected_indices
    working[SPLIT_COL] = ordered_manifest[SPLIT_COL].astype(str).to_numpy()

    result = {
        split_name: (
            working[working[SPLIT_COL] == split_name]
            .drop(columns=[SPLIT_COL])
            .copy()
        )
        for split_name in SPLIT_NAMES
    }

    return result


def get_or_create_shared_split(
    dataset_name: str,
    df: pd.DataFrame,
    split_root: Path,
    force_rebuild: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """
    Persist the split once so Random Forest, Isolation Forest, Autoencoder and
    LSTM can reuse exactly the same train/validation/test rows.
    """
    split_root = Path(split_root)
    split_root.mkdir(parents=True, exist_ok=True)

    manifest_path = split_root / f"{dataset_name}_source_aware_split.csv"
    metadata_path = split_root / f"{dataset_name}_source_aware_split.json"

    if manifest_path.exists() and metadata_path.exists() and not force_rebuild:
        manifest = pd.read_csv(manifest_path, low_memory=False)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        splits = apply_split_manifest(df, manifest)
        return splits, manifest, metadata

    manifest, metadata = construct_source_aware_split(df)

    manifest.to_csv(manifest_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    splits = apply_split_manifest(df, manifest)
    return splits, manifest, metadata