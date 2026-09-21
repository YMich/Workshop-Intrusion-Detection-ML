from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

# CPU-oriented TensorFlow defaults. These are set before TensorFlow import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Quiet TensorFlow's repeated informational output.
tf.get_logger().setLevel("ERROR")

try:
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    # TensorFlow may already be initialized when this module is imported.
    pass


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"

# Final same-hyperparameter benchmark: Dataset 2 + Dataset 3 only.
DATASET_PATHS = {
    "dataset2": INGESTED_ROOT / "dataset2_ingested.csv",
    "dataset3": INGESTED_ROOT / "dataset3_ingested.csv",
}
BENCHMARK_DATASETS = tuple(DATASET_PATHS.keys())

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models_baseline" / "AE"
RESULT_ROOT = PROJECT_ROOT / "results" / "models_baseline" / "AE"
SHARED_SPLIT_ROOT = PROJECT_ROOT / "artifacts" / "splits"
SHARED_CONFIG_PATH = ARTIFACT_ROOT / "shared_configuration.json"


# ============================================================
# REUSE STEP 2 + STEP 3
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent

for path in [
    MODEL_DIR,
    SRC_ROOT / "preprocessing",
    SRC_ROOT / "feature_engineering",
]:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from preprocessing import DatasetPreprocessor  # noqa: E402
from feature_selection import select_and_engineer_dataset  # noqa: E402
from source_aware_split import (  # noqa: E402
    ORIGINAL_ROW_COL,
    get_or_create_shared_split,
)


# ============================================================
# COLUMNS / LABELS
# ============================================================

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1


# ============================================================
# HYBRID SOURCEFILE / CHRONOLOGICAL REPEATED HOLDOUT
# ============================================================

# The SAME split policy is applied to Dataset 2 and Dataset 3.
#
# 1) Prefer a pure whole-SourceFile 70/15/15 split.
# 2) If no reasonable grouped split exists because one or more SourceFiles are
#    too large, split ONLY those oversized files chronologically.
# 3) Smaller SourceFiles remain indivisible.
# 4) No random row-level split is ever used.
REPEATED_HOLDOUTS = 3
SPLIT_RANDOM_STATE = 42
CANDIDATE_SPLITS = 5000

TARGET_TRAIN_FRACTION = 0.70
TARGET_VALIDATION_FRACTION = 0.15
TARGET_TEST_FRACTION = 0.15

# A pure grouped split is considered reasonable only when validation and test
# are not tiny and training remains large enough. If this cannot be achieved,
# the oversized-source chronological fallback is activated.
MIN_REASONABLE_TRAIN_FRACTION = 0.55
MAX_REASONABLE_TRAIN_FRACTION = 0.85
MIN_REASONABLE_EVAL_FRACTION = 0.08
WHOLE_SOURCE_MAX_OBJECTIVE = 0.42

# Autoencoder-specific minimum benign training support.
MIN_TRAIN_BENIGN_ABSOLUTE = 10000
MIN_TRAIN_BENIGN_FRACTION = 0.40

# Sources at or above this share of all rows are eligible for chronological
# splitting when a whole-group split is not reasonable. If none meet this
# threshold but grouped splitting still fails, the largest source is used as
# the fallback oversized source.
OVERSIZED_SOURCE_FRACTION = 0.20

# All repeated holdouts use the SAME target protocol: 70/15/15.
# If an oversized SourceFile must be split, it is always split chronologically
# with these same fractions: earliest -> train, middle -> validation, latest
# -> test. We do not change the experimental ratio merely to force different
# repeats.
CHRONOLOGICAL_FRACTIONS = (
    TARGET_TRAIN_FRACTION,
    TARGET_VALIDATION_FRACTION,
    TARGET_TEST_FRACTION,
)

# Repeat diversity is SOFT, not a hard uniqueness constraint. The same evolving
# RNG is used across the three searches, and candidate assignments that reuse
# intact validation/test SourceFiles from earlier repeats receive a penalty.
# This encourages different holdouts while preserving the primary goals of
# sensible row balance, class support, and valid training size.
# Chronologically split oversized SourceFiles necessarily overlap by design.
TEST_OVERLAP_PENALTY = 0.20
VALIDATION_OVERLAP_PENALTY = 0.10

ROW_INDEX_COL = "__AE_RowIndex"
SPLIT_MODE_COL = "SplitMode"
WHOLE_SOURCE_MODE = "whole_source"
CHRONOLOGICAL_MODE = "chronological_oversized_source"


# ============================================================
# FIXED SHARED AUTOENCODER CONFIGURATION
# ============================================================

# Frozen baseline configuration used identically for Dataset 2 and Dataset 3.
# No hyperparameter search, grid search, sensitivity sweep, or stability-based
# model selection is performed by the submitted baseline code.
BASE_PARAMS = {
    "encoder_hidden_dims": (64, 32),
    "bottleneck_dim": 8,
    "hidden_activation": "relu",
    "output_activation": "linear",
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "use_bias": True,
    "l2_regularization": 1e-5,
    "learning_rate": 2e-3,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-7,
    "adam_amsgrad": False,
    "loss": "huber",
    "huber_delta": 1.0,
    "batch_size": 256,
    "epochs": 20,
    "shuffle": True,
    "fit_verbose": 0,
    "internal_validation_fraction": 0.10,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-5,
    "denoising_noise_std": 0.02,
    "steps_per_execution": 16,
}

SHARED_SCORE_MODE = "top5_mse"
FINAL_RANDOM_STATE = 42

# Per-dataset operating threshold calibration remains validation-only.
FINAL_THRESHOLD_FPR_CANDIDATES = [0.01, 0.02, 0.05]
FINAL_THRESHOLD_BENIGN_QUANTILES = [0.95, 0.975, 0.98, 0.99, 0.995]
THRESHOLD_CALIBRATION_VERSION = "validation_f1_primary_v2"

# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ============================================================
# HYBRID SPLIT SEARCH
# ============================================================


def build_sourcefile_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {SOURCE_FILE_COL, LABEL_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns required for splitting: "
            f"{sorted(missing)}"
        )

    groups = (
        df.groupby(SOURCE_FILE_COL, sort=True)
        .agg(
            Label=(LABEL_COL, "first"),
            Label_Nunique=(LABEL_COL, "nunique"),
            Rows=(LABEL_COL, "size"),
        )
        .reset_index()
    )

    mixed = groups[groups["Label_Nunique"] != 1]
    if not mixed.empty:
        raise ValueError(
            "At least one SourceFile contains both labels:\n"
            + mixed.to_string(index=False)
        )

    groups = groups.drop(columns=["Label_Nunique"])
    groups["Label"] = groups["Label"].astype(int)

    if set(groups["Label"].unique()) != {BENIGN_LABEL, MALICIOUS_LABEL}:
        raise ValueError("Both benign and malicious SourceFiles are required.")

    return groups


def _random_group_assignment(
    groups: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, set[str]]:
    """Random whole-source assignment used only during candidate search."""
    result = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    for _, row in groups.iterrows():
        split_name = str(
            rng.choice(
                ["train", "validation", "test"],
                p=[
                    TARGET_TRAIN_FRACTION,
                    TARGET_VALIDATION_FRACTION,
                    TARGET_TEST_FRACTION,
                ],
            )
        )
        result[split_name].add(str(row[SOURCE_FILE_COL]))

    return result


def _whole_assignment_stats(
    groups: pd.DataFrame,
    assignment: dict[str, set[str]],
) -> dict:
    total_rows = int(groups["Rows"].sum())
    stats = {}

    for split_name, sources in assignment.items():
        part = groups[groups[SOURCE_FILE_COL].isin(sources)]
        benign_rows = int(
            part.loc[part["Label"] == BENIGN_LABEL, "Rows"].sum()
        )
        malicious_rows = int(
            part.loc[part["Label"] == MALICIOUS_LABEL, "Rows"].sum()
        )
        rows = benign_rows + malicious_rows
        stats[split_name] = {
            "Rows": rows,
            "RowFraction": rows / total_rows if total_rows else 0.0,
            "BenignRows": benign_rows,
            "MaliciousRows": malicious_rows,
            "SourceFiles": len(sources),
        }

    return stats


def _min_benign_training_rows(groups: pd.DataFrame) -> int:
    benign_total = int(
        groups.loc[groups["Label"] == BENIGN_LABEL, "Rows"].sum()
    )
    return min(
        benign_total - 1,
        max(
            MIN_TRAIN_BENIGN_ABSOLUTE,
            int(round(MIN_TRAIN_BENIGN_FRACTION * benign_total)),
        ),
    )


def _whole_assignment_valid(
    stats: dict,
    groups: pd.DataFrame,
) -> bool:
    if stats["train"]["BenignRows"] < _min_benign_training_rows(groups):
        return False

    for split_name in ["train", "validation", "test"]:
        if stats[split_name]["BenignRows"] <= 0:
            return False
        if stats[split_name]["MaliciousRows"] <= 0:
            return False

    if not (
        MIN_REASONABLE_TRAIN_FRACTION
        <= stats["train"]["RowFraction"]
        <= MAX_REASONABLE_TRAIN_FRACTION
    ):
        return False

    for split_name in ["validation", "test"]:
        if stats[split_name]["RowFraction"] < MIN_REASONABLE_EVAL_FRACTION:
            return False

    return True


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _whole_assignment_objective(
    stats: dict,
    assignment: dict[str, set[str]],
    previous_assignments: list[dict[str, set[str]]],
    groups: pd.DataFrame,
) -> float:
    row_error = (
        abs(stats["train"]["RowFraction"] - TARGET_TRAIN_FRACTION)
        + 1.25
        * abs(
            stats["validation"]["RowFraction"]
            - TARGET_VALIDATION_FRACTION
        )
        + 1.25
        * abs(stats["test"]["RowFraction"] - TARGET_TEST_FRACTION)
    )

    class_error = 0.0
    for label_value, column in [
        (BENIGN_LABEL, "BenignRows"),
        (MALICIOUS_LABEL, "MaliciousRows"),
    ]:
        total = max(
            1,
            int(groups.loc[groups["Label"] == label_value, "Rows"].sum()),
        )
        class_error += abs(
            stats["validation"][column] / total
            - TARGET_VALIDATION_FRACTION
        )
        class_error += abs(
            stats["test"][column] / total
            - TARGET_TEST_FRACTION
        )

    overlap_penalty = 0.0
    for previous in previous_assignments:
        overlap_penalty += TEST_OVERLAP_PENALTY * _jaccard(
            assignment["test"], previous["test"]
        )
        overlap_penalty += VALIDATION_OVERLAP_PENALTY * _jaccard(
            assignment["validation"], previous["validation"]
        )

    return row_error + 0.35 * class_error + overlap_penalty


def _search_reasonable_whole_source_assignment(
    groups: pd.DataFrame,
    rng: np.random.Generator,
    previous_assignments: list[dict[str, set[str]]],
) -> dict | None:
    best = None

    for _ in range(CANDIDATE_SPLITS):
        assignment = _random_group_assignment(groups, rng)
        stats = _whole_assignment_stats(groups, assignment)

        if not _whole_assignment_valid(stats, groups):
            continue

        objective = _whole_assignment_objective(
            stats,
            assignment,
            previous_assignments,
            groups,
        )

        if best is None or objective < best["objective"]:
            best = {
                "assignment": assignment,
                "stats": stats,
                "objective": float(objective),
            }

    if best is None:
        return None

    if best["objective"] > WHOLE_SOURCE_MAX_OBJECTIVE:
        return None

    return best


def _chronological_row_ids_for_source(
    df: pd.DataFrame,
    source: str,
    fractions: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    source_mask = df[SOURCE_FILE_COL].astype(str) == str(source)
    source_positions = np.flatnonzero(source_mask.to_numpy())

    if len(source_positions) < 3:
        raise ValueError(
            f"Oversized SourceFile {source!r} has fewer than 3 rows."
        )

    if "Timestamp" in df.columns:
        timestamps = pd.to_datetime(
            df.iloc[source_positions]["Timestamp"],
            errors="coerce",
        )
        if timestamps.isna().any():
            raise ValueError(
                f"Oversized SourceFile {source!r} contains unparsable "
                "timestamps; chronological splitting is unsafe."
            )
        ordering = np.lexsort(
            (source_positions, timestamps.astype("int64").to_numpy())
        )
        ordered_positions = source_positions[ordering]
    else:
        # Ingestion normally guarantees Timestamp. Original row order is only
        # used as a last-resort chronological proxy if the column is absent.
        ordered_positions = source_positions

    train_fraction, validation_fraction, _test_fraction = fractions
    n = len(ordered_positions)
    train_end = int(np.floor(n * train_fraction))
    validation_end = int(
        np.floor(n * (train_fraction + validation_fraction))
    )

    # Keep every temporal segment non-empty.
    train_end = min(max(train_end, 1), n - 2)
    validation_end = min(
        max(validation_end, train_end + 1),
        n - 1,
    )

    return {
        "train": ordered_positions[:train_end],
        "validation": ordered_positions[train_end:validation_end],
        "test": ordered_positions[validation_end:],
    }


def _hybrid_candidate_from_small_sources(
    df: pd.DataFrame,
    groups: pd.DataFrame,
    oversized_sources: set[str],
    fractions: tuple[float, float, float],
    rng: np.random.Generator,
    previous_small_assignments: list[dict[str, set[str]]],
) -> dict:
    """
    Optimize intact small-source placement using only the compact group table.
    Full dataframe row-index materialization happens ONCE after the best
    assignment is chosen, which keeps split generation fast.
    """
    total_rows = len(df)
    target_rows = {
        "train": TARGET_TRAIN_FRACTION * total_rows,
        "validation": TARGET_VALIDATION_FRACTION * total_rows,
        "test": TARGET_TEST_FRACTION * total_rows,
    }

    oversized_groups = groups[
        groups[SOURCE_FILE_COL].astype(str).isin(oversized_sources)
    ].copy()
    small_groups = groups[
        ~groups[SOURCE_FILE_COL].astype(str).isin(oversized_sources)
    ].copy()

    # Fixed chronological contribution from oversized sources, represented by
    # counts only during candidate search.
    fixed_stats = {
        split_name: {
            "Rows": 0,
            "BenignRows": 0,
            "MaliciousRows": 0,
        }
        for split_name in ["train", "validation", "test"]
    }

    for _, row in oversized_groups.iterrows():
        n = int(row["Rows"])
        train_fraction, validation_fraction, _ = fractions
        train_end = min(max(int(np.floor(n * train_fraction)), 1), n - 2)
        validation_end = min(
            max(
                int(np.floor(n * (train_fraction + validation_fraction))),
                train_end + 1,
            ),
            n - 1,
        )
        counts = {
            "train": train_end,
            "validation": validation_end - train_end,
            "test": n - validation_end,
        }
        label_column = (
            "BenignRows"
            if int(row["Label"]) == BENIGN_LABEL
            else "MaliciousRows"
        )
        for split_name, count in counts.items():
            fixed_stats[split_name]["Rows"] += count
            fixed_stats[split_name][label_column] += count

    best = None
    iterations = max(2000, CANDIDATE_SPLITS)

    # Converting to records once avoids pandas row construction in the hot loop.
    small_records = small_groups[
        [SOURCE_FILE_COL, "Label", "Rows"]
    ].to_dict(orient="records")

    for _ in range(iterations):
        assignment = {
            "train": set(),
            "validation": set(),
            "test": set(),
        }
        stats = {
            split_name: dict(values)
            for split_name, values in fixed_stats.items()
        }

        for row in small_records:
            split_name = str(
                rng.choice(
                    ["train", "validation", "test"],
                    p=[
                        TARGET_TRAIN_FRACTION,
                        TARGET_VALIDATION_FRACTION,
                        TARGET_TEST_FRACTION,
                    ],
                )
            )
            source = str(row[SOURCE_FILE_COL])
            label = int(row["Label"])
            count = int(row["Rows"])

            assignment[split_name].add(source)
            stats[split_name]["Rows"] += count
            if label == BENIGN_LABEL:
                stats[split_name]["BenignRows"] += count
            else:
                stats[split_name]["MaliciousRows"] += count

        valid = True
        objective = 0.0

        for split_name in ["train", "validation", "test"]:
            rows = stats[split_name]["Rows"]
            stats[split_name]["RowFraction"] = rows / max(1, total_rows)
            stats[split_name]["WholeSourceFiles"] = len(
                assignment[split_name]
            )

            if stats[split_name]["BenignRows"] <= 0:
                valid = False
            if stats[split_name]["MaliciousRows"] <= 0:
                valid = False

            objective += abs(rows - target_rows[split_name]) / max(1, total_rows)

        if not valid:
            continue

        if stats["train"]["BenignRows"] < _min_benign_training_rows(groups):
            continue

        if stats["validation"]["RowFraction"] < MIN_REASONABLE_EVAL_FRACTION:
            continue
        if stats["test"]["RowFraction"] < MIN_REASONABLE_EVAL_FRACTION:
            continue

        for previous in previous_small_assignments:
            objective += TEST_OVERLAP_PENALTY * _jaccard(
                assignment["test"], previous["test"]
            )
            objective += VALIDATION_OVERLAP_PENALTY * _jaccard(
                assignment["validation"], previous["validation"]
            )

        if best is None or objective < best["objective"]:
            best = {
                "assignment": {
                    key: set(value)
                    for key, value in assignment.items()
                },
                "stats": {
                    key: dict(value)
                    for key, value in stats.items()
                },
                "objective": float(objective),
            }

    if best is None:
        raise RuntimeError(
            "Could not build a valid hybrid split even after chronological "
            "splitting of oversized SourceFiles."
        )

    # Materialize exact row IDs only once for the winning assignment.
    row_ids = {"train": [], "validation": [], "test": []}

    for source in sorted(oversized_sources):
        pieces = _chronological_row_ids_for_source(
            df,
            source,
            fractions,
        )
        for split_name in row_ids:
            row_ids[split_name].extend(pieces[split_name].tolist())

    source_series = df[SOURCE_FILE_COL].astype(str)
    for split_name, sources in best["assignment"].items():
        if sources:
            positions = np.flatnonzero(
                source_series.isin(sources).to_numpy()
            )
            row_ids[split_name].extend(positions.tolist())

    best["row_ids"] = {
        key: np.asarray(sorted(set(value)), dtype=int)
        for key, value in row_ids.items()
    }
    return best


def _build_row_manifest(
    df: pd.DataFrame,
    repeat_index: int,
    row_ids: dict[str, np.ndarray],
    oversized_sources: set[str],
) -> pd.DataFrame:
    rows = []

    for split_name in ["train", "validation", "test"]:
        ids = np.asarray(row_ids[split_name], dtype=int)
        part = df.iloc[ids]

        frame = pd.DataFrame(
            {
                "Repeat": repeat_index,
                ROW_INDEX_COL: ids,
                SOURCE_FILE_COL: part[SOURCE_FILE_COL].astype(str).to_numpy(),
                "Label": part[LABEL_COL].astype(int).to_numpy(),
                "Split": split_name,
            }
        )

        frame[SPLIT_MODE_COL] = np.where(
            frame[SOURCE_FILE_COL].isin(oversized_sources),
            CHRONOLOGICAL_MODE,
            WHOLE_SOURCE_MODE,
        )

        if "Timestamp" in part.columns:
            frame["Timestamp"] = part["Timestamp"].astype(str).to_numpy()

        rows.append(frame)

    manifest = pd.concat(rows, ignore_index=True)

    if manifest[ROW_INDEX_COL].duplicated().any():
        raise RuntimeError(
            f"Repeat {repeat_index}: at least one row appears in more than "
            "one split."
        )

    if len(manifest) != len(df):
        missing = len(df) - len(manifest)
        raise RuntimeError(
            f"Repeat {repeat_index}: row coverage mismatch ({missing} rows)."
        )

    return manifest.sort_values(ROW_INDEX_COL).reset_index(drop=True)


def _frames_from_row_ids(
    df: pd.DataFrame,
    row_ids: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    return {
        split_name: df.iloc[np.asarray(ids, dtype=int)].copy()
        for split_name, ids in row_ids.items()
    }


def make_hybrid_repeated_holdouts(
    df: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Build three repeated 70/15/15 holdouts using one consistent policy.

    Prefer whole SourceFiles; fall back to chronological row splitting only
    for oversized SourceFiles when grouped splitting cannot produce a useful
    train/validation/test experiment. Repeats are encouraged (not forced) to
    differ through an evolving RNG plus soft validation/test SourceFile overlap
    penalties.
    """
    groups = build_sourcefile_table(df)
    rng = np.random.default_rng(SPLIT_RANDOM_STATE)

    total_rows = int(len(df))
    groups = groups.copy()
    groups["RowFraction"] = groups["Rows"] / max(1, total_rows)

    oversized_sources = set(
        groups.loc[
            groups["RowFraction"] >= OVERSIZED_SOURCE_FRACTION,
            SOURCE_FILE_COL,
        ].astype(str)
    )

    selected = []
    diagnostics = []
    previous_whole_assignments = []
    previous_small_assignments = []

    # First determine whether a genuinely useful pure grouped split exists.
    probe = _search_reasonable_whole_source_assignment(
        groups,
        rng,
        previous_assignments=[],
    )
    use_hybrid = probe is None

    if use_hybrid and not oversized_sources:
        largest_source = str(
            groups.sort_values("Rows", ascending=False).iloc[0][SOURCE_FILE_COL]
        )
        oversized_sources = {largest_source}

    print("Hybrid split policy")
    if use_hybrid:
        print(
            "  whole-SourceFile split was not reasonable; chronological "
            "fallback activated"
        )
        print(
            "  chronologically split oversized SourceFiles: "
            + ", ".join(sorted(oversized_sources))
        )
    else:
        print("  reasonable whole-SourceFile split available; no row split needed")

    for repeat_index in range(1, REPEATED_HOLDOUTS + 1):
        if not use_hybrid:
            best = _search_reasonable_whole_source_assignment(
                groups,
                rng,
                previous_assignments=previous_whole_assignments,
            )
            if best is None:
                # It is possible that overlap penalties make later repeats
                # impossible even though repeat 1 was valid. Fall back safely.
                use_hybrid = True
                if not oversized_sources:
                    largest_source = str(
                        groups.sort_values("Rows", ascending=False)
                        .iloc[0][SOURCE_FILE_COL]
                    )
                    oversized_sources = {largest_source}
            else:
                assignment = best["assignment"]
                previous_whole_assignments.append(assignment)

                row_ids = {}
                for split_name, sources in assignment.items():
                    row_ids[split_name] = np.flatnonzero(
                        df[SOURCE_FILE_COL].astype(str).isin(sources).to_numpy()
                    )

                frames = _frames_from_row_ids(df, row_ids)
                manifest = _build_row_manifest(
                    df,
                    repeat_index,
                    row_ids,
                    oversized_sources=set(),
                )

                selected.append(
                    {
                        "repeat": repeat_index,
                        "train_df": frames["train"],
                        "validation_df": frames["validation"],
                        "test_df": frames["test"],
                        "manifest": manifest,
                        "objective": best["objective"],
                        "split_strategy": "whole_source",
                        "oversized_sources": [],
                    }
                )

                for split_name in ["train", "validation", "test"]:
                    row = {
                        "Repeat": repeat_index,
                        "Split": split_name,
                        "SplitStrategy": "whole_source",
                        "Objective": best["objective"],
                        "ChronologicallySplitSourceFiles": "",
                        **best["stats"][split_name],
                    }
                    diagnostics.append(row)
                continue

        # Keep the same 70/15/15 experimental protocol in every repeat.
        # Diversity comes from the new candidate search and soft SourceFile
        # overlap penalties, not from changing the temporal cut fractions.
        fractions = CHRONOLOGICAL_FRACTIONS

        best = _hybrid_candidate_from_small_sources(
            df=df,
            groups=groups,
            oversized_sources=oversized_sources,
            fractions=fractions,
            rng=rng,
            previous_small_assignments=previous_small_assignments,
        )
        previous_small_assignments.append(best["assignment"])

        frames = _frames_from_row_ids(df, best["row_ids"])
        manifest = _build_row_manifest(
            df,
            repeat_index,
            best["row_ids"],
            oversized_sources=oversized_sources,
        )

        selected.append(
            {
                "repeat": repeat_index,
                "train_df": frames["train"],
                "validation_df": frames["validation"],
                "test_df": frames["test"],
                "manifest": manifest,
                "objective": best["objective"],
                "split_strategy": "hybrid_chronological_oversized",
                "oversized_sources": sorted(oversized_sources),
            }
        )

        for split_name in ["train", "validation", "test"]:
            row = {
                "Repeat": repeat_index,
                "Split": split_name,
                "SplitStrategy": "hybrid_chronological_oversized",
                "Objective": best["objective"],
                "ChronologicallySplitSourceFiles": ";".join(
                    sorted(oversized_sources)
                ),
                "ChronologicalTrainFraction": fractions[0],
                "ChronologicalValidationFraction": fractions[1],
                "ChronologicalTestFraction": fractions[2],
                **best["stats"][split_name],
            }
            diagnostics.append(row)

    return selected, pd.DataFrame(diagnostics)


# ============================================================
# DATA HELPERS
# ============================================================


def benign_only(df: pd.DataFrame) -> pd.DataFrame:
    result = df[df[LABEL_COL].astype(int) == BENIGN_LABEL].copy()
    if result.empty:
        raise ValueError("No benign rows are available for Autoencoder training.")
    return result


def transform_and_select(
    preprocessor: DatasetPreprocessor,
    df: pd.DataFrame,
    dataset_name: str,
    expected_features=None,
):
    normalized = preprocessor.transform(df)

    (
        selected_df,
        selected_features,
        _dropped,
        _manifest,
    ) = select_and_engineer_dataset(
        dataset_name=dataset_name,
        df=normalized,
        expected_selected_features=expected_features,
    )

    X = selected_df[selected_features].to_numpy(dtype=np.float32)
    y = selected_df[LABEL_COL].astype(int).to_numpy()

    return selected_df, X, y, selected_features






# ============================================================
# AUTOENCODER
# ============================================================


def _keras_loss(params: dict):
    loss_name = str(params["loss"]).lower()

    if loss_name == "mse":
        return keras.losses.MeanSquaredError()
    if loss_name == "mae":
        return keras.losses.MeanAbsoluteError()
    if loss_name == "huber":
        return keras.losses.Huber(delta=float(params["huber_delta"]))

    raise ValueError(f"Unsupported Autoencoder loss: {params['loss']}")


def build_autoencoder(input_dim: int, params: dict) -> keras.Model:
    regularizer = (
        regularizers.l2(float(params["l2_regularization"]))
        if float(params["l2_regularization"]) > 0
        else None
    )

    inputs = keras.Input(
        shape=(input_dim,),
        dtype="float32",
        name="features",
    )

    x = inputs

    for index, units in enumerate(params["encoder_hidden_dims"]):
        x = layers.Dense(
            units=int(units),
            activation=params["hidden_activation"],
            use_bias=bool(params["use_bias"]),
            kernel_initializer=params["kernel_initializer"],
            bias_initializer=params["bias_initializer"],
            kernel_regularizer=regularizer,
            name=f"encoder_dense_{index + 1}",
        )(x)

    x = layers.Dense(
        units=int(params["bottleneck_dim"]),
        activation=params["hidden_activation"],
        use_bias=bool(params["use_bias"]),
        kernel_initializer=params["kernel_initializer"],
        bias_initializer=params["bias_initializer"],
        kernel_regularizer=regularizer,
        name="bottleneck",
    )(x)

    for index, units in enumerate(reversed(params["encoder_hidden_dims"])):
        x = layers.Dense(
            units=int(units),
            activation=params["hidden_activation"],
            use_bias=bool(params["use_bias"]),
            kernel_initializer=params["kernel_initializer"],
            bias_initializer=params["bias_initializer"],
            kernel_regularizer=regularizer,
            name=f"decoder_dense_{index + 1}",
        )(x)

    outputs = layers.Dense(
        units=int(input_dim),
        activation=params["output_activation"],
        use_bias=bool(params["use_bias"]),
        kernel_initializer=params["kernel_initializer"],
        bias_initializer=params["bias_initializer"],
        name="reconstruction",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="flow_autoencoder")

    optimizer = keras.optimizers.Adam(
        learning_rate=float(params["learning_rate"]),
        beta_1=float(params["adam_beta_1"]),
        beta_2=float(params["adam_beta_2"]),
        epsilon=float(params["adam_epsilon"]),
        amsgrad=bool(params["adam_amsgrad"]),
    )

    model.compile(
        optimizer=optimizer,
        loss=_keras_loss(params),
        steps_per_execution=int(params.get("steps_per_execution", 1)),
    )

    return model


def _corrupt_with_noise(
    X: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_std <= 0:
        return X.copy()

    noise = rng.normal(
        loc=0.0,
        scale=float(noise_std),
        size=X.shape,
    ).astype(np.float32)

    return (X + noise).astype(np.float32)


def fit_autoencoder(
    model: keras.Model,
    X_benign_train: np.ndarray,
    params: dict,
    seed: int,
):
    fraction = float(params["internal_validation_fraction"])

    if len(X_benign_train) < 20:
        raise ValueError(
            "Too few benign training rows for Autoencoder fitting: "
            f"{len(X_benign_train)}"
        )

    X_fit, X_early_stop = train_test_split(
        X_benign_train,
        test_size=fraction,
        random_state=seed,
        shuffle=True,
    )

    rng = np.random.default_rng(seed)

    X_fit_input = _corrupt_with_noise(
        X_fit,
        float(params["denoising_noise_std"]),
        rng,
    )
    X_early_input = _corrupt_with_noise(
        X_early_stop,
        float(params["denoising_noise_std"]),
        rng,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=float(params["early_stopping_min_delta"]),
        patience=int(params["early_stopping_patience"]),
        mode="min",
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_fit_input,
        X_fit,
        validation_data=(X_early_input, X_early_stop),
        batch_size=int(params["batch_size"]),
        epochs=int(params["epochs"]),
        shuffle=bool(params["shuffle"]),
        verbose=int(params["fit_verbose"]),
        callbacks=[early_stopping],
    )

    val_loss = history.history.get("val_loss", [])
    if val_loss:
        best_epoch = int(np.argmin(val_loss) + 1)
        best_val_loss = float(np.min(val_loss))
    else:
        best_epoch = len(history.history.get("loss", []))
        best_val_loss = float("nan")

    return history, best_epoch, best_val_loss


def reconstruct_matrix(
    model: keras.Model,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs = []

    for start in range(0, len(X), int(batch_size)):
        batch = X[start:start + int(batch_size)]
        outputs.append(model(batch, training=False).numpy())

    return np.concatenate(outputs, axis=0).astype(np.float32)


def score_from_reconstruction(
    X: np.ndarray,
    reconstructed: np.ndarray,
    score_mode: str,
) -> np.ndarray:
    absolute_error = np.abs(X - reconstructed)
    squared_error = np.square(X - reconstructed)

    if score_mode == "mse":
        return np.mean(squared_error, axis=1)
    if score_mode == "mae":
        return np.mean(absolute_error, axis=1)

    if score_mode in {"top5_mse", "top10_mse"}:
        k = 5 if score_mode == "top5_mse" else 10
        k = min(k, squared_error.shape[1])
        top_k = np.partition(
            squared_error,
            kth=squared_error.shape[1] - k,
            axis=1,
        )[:, -k:]
        return np.mean(top_k, axis=1)

    raise ValueError(f"Unknown reconstruction score mode: {score_mode}")


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================


def _best_fbeta_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    beta: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if len(thresholds) == 0:
        raise ValueError("Unable to calibrate reconstruction threshold.")

    precision = precision[:-1]
    recall = recall[:-1]

    beta_sq = beta ** 2
    denominator = beta_sq * precision + recall

    fbeta = np.divide(
        (1.0 + beta_sq) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_value = np.nanmax(fbeta)
    candidates = np.flatnonzero(np.isclose(fbeta, best_value))

    if len(candidates) > 1:
        index = int(candidates[np.argmax(recall[candidates])])
    else:
        index = int(candidates[0])

    return float(thresholds[index])


def _recall_at_fpr_cap_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    fpr_cap: float,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eligible = np.flatnonzero(fpr <= float(fpr_cap))

    if len(eligible) == 0:
        return float(np.nextafter(np.max(scores), np.inf))

    finite = eligible[np.isfinite(thresholds[eligible])]
    if len(finite) > 0:
        eligible = finite

    best_tpr = np.max(tpr[eligible])
    candidates = eligible[np.isclose(tpr[eligible], best_tpr)]
    min_fpr = np.min(fpr[candidates])
    candidates = candidates[np.isclose(fpr[candidates], min_fpr)]
    index = int(candidates[np.argmax(thresholds[candidates])])

    return float(thresholds[index])


def choose_validation_threshold(
    y_true,
    scores,
    method: str,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if method == "validation_f1":
        return _best_fbeta_threshold(y_true, scores, beta=1.0)
    if method == "validation_f2":
        return _best_fbeta_threshold(y_true, scores, beta=2.0)
    if method == "recall_at_fpr_cap":
        return _recall_at_fpr_cap_threshold(
            y_true,
            scores,
            fpr_cap=VALIDATION_FPR_CAP,
        )

    raise ValueError(f"Unknown threshold method: {method}")


def _benign_quantile_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    quantile: float,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    benign_scores = scores[y_true == BENIGN_LABEL]

    if len(benign_scores) == 0:
        raise ValueError(
            "Validation split contains no benign rows for quantile calibration."
        )

    return float(np.quantile(benign_scores, float(quantile)))


def calibrate_final_threshold(
    y_true,
    scores,
) -> tuple[float, str, pd.DataFrame]:
    """
    Calibrate the FINAL operating threshold for one repeated holdout.

    The anomaly-score function is already fixed at dataset level. The primary
    threshold is the validation threshold that maximizes F1. This balances
    precision and recall without imposing an arbitrary hard FPR cap.

    Additional validation-only operating points are still computed and saved
    for diagnostics:
      - benign-score quantiles;
      - F2 threshold;
      - recall-maximizing thresholds under 1%, 2%, and 5% FPR caps.

    TEST is never used here.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    labels = set(np.unique(y_true))
    if labels != {BENIGN_LABEL, MALICIOUS_LABEL}:
        raise ValueError(
            "Final threshold calibration requires benign and malicious "
            "validation rows."
        )

    # Primary operating threshold: maximize validation F1. Put it first so if
    # another diagnostic rule yields the exact same numeric threshold, the
    # auditable candidate table retains the primary method name.
    primary_threshold = _best_fbeta_threshold(
        y_true,
        scores,
        beta=1.0,
    )

    candidate_specs: list[tuple[str, float]] = [
        ("validation_f1", primary_threshold),
    ]

    for quantile in FINAL_THRESHOLD_BENIGN_QUANTILES:
        threshold = _benign_quantile_threshold(
            y_true,
            scores,
            quantile,
        )
        candidate_specs.append(
            (f"benign_q{quantile:g}", threshold)
        )

    candidate_specs.append(
        (
            "validation_f2",
            _best_fbeta_threshold(y_true, scores, beta=2.0),
        )
    )

    for fpr_cap in FINAL_THRESHOLD_FPR_CANDIDATES:
        threshold = _recall_at_fpr_cap_threshold(
            y_true,
            scores,
            fpr_cap=fpr_cap,
        )
        candidate_specs.append(
            (f"recall_at_fpr_{fpr_cap:g}", threshold)
        )

    # Deduplicate numerically identical thresholds while keeping the first
    # descriptive method name.
    seen = set()
    rows = []
    for method, threshold in candidate_specs:
        key = round(float(threshold), 12)
        if key in seen:
            continue
        seen.add(key)

        metrics = calculate_metrics(
            y_true,
            scores,
            float(threshold),
        )

        rows.append(
            {
                "threshold_method": method,
                "threshold": float(threshold),
                "selected_for_final": method == "validation_f1",
                **metrics,
            }
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("No validation threshold candidates were generated.")

    # The selected threshold is exactly the F1-maximizing validation threshold.
    # Candidate diagnostics do not override it.
    candidates = candidates.sort_values(
        by=["selected_for_final", "f1", "fpr"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return (
        float(primary_threshold),
        "validation_f1",
        candidates,
    )


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    y_true,
    scores,
    threshold: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[BENIGN_LABEL, MALICIOUS_LABEL],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_tpr": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "specificity_tnr": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(
            fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


# ============================================================
# FINAL MODEL
# ============================================================


def train_final_model(
    dataset_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    best_params: dict,
    score_mode: str,
    expected_selected_features: list[str],
):
    benign_train_df = benign_only(train_df)

    preprocessor = DatasetPreprocessor(
        f"{dataset_name}_autoencoder_final"
    )
    preprocessor.fit(benign_train_df)

    (
        _selected_train,
        X_benign_train,
        y_benign_train,
        selected_features,
    ) = transform_and_select(
        preprocessor,
        benign_train_df,
        dataset_name,
        expected_features=expected_selected_features,
    )

    (
        _selected_validation,
        X_validation,
        y_validation,
        _,
    ) = transform_and_select(
        preprocessor,
        validation_df,
        dataset_name,
        expected_features=selected_features,
    )

    if set(np.unique(y_benign_train)) != {BENIGN_LABEL}:
        raise RuntimeError("Malicious rows reached final Autoencoder fitting.")

    keras.backend.clear_session()
    set_random_seed(FINAL_RANDOM_STATE)

    model = build_autoencoder(
        input_dim=X_benign_train.shape[1],
        params=best_params,
    )

    history, best_epoch, best_internal_val_loss = fit_autoencoder(
        model,
        X_benign_train,
        best_params,
        FINAL_RANDOM_STATE,
    )

    validation_reconstruction = reconstruct_matrix(
        model,
        X_validation,
        batch_size=int(best_params["batch_size"]),
    )

    validation_scores = score_from_reconstruction(
        X_validation,
        validation_reconstruction,
        score_mode,
    )

    threshold, threshold_method, threshold_candidates = calibrate_final_threshold(
        y_validation,
        validation_scores,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_scores,
        threshold,
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "threshold": threshold,
        "threshold_method": threshold_method,
        "threshold_candidates": threshold_candidates,
        "history": history,
        "validation_metrics": validation_metrics,
        "benign_training_rows": len(X_benign_train),
        "best_epoch": best_epoch,
        "best_internal_val_loss": best_internal_val_loss,
    }


# ============================================================
# SHARED HOLDOUT / ARTIFACT HELPERS
# ============================================================


def _shared_split_signature(manifest: pd.DataFrame) -> str:
    """Stable hash of the exact shared row-level train/validation/test manifest."""
    columns = [
        column
        for column in [
            ORIGINAL_ROW_COL,
            SOURCE_FILE_COL,
            LABEL_COL,
            "Timestamp",
            "__Split",
        ]
        if column in manifest.columns
    ]

    payload = (
        manifest[columns]
        .sort_values(ORIGINAL_ROW_COL)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _drop_shared_split_helper_columns(
    splits: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        split_name: split_df.drop(
            columns=[ORIGINAL_ROW_COL],
            errors="ignore",
        ).copy()
        for split_name, split_df in splits.items()
    }


def _jsonable_params(params: dict) -> dict:
    return {
        key: (
            list(value)
            if isinstance(value, tuple)
            else value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in params.items()
    }


def _load_dataset_split(
    dataset_name: str,
    force_resplit: bool = False,
):
    path = DATASET_PATHS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing ingested dataset: {path}")

    df = pd.read_csv(path, low_memory=False)
    splits, manifest, metadata = get_or_create_shared_split(
        dataset_name=dataset_name,
        df=df,
        split_root=SHARED_SPLIT_ROOT,
        force_rebuild=force_resplit,
    )
    return _drop_shared_split_helper_columns(splits), manifest, metadata


def _print_split_summary(
    dataset_name: str,
    splits: dict[str, pd.DataFrame],
    metadata: dict,
) -> None:
    print("\n" + "-" * 90)
    print(f"{dataset_name.upper()} SPLIT")
    print("-" * 90)
    print(
        f"Rows | train={len(splits['train']):,}, "
        f"validation={len(splits['validation']):,}, "
        f"test={len(splits['test']):,}"
    )
    print(f"Shared split mode: {metadata.get('mode')}")
    if metadata.get("oversized_source") is not None:
        print(
            "Chronologically split oversized SourceFile: "
            f"{metadata.get('oversized_source')}"
        )


# ============================================================
# PER-DATASET FINAL ARTIFACTS USING SHARED CONFIGURATION
# ============================================================


def save_dataset_training_outputs(
    dataset_name: str,
    manifest: pd.DataFrame,
    split_metadata: dict,
    shared_params: dict,
    score_mode: str,
    final: dict,
) -> None:
    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    signature = _shared_split_signature(manifest)

    final["model"].save(artifact_dir / "autoencoder.keras")
    final["preprocessor"].save(artifact_dir / "preprocessor.joblib")
    manifest.to_csv(artifact_dir / "split_manifest.csv", index=False)
    pd.DataFrame({"Feature": final["selected_features"]}).to_csv(
        artifact_dir / "selected_features.csv", index=False
    )

    with (artifact_dir / "model_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_jsonable_params(shared_params), handle, indent=2)

    with (artifact_dir / "decision_rule.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "split_signature": signature,
                "shared_split_manifest": str(
                    SHARED_SPLIT_ROOT / f"{dataset_name}_source_aware_split.csv"
                ),
                "split_strategy": split_metadata.get("mode"),
                "oversized_source": split_metadata.get("oversized_source"),
                "score_mode": score_mode,
                "threshold_method": final["threshold_method"],
                "threshold_calibration_version": THRESHOLD_CALIBRATION_VERSION,
                "threshold": float(final["threshold"]),
                "test_used_for_selection": False,
            },
            handle,
            indent=2,
        )

    with (artifact_dir / "training_strategy.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "model": "Dense Autoencoder",
                "benchmark_datasets": list(BENCHMARK_DATASETS),
                "hyperparameter_scope": "shared across Dataset 2 and Dataset 3",
                "configuration_mode": (
                    "fixed shared baseline configuration; no hyperparameter search "
                    "is performed during this run"
                ),
                "evaluation_strategy": (
                    "shared source-aware 70/15/15 holdout reused by LSTM, "
                    "Autoencoder, and Isolation Forest"
                ),
                "split_mode": split_metadata.get("mode"),
                "oversized_source": split_metadata.get("oversized_source"),
                "imbalance_strategy": "benign-only reconstruction training",
                "preprocessing_fit_scope": "this dataset's benign training rows only",
                "detector_fit_scope": "this dataset's benign training rows only",
                "internal_early_stopping_scope": (
                    "10% split from this dataset's benign training rows"
                ),
                "threshold_calibration_scope": (
                    "this dataset's full mixed validation split only"
                ),
                "final_threshold_selection": (
                    "maximize validation F1; diagnostic alternatives saved separately"
                ),
                "score_mode": score_mode,
                "benign_training_rows": int(final["benign_training_rows"]),
                "best_epoch": int(final["best_epoch"]),
                "best_internal_validation_loss": float(
                    final["best_internal_val_loss"]
                ),
                "shared_configuration_artifact": str(SHARED_CONFIG_PATH),
                "test_used_for_model_or_threshold_selection": False,
            },
            handle,
            indent=2,
        )

    pd.DataFrame(final["history"].history).to_csv(
        result_dir / "final_training_history.csv", index=False
    )
    pd.DataFrame([final["validation_metrics"]]).to_csv(
        result_dir / "final_validation_metrics.csv", index=False
    )
    final["threshold_candidates"].to_csv(
        result_dir / "validation_threshold_candidates.csv", index=False
    )

    pd.DataFrame(
        [
            {
                "Dataset": dataset_name,
                "Split": split_name,
                "Rows": int(split_metadata["row_counts"][split_name]),
                "Fraction": float(split_metadata["row_ratios"][split_name]),
            }
            for split_name in ["train", "validation", "test"]
        ]
    ).to_csv(result_dir / "shared_split_sizes.csv", index=False)


def _joint_artifacts_are_compatible(
    dataset_records: dict,
) -> bool:
    """Conservative resume check for a completed shared D2/D3 experiment."""
    if not SHARED_CONFIG_PATH.exists():
        return False

    try:
        with SHARED_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            shared = json.load(handle)

        if shared.get("scope") != "shared_dataset2_dataset3":
            return False
        if shared.get("score_mode") != SHARED_SCORE_MODE:
            return False

        for dataset_name in BENCHMARK_DATASETS:
            artifact_dir = ARTIFACT_ROOT / dataset_name
            required = [
                artifact_dir / "autoencoder.keras",
                artifact_dir / "preprocessor.joblib",
                artifact_dir / "model_config.json",
                artifact_dir / "decision_rule.json",
            ]
            if not all(path.exists() for path in required):
                return False

            with (artifact_dir / "model_config.json").open(
                "r", encoding="utf-8"
            ) as handle:
                local_params = json.load(handle)
            if local_params != shared.get("parameters"):
                return False

            with (artifact_dir / "decision_rule.json").open(
                "r", encoding="utf-8"
            ) as handle:
                rule = json.load(handle)
            if rule.get("score_mode") != shared.get("score_mode"):
                return False
            if rule.get("threshold_calibration_version") != THRESHOLD_CALIBRATION_VERSION:
                return False
            if rule.get("split_signature") != _shared_split_signature(
                dataset_records[dataset_name]["manifest"]
            ):
                return False
    except Exception:
        return False

    return True


# ============================================================
# JOINT D2/D3 TRAINING
# ============================================================


def train_all_datasets(
    force: bool = False,
    force_resplit: bool = False,
) -> None:
    """Train D2/D3 Autoencoders with one frozen shared configuration."""
    print("\n" + "=" * 90)
    print("AUTOENCODER - FIXED D2/D3 BASELINE")
    print("=" * 90)

    dataset_records = {}
    for dataset_name in BENCHMARK_DATASETS:
        splits, manifest, metadata = _load_dataset_split(
            dataset_name,
            force_resplit=force_resplit,
        )
        _print_split_summary(dataset_name, splits, metadata)
        dataset_records[dataset_name] = {
            "splits": splits,
            "manifest": manifest,
            "metadata": metadata,
        }

    if not force and not force_resplit and _joint_artifacts_are_compatible(
        dataset_records
    ):
        print(
            "\n[RESUME] Fixed D2/D3 Autoencoder artifacts already match the "
            "current split and configuration. Skipping retraining."
        )
        return

    shared_params = deepcopy(BASE_PARAMS)
    score_mode = SHARED_SCORE_MODE

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with SHARED_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scope": "shared_dataset2_dataset3",
                "configuration_mode": "fixed_baseline_no_hyperparameter_search",
                "parameters": _jsonable_params(shared_params),
                "score_mode": score_mode,
            },
            handle,
            indent=2,
        )

    print("\nFrozen shared Autoencoder configuration:")
    print(f"  encoder_hidden_dims: {shared_params['encoder_hidden_dims']}")
    print(f"  bottleneck_dim: {shared_params['bottleneck_dim']}")
    print(f"  learning_rate: {shared_params['learning_rate']}")
    print(f"  loss: {shared_params['loss']}")
    print(f"  l2_regularization: {shared_params['l2_regularization']}")
    print(f"  denoising_noise_std: {shared_params['denoising_noise_std']}")
    print(f"  batch_size: {shared_params['batch_size']}")
    print(f"  score_mode: {score_mode}")

    for dataset_name in BENCHMARK_DATASETS:
        record = dataset_records[dataset_name]
        print("\n" + "-" * 90)
        print(f"TRAIN FIXED BASELINE - {dataset_name.upper()}")
        print("-" * 90)

        final = train_final_model(
            dataset_name=dataset_name,
            train_df=record["splits"]["train"],
            validation_df=record["splits"]["validation"],
            best_params=shared_params,
            score_mode=score_mode,
            expected_selected_features=None,
        )

        print(
            f"Validation threshold: {final['threshold']:.8f} "
            f"({final['threshold_method']})"
        )

        save_dataset_training_outputs(
            dataset_name=dataset_name,
            manifest=record["manifest"],
            split_metadata=record["metadata"],
            shared_params=shared_params,
            score_mode=score_mode,
            final=final,
        )

        print(f"Artifacts: {ARTIFACT_ROOT / dataset_name}")
        print(f"Results: {RESULT_ROOT / dataset_name}")


