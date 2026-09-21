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

def _find_project_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "data" / "ingested").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate Workshop_final project root. Expected a parent containing "
        "both src/ and data/ingested/."
    )


PROJECT_ROOT = _find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"

# Final benchmark datasets only.
DATASET_PATHS = {
    "dataset2": INGESTED_ROOT / "dataset2_ingested.csv",
    "dataset3": INGESTED_ROOT / "dataset3_ingested.csv",
}

BASELINE_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models_same_hyper" / "Autoencoder" / "baseline"
BASELINE_RESULT_ROOT = PROJECT_ROOT / "results" / "models_same_hyper" / "Autoencoder" / "baseline"
OPTIMIZATION_NAME = "optimization_1_brittle_feature_removal"
ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts" / "models_same_hyper" / "Autoencoder" / OPTIMIZATION_NAME
)
RESULT_ROOT = (
    PROJECT_ROOT / "results" / "models_same_hyper" / "Autoencoder" / OPTIMIZATION_NAME
)
SHARED_SPLIT_ROOT = PROJECT_ROOT / "artifacts" / "splits"



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
# OPTIMIZATION 1 - BRITTLE FEATURE REMOVAL
# ============================================================

O1_EXPLICIT_DROP_FEATURES = (
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Total Connection Flow Time",
    "Fwd Header Length",
    "Bwd Header Length",
)
O1_FLAG_KEEP = "URG Flag Count"
EXPECTED_BASELINE_FEATURE_COUNT = 57
EXPECTED_O1_FEATURE_COUNT = 43


def apply_o1_feature_pruning(
    selected_df: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Apply exactly the accepted brittle-feature rule used by LSTM O1."""
    explicit = set(O1_EXPLICIT_DROP_FEATURES)
    dropped: list[str] = []
    kept: list[str] = []

    for feature in selected_features:
        drop_explicit = feature in explicit
        drop_flag = ("Flag" in feature) and feature != O1_FLAG_KEEP
        if drop_explicit or drop_flag:
            dropped.append(feature)
        else:
            kept.append(feature)

    if (
        len(selected_features) == EXPECTED_BASELINE_FEATURE_COUNT
        and len(kept) != EXPECTED_O1_FEATURE_COUNT
    ):
        raise RuntimeError(
            "O1 pruning expected 43 features from the 57-feature baseline, "
            f"but produced {len(kept)}. Dropped={dropped}"
        )

    return selected_df, kept, dropped


# ============================================================
# HYBRID SOURCEFILE / CHRONOLOGICAL REPEATED HOLDOUT
# ============================================================

# The SAME split policy is applied to dataset1, dataset2 and dataset3.
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
# AUTOENCODER BASELINE + FASTER SEARCH
# ============================================================

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

# Faster staged search: fewer low-value candidates than the previous version,
# while still varying architecture, optimizer/loss, regularization and noise.
ARCHITECTURE_CANDIDATES = [
    {
        "name": "baseline_64_32_b8",
        "changes": {
            "encoder_hidden_dims": (64, 32),
            "bottleneck_dim": 8,
        },
    },
    {
        "name": "compact_32_16_b4",
        "changes": {
            "encoder_hidden_dims": (32, 16),
            "bottleneck_dim": 4,
        },
    },
    {
        "name": "deep_64_32_16_b8",
        "changes": {
            "encoder_hidden_dims": (64, 32, 16),
            "bottleneck_dim": 8,
        },
    },
]

OPTIMIZATION_VALUES = {
    "learning_rate": [5e-4, 2e-3],
    "loss": ["huber"],
    "l2_regularization": [0.0, 1e-4],
}

REFINEMENT_VALUES = {
    "denoising_noise_std": [0.02],
    "batch_size": [512],
}

SCREENING_RANDOM_STATE = 42
STABILITY_RANDOM_STATES = [17, 42, 73]
STABILITY_FINALISTS = 2
FINAL_RANDOM_STATE = 42

# Sensitivity search uses representative samples. The final selected model is
# still fitted on ALL benign training rows and calibrated on ALL validation rows.
SENSITIVITY_MAX_BENIGN_TRAIN_ROWS = 20000
SENSITIVITY_MAX_VALIDATION_ROWS = 8000
SENSITIVITY_MAX_VALIDATION_MALICIOUS = 3000

SCORE_MODES = [
    "mse",
    "mae",
    "top5_mse",
    "top10_mse",
]

# During the shared sensitivity search we use one consistent operating point
# so network/score selection is not driven by whichever threshold objective
# happened to win. The final per-repeat threshold is calibrated separately
# below using the full validation split.
THRESHOLD_METHODS = [
    "recall_at_fpr_cap",
]

VALIDATION_FPR_CAP = 0.02

# Final per-repeat operating-threshold calibration. The selected anomaly score
# is fixed once per dataset; only the threshold is re-calibrated for each
# repeat. The PRIMARY operating threshold maximizes validation F1. Additional
# benign-quantile and FPR-constrained thresholds are saved for diagnostics only.
FINAL_THRESHOLD_FPR_CANDIDATES = [0.01, 0.02, 0.05]
FINAL_THRESHOLD_BENIGN_QUANTILES = [0.95, 0.975, 0.98, 0.99, 0.995]
THRESHOLD_CALIBRATION_VERSION = "validation_f1_per_repeat_v2"


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
        baseline_features,
        _dropped,
        _manifest,
    ) = select_and_engineer_dataset(
        dataset_name=dataset_name,
        df=normalized,
        expected_selected_features=None,
    )

    selected_df, selected_features, _o1_dropped = apply_o1_feature_pruning(
        selected_df,
        list(baseline_features),
    )

    if expected_features is not None and list(selected_features) != list(expected_features):
        missing = [f for f in expected_features if f not in selected_features]
        extra = [f for f in selected_features if f not in expected_features]
        raise ValueError(
            f"{dataset_name}: O1 feature schema differs from training. "
            f"Missing={missing}; Extra={extra}"
        )

    X = selected_df[selected_features].to_numpy(dtype=np.float32)
    y = selected_df[LABEL_COL].astype(int).to_numpy()

    return selected_df, X, y, selected_features


def _sample_rows(
    X: np.ndarray,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if len(X) <= max_rows:
        return X

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=max_rows, replace=False)
    return X[np.sort(indices)]


def _sample_validation(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    max_malicious: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_rows:
        return X, y

    rng = np.random.default_rng(seed)
    benign_idx = np.flatnonzero(y == BENIGN_LABEL)
    malicious_idx = np.flatnonzero(y == MALICIOUS_LABEL)

    malicious_keep = min(
        len(malicious_idx),
        max_malicious,
        max_rows // 2,
    )
    benign_keep = min(
        len(benign_idx),
        max_rows - malicious_keep,
    )

    # If benign is scarce, give the remaining capacity to malicious.
    remaining = max_rows - benign_keep - malicious_keep
    if remaining > 0:
        malicious_keep = min(
            len(malicious_idx),
            malicious_keep + remaining,
        )

    chosen_benign = rng.choice(
        benign_idx,
        size=benign_keep,
        replace=False,
    )
    chosen_malicious = rng.choice(
        malicious_idx,
        size=malicious_keep,
        replace=False,
    )

    chosen = np.concatenate([chosen_benign, chosen_malicious])
    rng.shuffle(chosen)

    return X[chosen], y[chosen]


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
# FAST STAGED SENSITIVITY SEARCH
# ============================================================


def _serialize_param(value):
    if isinstance(value, tuple):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _params_signature(params: dict) -> str:
    return "|".join(
        f"{key}={repr(params[key])}"
        for key in sorted(params.keys())
    )


def _rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["pr_auc", "f2", "recall_tpr", "f1", "fpr"],
        ascending=[False, False, False, False, True],
    )


def _stage_a_configurations() -> list[dict]:
    configurations = []

    for index, candidate in enumerate(ARCHITECTURE_CANDIDATES, start=1):
        params = deepcopy(BASE_PARAMS)
        params.update(candidate["changes"])
        configurations.append(
            {
                "stage": "A_architecture",
                "config_id": f"A{index:02d}",
                "config_name": candidate["name"],
                "params": params,
            }
        )

    return configurations


def _one_factor_variants(
    base_params: dict,
    values_by_parameter: dict,
    stage: str,
    prefix: str,
) -> list[dict]:
    configurations = []
    seen = {_params_signature(base_params)}
    counter = 1

    for parameter, values in values_by_parameter.items():
        for value in values:
            params = deepcopy(base_params)
            params[parameter] = value
            signature = _params_signature(params)

            if signature in seen:
                continue

            seen.add(signature)
            configurations.append(
                {
                    "stage": stage,
                    "config_id": f"{prefix}{counter:02d}",
                    "config_name": f"{parameter}={value}",
                    "params": params,
                }
            )
            counter += 1

    return configurations


def _lookup_params(
    configurations: list[dict],
    signature: str,
) -> dict:
    for config in configurations:
        if _params_signature(config["params"]) == signature:
            return deepcopy(config["params"])
    raise KeyError(f"Could not recover parameters for signature: {signature}")


def _evaluate_network_decisions(
    model: keras.Model,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    params: dict,
    stage: str,
    config_id: str,
    config_name: str,
    seed: int,
    best_epoch: int,
    best_internal_val_loss: float,
    final_training_loss: float,
) -> list[dict]:
    reconstruction = reconstruct_matrix(
        model,
        X_validation,
        batch_size=int(params["batch_size"]),
    )

    rows = []
    signature = _params_signature(params)

    for score_mode in SCORE_MODES:
        scores = score_from_reconstruction(
            X_validation,
            reconstruction,
            score_mode,
        )

        for threshold_method in THRESHOLD_METHODS:
            threshold = choose_validation_threshold(
                y_validation,
                scores,
                threshold_method,
            )

            metrics = calculate_metrics(
                y_validation,
                scores,
                threshold,
            )

            row = {
                "stage": stage,
                "config_id": config_id,
                "config_name": config_name,
                "param_signature": signature,
                "random_state": int(seed),
                "score_mode": score_mode,
                "threshold_method": threshold_method,
                "best_epoch": int(best_epoch),
                "best_internal_val_loss": float(best_internal_val_loss),
                "final_training_loss": float(final_training_loss),
            }

            for parameter, value in params.items():
                row[f"param_{parameter}"] = _serialize_param(value)

            row.update(metrics)
            rows.append(row)

    return rows


def _train_and_evaluate_configuration(
    X_benign_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    params: dict,
    stage: str,
    config_id: str,
    config_name: str,
    seed: int,
) -> list[dict]:
    keras.backend.clear_session()
    set_random_seed(seed)

    model = build_autoencoder(
        input_dim=X_benign_train.shape[1],
        params=params,
    )

    history, best_epoch, best_val_loss = fit_autoencoder(
        model,
        X_benign_train,
        params,
        seed,
    )

    final_training_loss = float(history.history["loss"][-1])

    return _evaluate_network_decisions(
        model=model,
        X_validation=X_validation,
        y_validation=y_validation,
        params=params,
        stage=stage,
        config_id=config_id,
        config_name=config_name,
        seed=seed,
        best_epoch=best_epoch,
        best_internal_val_loss=best_val_loss,
        final_training_loss=final_training_loss,
    )


def _best_row_per_network(raw_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, group in raw_runs.groupby("param_signature", sort=False):
        rows.append(_rank_rows(group).iloc[0].to_dict())

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["pr_auc", "f2", "recall_tpr", "f1", "fpr"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )


def _stability_summary(stability_runs: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_tpr",
        "specificity_tnr",
        "fpr",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
        "mcc",
        "threshold",
        "best_epoch",
        "best_internal_val_loss",
        "final_training_loss",
    ]

    rows = []

    grouped = stability_runs.groupby(
        [
            "finalist_id",
            "config_name",
            "param_signature",
            "score_mode",
            "threshold_method",
        ],
        dropna=False,
    )

    for group_key, group_df in grouped:
        first = group_df.iloc[0]
        row = {
            "finalist_id": group_key[0],
            "config_name": group_key[1],
            "param_signature": group_key[2],
            "score_mode": group_key[3],
            "threshold_method": group_key[4],
            "runs": int(len(group_df)),
        }

        for parameter in BASE_PARAMS:
            row[f"param_{parameter}"] = first[f"param_{parameter}"]

        for metric_name in metric_columns:
            row[f"{metric_name}_mean"] = float(
                group_df[metric_name].mean()
            )
            row[f"{metric_name}_std"] = float(
                group_df[metric_name].std(ddof=0)
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        by=[
            "pr_auc_mean",
            "f2_mean",
            "recall_tpr_mean",
            "f1_mean",
            "fpr_mean",
            "pr_auc_std",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)


def run_sensitivity_analysis(
    dataset_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    benign_train_df = benign_only(train_df)

    preprocessor = DatasetPreprocessor(
        f"{dataset_name}_autoencoder_sensitivity"
    )
    preprocessor.fit(benign_train_df)

    (
        _selected_train,
        X_benign_train_full,
        y_benign_train,
        selected_features,
    ) = transform_and_select(
        preprocessor,
        benign_train_df,
        dataset_name,
    )

    (
        _selected_validation,
        X_validation_full,
        y_validation_full,
        _,
    ) = transform_and_select(
        preprocessor,
        validation_df,
        dataset_name,
        expected_features=selected_features,
    )

    if set(np.unique(y_benign_train)) != {BENIGN_LABEL}:
        raise RuntimeError("Malicious rows reached Autoencoder fitting.")

    X_benign_train = _sample_rows(
        X_benign_train_full,
        SENSITIVITY_MAX_BENIGN_TRAIN_ROWS,
        SCREENING_RANDOM_STATE,
    )

    X_validation, y_validation = _sample_validation(
        X_validation_full,
        y_validation_full,
        SENSITIVITY_MAX_VALIDATION_ROWS,
        SENSITIVITY_MAX_VALIDATION_MALICIOUS,
        SCREENING_RANDOM_STATE,
    )

    print(
        f"Sensitivity benign rows: {len(X_benign_train):,} "
        f"(full train benign={len(X_benign_train_full):,})"
    )
    print(
        f"Sensitivity validation rows: {len(X_validation):,} "
        f"(full validation={len(X_validation_full):,})"
    )

    all_configurations = []
    screening_rows = []

    stage_a = _stage_a_configurations()
    all_configurations.extend(stage_a)

    print(f"Stage A - architecture: {len(stage_a)} networks")
    for config in stage_a:
        print(f"  {config['config_id']}: {config['config_name']}")
        screening_rows.extend(
            _train_and_evaluate_configuration(
                X_benign_train,
                X_validation,
                y_validation,
                config["params"],
                config["stage"],
                config["config_id"],
                config["config_name"],
                SCREENING_RANDOM_STATE,
            )
        )

    screening_raw = pd.DataFrame(screening_rows)
    stage_a_best = _best_row_per_network(
        screening_raw[screening_raw["stage"] == "A_architecture"]
    )

    best_a_signature = str(stage_a_best.iloc[0]["param_signature"])
    best_a_params = _lookup_params(stage_a, best_a_signature)

    stage_b = _one_factor_variants(
        best_a_params,
        OPTIMIZATION_VALUES,
        stage="B_optimization",
        prefix="B",
    )
    all_configurations.extend(stage_b)

    print(f"Stage B - optimization: {len(stage_b)} new networks")
    for config in stage_b:
        print(f"  {config['config_id']}: {config['config_name']}")
        screening_rows.extend(
            _train_and_evaluate_configuration(
                X_benign_train,
                X_validation,
                y_validation,
                config["params"],
                config["stage"],
                config["config_id"],
                config["config_name"],
                SCREENING_RANDOM_STATE,
            )
        )

    screening_raw = pd.DataFrame(screening_rows)
    screened_best = _best_row_per_network(screening_raw)
    best_b_signature = str(screened_best.iloc[0]["param_signature"])
    best_b_params = _lookup_params(all_configurations, best_b_signature)

    stage_c = _one_factor_variants(
        best_b_params,
        REFINEMENT_VALUES,
        stage="C_refinement",
        prefix="C",
    )
    all_configurations.extend(stage_c)

    print(f"Stage C - refinement: {len(stage_c)} new networks")
    for config in stage_c:
        print(f"  {config['config_id']}: {config['config_name']}")
        screening_rows.extend(
            _train_and_evaluate_configuration(
                X_benign_train,
                X_validation,
                y_validation,
                config["params"],
                config["stage"],
                config["config_id"],
                config["config_name"],
                SCREENING_RANDOM_STATE,
            )
        )

    screening_raw = pd.DataFrame(screening_rows)
    screening_best = _best_row_per_network(screening_raw)

    finalist_best = screening_best.head(STABILITY_FINALISTS).copy()
    finalist_configs = []

    for index, row in finalist_best.iterrows():
        signature = str(row["param_signature"])
        finalist_configs.append(
            {
                "finalist_id": f"F{index + 1:02d}",
                "config_name": str(row["config_name"]),
                "param_signature": signature,
                "params": _lookup_params(all_configurations, signature),
            }
        )

    stability_rows = []

    print(
        "Stage D - stability: "
        f"{len(finalist_configs)} finalists x "
        f"{len(STABILITY_RANDOM_STATES)} seeds"
    )

    for finalist in finalist_configs:
        print(
            f"  {finalist['finalist_id']}: "
            f"{finalist['config_name']}"
        )

        seed42_rows = screening_raw[
            (screening_raw["param_signature"] == finalist["param_signature"])
            & (screening_raw["random_state"] == SCREENING_RANDOM_STATE)
        ].copy()

        seed42_rows["finalist_id"] = finalist["finalist_id"]
        seed42_rows["stage"] = "D_stability"
        stability_rows.extend(seed42_rows.to_dict(orient="records"))

        for seed in STABILITY_RANDOM_STATES:
            if seed == SCREENING_RANDOM_STATE:
                continue

            new_rows = _train_and_evaluate_configuration(
                X_benign_train,
                X_validation,
                y_validation,
                finalist["params"],
                "D_stability",
                finalist["finalist_id"],
                finalist["config_name"],
                seed,
            )

            for row in new_rows:
                row["finalist_id"] = finalist["finalist_id"]

            stability_rows.extend(new_rows)

    stability_raw = pd.DataFrame(stability_rows)
    stability_summary = _stability_summary(stability_raw)

    best_row = stability_summary.iloc[0]
    best_finalist = next(
        item
        for item in finalist_configs
        if item["finalist_id"] == str(best_row["finalist_id"])
    )

    network_counts = {
        "stage_a": len(stage_a),
        "stage_b": len(stage_b),
        "stage_c": len(stage_c),
        "stability_extra_seed_runs": (
            len(finalist_configs)
            * (len(STABILITY_RANDOM_STATES) - 1)
        ),
        "total_screening_networks": (
            len(stage_a) + len(stage_b) + len(stage_c)
        ),
    }

    return {
        "screening_runs": screening_raw,
        "screening_best": screening_best,
        "stability_runs": stability_raw,
        "stability_summary": stability_summary,
        "best_params": deepcopy(best_finalist["params"]),
        "best_score_mode": str(best_row["score_mode"]),
        "best_threshold_method": str(best_row["threshold_method"]),
        "selected_features": selected_features,
        "network_counts": network_counts,
    }


# ============================================================
# FROZEN SHARED CONFIGURATION FOR CONTROLLED O1
# ============================================================

# Agreed shared Autoencoder configuration for Dataset 2 and Dataset 3.
# O1 MUST NOT re-tune any of these values.
SHARED_FIXED_PARAMS = {
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
SHARED_FIXED_SCORE_MODE = "top5_mse"


def load_shared_fixed_configuration() -> tuple[dict, str]:
    return deepcopy(SHARED_FIXED_PARAMS), SHARED_FIXED_SCORE_MODE


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
# CACHE / RESUME
# ============================================================


def _manifest_signature(manifest: pd.DataFrame) -> str:
    columns = [
        "Repeat",
        ROW_INDEX_COL,
        SOURCE_FILE_COL,
        "Label",
        "Split",
        SPLIT_MODE_COL,
    ]
    available = [col for col in columns if col in manifest.columns]
    payload = (
        manifest[available]
        .sort_values(available)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _jsonable_params(params: dict) -> dict:
    return {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in params.items()
    }


def _params_from_json(params: dict) -> dict:
    params = dict(params)
    if "encoder_hidden_dims" in params:
        params["encoder_hidden_dims"] = tuple(params["encoder_hidden_dims"])
    return params


def _save_sensitivity_cache(
    result_dir: Path,
    signature: str,
    sensitivity: dict,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    sensitivity["screening_runs"].to_csv(
        result_dir / "sensitivity_screening_runs.csv",
        index=False,
    )
    sensitivity["screening_best"].to_csv(
        result_dir / "sensitivity_screening_best.csv",
        index=False,
    )
    sensitivity["stability_runs"].to_csv(
        result_dir / "sensitivity_stability_runs.csv",
        index=False,
    )
    sensitivity["stability_summary"].to_csv(
        result_dir / "sensitivity_stability_summary.csv",
        index=False,
    )

    with (result_dir / "sensitivity_selection.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "split_signature": signature,
                "best_params": _jsonable_params(
                    sensitivity["best_params"]
                ),
                "best_score_mode": sensitivity["best_score_mode"],
                "best_threshold_method": sensitivity[
                    "best_threshold_method"
                ],
                "selected_features": sensitivity["selected_features"],
                "network_counts": sensitivity["network_counts"],
            },
            handle,
            indent=2,
        )


def _load_sensitivity_cache(
    result_dir: Path,
    signature: str,
) -> dict | None:
    path = result_dir / "sensitivity_selection.json"
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("split_signature") != signature:
        return None

    print("[RESUME] Reusing completed sensitivity analysis.")

    return {
        "best_params": _params_from_json(payload["best_params"]),
        "best_score_mode": payload["best_score_mode"],
        "best_threshold_method": payload["best_threshold_method"],
        "selected_features": payload["selected_features"],
        "network_counts": payload.get("network_counts", {}),
    }


def recalibrate_saved_repeat_threshold(
    dataset_name: str,
    validation_df: pd.DataFrame,
    artifact_dir: Path,
    result_dir: Path,
    score_mode: str,
    split_signature: str,
) -> dict:
    """Recalibrate an already-trained repeat without fitting any network."""
    model_path = artifact_dir / "autoencoder.keras"
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    selected_features_path = artifact_dir / "selected_features.csv"
    decision_rule_path = artifact_dir / "decision_rule.json"

    for path in [model_path, preprocessor_path, selected_features_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot recalibrate; missing saved artifact: {path}"
            )

    model = tf.keras.models.load_model(model_path)
    preprocessor = DatasetPreprocessor.load(preprocessor_path)
    selected_features = pd.read_csv(
        selected_features_path
    )["Feature"].tolist()

    (
        _selected_validation,
        X_validation,
        y_validation,
        actual_features,
    ) = transform_and_select(
        preprocessor,
        validation_df,
        f"{dataset_name}_threshold_recalibration",
        expected_features=selected_features,
    )

    if actual_features != selected_features:
        raise ValueError(
            "Saved repeat feature schema differs during threshold recalibration."
        )

    reconstruction = reconstruct_matrix(
        model,
        X_validation,
        batch_size=512,
    )
    scores = score_from_reconstruction(
        X_validation,
        reconstruction,
        score_mode,
    )

    threshold, threshold_method, candidates = calibrate_final_threshold(
        y_validation,
        scores,
    )
    validation_metrics = calculate_metrics(
        y_validation,
        scores,
        threshold,
    )

    rule = {}
    if decision_rule_path.exists():
        try:
            with decision_rule_path.open("r", encoding="utf-8") as handle:
                rule = json.load(handle)
        except Exception:
            rule = {}

    rule.update(
        {
            "split_signature": split_signature,
            "score_mode": score_mode,
            "threshold_method": threshold_method,
            "threshold_calibration_version": THRESHOLD_CALIBRATION_VERSION,
                        "threshold": float(threshold),
            "anomaly_score_selected_once_per_dataset": True,
            "threshold_recalibrated_per_repeat": True,
            "test_used_for_selection": False,
        }
    )

    with decision_rule_path.open("w", encoding="utf-8") as handle:
        json.dump(rule, handle, indent=2)

    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([validation_metrics]).to_csv(
        result_dir / "final_validation_metrics.csv",
        index=False,
    )
    candidates.to_csv(
        result_dir / "validation_threshold_candidates.csv",
        index=False,
    )

    keras.backend.clear_session()

    return {
        "threshold": float(threshold),
        "threshold_method": threshold_method,
        "validation_metrics": validation_metrics,
    }


# ============================================================
# TRAINING OUTPUTS
# ============================================================


def _save_final_outputs(
    artifact_dir: Path,
    result_dir: Path,
    manifest: pd.DataFrame,
    signature: str,
    sensitivity_selection: dict,
    final: dict,
    repeat_number: int,
    split_strategy: str,
    oversized_sources: list[str],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    final["model"].save(artifact_dir / "autoencoder.keras")
    final["preprocessor"].save(artifact_dir / "preprocessor.joblib")

    manifest.to_csv(
        artifact_dir / "split_manifest.csv",
        index=False,
    )

    pd.DataFrame(
        {"Feature": final["selected_features"]}
    ).to_csv(
        artifact_dir / "selected_features.csv",
        index=False,
    )

    with (artifact_dir / "best_hyperparameters.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            _jsonable_params(sensitivity_selection["best_params"]),
            handle,
            indent=2,
        )

    with (artifact_dir / "decision_rule.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "repeat": int(repeat_number),
                "split_signature": signature,
                "split_strategy": split_strategy,
                "chronologically_split_sources": oversized_sources,
                "score_mode": sensitivity_selection["best_score_mode"],
                "threshold_method": final["threshold_method"],
                "threshold_calibration_version": THRESHOLD_CALIBRATION_VERSION,
                                "threshold": float(final["threshold"]),
                "anomaly_score_selected_once_per_dataset": True,
                "threshold_recalibrated_per_repeat": True,
                "test_used_for_selection": False,
            },
            handle,
            indent=2,
        )

    with (artifact_dir / "training_strategy.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "model": "Dense Autoencoder",
                "evaluation_strategy": (
                    "whole-SourceFile split with chronological oversized-"
                    "SourceFile fallback"
                ),
                "same_split_policy_for_all_datasets": True,
                "repeat": int(repeat_number),
                "split_strategy_for_repeat": split_strategy,
                "chronologically_split_sources": oversized_sources,
                "chronological_rule": (
                    "earliest rows=train, middle rows=validation, latest rows=test"
                ),
                "random_row_split_used": False,
                "imbalance_strategy": (
                    "benign-only reconstruction training"
                ),
                "benign_training_rows": int(
                    final["benign_training_rows"]
                ),
                "preprocessing_fit_scope": (
                    "benign training rows of this repeat only"
                ),
                "sensitivity_selected_once_per_dataset": True,
                "sensitivity_tuning_repeat": 1,
                "sensitivity_training_sample_cap": (
                    SENSITIVITY_MAX_BENIGN_TRAIN_ROWS
                ),
                "sensitivity_validation_sample_cap": (
                    SENSITIVITY_MAX_VALIDATION_ROWS
                ),
                "final_model_uses_all_benign_training_rows": True,
                "final_threshold_uses_full_validation_split": True,
                "final_threshold_calibration_version": THRESHOLD_CALIBRATION_VERSION,
                "final_threshold_selection": (
                    "maximize validation F1; benign-quantile, F2, and "
                    "FPR-constrained thresholds are diagnostic only"
                ),
                "anomaly_score_selected_once_per_dataset": True,
                "threshold_recalibrated_independently_per_repeat": True,
                "best_epoch": int(final["best_epoch"]),
                "best_internal_validation_loss": float(
                    final["best_internal_val_loss"]
                ),
                "test_used_for_model_selection": False,
            },
            handle,
            indent=2,
        )

    pd.DataFrame(final["history"].history).to_csv(
        result_dir / "final_training_history.csv",
        index=False,
    )

    pd.DataFrame([final["validation_metrics"]]).to_csv(
        result_dir / "final_validation_metrics.csv",
        index=False,
    )

    final["threshold_candidates"].to_csv(
        result_dir / "validation_threshold_candidates.csv",
        index=False,
    )


# ============================================================
# TRAIN DATASET(S)
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
    """Remove manifest-only row identifiers before preprocessing/model input."""
    return {
        split_name: split_df.drop(
            columns=[ORIGINAL_ROW_COL],
            errors="ignore",
        ).copy()
        for split_name, split_df in splits.items()
    }



# ============================================================
# SENSITIVITY / STABILITY VISUALIZATION
# ============================================================


def _rank_sensitivity_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(
        by=["pr_auc", "f2", "recall_tpr", "f1", "fpr"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def _deduplicate_parameter_rows(
    rows: pd.DataFrame,
    parameter: str,
) -> pd.DataFrame:
    """Keep the strongest evaluated row for each plotted parameter value."""
    if rows.empty:
        return rows

    value_col = f"param_{parameter}"
    rows = _rank_sensitivity_rows(rows)
    return rows.drop_duplicates(subset=[value_col], keep="first")


def _plot_autoencoder_parameter(
    rows: pd.DataFrame,
    parameter: str,
    output_path: Path,
    title: str,
) -> None:
    if rows.empty:
        return

    value_col = f"param_{parameter}"
    rows = _deduplicate_parameter_rows(rows, parameter)
    labels = [str(value) for value in rows[value_col]]
    x = np.arange(len(rows))
    y = pd.to_numeric(rows["pr_auc"], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel(parameter)
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_sensitivity_plots(
    sensitivity: dict,
    output_dir: Path,
) -> None:
    """
    Create report-ready Autoencoder sensitivity curves and a stability plot.

    Screening uses seed 42. Stability reruns the finalists with seeds
    17/42/73 and reports mean ± standard deviation.
    """
    screening_best = sensitivity["screening_best"].copy()
    stability_summary = sensitivity["stability_summary"].copy()

    if screening_best.empty:
        return

    plot_dir = output_dir / "sensitivity_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: architecture and bottleneck are changed together, so show the
    # evaluated architecture candidates as categorical sensitivity values.
    architecture = screening_best[
        screening_best["stage"] == "A_architecture"
    ].sort_values("config_id")
    if not architecture.empty:
        x = np.arange(len(architecture))
        y = pd.to_numeric(architecture["pr_auc"], errors="coerce").to_numpy()
        labels = architecture["config_name"].astype(str).tolist()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, y, marker="o")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Validation PR-AUC")
        ax.set_title("Autoencoder Sensitivity - Architecture / Bottleneck")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "sensitivity_architecture.png", dpi=300)
        plt.close(fig)

    stage_a_best = _rank_sensitivity_rows(
        screening_best[screening_best["stage"] == "A_architecture"]
    )
    reference_b = stage_a_best.iloc[[0]].copy() if not stage_a_best.empty else pd.DataFrame()

    for parameter in OPTIMIZATION_VALUES:
        varied = screening_best[
            (screening_best["stage"] == "B_optimization")
            & screening_best["config_name"].astype(str).str.startswith(
                f"{parameter}="
            )
        ].copy()
        rows = pd.concat([reference_b, varied], ignore_index=True)
        _plot_autoencoder_parameter(
            rows,
            parameter,
            plot_dir / f"sensitivity_{parameter}.png",
            f"Autoencoder Sensitivity - {parameter}",
        )

    prior_c = _rank_sensitivity_rows(
        screening_best[
            screening_best["stage"].isin(["A_architecture", "B_optimization"])
        ]
    )
    reference_c = prior_c.iloc[[0]].copy() if not prior_c.empty else pd.DataFrame()

    for parameter in REFINEMENT_VALUES:
        varied = screening_best[
            (screening_best["stage"] == "C_refinement")
            & screening_best["config_name"].astype(str).str.startswith(
                f"{parameter}="
            )
        ].copy()
        rows = pd.concat([reference_c, varied], ignore_index=True)
        _plot_autoencoder_parameter(
            rows,
            parameter,
            plot_dir / f"sensitivity_{parameter}.png",
            f"Autoencoder Sensitivity - {parameter}",
        )

    # Best score-mode row per finalist, with seed variability shown as error bars.
    if not stability_summary.empty:
        finalists = (
            stability_summary
            .drop_duplicates(subset=["finalist_id"], keep="first")
            .copy()
        )
        labels = [
            f"{row['finalist_id']}: {row['config_name']} / {row['score_mode']}"
            for _, row in finalists.iterrows()
        ]
        x = np.arange(len(finalists))
        y = pd.to_numeric(finalists["pr_auc_mean"], errors="coerce").to_numpy()
        yerr = pd.to_numeric(
            finalists["pr_auc_std"], errors="coerce"
        ).fillna(0.0).to_numpy()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Validation PR-AUC (mean ± std across seeds)")
        ax.set_title("Autoencoder Stability - Finalists")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "stability_finalists.png", dpi=300)
        plt.close(fig)

def _save_shared_holdout_outputs(
    dataset_name: str,
    manifest: pd.DataFrame,
    split_metadata: dict,
    sensitivity: dict,
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
        artifact_dir / "selected_features.csv",
        index=False,
    )

    with (artifact_dir / "best_hyperparameters.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            _jsonable_params(sensitivity["best_params"]),
            handle,
            indent=2,
        )

    with (artifact_dir / "decision_rule.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "split_signature": signature,
                "shared_split_manifest": str(
                    SHARED_SPLIT_ROOT
                    / f"{dataset_name}_source_aware_split.csv"
                ),
                "split_strategy": split_metadata.get("mode"),
                "oversized_source": split_metadata.get("oversized_source"),
                "score_mode": sensitivity["best_score_mode"],
                "threshold_method": final["threshold_method"],
                "threshold_calibration_version": THRESHOLD_CALIBRATION_VERSION,
                "threshold": float(final["threshold"]),
                "test_used_for_selection": False,
            },
            handle,
            indent=2,
        )

    with (artifact_dir / "training_strategy.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "model": "Dense Autoencoder",
                "evaluation_strategy": (
                    "shared source-aware 70/15/15 holdout reused by LSTM, "
                    "Autoencoder, and Isolation Forest"
                ),
                "split_mode": split_metadata.get("mode"),
                "oversized_source": split_metadata.get("oversized_source"),
                "imbalance_strategy": "benign-only reconstruction training",
                "preprocessing_fit_scope": "benign shared training rows only",
                "detector_fit_scope": "benign shared training rows only",
                "hyperparameter_selection_scope": (
                    "frozen shared D2/D3 Autoencoder configuration; no O1 retuning"
                ),
                "threshold_calibration_scope": "O1 validation split only",
                "optimization_stage": "O1_brittle_feature_removal",
                "shared_configuration": "64-32_b8_lr0.002_huber_l2_1e-5_noise0.02_batch256_top5_mse",
                "dropped_feature_rule": (
                    "drop FWD/Bwd Init Win Bytes, Total Connection Flow Time, "
                    "Fwd/Bwd Header Length, and every *Flag* feature except "
                    "exact URG Flag Count if present"
                ),
                "test_used_for_model_or_threshold_selection": False,
                "benign_training_rows": int(final["benign_training_rows"]),
                "best_epoch": int(final["best_epoch"]),
                "best_internal_validation_loss": float(
                    final["best_internal_val_loss"]
                ),
            },
            handle,
            indent=2,
        )

    sensitivity["screening_runs"].to_csv(
        result_dir / "sensitivity_screening_runs.csv",
        index=False,
    )
    sensitivity["screening_best"].to_csv(
        result_dir / "sensitivity_screening_best.csv",
        index=False,
    )
    sensitivity["stability_runs"].to_csv(
        result_dir / "sensitivity_stability_runs.csv",
        index=False,
    )
    sensitivity["stability_summary"].to_csv(
        result_dir / "sensitivity_stability_summary.csv",
        index=False,
    )
    save_sensitivity_plots(
        sensitivity=sensitivity,
        output_dir=result_dir,
    )

    pd.DataFrame(final["history"].history).to_csv(
        result_dir / "final_training_history.csv",
        index=False,
    )
    pd.DataFrame([final["validation_metrics"]]).to_csv(
        result_dir / "final_validation_metrics.csv",
        index=False,
    )
    final["threshold_candidates"].to_csv(
        result_dir / "validation_threshold_candidates.csv",
        index=False,
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
    ).to_csv(
        result_dir / "shared_split_sizes.csv",
        index=False,
    )


def train_dataset(
    dataset_name: str,
    force: bool = False,
    force_resplit: bool = False,
) -> None:
    """
    Train one Autoencoder using the persisted shared 70/15/15 split.

    The split is exactly the same artifact consumed by LSTM and the revised
    Isolation Forest. Random Forest is intentionally untouched.
    """
    print("\\n" + "=" * 90)
    print(f"AUTOENCODER O1 - BRITTLE FEATURE REMOVAL - {dataset_name.upper()}")
    print("=" * 90)

    input_path = DATASET_PATHS[dataset_name]
    if not input_path.exists():
        raise FileNotFoundError(f"Missing ingested dataset: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    splits, manifest, split_metadata = get_or_create_shared_split(
        dataset_name=dataset_name,
        df=df,
        split_root=SHARED_SPLIT_ROOT,
        force_rebuild=force_resplit,
    )
    splits = _drop_shared_split_helper_columns(splits)

    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    print(
        f"Rows | train={len(train_df):,}, "
        f"validation={len(validation_df):,}, "
        f"test={len(test_df):,}"
    )
    print(
        "Shared split mode: "
        f"{split_metadata.get('mode')}"
    )
    if split_metadata.get("oversized_source") is not None:
        print(
            "Chronologically split oversized SourceFile: "
            f"{split_metadata.get('oversized_source')}"
        )

    artifact_dir = ARTIFACT_ROOT / dataset_name
    decision_rule_path = artifact_dir / "decision_rule.json"
    model_path = artifact_dir / "autoencoder.keras"
    signature = _shared_split_signature(manifest)

    if not force and decision_rule_path.exists() and model_path.exists():
        try:
            with decision_rule_path.open("r", encoding="utf-8") as handle:
                existing_rule = json.load(handle)
            if (
                existing_rule.get("split_signature") == signature
                and existing_rule.get("threshold_calibration_version")
                == THRESHOLD_CALIBRATION_VERSION
            ):
                print(
                    "[RESUME] Autoencoder artifacts already match the exact "
                    "shared split. Skipping retraining."
                )
                return
        except Exception:
            pass

    baseline_params, baseline_score_mode = load_shared_fixed_configuration()

    print("\nO1 controlled experiment - frozen shared D2/D3 configuration:")
    for key, value in baseline_params.items():
        print(f"  {key}: {value}")
    print(f"Frozen anomaly score mode: {baseline_score_mode}")
    print("O1 change: brittle feature pruning only; shared model hyperparameters and Top-5 MSE score are frozen.")

    benign_train_df = benign_only(train_df)
    schema_preprocessor = DatasetPreprocessor(f"{dataset_name}_o1_schema")
    schema_preprocessor.fit(benign_train_df)
    _tmp_df, _tmp_X, _tmp_y, o1_features = transform_and_select(
        schema_preprocessor,
        benign_train_df,
        dataset_name=f"{dataset_name}_o1_schema",
        expected_features=None,
    )

    print(f"O1 feature count: {len(o1_features)}")
    print("Mandatory O1 drops:")
    for feature in O1_EXPLICIT_DROP_FEATURES:
        print(f"  - {feature}")
    print(
        "Flag rule: drop every *Flag* feature except exact aggregate "
        f"{O1_FLAG_KEEP!r}, if present."
    )

    final = train_final_model(
        dataset_name=dataset_name,
        train_df=train_df,
        validation_df=validation_df,
        best_params=baseline_params,
        score_mode=baseline_score_mode,
        expected_selected_features=o1_features,
    )

    empty = pd.DataFrame()
    sensitivity = {
        "best_params": baseline_params,
        "best_score_mode": baseline_score_mode,
        "screening_runs": empty.copy(),
        "screening_best": empty.copy(),
        "stability_runs": empty.copy(),
        "stability_summary": empty.copy(),
    }


    _save_shared_holdout_outputs(
        dataset_name=dataset_name,
        manifest=manifest,
        split_metadata=split_metadata,
        sensitivity=sensitivity,
        final=final,
    )

    print(f"Validation threshold: {final['threshold']:.8f}")
    print(f"Benign training rows: {final['benign_training_rows']:,}")
    print(f"Untouched test rows: {len(test_df):,}")


def train_all_datasets(
    force: bool = False,
    force_resplit: bool = False,
) -> None:
    print(
        "Using frozen shared Autoencoder configuration for Dataset 2 and Dataset 3: "
        "64-32, bottleneck=8, LR=0.002, Huber, L2=1e-5, noise=0.02, "
        "batch=256, score=Top-5 MSE."
    )
    for dataset_name in DATASET_PATHS:
        train_dataset(
            dataset_name,
            force=force,
            force_resplit=force_resplit,
        )
