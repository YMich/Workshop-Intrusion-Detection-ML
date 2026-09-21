from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

# This file:
#   <PROJECT_ROOT>/src/models_baseline/IF/isolation_forest.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]

INGESTED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ingested"
)

DATASET_PATHS = {
    "dataset2": INGESTED_ROOT / "dataset2_ingested.csv",
    "dataset3": INGESTED_ROOT / "dataset3_ingested.csv",
}
BENCHMARK_DATASETS = tuple(DATASET_PATHS.keys())

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "models_baseline"
    / "IF"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "models_baseline"
    / "IF"
)

SHARED_SPLIT_ROOT = PROJECT_ROOT / "artifacts" / "splits"
SHARED_HYPERPARAMETER_PATH = ARTIFACT_ROOT / "shared_configuration.json"


# =============================================================================
# REUSE STEP 2 + STEP 3
# =============================================================================

MODEL_DIR = Path(__file__).resolve().parent

for path in [
    MODEL_DIR,
    SRC_ROOT / "preprocessing",
    SRC_ROOT / "feature_engineering",
]:
    path_text = str(path)

    if path_text not in sys.path:
        sys.path.insert(
            0,
            path_text,
        )


from preprocessing import DatasetPreprocessor  # noqa: E402
from feature_selection import select_and_engineer_dataset  # noqa: E402
from source_aware_split import (  # noqa: E402
    ORIGINAL_ROW_COL,
    get_or_create_shared_split,
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
TIMESTAMP_COL = "Timestamp"

BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


# Explicit fixed Isolation Forest baseline configuration.
# The same values are used for Dataset 2 and Dataset 3. No hyperparameter
# search or sensitivity sweep is performed during execution.
BASE_PARAMS = {
    "n_estimators": 200,
    "max_samples": 512,
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "n_jobs": -1,
    "verbose": 0,
    "warm_start": False,
}

FINAL_RANDOM_STATE = 42

# =============================================================================
# CHRONOLOGICAL SPLIT WITHIN SOURCEFILE
# =============================================================================

def _validate_split_fractions() -> None:
    total = (
        TRAIN_FRACTION
        + VALIDATION_FRACTION
        + TEST_FRACTION
    )

    if not np.isclose(
        total,
        1.0,
    ):
        raise ValueError(
            "Train/validation/test fractions must sum to 1.0; "
            f"got {total}."
        )


def _allocate_sourcefile_counts(
    n_rows: int,
) -> tuple[int, int, int]:
    """
    Allocate one SourceFile chronologically.

    For normal-sized SourceFiles:
        earliest 70% -> train
        next 15%     -> validation
        latest 15%   -> test

    Very small SourceFiles cannot populate every fold perfectly:
        n=1 -> train
        n=2 -> train/test
        n>=3 -> at least one row in each fold
    """
    if n_rows <= 0:
        raise ValueError(
            "SourceFile row count must be positive."
        )

    if n_rows == 1:
        return (
            1,
            0,
            0,
        )

    if n_rows == 2:
        return (
            1,
            0,
            1,
        )

    train_count = int(
        np.floor(
            TRAIN_FRACTION
            * n_rows
        )
    )

    validation_count = int(
        np.floor(
            VALIDATION_FRACTION
            * n_rows
        )
    )

    train_count = max(
        1,
        train_count,
    )

    validation_count = max(
        1,
        validation_count,
    )

    test_count = (
        n_rows
        - train_count
        - validation_count
    )

    if test_count < 1:
        shortage = (
            1
            - test_count
        )

        removable_from_train = max(
            0,
            train_count
            - 1,
        )

        take_from_train = min(
            shortage,
            removable_from_train,
        )

        train_count -= (
            take_from_train
        )

        shortage -= (
            take_from_train
        )

        if shortage > 0:
            removable_from_validation = max(
                0,
                validation_count
                - 1,
            )

            take_from_validation = min(
                shortage,
                removable_from_validation,
            )

            validation_count -= (
                take_from_validation
            )

            shortage -= (
                take_from_validation
            )

        test_count = (
            n_rows
            - train_count
            - validation_count
        )

    if (
        train_count
        + validation_count
        + test_count
        != n_rows
    ):
        raise RuntimeError(
            "Internal chronological split allocation error."
        )

    if min(
        train_count,
        validation_count,
        test_count,
    ) < 1:
        raise RuntimeError(
            "SourceFile with >=3 rows did not populate all folds."
        )

    return (
        train_count,
        validation_count,
        test_count,
    )


def chronological_sourcefile_split(
    df: pd.DataFrame,
):
    """
    Chronological 70/15/15 split INSIDE each SourceFile.

    This solves the problem where one enormous SourceFile made a whole-file
    70/15/15 partition mathematically impossible.

    For every SourceFile:
        earliest rows -> train
        middle rows   -> validation
        latest rows   -> test

    The order is determined by Timestamp using stable sorting.

    Important tradeoff:
        The same SourceFile can contribute rows to multiple folds, but a later
        flow from that SourceFile can never appear in train before an earlier
        flow from that same SourceFile appears in validation/test.
    """
    _validate_split_fractions()

    required = {
        SOURCE_FILE_COL,
        TIMESTAMP_COL,
        LABEL_COL,
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing columns required for chronological splitting: "
            f"{sorted(missing)}"
        )

    working = df.copy()

    working[
        TIMESTAMP_COL
    ] = pd.to_datetime(
        working[
            TIMESTAMP_COL
        ],
        errors="coerce",
    )

    invalid_timestamps = int(
        working[
            TIMESTAMP_COL
        ]
        .isna()
        .sum()
    )

    if invalid_timestamps:
        raise ValueError(
            f"{invalid_timestamps} rows contain invalid Timestamp values."
        )

    split_parts = {
        "train": [],
        "validation": [],
        "test": [],
    }

    manifest_rows = []

    for (
        source_file,
        source_df,
    ) in working.groupby(
        SOURCE_FILE_COL,
        sort=True,
    ):
        labels = (
            source_df[
                LABEL_COL
            ]
            .astype(int)
            .unique()
        )

        if len(
            labels
        ) != 1:
            raise ValueError(
                f"SourceFile '{source_file}' contains mixed labels: "
                f"{labels.tolist()}"
            )

        source_df = (
            source_df
            .sort_values(
                TIMESTAMP_COL,
                kind="mergesort",
            )
            .copy()
        )

        n_rows = len(
            source_df
        )

        (
            n_train,
            n_validation,
            n_test,
        ) = (
            _allocate_sourcefile_counts(
                n_rows
            )
        )

        train_end = (
            n_train
        )

        validation_end = (
            n_train
            + n_validation
        )

        train_part = (
            source_df.iloc[
                :train_end
            ]
            .copy()
        )

        validation_part = (
            source_df.iloc[
                train_end:
                validation_end
            ]
            .copy()
        )

        test_part = (
            source_df.iloc[
                validation_end:
            ]
            .copy()
        )

        split_parts[
            "train"
        ].append(
            train_part
        )

        if not validation_part.empty:
            split_parts[
                "validation"
            ].append(
                validation_part
            )

        if not test_part.empty:
            split_parts[
                "test"
            ].append(
                test_part
            )

        for (
            split_name,
            part,
        ) in [
            (
                "train",
                train_part,
            ),
            (
                "validation",
                validation_part,
            ),
            (
                "test",
                test_part,
            ),
        ]:
            if part.empty:
                continue

            manifest_rows.append({
                SOURCE_FILE_COL: (
                    source_file
                ),
                "Label": int(
                    labels[
                        0
                    ]
                ),
                "Split": (
                    split_name
                ),
                "Rows": int(
                    len(
                        part
                    )
                ),
                "Start_Timestamp": (
                    part[
                        TIMESTAMP_COL
                    ]
                    .iloc[
                        0
                    ]
                    .isoformat()
                ),
                "End_Timestamp": (
                    part[
                        TIMESTAMP_COL
                    ]
                    .iloc[
                        -1
                    ]
                    .isoformat()
                ),
            })

    folds = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        if not split_parts[
            split_name
        ]:
            raise ValueError(
                f"{split_name} fold is empty."
            )

        folds[
            split_name
        ] = (
            pd.concat(
                split_parts[
                    split_name
                ],
                ignore_index=True,
            )
            .sort_values(
                [
                    SOURCE_FILE_COL,
                    TIMESTAMP_COL,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        labels = set(
            folds[
                split_name
            ][
                LABEL_COL
            ]
            .astype(int)
            .unique()
        )

        if labels != {
            BENIGN_LABEL,
            MALICIOUS_LABEL,
        }:
            raise ValueError(
                f"{split_name} fold does not contain both classes."
            )

    manifest = (
        pd.DataFrame(
            manifest_rows
        )
        .sort_values(
            [
                SOURCE_FILE_COL,
                "Split",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary = (
        build_split_summary(
            folds
        )
    )

    return (
        folds[
            "train"
        ],
        folds[
            "validation"
        ],
        folds[
            "test"
        ],
        manifest,
        summary,
    )


def build_split_summary(
    folds: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    total_rows = sum(
        len(
            frame
        )
        for frame in (
            folds.values()
        )
    )

    rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        frame = (
            folds[
                split_name
            ]
        )

        labels = (
            frame[
                LABEL_COL
            ]
            .astype(int)
        )

        benign_rows = int(
            (
                labels
                == BENIGN_LABEL
            )
            .sum()
        )

        malicious_rows = int(
            (
                labels
                == MALICIOUS_LABEL
            )
            .sum()
        )

        rows.append({
            "Split": (
                split_name
            ),
            "Rows": int(
                len(
                    frame
                )
            ),
            "Row_Fraction": float(
                len(
                    frame
                )
                / total_rows
            ),
            "Benign_Rows": (
                benign_rows
            ),
            "Malicious_Rows": (
                malicious_rows
            ),
            "Malicious_Percent": float(
                100.0
                * malicious_rows
                / len(
                    frame
                )
            ),
            "SourceFiles": int(
                frame[
                    SOURCE_FILE_COL
                ]
                .nunique()
            ),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# ONE-CLASS TRAINING + EXISTING PIPELINE
# =============================================================================

def benign_only(
    df: pd.DataFrame,
) -> pd.DataFrame:
    benign = (
        df[
            df[
                LABEL_COL
            ]
            .astype(int)
            == BENIGN_LABEL
        ]
        .copy()
    )

    if benign.empty:
        raise ValueError(
            "No benign rows are available for Isolation Forest fitting."
        )

    return benign


def transform_and_select(
    preprocessor,
    df,
    dataset_name,
    expected_features=None,
):
    """
    Reuse Step 2 and Step 3 exactly as implemented elsewhere.
    """
    normalized = (
        preprocessor.transform(
            df
        )
    )

    (
        selected_df,
        selected_features,
        _dropped,
        _manifest,
    ) = (
        select_and_engineer_dataset(
            dataset_name=(
                dataset_name
            ),
            df=(
                normalized
            ),
            expected_selected_features=(
                expected_features
            ),
        )
    )

    X = (
        selected_df[
            selected_features
        ]
        .copy()
    )

    y = (
        selected_df[
            LABEL_COL
        ]
        .astype(int)
        .copy()
    )

    return (
        selected_df,
        X,
        y,
        selected_features,
    )


# =============================================================================
# MODEL + ANOMALY SCORE
# =============================================================================

def build_model(
    params: dict,
    random_state: int,
) -> IsolationForest:
    missing = (
        set(
            BASE_PARAMS
        )
        - set(
            params
        )
    )

    if missing:
        raise ValueError(
            "Missing explicit Isolation Forest parameters: "
            f"{sorted(missing)}"
        )

    return IsolationForest(
        n_estimators=(
            params[
                "n_estimators"
            ]
        ),
        max_samples=(
            params[
                "max_samples"
            ]
        ),
        contamination=(
            params[
                "contamination"
            ]
        ),
        max_features=(
            params[
                "max_features"
            ]
        ),
        bootstrap=(
            params[
                "bootstrap"
            ]
        ),
        n_jobs=(
            params[
                "n_jobs"
            ]
        ),
        random_state=(
            random_state
        ),
        verbose=(
            params[
                "verbose"
            ]
        ),
        warm_start=(
            params[
                "warm_start"
            ]
        ),
    )


def anomaly_scores(
    model: IsolationForest,
    X,
) -> np.ndarray:
    """
    Higher score = more anomalous.

    score_samples() is used rather than decision_function() so the anomaly
    ranking is independent of sklearn's internal offset.
    """
    return (
        -model.score_samples(
            X
        )
    )


# =============================================================================
# METRICS + MAX-F1 THRESHOLD
# =============================================================================

def calculate_metrics(
    y_true,
    scores,
    threshold: float,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    y_pred = (
        scores
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[
                BENIGN_LABEL,
                MALICIOUS_LABEL,
            ],
        )
        .ravel()
    )

    specificity = (
        tn
        / (
            tn
            + fp
        )
        if (
            tn
            + fp
        )
        else 0.0
    )

    fpr = (
        fp
        / (
            fp
            + tn
        )
        if (
            fp
            + tn
        )
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall_tpr": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity_tnr": float(
            specificity
        ),
        "fpr": float(
            fpr
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "tn": int(
            tn
        ),
        "fp": int(
            fp
        ),
        "fn": int(
            fn
        ),
        "tp": int(
            tp
        ),
        "threshold": float(
            threshold
        ),
    }


def choose_threshold_max_f1(
    y_validation,
    validation_scores,
):
    """
    Choose the anomaly-score threshold that maximizes F1 on VALIDATION.

    Test data is never used.

    Tie-breaking:
        1. highest F1
        2. highest recall
        3. highest precision
        4. lower FPR
    """
    y_validation = np.asarray(
        y_validation,
        dtype=int,
    )

    validation_scores = np.asarray(
        validation_scores,
        dtype=float,
    )

    if set(
        np.unique(
            y_validation
        )
    ) != {
        BENIGN_LABEL,
        MALICIOUS_LABEL,
    }:
        raise ValueError(
            "Validation labels must contain both classes."
        )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_validation,
            validation_scores,
            pos_label=(
                MALICIOUS_LABEL
            ),
        )
    )

    if len(
        thresholds
    ) == 0:
        raise ValueError(
            "No threshold candidates were produced."
        )

    # precision/recall have one more element than thresholds.
    precision = (
        precision[
            :-1
        ]
    )

    recall = (
        recall[
            :-1
        ]
    )

    denominator = (
        precision
        + recall
    )

    f1_values = np.divide(
        2.0
        * precision
        * recall,
        denominator,
        out=np.zeros_like(
            denominator,
            dtype=float,
        ),
        where=(
            denominator
            > 0
        ),
    )

    best_f1 = float(
        np.max(
            f1_values
        )
    )

    candidate_indices = np.flatnonzero(
        np.isclose(
            f1_values,
            best_f1,
            rtol=1e-12,
            atol=1e-12,
        )
    )

    candidate_rows = []

    for index in (
        candidate_indices
    ):
        threshold = float(
            thresholds[
                index
            ]
        )

        result = (
            calculate_metrics(
                y_validation,
                validation_scores,
                threshold,
            )
        )

        candidate_rows.append(
            result
        )

    candidate_rows = sorted(
        candidate_rows,
        key=lambda result: (
            result[
                "f1"
            ],
            result[
                "recall_tpr"
            ],
            result[
                "precision"
            ],
            -result[
                "fpr"
            ],
        ),
        reverse=True,
    )

    best_result = (
        candidate_rows[
            0
        ]
    )

    threshold = float(
        best_result[
            "threshold"
        ]
    )

    return (
        threshold,
        best_result,
    )



# =============================================================================
# FINAL MODEL
# =============================================================================

def train_final_model(
    dataset_name,
    train_df,
    validation_df,
    best_params,
    selected_features,
):
    """
    Final model:
        Step 2 fit on benign TRAIN
        Isolation Forest fit on benign TRAIN
        max-F1 threshold learned on VALIDATION
        TEST untouched

    We intentionally do not refit after threshold calibration because changing
    the scaler or model would change the numerical anomaly-score distribution.
    """
    benign_train_df = (
        benign_only(
            train_df
        )
    )

    preprocessor = (
        DatasetPreprocessor(
            f"{dataset_name}_"
            "if_final"
        )
    )

    preprocessor.fit(
        benign_train_df
    )

    (
        _selected_train,
        X_train,
        y_train,
        final_features,
    ) = (
        transform_and_select(
            preprocessor,
            benign_train_df,
            dataset_name,
            expected_features=(
                selected_features
            ),
        )
    )

    if set(
        y_train.unique()
    ) != {
        BENIGN_LABEL
    }:
        raise RuntimeError(
            "Malicious rows reached final Isolation Forest fitting."
        )

    model = (
        build_model(
            best_params,
            FINAL_RANDOM_STATE,
        )
    )

    model.fit(
        X_train
    )

    (
        _selected_validation,
        X_validation,
        y_validation,
        _,
    ) = (
        transform_and_select(
            preprocessor,
            validation_df,
            dataset_name,
            expected_features=(
                final_features
            ),
        )
    )

    validation_scores = (
        anomaly_scores(
            model,
            X_validation,
        )
    )

    (
        operating_threshold,
        validation_metrics,
    ) = (
        choose_threshold_max_f1(
            y_validation,
            validation_scores,
        )
    )

    return (
        model,
        preprocessor,
        final_features,
        operating_threshold,
        validation_metrics,
        len(
            X_train
        ),
    )


# =============================================================================
# TRAIN + SAVE
# =============================================================================

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


def _load_dataset_split(
    dataset_name: str,
    force_resplit: bool = False,
):
    path = DATASET_PATHS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing ingested dataset: {path}")

    df = pd.read_csv(path, low_memory=False)
    splits, split_manifest, split_metadata = get_or_create_shared_split(
        dataset_name=dataset_name,
        df=df,
        split_root=SHARED_SPLIT_ROOT,
        force_rebuild=force_resplit,
    )
    splits = _drop_shared_split_helper_columns(splits)
    return splits, split_manifest, split_metadata


def _print_split_summary(
    dataset_name: str,
    splits: dict[str, pd.DataFrame],
    split_metadata: dict,
) -> None:
    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    print("\\n" + "-" * 90)
    print(f"{dataset_name.upper()} SPLIT")
    print("-" * 90)
    print(
        f"Rows | train={len(train_df):,}, "
        f"validation={len(validation_df):,}, "
        f"test={len(test_df):,}"
    )
    print(f"Shared split mode: {split_metadata.get('mode')}")
    if split_metadata.get("oversized_source") is not None:
        print(
            "Chronologically split oversized SourceFile: "
            f"{split_metadata.get('oversized_source')}"
        )






def save_dataset_training_outputs(
    dataset_name: str,
    split_manifest: pd.DataFrame,
    split_metadata: dict,
    best_params: dict,
    model,
    preprocessor,
    final_features: list[str],
    operating_threshold: float,
    validation_metrics: dict,
    benign_training_rows: int,
) -> None:
    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_dir / "isolation_forest.joblib")
    preprocessor.save(artifact_dir / "preprocessor.joblib")
    split_manifest.to_csv(artifact_dir / "split_manifest.csv", index=False)
    pd.DataFrame({"Feature": final_features}).to_csv(
        artifact_dir / "selected_features.csv", index=False
    )

    # Exact same file contents for D2 and D3.
    with (artifact_dir / "model_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(best_params, handle, indent=2)

    operating_point = {
        "threshold": float(operating_threshold),
        "threshold_rule": "Anomaly_Score >= threshold => Malicious",
        "threshold_objective": "maximize validation F1",
        "score_definition": "-IsolationForest.score_samples(X)",
        "validation_metrics": validation_metrics,
    }
    with (artifact_dir / "operating_threshold.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(operating_point, handle, indent=2)

    with (artifact_dir / "training_strategy.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "benchmark_datasets": list(BENCHMARK_DATASETS),
                "hyperparameter_scope": "shared across Dataset 2 and Dataset 3",
                "configuration_mode": (
                    "fixed shared baseline configuration; no hyperparameter search "
                    "is performed during this run"
                ),
                "imbalance_strategy": "benign-only one-class Isolation Forest",
                "split_strategy": (
                    "shared source-aware 70/15/15 holdout reused by sequence/one-class models"
                ),
                "split_mode": split_metadata.get("mode"),
                "oversized_source": split_metadata.get("oversized_source"),
                "preprocessing_fit_scope": "this dataset's benign training rows only",
                "detector_fit_scope": "this dataset's benign training rows only",
                "threshold_calibration_scope": (
                    "this dataset's mixed validation split only"
                ),
                "threshold_objective": "maximize validation F1",
                "malicious_rows_used_to_fit_detector": 0,
                "benign_training_rows": int(benign_training_rows),
                "test_used_for_model_or_threshold_selection": False,
                "shared_hyperparameter_artifact": str(SHARED_HYPERPARAMETER_PATH),
            },
            handle,
            indent=2,
        )

    pd.DataFrame(
        [{"Dataset": dataset_name, **validation_metrics}]
    ).to_csv(result_dir / "validation_operating_point.csv", index=False)

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


def train_all_datasets(
    force_resplit: bool = False,
):
    """Train one fixed-configuration Isolation Forest per benchmark dataset."""
    print("\n" + "=" * 90)
    print("ISOLATION FOREST - FIXED D2/D3 BASELINE")
    print("=" * 90)

    best_params = dict(BASE_PARAMS)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with SHARED_HYPERPARAMETER_PATH.open("w", encoding="utf-8") as handle:
        json.dump(best_params, handle, indent=2)

    for dataset_name in BENCHMARK_DATASETS:
        splits, split_manifest, split_metadata = _load_dataset_split(
            dataset_name,
            force_resplit=force_resplit,
        )
        _print_split_summary(dataset_name, splits, split_metadata)

        print("\n" + "-" * 90)
        print(f"TRAIN FIXED BASELINE - {dataset_name.upper()}")
        print("-" * 90)

        (
            model,
            preprocessor,
            final_features,
            operating_threshold,
            validation_metrics,
            benign_training_rows,
        ) = train_final_model(
            dataset_name,
            splits["train"],
            splits["validation"],
            best_params,
            None,
        )

        print("Validation operating point (maximum F1):")
        print(f"  threshold: {operating_threshold:.8f}")
        print(f"  Precision: {validation_metrics['precision']:.4f}")
        print(f"  Recall / TPR: {validation_metrics['recall_tpr']:.4f}")
        print(f"  FPR: {validation_metrics['fpr']:.4f}")

        save_dataset_training_outputs(
            dataset_name=dataset_name,
            split_manifest=split_manifest,
            split_metadata=split_metadata,
            best_params=best_params,
            model=model,
            preprocessor=preprocessor,
            final_features=final_features,
            operating_threshold=operating_threshold,
            validation_metrics=validation_metrics,
            benign_training_rows=benign_training_rows,
        )

        print(f"Artifacts: {ARTIFACT_ROOT / dataset_name}")
        print(f"Results: {RESULT_ROOT / dataset_name}")


