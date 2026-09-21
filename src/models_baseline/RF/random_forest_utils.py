from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from random_forest_config import (
    BASE_RF_PARAMS,
    CV_RANDOM_STATE,
    CV_SHUFFLE,
    LABEL_COL,
    MODEL_RANDOM_STATE,
    PREDICTION_THRESHOLD,
    SOURCE_FILE_COL,
)


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_tpr",
    "specificity_tnr",
    "fpr",
    "f1",
    "roc_auc",
    "pr_auc",
    "mcc",
]


def validate_source_groups(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Validate that SourceFile is safe to use as a group boundary.

    Every source/capture must contain exactly one class.
    """
    required = {
        SOURCE_FILE_COL,
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
            f"{dataset_name}: "
            f"missing group columns: {sorted(missing)}"
        )

    if (
        df[
            SOURCE_FILE_COL
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            f"{dataset_name}: "
            f"{SOURCE_FILE_COL} contains missing values."
        )

    group_table = (
        df.groupby(
            SOURCE_FILE_COL,
            sort=True,
            dropna=False,
        )
        .agg(
            Label=(
                LABEL_COL,
                "first",
            ),
            Label_Nunique=(
                LABEL_COL,
                "nunique",
            ),
            Rows=(
                LABEL_COL,
                "size",
            ),
        )
        .reset_index()
    )

    mixed = (
        group_table[
            group_table[
                "Label_Nunique"
            ]
            != 1
        ]
    )

    if not mixed.empty:
        raise ValueError(
            f"{dataset_name}: "
            "at least one SourceFile contains both classes:\n"
            f"{mixed.to_string(index=False)}"
        )

    group_table = (
        group_table.drop(
            columns=[
                "Label_Nunique",
            ]
        )
    )

    group_table[
        LABEL_COL
    ] = (
        pd.to_numeric(
            group_table[
                LABEL_COL
            ],
            errors="raise",
        )
        .astype(int)
    )

    labels = set(
        group_table[
            LABEL_COL
        ]
        .unique()
        .tolist()
    )

    if labels != {
        0,
        1,
    }:
        raise ValueError(
            f"{dataset_name}: "
            f"expected labels {{0, 1}}, found {sorted(labels)}"
        )

    return group_table


def feasible_group_fold_count(
    df: pd.DataFrame,
    requested_folds: int,
    dataset_name: str,
) -> int:
    """
    The number of group-CV folds cannot exceed the number of SourceFiles in
    the smaller class if every fold is expected to contain both classes.
    """
    groups = (
        validate_source_groups(
            df,
            dataset_name,
        )
    )

    class_group_counts = (
        groups[
            LABEL_COL
        ]
        .value_counts()
        .sort_index()
    )

    maximum = int(
        class_group_counts.min()
    )

    folds = min(
        int(
            requested_folds
        ),
        maximum,
    )

    if folds < 2:
        raise ValueError(
            f"{dataset_name}: "
            "not enough SourceFiles per class for group cross-validation. "
            f"SourceFile counts={class_group_counts.to_dict()}"
        )

    return folds


def make_stratified_group_folds(
    df: pd.DataFrame,
    requested_folds: int,
    dataset_name: str,
    random_state: int = CV_RANDOM_STATE,
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
    ]
]:
    """
    Create SourceFile-disjoint stratified group folds.

    StratifiedGroupKFold is heuristic: even when enough benign and malicious
    SourceFiles exist, a particular shuffled assignment can occasionally leave
    one held-out fold with only one class. That is invalid for this project's
    binary metrics and source-aware group-OOF protocol.

    Therefore we deterministically retry consecutive random seeds until we find
    an assignment in which:
      * every train side contains both classes;
      * every validation/test side contains both classes;
      * no SourceFile crosses the fold boundary.

    The number of folds is NOT reduced merely because one shuffled assignment
    is invalid. Positions returned by sklearn are positional indices and must
    be used with .iloc.
    """
    n_splits = feasible_group_fold_count(
        df,
        requested_folds,
        dataset_name,
    )

    y = (
        pd.to_numeric(
            df[LABEL_COL],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    groups = (
        df[SOURCE_FILE_COL]
        .astype(str)
        .to_numpy()
    )

    max_attempts = 1000
    last_failure = None

    for attempt in range(max_attempts):
        effective_seed = int(random_state) + attempt

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=CV_SHUFFLE,
            random_state=(
                effective_seed
                if CV_SHUFFLE
                else None
            ),
        )

        candidate_folds = list(
            splitter.split(
                X=np.zeros(
                    (len(df), 1),
                    dtype=np.int8,
                ),
                y=y,
                groups=groups,
            )
        )

        valid = True

        for fold_number, (
            train_positions,
            test_positions,
        ) in enumerate(candidate_folds, start=1):

            train_df = df.iloc[train_positions]
            test_df = df.iloc[test_positions]

            train_sources = set(
                train_df[SOURCE_FILE_COL]
                .astype(str)
                .unique()
            )
            test_sources = set(
                test_df[SOURCE_FILE_COL]
                .astype(str)
                .unique()
            )

            overlap = train_sources & test_sources
            if overlap:
                raise RuntimeError(
                    f"{dataset_name}: SourceFile leakage in fold "
                    f"{fold_number}: {sorted(overlap)}"
                )

            train_labels = set(
                pd.to_numeric(
                    train_df[LABEL_COL],
                    errors="raise",
                )
                .astype(int)
                .unique()
                .tolist()
            )

            test_labels = set(
                pd.to_numeric(
                    test_df[LABEL_COL],
                    errors="raise",
                )
                .astype(int)
                .unique()
                .tolist()
            )

            if train_labels != {0, 1} or test_labels != {0, 1}:
                valid = False
                last_failure = (
                    f"seed={effective_seed}, fold={fold_number}, "
                    f"train_labels={sorted(train_labels)}, "
                    f"validation/test_labels={sorted(test_labels)}"
                )
                break

        if valid:
            if attempt > 0:
                print(
                    f"{dataset_name}: StratifiedGroupKFold seed "
                    f"{random_state} produced an invalid class allocation; "
                    f"using deterministic retry seed {effective_seed}."
                )
            return candidate_folds

    raise ValueError(
        f"{dataset_name}: could not construct {n_splits} valid "
        f"StratifiedGroupKFold folds after {max_attempts} deterministic "
        f"seed attempts. Last failure: {last_failure}"
    )


def build_fold_manifest(
    df: pd.DataFrame,
    folds: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    row_index_col: str,
) -> pd.DataFrame:
    """
    Every row receives exactly one OUTER test-fold assignment.
    """
    outer_fold = np.full(
        len(
            df
        ),
        -1,
        dtype=np.int16,
    )

    for fold_number, (
        _train_positions,
        test_positions,
    ) in enumerate(
        folds,
        start=1,
    ):
        if np.any(
            outer_fold[
                test_positions
            ]
            != -1
        ):
            raise RuntimeError(
                "A row was assigned to more than one outer test fold."
            )

        outer_fold[
            test_positions
        ] = fold_number

    if np.any(
        outer_fold
        == -1
    ):
        raise RuntimeError(
            "At least one row was never assigned to an outer test fold."
        )

    manifest = pd.DataFrame({
        row_index_col: np.arange(
            len(
                df
            ),
            dtype=np.int64,
        ),
        "Outer_Fold": (
            outer_fold
        ),
        LABEL_COL: (
            pd.to_numeric(
                df[
                    LABEL_COL
                ],
                errors="raise",
            )
            .astype(int)
            .to_numpy()
        ),
        SOURCE_FILE_COL: (
            df[
                SOURCE_FILE_COL
            ]
            .astype(str)
            .to_numpy()
        ),
    })

    # Hard group integrity check:
    # each SourceFile may map to only one outer test fold.
    per_source_fold_counts = (
        manifest.groupby(
            SOURCE_FILE_COL
        )[
            "Outer_Fold"
        ]
        .nunique()
    )

    if (
        per_source_fold_counts
        > 1
    ).any():
        bad = (
            per_source_fold_counts[
                per_source_fold_counts
                > 1
            ]
            .index
            .tolist()
        )

        raise RuntimeError(
            "A SourceFile was assigned to multiple outer test folds: "
            f"{bad}"
        )

    return manifest


def compute_training_class_weights(
    y: pd.Series,
) -> dict[
    int,
    float,
]:
    """
    Cost-sensitive learning using the CURRENT training fold only.
    """
    y = (
        pd.Series(
            y
        )
        .astype(int)
    )

    classes = np.array(
        [
            0,
            1,
        ],
        dtype=int,
    )

    present = set(
        y.unique()
        .tolist()
    )

    if present != {
        0,
        1,
    }:
        raise ValueError(
            f"Training labels must contain 0 and 1. Found {sorted(present)}"
        )

    weights = (
        compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y.to_numpy(),
        )
    )

    return {
        int(label): float(weight)
        for label, weight in zip(
            classes,
            weights,
        )
    }


def build_random_forest(
    params: dict,
    class_weights: dict[
        int,
        float,
    ],
    random_state: int = MODEL_RANDOM_STATE,
) -> RandomForestClassifier:
    """
    Construct the RF with explicit project-controlled hyperparameters.
    """
    required = set(
        BASE_RF_PARAMS.keys()
    )

    missing = (
        required
        - set(
            params.keys()
        )
    )

    if missing:
        raise ValueError(
            f"Missing explicit RF parameters: {sorted(missing)}"
        )

    return RandomForestClassifier(
        n_estimators=(
            params[
                "n_estimators"
            ]
        ),
        criterion=(
            params[
                "criterion"
            ]
        ),
        max_depth=(
            params[
                "max_depth"
            ]
        ),
        min_samples_split=(
            params[
                "min_samples_split"
            ]
        ),
        min_samples_leaf=(
            params[
                "min_samples_leaf"
            ]
        ),
        min_weight_fraction_leaf=(
            params[
                "min_weight_fraction_leaf"
            ]
        ),
        max_features=(
            params[
                "max_features"
            ]
        ),
        max_leaf_nodes=(
            params[
                "max_leaf_nodes"
            ]
        ),
        min_impurity_decrease=(
            params[
                "min_impurity_decrease"
            ]
        ),
        bootstrap=(
            params[
                "bootstrap"
            ]
        ),
        oob_score=(
            params[
                "oob_score"
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
        class_weight=(
            class_weights
        ),
        ccp_alpha=(
            params[
                "ccp_alpha"
            ]
        ),
        max_samples=(
            params[
                "max_samples"
            ]
        ),
    )


def calculate_binary_metrics(
    y_true,
    y_probability,
    threshold: float = PREDICTION_THRESHOLD,
) -> dict[
    str,
    float,
]:
    """
    Imbalance-aware binary classification metrics.
    """
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

    y_pred = (
        y_probability
        >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
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
                y_probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability,
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








