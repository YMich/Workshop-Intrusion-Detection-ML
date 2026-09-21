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


def _validate_group_folds(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    dataset_name: str,
) -> tuple[bool, str | None]:
    """Validate SourceFile disjointness and two-class presence on both sides."""
    for fold_number, (train_positions, test_positions) in enumerate(folds, start=1):
        train_df = df.iloc[train_positions]
        test_df = df.iloc[test_positions]

        train_sources = set(train_df[SOURCE_FILE_COL].astype(str).unique())
        test_sources = set(test_df[SOURCE_FILE_COL].astype(str).unique())

        overlap = train_sources & test_sources
        if overlap:
            return (
                False,
                f"{dataset_name}: SourceFile leakage in fold {fold_number}: "
                f"{sorted(overlap)}",
            )

        for side_name, side_df in [
            ("train", train_df),
            ("validation/test", test_df),
        ]:
            labels = set(
                pd.to_numeric(side_df[LABEL_COL], errors="raise")
                .astype(int)
                .unique()
                .tolist()
            )
            if labels != {0, 1}:
                return (
                    False,
                    f"{dataset_name}: fold {fold_number} {side_name} side does "
                    f"not contain both classes. Found {sorted(labels)}",
                )

    return True, None


def _make_class_balanced_group_folds(
    df: pd.DataFrame,
    n_splits: int,
    dataset_name: str,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Deterministic fallback for highly imbalanced numbers/sizes of SourceFiles.

    SourceFiles are assigned separately inside each class so every test fold
    receives at least one benign and one malicious SourceFile. Remaining groups
    are greedily assigned to the fold with the smallest row count for that class.
    """
    group_table = validate_source_groups(df, dataset_name).copy()
    group_table[SOURCE_FILE_COL] = group_table[SOURCE_FILE_COL].astype(str)

    rng = np.random.default_rng(int(random_state))
    source_to_fold: dict[str, int] = {}

    for label in (0, 1):
        class_groups = group_table[group_table[LABEL_COL] == label].copy()
        if len(class_groups) < n_splits:
            raise ValueError(
                f"{dataset_name}: class {label} has only {len(class_groups)} "
                f"SourceFiles for {n_splits} folds."
            )

        # Stable but seed-dependent tie breaking. Large groups are placed first.
        class_groups["_tie"] = rng.random(len(class_groups))
        class_groups = class_groups.sort_values(
            ["Rows", "_tie"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        class_rows = np.zeros(n_splits, dtype=np.int64)
        initial_fold_order = rng.permutation(n_splits)

        # Guarantee one SourceFile of this class in every fold.
        for idx in range(n_splits):
            row = class_groups.iloc[idx]
            fold_idx = int(initial_fold_order[idx])
            source = str(row[SOURCE_FILE_COL])
            source_to_fold[source] = fold_idx
            class_rows[fold_idx] += int(row["Rows"])

        # Balance the remaining SourceFiles by row count within this class.
        for idx in range(n_splits, len(class_groups)):
            row = class_groups.iloc[idx]
            minimum = class_rows.min()
            candidates = np.flatnonzero(class_rows == minimum)
            fold_idx = int(rng.choice(candidates))
            source = str(row[SOURCE_FILE_COL])
            source_to_fold[source] = fold_idx
            class_rows[fold_idx] += int(row["Rows"])

    source_values = df[SOURCE_FILE_COL].astype(str).to_numpy()
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_idx in range(n_splits):
        test_mask = np.fromiter(
            (source_to_fold[source] == fold_idx for source in source_values),
            dtype=bool,
            count=len(source_values),
        )
        test_positions = np.flatnonzero(test_mask)
        train_positions = np.flatnonzero(~test_mask)
        folds.append((train_positions, test_positions))

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
    Create SourceFile-disjoint folds with both classes on every side.

    StratifiedGroupKFold is attempted first to preserve the original baseline
    behavior. If sklearn produces a fold whose test/train side contains only one
    class, a deterministic source-level class-balanced allocator is used instead.

    This fallback changes only fold construction; RF features, preprocessing,
    class weighting, hyperparameters and threshold remain unchanged.
    """
    n_splits = feasible_group_fold_count(
        df,
        requested_folds,
        dataset_name,
    )

    y = (
        pd.to_numeric(df[LABEL_COL], errors="raise")
        .astype(int)
        .to_numpy()
    )
    groups = df[SOURCE_FILE_COL].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=CV_SHUFFLE,
        random_state=(random_state if CV_SHUFFLE else None),
    )

    folds = list(
        splitter.split(
            X=np.zeros((len(df), 1), dtype=np.int8),
            y=y,
            groups=groups,
        )
    )

    valid, reason = _validate_group_folds(df, folds, dataset_name)
    if not valid:
        print(
            f"[FOLD FALLBACK] {reason}\n"
            f"[FOLD FALLBACK] Rebuilding {dataset_name} with deterministic "
            "class-balanced SourceFile folds."
        )
        folds = _make_class_balanced_group_folds(
            df=df,
            n_splits=n_splits,
            dataset_name=dataset_name,
            random_state=random_state,
        )

        valid, reason = _validate_group_folds(df, folds, dataset_name)
        if not valid:
            raise RuntimeError(
                f"{dataset_name}: deterministic group-fold fallback failed: {reason}"
            )

    return folds

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


def make_sensitivity_configurations(
    sensitivity_values: dict,
) -> list[
    dict
]:
    """
    One-factor-at-a-time sensitivity analysis around BASE_RF_PARAMS.

    The baseline is included exactly once.
    """
    baseline = deepcopy(
        BASE_RF_PARAMS
    )

    configurations = []
    seen = set()

    def add_configuration(
        varied_parameter,
        varied_value,
        params,
    ):
        signature = tuple(
            (
                key,
                repr(
                    params[
                        key
                    ]
                ),
            )
            for key in sorted(
                params.keys()
            )
        )

        if signature in seen:
            return

        seen.add(
            signature
        )

        configurations.append({
            "config_id": (
                f"config_{len(configurations) + 1:02d}"
            ),
            "varied_parameter": (
                varied_parameter
            ),
            "varied_value": (
                varied_value
            ),
            "params": params,
        })

    add_configuration(
        "baseline",
        "baseline",
        deepcopy(
            baseline
        ),
    )

    for parameter, values in (
        sensitivity_values.items()
    ):
        if parameter not in baseline:
            raise ValueError(
                f"Unknown sensitivity parameter: {parameter}"
            )

        for value in values:
            params = deepcopy(
                baseline
            )

            params[
                parameter
            ] = value

            add_configuration(
                parameter,
                value,
                params,
            )

    return configurations


def summarize_sensitivity_runs(
    raw_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize inner-validation performance/stability for each configuration.
    """
    if raw_results.empty:
        raise ValueError(
            "Sensitivity result table is empty."
        )

    group_columns = [
        "config_id",
        "varied_parameter",
        "varied_value",
    ]

    summary_rows = []

    for group_key, group_df in (
        raw_results.groupby(
            group_columns,
            dropna=False,
        )
    ):
        row = {
            "config_id": (
                group_key[
                    0
                ]
            ),
            "varied_parameter": (
                group_key[
                    1
                ]
            ),
            "varied_value": (
                group_key[
                    2
                ]
            ),
            "runs": len(
                group_df
            ),
        }

        first = (
            group_df.iloc[
                0
            ]
        )

        for parameter in (
            BASE_RF_PARAMS.keys()
        ):
            row[
                f"param_{parameter}"
            ] = first[
                f"param_{parameter}"
            ]

        for metric in (
            METRIC_COLUMNS
        ):
            row[
                f"{metric}_mean"
            ] = float(
                group_df[
                    metric
                ]
                .mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                group_df[
                    metric
                ]
                .std(
                    ddof=0
                )
            )

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )


def rank_sensitivity_summary(
    summary: pd.DataFrame,
    primary_metric: str,
) -> pd.DataFrame:
    """
    Higher PR-AUC, Recall and F1 are preferred; lower PR-AUC std is preferred
    as the stability tie-breaker.
    """
    if primary_metric not in (
        summary.columns
    ):
        raise ValueError(
            f"Unknown primary selection metric: {primary_metric}"
        )

    return (
        summary.sort_values(
            by=[
                primary_metric,
                "recall_tpr_mean",
                "f1_mean",
                "pr_auc_std",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def recover_config_by_id(
    configurations: list[
        dict
    ],
    config_id: str,
) -> dict:
    """
    Recover original Python parameter types directly from the configuration
    list. This avoids pandas converting max_depth=10 into 10.0.
    """
    matches = [
        configuration
        for configuration in configurations
        if configuration[
            "config_id"
        ]
        == config_id
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Could not uniquely recover configuration {config_id}"
        )

    return deepcopy(
        matches[
            0
        ][
            "params"
        ]
    )
