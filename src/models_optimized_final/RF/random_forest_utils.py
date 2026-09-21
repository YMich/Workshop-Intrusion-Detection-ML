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
    CV_RANDOM_STATE,
    CV_SHUFFLE,
    FINAL_RF_PARAMS,
    LABEL_COL,
    MODEL_RANDOM_STATE,
    PREDICTION_THRESHOLD,
    SOURCE_FILE_COL,
)


def validate_source_groups(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required = {SOURCE_FILE_COL, LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_name}: missing columns {sorted(missing)}")
    if df[SOURCE_FILE_COL].isna().any():
        raise ValueError(f"{dataset_name}: SourceFile contains missing values.")

    table = (
        df.groupby(SOURCE_FILE_COL, sort=True, dropna=False)
        .agg(
            Label=(LABEL_COL, "first"),
            Label_Nunique=(LABEL_COL, "nunique"),
            Rows=(LABEL_COL, "size"),
        )
        .reset_index()
    )
    mixed = table[table["Label_Nunique"] != 1]
    if not mixed.empty:
        raise ValueError(
            f"{dataset_name}: at least one SourceFile contains both classes:\n"
            f"{mixed.to_string(index=False)}"
        )

    table = table.drop(columns=["Label_Nunique"])
    table[LABEL_COL] = pd.to_numeric(table[LABEL_COL], errors="raise").astype(int)
    labels = set(table[LABEL_COL].unique().tolist())
    if labels != {0, 1}:
        raise ValueError(f"{dataset_name}: expected labels {{0,1}}, found {labels}")
    return table


def feasible_group_fold_count(
    df: pd.DataFrame, requested_folds: int, dataset_name: str
) -> int:
    groups = validate_source_groups(df, dataset_name)
    maximum = int(groups[LABEL_COL].value_counts().min())
    folds = min(int(requested_folds), maximum)
    if folds < 2:
        raise ValueError(
            f"{dataset_name}: not enough SourceFiles per class for group CV."
        )
    return folds


def _validate_group_folds(df, folds, dataset_name):
    for fold_number, (train_pos, test_pos) in enumerate(folds, start=1):
        train_df = df.iloc[train_pos]
        test_df = df.iloc[test_pos]
        train_sources = set(train_df[SOURCE_FILE_COL].astype(str).unique())
        test_sources = set(test_df[SOURCE_FILE_COL].astype(str).unique())
        overlap = train_sources & test_sources
        if overlap:
            return False, f"{dataset_name}: SourceFile leakage in fold {fold_number}"

        for side_name, side_df in (("train", train_df), ("test", test_df)):
            labels = set(
                pd.to_numeric(side_df[LABEL_COL], errors="raise")
                .astype(int)
                .unique()
                .tolist()
            )
            if labels != {0, 1}:
                return (
                    False,
                    f"{dataset_name}: fold {fold_number} {side_name} side does not "
                    f"contain both classes. Found {sorted(labels)}",
                )
    return True, None


def _make_class_balanced_group_folds(
    df: pd.DataFrame, n_splits: int, dataset_name: str, random_state: int
):
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

        class_groups["_tie"] = rng.random(len(class_groups))
        class_groups = class_groups.sort_values(
            ["Rows", "_tie"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)

        class_rows = np.zeros(n_splits, dtype=np.int64)
        initial_order = rng.permutation(n_splits)
        for idx in range(n_splits):
            row = class_groups.iloc[idx]
            fold_idx = int(initial_order[idx])
            source_to_fold[str(row[SOURCE_FILE_COL])] = fold_idx
            class_rows[fold_idx] += int(row["Rows"])

        for idx in range(n_splits, len(class_groups)):
            row = class_groups.iloc[idx]
            candidates = np.flatnonzero(class_rows == class_rows.min())
            fold_idx = int(rng.choice(candidates))
            source_to_fold[str(row[SOURCE_FILE_COL])] = fold_idx
            class_rows[fold_idx] += int(row["Rows"])

    source_values = df[SOURCE_FILE_COL].astype(str).to_numpy()
    folds = []
    for fold_idx in range(n_splits):
        test_mask = np.fromiter(
            (source_to_fold[s] == fold_idx for s in source_values),
            dtype=bool,
            count=len(source_values),
        )
        test_pos = np.flatnonzero(test_mask)
        train_pos = np.flatnonzero(~test_mask)
        folds.append((train_pos, test_pos))
    return folds


def make_source_aware_folds(
    df: pd.DataFrame,
    requested_folds: int,
    dataset_name: str,
    random_state: int = CV_RANDOM_STATE,
):
    n_splits = feasible_group_fold_count(df, requested_folds, dataset_name)
    y = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int).to_numpy()
    groups = df[SOURCE_FILE_COL].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=CV_SHUFFLE,
        random_state=(random_state if CV_SHUFFLE else None),
    )
    folds = list(
        splitter.split(
            X=np.zeros((len(df), 1), dtype=np.int8), y=y, groups=groups
        )
    )

    valid, reason = _validate_group_folds(df, folds, dataset_name)
    if not valid:
        print(f"[FOLD FALLBACK] {reason}")
        print(
            f"[FOLD FALLBACK] Rebuilding {dataset_name} with deterministic "
            "class-balanced SourceFile folds."
        )
        folds = _make_class_balanced_group_folds(
            df, n_splits, dataset_name, random_state
        )
        valid, reason = _validate_group_folds(df, folds, dataset_name)
        if not valid:
            raise RuntimeError(f"{dataset_name}: fold fallback failed: {reason}")
    return folds


def compute_training_class_weights(y: pd.Series) -> dict[int, float]:
    y = pd.Series(y).astype(int)
    if set(y.unique().tolist()) != {0, 1}:
        raise ValueError("Training labels must contain both classes.")
    classes = np.array([0, 1], dtype=int)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y.to_numpy()
    )
    return {int(k): float(v) for k, v in zip(classes, weights)}


def build_final_random_forest(
    class_weights: dict[int, float], random_state: int = MODEL_RANDOM_STATE
) -> RandomForestClassifier:
    p = deepcopy(FINAL_RF_PARAMS)
    return RandomForestClassifier(
        n_estimators=p["n_estimators"],
        criterion=p["criterion"],
        max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
        min_weight_fraction_leaf=p["min_weight_fraction_leaf"],
        max_features=p["max_features"],
        max_leaf_nodes=p["max_leaf_nodes"],
        min_impurity_decrease=p["min_impurity_decrease"],
        bootstrap=p["bootstrap"],
        oob_score=p["oob_score"],
        n_jobs=p["n_jobs"],
        random_state=random_state,
        verbose=p["verbose"],
        warm_start=p["warm_start"],
        class_weight=class_weights,
        ccp_alpha=p["ccp_alpha"],
        max_samples=p["max_samples"],
    )


def calculate_binary_metrics(
    y_true, y_probability, threshold: float = PREDICTION_THRESHOLD
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_probability, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, pred, zero_division=0)),
        "specificity_tnr": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }
