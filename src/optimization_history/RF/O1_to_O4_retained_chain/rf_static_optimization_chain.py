from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from random_forest_config import (
    ARTIFACT_ROOT,
    BASE_RF_PARAMS,
    BENCHMARK_DATASETS,
    CV_RANDOM_STATE,
    DATASET_PATHS,
    INNER_CV_FOLDS,
    LABEL_COL,
    MODEL_RANDOM_STATE,
    OUTER_CV_FOLDS,
    PREDICTION_THRESHOLD,
    PROJECT_ROOT,
    RESULT_ROOT,
    ROW_INDEX_COL,
    SOURCE_FILE_COL,
)
from random_forest_preprocessing import FoldMedianImputer
from random_forest_utils import (
    build_fold_manifest,
    build_random_forest,
    calculate_binary_metrics,
    compute_training_class_weights,
    feasible_group_fold_count,
    make_stratified_group_folds,
    validate_source_groups,
)

# ============================================================================
# FEATURE SCHEMAS
# ============================================================================

BASE_10_FEATURES = [
    "Fwd IAT Mean",
    "Flow IAT Std",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Down/Up Ratio",
    "Fwd Act Data Pkts",
]

FWD_PACKET_LENGTH_CV = "Fwd Packet Length CV"
TOTAL_FWD_LENGTH = "Total Length of Fwd Packet"
BASE_12_FEATURES = BASE_10_FEATURES + [
    FWD_PACKET_LENGTH_CV,
    TOTAL_FWD_LENGTH,
]

TOTAL_FWD_LENGTH_ALIASES = [
    "Total Length of Fwd Packet",
    "Total Length of Fwd Packets",
    "TotLen Fwd Pkts",
]

REPORT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_tpr",
    "fpr",
    "f1",
    "roc_auc",
    "pr_auc",
    "mcc",
    "tn",
    "fp",
    "fn",
    "tp",
]

# ============================================================================
# LOAD ORIGINAL 57-FEATURE STEP-3 SCHEMA
# ============================================================================

FEATURE_ENGINEERING_DIR = PROJECT_ROOT / "src" / "feature_engineering"
if str(FEATURE_ENGINEERING_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_ENGINEERING_DIR))

from feature_config import EXPECTED_FINAL_FEATURE_COUNT  # noqa: E402
from feature_selection import (  # noqa: E402
    clean_column_names,
    select_feature_columns,
    validate_required_columns,
)


def load_original_57_features() -> list[str]:
    """Derive the exact original Step-3 schema from dataset headers only."""
    reference: list[str] | None = None
    reference_dataset: str | None = None

    for dataset_name in BENCHMARK_DATASETS:
        path = DATASET_PATHS[dataset_name]
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        header = pd.read_csv(path, nrows=0)
        header = clean_column_names(header)
        validate_required_columns(header, f"{dataset_name}_rf_static_chain")
        features, _dropped, _manifest = select_feature_columns(header)

        if len(features) != EXPECTED_FINAL_FEATURE_COUNT:
            raise RuntimeError(
                f"{dataset_name}: expected {EXPECTED_FINAL_FEATURE_COUNT} Step-3 "
                f"features, found {len(features)}."
            )
        if len(features) != 57:
            raise RuntimeError(
                f"Controlled RF chain expects a 57-feature baseline, found {len(features)}."
            )

        if reference is None:
            reference = list(features)
            reference_dataset = dataset_name
        elif list(features) != reference:
            raise RuntimeError(
                f"Step-3 schema differs between {reference_dataset} and {dataset_name}."
            )

    if reference is None:
        raise RuntimeError("No benchmark datasets configured.")
    return reference


# ============================================================================
# DATA / FEATURE PREPARATION
# ============================================================================


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _resolve_alias(columns, aliases: list[str], role: str) -> str:
    columns = list(columns)
    exact = set(columns)
    for alias in aliases:
        if alias in exact:
            return alias

    normalized = {_normalize_name(c): c for c in columns}
    for alias in aliases:
        key = _normalize_name(alias)
        if key in normalized:
            return normalized[key]

    raise ValueError(f"Could not resolve {role}. Tried {aliases}.")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = _numeric(numerator)
    denominator = _numeric(denominator)
    result = pd.Series(np.nan, index=numerator.index, dtype=float)

    valid = numerator.notna() & denominator.notna()
    nonzero = valid & (denominator.abs() > 1e-12)
    result.loc[nonzero] = numerator.loc[nonzero] / denominator.loc[nonzero]
    result.loc[valid & ~nonzero] = 0.0
    return result.replace([np.inf, -np.inf], np.nan)


def load_raw_dataset(dataset_name: str) -> pd.DataFrame:
    path = DATASET_PATHS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name}: missing dataset: {path}")

    df = pd.read_csv(path, low_memory=False).copy()
    df[ROW_INDEX_COL] = np.arange(len(df), dtype=np.int64)
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int)
    print(f"Loaded {dataset_name}: {path} | shape={df.shape}")
    return df


def add_refined_features(raw_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required = {"Fwd Packet Length Mean", "Fwd Packet Length Std"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"{dataset_name}: missing CV source columns: {missing}")

    total_col = _resolve_alias(
        raw_df.columns,
        TOTAL_FWD_LENGTH_ALIASES,
        "total forward-packet length",
    )

    out = raw_df.copy()
    out[FWD_PACKET_LENGTH_CV] = _safe_ratio(
        out["Fwd Packet Length Std"],
        _numeric(out["Fwd Packet Length Mean"]).abs(),
    )
    out[TOTAL_FWD_LENGTH] = _numeric(out[total_col])
    return out


def prepare_representation(
    raw_df: pd.DataFrame,
    dataset_name: str,
    feature_names: list[str],
) -> pd.DataFrame:
    df = add_refined_features(raw_df, dataset_name)
    required = [ROW_INDEX_COL, SOURCE_FILE_COL, LABEL_COL] + list(feature_names)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing model columns: {missing}")

    model_df = df[required].copy()
    for feature in feature_names:
        model_df[feature] = _numeric(model_df[feature])

    validate_source_groups(model_df, dataset_name)
    return model_df


# ============================================================================
# OUTER FOLDS -- CREATED ONCE AND REUSED BY EVERY STAGE
# ============================================================================


def build_shared_experiment_folds(
    frames_57: dict[str, pd.DataFrame],
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    feasible = {
        name: feasible_group_fold_count(
            frames_57[name],
            requested_folds=OUTER_CV_FOLDS,
            dataset_name=f"{name}_outer_feasibility",
        )
        for name in BENCHMARK_DATASETS
    }
    common = min(feasible.values())
    if common < 2:
        raise RuntimeError("Not enough SourceFile groups for outer CV.")

    if common != OUTER_CV_FOLDS:
        print(
            f"Requested {OUTER_CV_FOLDS} outer folds; using {common} because of "
            "SourceFile availability."
        )

    folds = {}
    for name in BENCHMARK_DATASETS:
        folds[name] = make_stratified_group_folds(
            frames_57[name],
            requested_folds=common,
            dataset_name=f"{name}_outer",
            random_state=CV_RANDOM_STATE,
        )

        manifest = build_fold_manifest(
            frames_57[name],
            folds[name],
            ROW_INDEX_COL,
        )
        out_dir = ARTIFACT_ROOT / "folds"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_dir / f"{name}_outer_fold_manifest.csv", index=False)

    return folds


# ============================================================================
# COMMON RF EVALUATION
# ============================================================================


def evaluate_stage_fixed_params(
    stage_id: str,
    dataset_name: str,
    model_df: pd.DataFrame,
    features: list[str],
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    params_by_fold: dict[int, dict],
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof_parts = []
    fold_rows = []
    importance_parts = []

    for fold_number, (train_pos, test_pos) in enumerate(outer_folds, start=1):
        if fold_number not in params_by_fold:
            raise RuntimeError(f"{stage_id}: missing parameters for fold {fold_number}.")

        train_df = model_df.iloc[train_pos]
        test_df = model_df.iloc[test_pos]

        train_sources = set(train_df[SOURCE_FILE_COL].astype(str).unique())
        test_sources = set(test_df[SOURCE_FILE_COL].astype(str).unique())
        if train_sources & test_sources:
            raise RuntimeError(
                f"{stage_id}/{dataset_name}/fold{fold_number}: SourceFile leakage."
            )

        y_train = train_df[LABEL_COL].astype(int)
        y_test = test_df[LABEL_COL].astype(int)

        imputer = FoldMedianImputer(f"{stage_id}_{dataset_name}_fold{fold_number}")
        X_train = imputer.fit_transform(train_df[features])
        X_test = imputer.transform(test_df[features])

        weights = compute_training_class_weights(y_train)
        params = deepcopy(params_by_fold[fold_number])
        model = build_random_forest(
            params=params,
            class_weights=weights,
            random_state=MODEL_RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= PREDICTION_THRESHOLD).astype(int)
        metrics = calculate_binary_metrics(
            y_test,
            prob,
            threshold=PREDICTION_THRESHOLD,
        )

        fold_rows.append(
            {
                "Stage": stage_id,
                "Dataset": dataset_name,
                "Outer_Fold": fold_number,
                "Feature_Count": len(features),
                **metrics,
            }
        )

        oof_parts.append(
            pd.DataFrame(
                {
                    ROW_INDEX_COL: test_df[ROW_INDEX_COL].to_numpy(dtype=np.int64),
                    SOURCE_FILE_COL: test_df[SOURCE_FILE_COL].astype(str).to_numpy(),
                    "Outer_Fold": fold_number,
                    "Actual_Label": y_test.to_numpy(dtype=int),
                    "Malicious_Probability": prob,
                    "Predicted_Label": pred,
                }
            )
        )

        importance_parts.append(
            pd.DataFrame(
                {
                    "Stage": stage_id,
                    "Dataset": dataset_name,
                    "Outer_Fold": fold_number,
                    "Feature": features,
                    "MDI_Importance": model.feature_importances_,
                }
            )
        )

    oof = (
        pd.concat(oof_parts, ignore_index=True)
        .sort_values(ROW_INDEX_COL)
        .reset_index(drop=True)
    )
    if len(oof) != len(model_df) or oof[ROW_INDEX_COL].nunique() != len(model_df):
        raise RuntimeError(f"{stage_id}/{dataset_name}: invalid OOF coverage.")

    overall = calculate_binary_metrics(
        oof["Actual_Label"].to_numpy(dtype=int),
        oof["Malicious_Probability"].to_numpy(dtype=float),
        threshold=PREDICTION_THRESHOLD,
    )
    summary = {
        "Stage": stage_id,
        "Dataset": dataset_name,
        "Feature_Count": len(features),
        **overall,
    }

    return (
        summary,
        oof,
        pd.DataFrame(fold_rows),
        pd.concat(importance_parts, ignore_index=True),
    )


# ============================================================================
# O3 SHARED D2/D3 NESTED HYPERPARAMETER SEARCH ON FROZEN 12 FEATURES
# ============================================================================


def make_o3_grid(grid_mode: str) -> list[dict]:
    if grid_mode not in {"focused", "full"}:
        raise ValueError("grid_mode must be 'focused' or 'full'.")

    if grid_mode == "focused":
        candidates = [
            (128, 10, 1, 0.5),       # B0 configuration
            (256, 10, 1, "sqrt"),
            (256, None, 1, "sqrt"),
            (256, None, 2, "sqrt"),
        ]
    else:
        candidates = [
            (n, depth, leaf, mf)
            for n in (128, 256)
            for depth in (10, None)
            for leaf in (1, 2)
            for mf in ("sqrt", 0.5)
        ]

    configs = []
    for idx, (n, depth, leaf, mf) in enumerate(candidates, start=1):
        params = deepcopy(BASE_RF_PARAMS)
        params.update(
            {
                "n_estimators": int(n),
                "max_depth": depth,
                "min_samples_leaf": int(leaf),
                "max_features": mf,
                "bootstrap": True,
            }
        )
        configs.append({"config_id": f"o3_config_{idx:02d}", "params": params})
    return configs


def _evaluate_inner_dataset(
    dataset_name: str,
    outer_fold_number: int,
    outer_train_df: pd.DataFrame,
    features: list[str],
    configs: list[dict],
) -> pd.DataFrame:
    inner_folds = make_stratified_group_folds(
        outer_train_df,
        requested_folds=INNER_CV_FOLDS,
        dataset_name=f"o3_{dataset_name}_outer{outer_fold_number}_inner",
        random_state=CV_RANDOM_STATE + 100 * outer_fold_number,
    )

    rows = []
    for config in configs:
        for inner_fold, (train_pos, val_pos) in enumerate(inner_folds, start=1):
            train_df = outer_train_df.iloc[train_pos]
            val_df = outer_train_df.iloc[val_pos]
            y_train = train_df[LABEL_COL].astype(int)
            y_val = val_df[LABEL_COL].astype(int)

            imputer = FoldMedianImputer(
                f"o3_{dataset_name}_outer{outer_fold_number}_inner{inner_fold}_"
                f"{config['config_id']}"
            )
            X_train = imputer.fit_transform(train_df[features])
            X_val = imputer.transform(val_df[features])
            weights = compute_training_class_weights(y_train)

            model = build_random_forest(
                params=config["params"],
                class_weights=weights,
                random_state=MODEL_RANDOM_STATE,
            )
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_val)[:, 1]
            metrics = calculate_binary_metrics(
                y_val,
                prob,
                threshold=PREDICTION_THRESHOLD,
            )

            row = {
                "Dataset": dataset_name,
                "Outer_Fold": outer_fold_number,
                "Inner_Fold": inner_fold,
                "config_id": config["config_id"],
            }
            for key, value in config["params"].items():
                row[f"param_{key}"] = value
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def _summarize_o3_inner(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_id, group in raw.groupby("config_id", sort=True):
        per_dataset = group.groupby("Dataset", sort=True)[
            ["pr_auc", "recall_tpr", "f1", "roc_auc", "fpr"]
        ].mean()
        if set(per_dataset.index.astype(str)) != set(BENCHMARK_DATASETS):
            raise RuntimeError(f"{config_id}: missing one benchmark dataset.")

        first = group.iloc[0]
        row = {
            "config_id": config_id,
            "n_estimators": first["param_n_estimators"],
            "max_depth": first["param_max_depth"],
            "min_samples_leaf": first["param_min_samples_leaf"],
            "max_features": first["param_max_features"],
            "bootstrap": first["param_bootstrap"],
            "pr_auc_mean_across_datasets": float(per_dataset["pr_auc"].mean()),
            "pr_auc_min_across_datasets": float(per_dataset["pr_auc"].min()),
            "recall_mean_across_datasets": float(per_dataset["recall_tpr"].mean()),
            "f1_mean_across_datasets": float(per_dataset["f1"].mean()),
            "roc_auc_mean_across_datasets": float(per_dataset["roc_auc"].mean()),
            "fpr_mean_across_datasets": float(per_dataset["fpr"].mean()),
            "pr_auc_std_across_inner_runs": float(group["pr_auc"].std(ddof=0)),
        }
        for dataset_name in BENCHMARK_DATASETS:
            d = group[group["Dataset"] == dataset_name]
            row[f"{dataset_name}_pr_auc_mean"] = float(d["pr_auc"].mean())
            row[f"{dataset_name}_recall_mean"] = float(d["recall_tpr"].mean())
            row[f"{dataset_name}_f1_mean"] = float(d["f1"].mean())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        [
            "pr_auc_mean_across_datasets",
            "pr_auc_min_across_datasets",
            "recall_mean_across_datasets",
            "f1_mean_across_datasets",
            "pr_auc_std_across_inner_runs",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary["selection_rank"] = np.arange(1, len(summary) + 1)
    summary["selected"] = summary["selection_rank"] == 1
    return summary


def _config_by_id(configs: list[dict], config_id: str) -> dict:
    matches = [c for c in configs if c["config_id"] == config_id]
    if len(matches) != 1:
        raise RuntimeError(f"Could not recover {config_id}.")
    return deepcopy(matches[0]["params"])


def run_o3_nested_tuning(
    frames_12: dict[str, pd.DataFrame],
    outer_folds: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    features: list[str],
    grid_mode: str,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    configs = make_o3_grid(grid_mode)
    common_fold_count = min(len(outer_folds[name]) for name in BENCHMARK_DATASETS)

    all_inner_parts = []
    selection_rows = []
    oof_parts_by_dataset = {name: [] for name in BENCHMARK_DATASETS}

    for outer_fold_number in range(1, common_fold_count + 1):
        print("\n" + "-" * 90)
        print(f"O3 SHARED TUNING - OUTER FOLD {outer_fold_number}")
        print("-" * 90)

        inner_parts = []
        for dataset_name in BENCHMARK_DATASETS:
            train_pos, _test_pos = outer_folds[dataset_name][outer_fold_number - 1]
            outer_train_df = frames_12[dataset_name].iloc[train_pos].reset_index(drop=True)
            inner_parts.append(
                _evaluate_inner_dataset(
                    dataset_name,
                    outer_fold_number,
                    outer_train_df,
                    features,
                    configs,
                )
            )

        fold_inner = pd.concat(inner_parts, ignore_index=True)
        all_inner_parts.append(fold_inner)
        fold_summary = _summarize_o3_inner(fold_inner)
        best_id = str(fold_summary.iloc[0]["config_id"])
        best_params = _config_by_id(configs, best_id)

        selection_rows.append(
            {
                "Outer_Fold": outer_fold_number,
                "config_id": best_id,
                "n_estimators": best_params["n_estimators"],
                "max_depth": best_params["max_depth"],
                "min_samples_leaf": best_params["min_samples_leaf"],
                "max_features": best_params["max_features"],
                "bootstrap": best_params["bootstrap"],
                "inner_pr_auc_mean_across_datasets": float(
                    fold_summary.iloc[0]["pr_auc_mean_across_datasets"]
                ),
            }
        )

        print(
            f"Selected shared config: {best_id} | n={best_params['n_estimators']}, "
            f"depth={best_params['max_depth']}, leaf={best_params['min_samples_leaf']}, "
            f"max_features={best_params['max_features']}"
        )

        # Use the one shared D2/D3 configuration selected from INNER validation
        # to evaluate the untouched outer test folds.
        for dataset_name in BENCHMARK_DATASETS:
            train_pos, test_pos = outer_folds[dataset_name][outer_fold_number - 1]
            train_df = frames_12[dataset_name].iloc[train_pos]
            test_df = frames_12[dataset_name].iloc[test_pos]
            y_train = train_df[LABEL_COL].astype(int)
            y_test = test_df[LABEL_COL].astype(int)

            imputer = FoldMedianImputer(f"o3_{dataset_name}_outer{outer_fold_number}")
            X_train = imputer.fit_transform(train_df[features])
            X_test = imputer.transform(test_df[features])
            weights = compute_training_class_weights(y_train)
            model = build_random_forest(
                params=best_params,
                class_weights=weights,
                random_state=MODEL_RANDOM_STATE,
            )
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= PREDICTION_THRESHOLD).astype(int)

            oof_parts_by_dataset[dataset_name].append(
                pd.DataFrame(
                    {
                        ROW_INDEX_COL: test_df[ROW_INDEX_COL].to_numpy(dtype=np.int64),
                        SOURCE_FILE_COL: test_df[SOURCE_FILE_COL].astype(str).to_numpy(),
                        "Outer_Fold": outer_fold_number,
                        "Actual_Label": y_test.to_numpy(dtype=int),
                        "Malicious_Probability": prob,
                        "Predicted_Label": pred,
                        "Selected_O3_Config_ID": best_id,
                    }
                )
            )

    all_inner = pd.concat(all_inner_parts, ignore_index=True)
    final_summary = _summarize_o3_inner(all_inner)
    final_id = str(final_summary.iloc[0]["config_id"])
    final_params = _config_by_id(configs, final_id)

    oof_summary_rows = []
    completed_oof = {}
    for dataset_name in BENCHMARK_DATASETS:
        oof = (
            pd.concat(oof_parts_by_dataset[dataset_name], ignore_index=True)
            .sort_values(ROW_INDEX_COL)
            .reset_index(drop=True)
        )
        if len(oof) != len(frames_12[dataset_name]):
            raise RuntimeError(f"O3/{dataset_name}: invalid OOF coverage.")

        metrics = calculate_binary_metrics(
            oof["Actual_Label"].to_numpy(dtype=int),
            oof["Malicious_Probability"].to_numpy(dtype=float),
            threshold=PREDICTION_THRESHOLD,
        )
        oof_summary_rows.append(
            {
                "Stage": "O3_12_TUNED",
                "Dataset": dataset_name,
                "Feature_Count": 12,
                **metrics,
            }
        )
        completed_oof[dataset_name] = oof

    return (
        oof_summary_rows,
        all_inner,
        final_summary,
        pd.DataFrame(selection_rows),
        final_params,
        completed_oof,
    )


# ============================================================================
# REPORT OUTPUTS
# ============================================================================


def build_incremental_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    stage_order = [
        "B0_57_FIXED",
        "O1_10_LEAN",
        "O2_12_REFINED",
        "O3_12_TUNED",
    ]
    rows = []
    for dataset_name in BENCHMARK_DATASETS:
        subset = summary[summary["Dataset"] == dataset_name].set_index("Stage")
        for previous, current in zip(stage_order[:-1], stage_order[1:]):
            if previous not in subset.index or current not in subset.index:
                continue
            row = {
                "Dataset": dataset_name,
                "From_Stage": previous,
                "To_Stage": current,
            }
            for metric in ["precision", "recall_tpr", "fpr", "f1", "roc_auc", "pr_auc", "fp", "fn", "tp"]:
                row[f"Delta_{metric}"] = float(subset.loc[current, metric]) - float(
                    subset.loc[previous, metric]
                )
            # Positive means improvement for every *_Improvement field.
            row["Recall_Improvement"] = row["Delta_recall_tpr"]
            row["F1_Improvement"] = row["Delta_f1"]
            row["PR_AUC_Improvement"] = row["Delta_pr_auc"]
            row["FPR_Improvement"] = -row["Delta_fpr"]
            row["FP_Improvement"] = -row["Delta_fp"]
            row["FN_Improvement"] = -row["Delta_fn"]
            rows.append(row)
    return pd.DataFrame(rows)


def save_stage_outputs(
    stage_id: str,
    dataset_name: str,
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame | None = None,
    importance: pd.DataFrame | None = None,
) -> None:
    stage_dir = RESULT_ROOT / stage_id / dataset_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(stage_dir / "oof_predictions.csv", index=False)
    if fold_metrics is not None:
        fold_metrics.to_csv(stage_dir / "outer_fold_metrics.csv", index=False)
    if importance is not None:
        importance.to_csv(stage_dir / "feature_importance_by_fold.csv", index=False)
        (
            importance.groupby("Feature")["MDI_Importance"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "MDI_Importance_Mean", "std": "MDI_Importance_Std"})
            .sort_values("MDI_Importance_Mean", ascending=False)
            .to_csv(stage_dir / "feature_importance_mean.csv", index=False)
        )


# ============================================================================
# TOP LEVEL
# ============================================================================


def run_static_optimization(grid_mode: str = "full") -> pd.DataFrame:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("RANDOM FOREST - CONTROLLED STATIC OPTIMIZATION CHAIN")
    print("=" * 90)
    print("B0: 57 features + fixed baseline HP")
    print("O1: 10 features + exact same HP/folds")
    print("O2: 12 features + exact same HP/folds")
    print("O3: shared D2/D3 nested HP tuning on frozen 12-feature representation")
    print(f"O3 grid: {grid_mode}")
    print("Baseline HP:")
    print("  n_estimators=128")
    print("  max_depth=10")
    print("  min_samples_leaf=1")
    print("  max_features=0.5")
    print("  bootstrap=True")
    print(f"  threshold={PREDICTION_THRESHOLD}")

    features_57 = load_original_57_features()
    raw = {name: load_raw_dataset(name) for name in BENCHMARK_DATASETS}

    frames_57 = {
        name: prepare_representation(raw[name], name, features_57)
        for name in BENCHMARK_DATASETS
    }
    frames_10 = {
        name: prepare_representation(raw[name], name, BASE_10_FEATURES)
        for name in BENCHMARK_DATASETS
    }
    frames_12 = {
        name: prepare_representation(raw[name], name, BASE_12_FEATURES)
        for name in BENCHMARK_DATASETS
    }

    outer_folds = build_shared_experiment_folds(frames_57)
    fixed_params_by_dataset = {
        name: {
            fold_number: deepcopy(BASE_RF_PARAMS)
            for fold_number in range(1, len(outer_folds[name]) + 1)
        }
        for name in BENCHMARK_DATASETS
    }

    # Save schemas and experiment definition.
    pd.DataFrame({"Feature": features_57}).to_csv(
        ARTIFACT_ROOT / "B0_57_features.csv", index=False
    )
    pd.DataFrame({"Feature": BASE_10_FEATURES}).to_csv(
        ARTIFACT_ROOT / "O1_10_features.csv", index=False
    )
    pd.DataFrame({"Feature": BASE_12_FEATURES}).to_csv(
        ARTIFACT_ROOT / "O2_O3_12_features.csv", index=False
    )
    with (ARTIFACT_ROOT / "B0_fixed_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "parameters": BASE_RF_PARAMS,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "scope": "fixed_shared_D2_D3_for_B0_O1_O2",
            },
            handle,
            indent=2,
        )

    summary_rows = []

    # B0, O1, O2 -- one controlled change at a time.
    for stage_id, frames, features in [
        ("B0_57_FIXED", frames_57, features_57),
        ("O1_10_LEAN", frames_10, BASE_10_FEATURES),
        ("O2_12_REFINED", frames_12, BASE_12_FEATURES),
    ]:
        print("\n" + "=" * 90)
        print(stage_id)
        print("=" * 90)
        for dataset_name in BENCHMARK_DATASETS:
            summary, oof, fold_metrics, importance = evaluate_stage_fixed_params(
                stage_id=stage_id,
                dataset_name=dataset_name,
                model_df=frames[dataset_name],
                features=list(features),
                outer_folds=outer_folds[dataset_name],
                params_by_fold=fixed_params_by_dataset[dataset_name],
            )
            summary["Hyperparameter_Regime"] = "fixed_B0_128_10_1_0.5_true"
            summary_rows.append(summary)
            save_stage_outputs(stage_id, dataset_name, oof, fold_metrics, importance)
            print(
                f"{dataset_name}: recall={summary['recall_tpr']:.4f}, "
                f"fpr={summary['fpr']:.6f}, f1={summary['f1']:.4f}, "
                f"pr_auc={summary['pr_auc']:.4f}"
            )

    # O3 -- hyperparameter optimization only, same 12 features and same outer folds.
    print("\n" + "=" * 90)
    print("O3_12_TUNED")
    print("=" * 90)
    (
        o3_summary_rows,
        o3_inner_runs,
        o3_tuning_summary,
        o3_outer_selections,
        o3_final_params,
        o3_oof,
    ) = run_o3_nested_tuning(
        frames_12=frames_12,
        outer_folds=outer_folds,
        features=BASE_12_FEATURES,
        grid_mode=grid_mode,
    )

    for row in o3_summary_rows:
        row["Hyperparameter_Regime"] = "nested_shared_tuning_on_frozen_12"
        summary_rows.append(row)
        save_stage_outputs("O3_12_TUNED", row["Dataset"], o3_oof[row["Dataset"]])
        print(
            f"{row['Dataset']}: recall={row['recall_tpr']:.4f}, "
            f"fpr={row['fpr']:.6f}, f1={row['f1']:.4f}, "
            f"pr_auc={row['pr_auc']:.4f}"
        )

    o3_inner_runs.to_csv(RESULT_ROOT / "O3_inner_validation_runs.csv", index=False)
    o3_tuning_summary.to_csv(RESULT_ROOT / "O3_shared_tuning_summary.csv", index=False)
    o3_outer_selections.to_csv(RESULT_ROOT / "O3_outer_fold_selections.csv", index=False)

    with (ARTIFACT_ROOT / "O3_final_shared_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "parameters": o3_final_params,
                "feature_count": 12,
                "feature_schema": BASE_12_FEATURES,
                "selection": "equal-weight D2/D3 nested inner validation",
                "grid_mode": grid_mode,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "note": (
                    "For unbiased temporal OOF ablations, reuse the per-outer-fold "
                    "shared selections in O3_outer_fold_selections.csv. Use this final "
                    "configuration for final deployment training."
                ),
            },
            handle,
            indent=2,
        )

    summary = pd.DataFrame(summary_rows)
    stage_order = {
        "B0_57_FIXED": 0,
        "O1_10_LEAN": 1,
        "O2_12_REFINED": 2,
        "O3_12_TUNED": 3,
    }
    summary["_order"] = summary["Stage"].map(stage_order)
    summary = summary.sort_values(["Dataset", "_order"]).drop(columns="_order").reset_index(drop=True)
    summary.to_csv(RESULT_ROOT / "phase1_static_comparison.csv", index=False)

    report_columns = [
        "Stage",
        "Dataset",
        "Feature_Count",
        "Hyperparameter_Regime",
        "precision",
        "recall_tpr",
        "fpr",
        "f1",
        "roc_auc",
        "pr_auc",
        "fp",
        "fn",
        "tp",
    ]
    summary[report_columns].to_csv(RESULT_ROOT / "phase1_report_table.csv", index=False)

    deltas = build_incremental_deltas(summary)
    deltas.to_csv(RESULT_ROOT / "phase1_incremental_deltas.csv", index=False)

    with (ARTIFACT_ROOT / "experiment_definition.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "optimization_chain": [
                    "B0: 57 features, fixed reported baseline HP",
                    "O1: 10 features, same folds and fixed baseline HP",
                    "O2: 12 features, same folds and fixed baseline HP",
                    "O3: 12 features, shared D2/D3 nested HP tuning",
                ],
                "baseline_parameters": BASE_RF_PARAMS,
                "outer_folds_reused_across_all_stages": True,
                "source_grouping": SOURCE_FILE_COL,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "o3_primary_selection_metric": "mean PR-AUC across D2 and D3",
            },
            handle,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("STATIC RF OPTIMIZATION COMPLETE")
    print("=" * 90)
    print(summary[report_columns].to_string(index=False))
    print("\nO3 final shared deployment configuration:")
    for key in ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "bootstrap"]:
        print(f"  {key}={o3_final_params[key]}")
    print(f"  threshold={PREDICTION_THRESHOLD}")
    print("\nSaved key files:")
    print(RESULT_ROOT / "phase1_report_table.csv")
    print(RESULT_ROOT / "phase1_incremental_deltas.csv")
    print(RESULT_ROOT / "O3_outer_fold_selections.csv")
    print(ARTIFACT_ROOT / "O3_final_shared_hyperparameters.json")

    return summary
