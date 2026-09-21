from __future__ import annotations

import json
import sys
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

from random_forest_config import (
    BASE_RF_PARAMS,
    BENCHMARK_DATASETS,
    CV_RANDOM_STATE,
    DATASET_PATHS,
    FOCUSED_MAX_FEATURES,
    FOCUSED_N_ESTIMATORS,
    INNER_CV_FOLDS,
    LABEL_COL,
    MODEL_ARTIFACT_ROOT,
    MODEL_RANDOM_STATE,
    OUTER_CV_FOLDS,
    PREDICTION_THRESHOLD,
    PROJECT_ROOT,
    RESULT_ROOT,
    ROW_INDEX_COL,
    SHARED_HYPERPARAMETER_PATH,
    SHARED_SELECTION_RESULT_DIR,
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


FEATURE_ENGINEERING_DIR = PROJECT_ROOT / "src" / "feature_engineering"
if str(FEATURE_ENGINEERING_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_ENGINEERING_DIR))

from feature_config import EXPECTED_FINAL_FEATURE_COUNT  # noqa: E402
from feature_selection import (  # noqa: E402
    clean_column_names,
    select_feature_columns,
    validate_required_columns,
)


# ============================================================
# FIXED MODEL INPUT SCHEMA -- DERIVED FROM STEP-3 SOURCE RULES
# ============================================================

# The baseline 57-feature schema is a deterministic consequence of the
# model-agnostic Step-3 rules in src/feature_engineering.  It is derived from
# dataset headers only, so RF training has no dependency on a pre-existing
# artifacts/feature_selection/selected_features.csv file.

METRIC_NAMES = [
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


def load_ingested_dataset(dataset_name: str) -> pd.DataFrame:
    path = DATASET_PATHS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name}: ingested dataset not found:\n{path}"
        )

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {dataset_name}: {path} | shape={df.shape}")
    return df


def load_selected_features() -> list[str]:
    """
    Derive the fixed Step-3 feature schema from the shared source rules.

    Only CSV headers are read here.  No labels, feature values, validation rows,
    or test rows influence the schema.  Dataset 2 and Dataset 3 must derive the
    exact same ordered feature list.
    """
    reference_features: list[str] | None = None
    reference_dataset: str | None = None

    for dataset_name in BENCHMARK_DATASETS:
        path = DATASET_PATHS[dataset_name]
        if not path.exists():
            raise FileNotFoundError(
                f"{dataset_name}: ingested dataset not found:\n{path}"
            )

        header = pd.read_csv(path, nrows=0)
        header = clean_column_names(header)
        validate_required_columns(header, f"{dataset_name}_rf_schema")

        features, _dropped, _manifest = select_feature_columns(header)

        if len(features) != EXPECTED_FINAL_FEATURE_COUNT:
            raise ValueError(
                f"{dataset_name}: Step-3 schema should contain "
                f"{EXPECTED_FINAL_FEATURE_COUNT} features, found {len(features)}."
            )
        if len(features) != len(set(features)):
            raise ValueError(
                f"{dataset_name}: duplicate entries in derived Step-3 schema."
            )

        if reference_features is None:
            reference_features = list(features)
            reference_dataset = dataset_name
            continue

        if features != reference_features:
            missing = [f for f in reference_features if f not in features]
            extra = [f for f in features if f not in reference_features]
            raise ValueError(
                f"{dataset_name}: Step-3 feature schema differs from "
                f"{reference_dataset}. Missing={missing}; Extra={extra}; "
                f"OrderMatches={features == reference_features}"
            )

    if reference_features is None:
        raise RuntimeError("No benchmark datasets are configured.")

    return reference_features


def prepare_model_input(
    df: pd.DataFrame,
    selected_features: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    """
    Preserve the original RF preparation logic.

    Before CV we only select the fixed columns, convert features to numeric and
    convert +/-inf to NaN. Median imputation remains inside each training fold.
    """
    required = [SOURCE_FILE_COL, LABEL_COL] + selected_features
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: missing required model-input columns: {missing}"
        )

    model_df = df[required].copy()
    for column in selected_features:
        model_df[column] = (
            pd.to_numeric(model_df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

    model_df[LABEL_COL] = pd.to_numeric(
        model_df[LABEL_COL], errors="raise"
    ).astype(int)
    return model_df


# ============================================================
# FOCUSED SHARED CONFIGURATIONS
# ============================================================


def make_focused_shared_configurations() -> list[dict]:
    """
    Four combinations covering the only D2/D3 disagreements from the previous
    validation-only RF sensitivity analysis.
    """
    configurations = []
    counter = 1

    for n_estimators in FOCUSED_N_ESTIMATORS:
        for max_features in FOCUSED_MAX_FEATURES:
            params = deepcopy(BASE_RF_PARAMS)
            params["n_estimators"] = int(n_estimators)
            params["max_features"] = max_features

            configurations.append(
                {
                    "config_id": f"shared_config_{counter:02d}",
                    "n_estimators": int(n_estimators),
                    "max_features": max_features,
                    "params": params,
                }
            )
            counter += 1

    return configurations


def recover_shared_config(
    configurations: list[dict],
    config_id: str,
) -> dict:
    matches = [c for c in configurations if c["config_id"] == config_id]
    if len(matches) != 1:
        raise RuntimeError(f"Could not uniquely recover shared config {config_id}")
    return deepcopy(matches[0]["params"])


# ============================================================
# INNER-CV EVALUATION OF ONE DATASET / ONE OUTER FOLD
# ============================================================


def evaluate_inner_configurations(
    dataset_name: str,
    outer_fold_number: int,
    outer_train_df: pd.DataFrame,
    selected_features: list[str],
    configurations: list[dict],
) -> pd.DataFrame:
    """
    Evaluate all focused configurations on one dataset's INNER folds only.

    Imputation and balanced class weights are recomputed from each inner
    training fold exactly as in the previous RF implementation.
    """
    inner_folds = make_stratified_group_folds(
        outer_train_df,
        requested_folds=INNER_CV_FOLDS,
        dataset_name=f"{dataset_name}_outer{outer_fold_number}_inner",
        random_state=CV_RANDOM_STATE + 100 * outer_fold_number,
    )

    rows = []

    for config in configurations:
        print(
            f"      {dataset_name}: {config['config_id']} | "
            f"n_estimators={config['n_estimators']}, "
            f"max_features={config['max_features']}"
        )

        for inner_fold_number, (
            inner_train_positions,
            inner_validation_positions,
        ) in enumerate(inner_folds, start=1):
            inner_train_df = outer_train_df.iloc[inner_train_positions]
            inner_validation_df = outer_train_df.iloc[inner_validation_positions]

            X_train_raw = inner_train_df[selected_features]
            X_validation_raw = inner_validation_df[selected_features]
            y_train = inner_train_df[LABEL_COL].astype(int)
            y_validation = inner_validation_df[LABEL_COL].astype(int)

            median_imputer = FoldMedianImputer(
                f"{dataset_name}_outer{outer_fold_number}_inner{inner_fold_number}"
            )
            X_train = median_imputer.fit_transform(X_train_raw)
            X_validation = median_imputer.transform(X_validation_raw)

            class_weights = compute_training_class_weights(y_train)
            model = build_random_forest(
                params=config["params"],
                class_weights=class_weights,
                random_state=MODEL_RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_validation)[:, 1]
            metrics = calculate_binary_metrics(
                y_validation,
                probabilities,
                threshold=PREDICTION_THRESHOLD,
            )

            row = {
                "Dataset": dataset_name,
                "outer_fold": int(outer_fold_number),
                "inner_fold": int(inner_fold_number),
                "config_id": config["config_id"],
                "inner_train_rows": int(len(inner_train_df)),
                "inner_validation_rows": int(len(inner_validation_df)),
                "inner_train_sources": int(
                    inner_train_df[SOURCE_FILE_COL].nunique()
                ),
                "inner_validation_sources": int(
                    inner_validation_df[SOURCE_FILE_COL].nunique()
                ),
            }

            for parameter, value in config["params"].items():
                row[f"param_{parameter}"] = value

            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# EQUAL-WEIGHT D2/D3 SHARED SELECTION
# ============================================================


def summarize_shared_inner_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize configurations with equal dataset weighting.

    We first average each metric inside each dataset, then average the D2 and D3
    means. Therefore Dataset 2 cannot dominate simply because it has more rows.
    """
    if raw_results.empty:
        raise ValueError("Shared inner-CV result table is empty.")

    summary_rows = []

    for config_id, config_group in raw_results.groupby("config_id", sort=True):
        dataset_means = config_group.groupby("Dataset", sort=True)[METRIC_NAMES].mean()

        if set(dataset_means.index.astype(str)) != set(BENCHMARK_DATASETS):
            raise RuntimeError(
                f"{config_id}: results do not include both benchmark datasets."
            )

        first = config_group.iloc[0]
        row = {
            "config_id": str(config_id),
            "runs": int(len(config_group)),
            "datasets": int(config_group["Dataset"].nunique()),
        }

        for parameter in BASE_RF_PARAMS:
            row[f"param_{parameter}"] = first[f"param_{parameter}"]

        for dataset_name in BENCHMARK_DATASETS:
            dataset_group = config_group[
                config_group["Dataset"] == dataset_name
            ]
            for metric in METRIC_NAMES:
                row[f"{dataset_name}_{metric}_mean"] = float(
                    dataset_group[metric].mean()
                )
                row[f"{dataset_name}_{metric}_std"] = float(
                    dataset_group[metric].std(ddof=0)
                )

        for metric in METRIC_NAMES:
            per_dataset = dataset_means[metric].to_numpy(dtype=float)
            row[f"{metric}_mean_across_datasets"] = float(np.mean(per_dataset))
            row[f"{metric}_min_across_datasets"] = float(np.min(per_dataset))
            row[f"{metric}_max_across_datasets"] = float(np.max(per_dataset))
            row[f"{metric}_std_across_all_inner_runs"] = float(
                config_group[metric].std(ddof=0)
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        by=[
            "pr_auc_mean_across_datasets",
            "pr_auc_min_across_datasets",
            "recall_tpr_mean_across_datasets",
            "f1_mean_across_datasets",
            "pr_auc_std_across_all_inner_runs",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    summary["selection_rank"] = np.arange(1, len(summary) + 1, dtype=int)
    summary["selected"] = summary["selection_rank"] == 1
    return summary


def run_joint_inner_selection(
    outer_fold_number: int,
    dataset_records: dict,
    selected_features: list[str],
    configurations: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict]:
    """
    Select ONE configuration for a paired D2/D3 outer fold.

    Only the two datasets' OUTER-TRAIN partitions participate. Neither outer
    test partition is inspected during this selection.
    """
    raw_parts = []

    for dataset_name in BENCHMARK_DATASETS:
        record = dataset_records[dataset_name]
        outer_train_positions, _outer_test_positions = record["outer_folds"][
            outer_fold_number - 1
        ]
        outer_train_df = record["df"].iloc[outer_train_positions]

        raw_parts.append(
            evaluate_inner_configurations(
                dataset_name=dataset_name,
                outer_fold_number=outer_fold_number,
                outer_train_df=outer_train_df,
                selected_features=selected_features,
                configurations=configurations,
            )
        )

    raw = pd.concat(raw_parts, ignore_index=True)
    summary = summarize_shared_inner_results(raw)
    best_config_id = str(summary.iloc[0]["config_id"])
    best_params = recover_shared_config(configurations, best_config_id)
    return raw, summary, best_config_id, best_params


# ============================================================
# OUTER-FOLD TRAIN / EVALUATE -- SAME RF LOGIC, SHARED PARAMS
# ============================================================


def train_outer_fold_with_shared_params(
    dataset_name: str,
    outer_fold_number: int,
    full_df: pd.DataFrame,
    outer_train_positions: np.ndarray,
    outer_test_positions: np.ndarray,
    selected_features: list[str],
    shared_config_id: str,
    shared_params: dict,
) -> dict:
    outer_train_df = full_df.iloc[outer_train_positions]
    outer_test_df = full_df.iloc[outer_test_positions]

    train_sources = set(
        outer_train_df[SOURCE_FILE_COL].astype(str).unique()
    )
    test_sources = set(
        outer_test_df[SOURCE_FILE_COL].astype(str).unique()
    )
    if train_sources & test_sources:
        raise RuntimeError(
            f"{dataset_name}: outer fold {outer_fold_number} has SourceFile leakage."
        )

    X_outer_train_raw = outer_train_df[selected_features]
    X_outer_test_raw = outer_test_df[selected_features]
    y_outer_train = outer_train_df[LABEL_COL].astype(int)
    y_outer_test = outer_test_df[LABEL_COL].astype(int)

    median_imputer = FoldMedianImputer(
        f"{dataset_name}_outer{outer_fold_number}"
    )
    X_outer_train = median_imputer.fit_transform(X_outer_train_raw)
    X_outer_test = median_imputer.transform(X_outer_test_raw)

    class_weights = compute_training_class_weights(y_outer_train)
    model = build_random_forest(
        params=shared_params,
        class_weights=class_weights,
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(X_outer_train, y_outer_train)

    probabilities = model.predict_proba(X_outer_test)[:, 1]
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)
    metrics = calculate_binary_metrics(
        y_outer_test,
        probabilities,
        threshold=PREDICTION_THRESHOLD,
    )

    metric_row = {
        "outer_fold": int(outer_fold_number),
        "selected_shared_config_id": shared_config_id,
        "train_rows": int(len(outer_train_df)),
        "test_rows": int(len(outer_test_df)),
        "train_sources": int(len(train_sources)),
        "test_sources": int(len(test_sources)),
        "train_benign": int((y_outer_train == 0).sum()),
        "train_malicious": int((y_outer_train == 1).sum()),
        "test_benign": int((y_outer_test == 0).sum()),
        "test_malicious": int((y_outer_test == 1).sum()),
        "class_weight_benign": float(class_weights[0]),
        "class_weight_malicious": float(class_weights[1]),
    }
    metric_row.update(metrics)

    oof_rows = pd.DataFrame(
        {
            ROW_INDEX_COL: outer_test_positions.astype(np.int64),
            "Outer_Fold": int(outer_fold_number),
            SOURCE_FILE_COL: outer_test_df[SOURCE_FILE_COL].astype(str).to_numpy(),
            "Actual_Label": y_outer_test.to_numpy(dtype=int),
            "Malicious_Probability": probabilities,
            "Predicted_Label": predictions,
            "Selected_Shared_Config_ID": shared_config_id,
        }
    )

    importance = pd.DataFrame(
        {
            "outer_fold": int(outer_fold_number),
            "Feature": selected_features,
            "MDI_Importance": model.feature_importances_,
        }
    )

    return {
        "model": model,
        "median_imputer": median_imputer,
        "selected_features": selected_features,
        "shared_config_id": shared_config_id,
        "shared_params": deepcopy(shared_params),
        "class_weights": class_weights,
        "metric_row": metric_row,
        "oof_rows": oof_rows,
        "importance": importance,
    }


# ============================================================
# FINAL DEPLOYMENT MODEL -- SAME SHARED CONFIGURATION FOR D2/D3
# ============================================================


def train_final_deployment_model(
    dataset_name: str,
    df: pd.DataFrame,
    selected_features: list[str],
    shared_config_id: str,
    shared_params: dict,
):
    """
    Fit one deployment RF on all rows after nested-CV OOF evaluation is done.
    The final shared hyperparameters are identical for D2 and D3; only learned
    medians, class weights and tree parameters differ because data differ.
    """
    X_raw = df[selected_features]
    y = df[LABEL_COL].astype(int)

    median_imputer = FoldMedianImputer(f"{dataset_name}_final_deployment")
    X = median_imputer.fit_transform(X_raw)
    class_weights = compute_training_class_weights(y)

    model = build_random_forest(
        params=shared_params,
        class_weights=class_weights,
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(X, y)

    return model, median_imputer, class_weights


# ============================================================
# ARTIFACT / RESULT SAVING
# ============================================================


def save_dataset_artifacts(
    dataset_name: str,
    fold_manifest: pd.DataFrame,
    outer_metrics: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    importance_by_fold: pd.DataFrame,
    fold_results: list[dict],
    final_model,
    final_median_imputer: FoldMedianImputer,
    selected_features: list[str],
    final_shared_config_id: str,
    final_shared_params: dict,
    final_class_weights: dict,
) -> None:
    artifact_dir = MODEL_ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    fold_manifest.to_csv(artifact_dir / "outer_fold_manifest.csv", index=False)
    pd.DataFrame({"Feature": selected_features}).to_csv(
        artifact_dir / "selected_features.csv", index=False
    )

    for fold_number, fold_result in enumerate(fold_results, start=1):
        fold_dir = artifact_dir / f"outer_fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            fold_result["model"],
            fold_dir / "random_forest.joblib",
        )
        fold_result["median_imputer"].save(
            fold_dir / "median_imputer.joblib"
        )
        with (fold_dir / "selected_shared_hyperparameters.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "config_id": fold_result["shared_config_id"],
                    "parameters": fold_result["shared_params"],
                },
                handle,
                indent=2,
            )
        with (fold_dir / "class_weights.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {str(k): float(v) for k, v in fold_result["class_weights"].items()},
                handle,
                indent=2,
            )

    joblib.dump(final_model, artifact_dir / "random_forest_final.joblib")
    final_median_imputer.save(artifact_dir / "median_imputer_final.joblib")

    with (artifact_dir / "final_hyperparameters.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "config_id": final_shared_config_id,
                "parameters": final_shared_params,
                "scope": "shared_dataset2_dataset3",
            },
            handle,
            indent=2,
        )

    with (artifact_dir / "final_class_weights.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {str(k): float(v) for k, v in final_class_weights.items()},
            handle,
            indent=2,
        )

    outer_metrics.to_csv(result_dir / "outer_fold_metrics.csv", index=False)
    oof_predictions.to_csv(result_dir / "oof_predictions.csv", index=False)
    importance_by_fold.to_csv(
        result_dir / "feature_importance_by_outer_fold.csv", index=False
    )

    (
        importance_by_fold.groupby("Feature")["MDI_Importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "MDI_Importance_Mean",
                "std": "MDI_Importance_Std",
            }
        )
        .sort_values("MDI_Importance_Mean", ascending=False)
        .to_csv(result_dir / "feature_importance_mean.csv", index=False)
    )


def save_shared_selection_outputs(
    shared_inner_runs: pd.DataFrame,
    shared_outer_summaries: pd.DataFrame,
    shared_outer_selections: pd.DataFrame,
    final_summary: pd.DataFrame,
    final_config_id: str,
    final_params: dict,
) -> None:
    MODEL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    SHARED_SELECTION_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with SHARED_HYPERPARAMETER_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_id": final_config_id,
                "parameters": final_params,
                "scope": "shared_dataset2_dataset3",
            },
            handle,
            indent=2,
        )

    shared_inner_runs.to_csv(
        SHARED_SELECTION_RESULT_DIR / "shared_inner_runs.csv",
        index=False,
    )
    shared_outer_summaries.to_csv(
        SHARED_SELECTION_RESULT_DIR / "shared_sensitivity_by_outer_fold.csv",
        index=False,
    )
    shared_outer_selections.to_csv(
        SHARED_SELECTION_RESULT_DIR / "shared_outer_fold_selections.csv",
        index=False,
    )
    final_summary.to_csv(
        SHARED_SELECTION_RESULT_DIR / "shared_final_sensitivity_summary.csv",
        index=False,
    )

    # Provenance for report Section 1.4.
    pd.DataFrame(
        [
            {
                "Hyperparameter": "n_estimators",
                "Dataset2Previous": 128,
                "Dataset3Previous": 256,
                "SharedTreatment": "jointly evaluate 128 vs 256",
            },
            {
                "Hyperparameter": "max_depth",
                "Dataset2Previous": 10,
                "Dataset3Previous": 10,
                "SharedTreatment": "fixed at 10",
            },
            {
                "Hyperparameter": "min_samples_leaf",
                "Dataset2Previous": 1,
                "Dataset3Previous": 1,
                "SharedTreatment": "fixed at 1",
            },
            {
                "Hyperparameter": "max_features",
                "Dataset2Previous": "0.5",
                "Dataset3Previous": "sqrt",
                "SharedTreatment": "jointly evaluate 0.5 vs sqrt",
            },
            {
                "Hyperparameter": "bootstrap",
                "Dataset2Previous": True,
                "Dataset3Previous": True,
                "SharedTreatment": "fixed at True",
            },
        ]
    ).to_csv(
        SHARED_SELECTION_RESULT_DIR / "prior_d2_d3_hyperparameter_consensus.csv",
        index=False,
    )

    with (SHARED_SELECTION_RESULT_DIR / "selection_strategy.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "benchmark_datasets": list(BENCHMARK_DATASETS),
                "validation_strategy": (
                    "nested StratifiedGroupKFold using SourceFile groups; "
                    "outer folds for unbiased OOF evaluation and inner folds for selection"
                ),
                "outer_cv_requested_folds": OUTER_CV_FOLDS,
                "inner_cv_requested_folds": INNER_CV_FOLDS,
                "prior_consensus_fixed": {
                    "max_depth": 10,
                    "min_samples_leaf": 1,
                    "bootstrap": True,
                },
                "focused_joint_search": {
                    "n_estimators": FOCUSED_N_ESTIMATORS,
                    "max_features": FOCUSED_MAX_FEATURES,
                    "cartesian_combinations": 4,
                },
                "dataset_weighting": (
                    "equal: metrics are averaged inside each dataset first, then across D2/D3"
                ),
                "primary_selection_metric": "mean PR-AUC across D2 and D3",
                "first_tiebreak": "minimum PR-AUC across D2 and D3",
                "class_imbalance_strategy_changed": False,
                "median_imputation_strategy_changed": False,
                "prediction_threshold_changed": False,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "outer_test_used_for_hyperparameter_selection": False,
            },
            handle,
            indent=2,
        )


# ============================================================
# JOINT D2/D3 NESTED-CV EXPERIMENT
# ============================================================


def train_all_datasets() -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("RANDOM FOREST - D2/D3 SHARED-HYPERPARAMETER NESTED GROUP CV")
    print("=" * 90)

    selected_features = load_selected_features()
    configurations = make_focused_shared_configurations()

    dataset_records = {}
    feasible_outer_counts = {}

    for dataset_name in BENCHMARK_DATASETS:
        raw_df = load_ingested_dataset(dataset_name)
        df = prepare_model_input(raw_df, selected_features, dataset_name)
        group_table = validate_source_groups(df, dataset_name)

        print(f"\n{dataset_name.upper()}")
        print(f"Fixed Step 3 features: {len(selected_features)}")
        print(
            group_table.groupby(LABEL_COL)
            .agg(
                SourceFiles=(SOURCE_FILE_COL, "size"),
                Rows=("Rows", "sum"),
            )
            .to_string()
        )

        feasible_outer_counts[dataset_name] = feasible_group_fold_count(
            df,
            requested_folds=OUTER_CV_FOLDS,
            dataset_name=f"{dataset_name}_outer_feasibility",
        )
        dataset_records[dataset_name] = {"df": df}

    # Use the same number of outer folds for both datasets. In the expected
    # project data this remains 5; the minimum is a defensive fallback.
    common_outer_folds = min(feasible_outer_counts.values())
    if common_outer_folds < 2:
        raise RuntimeError("Not enough SourceFile groups for paired nested CV.")

    if common_outer_folds != OUTER_CV_FOLDS:
        print(
            f"Requested {OUTER_CV_FOLDS} outer folds, but paired D2/D3 CV "
            f"is limited to {common_outer_folds} by SourceFile availability."
        )

    for dataset_name in BENCHMARK_DATASETS:
        df = dataset_records[dataset_name]["df"]
        outer_folds = make_stratified_group_folds(
            df,
            requested_folds=common_outer_folds,
            dataset_name=f"{dataset_name}_outer",
            random_state=CV_RANDOM_STATE,
        )
        dataset_records[dataset_name]["outer_folds"] = outer_folds
        dataset_records[dataset_name]["fold_manifest"] = build_fold_manifest(
            df,
            outer_folds,
            ROW_INDEX_COL,
        )
        dataset_records[dataset_name]["fold_results"] = []
        dataset_records[dataset_name]["outer_metric_rows"] = []
        dataset_records[dataset_name]["oof_parts"] = []
        dataset_records[dataset_name]["importance_parts"] = []

    print(f"\nOuter folds: {common_outer_folds}")
    print(f"Inner folds requested: {INNER_CV_FOLDS}")
    print("Focused shared configurations: 4")
    for config in configurations:
        print(
            f"  {config['config_id']}: n_estimators={config['n_estimators']}, "
            f"max_features={config['max_features']}"
        )

    all_shared_inner_runs = []
    all_shared_outer_summaries = []
    shared_outer_selection_rows = []

    for outer_fold_number in range(1, common_outer_folds + 1):
        print("\n" + "-" * 90)
        print(f"PAIRED OUTER FOLD {outer_fold_number}")
        print("-" * 90)
        print("  Joint D2/D3 inner-CV hyperparameter selection:")

        raw, summary, shared_config_id, shared_params = run_joint_inner_selection(
            outer_fold_number=outer_fold_number,
            dataset_records=dataset_records,
            selected_features=selected_features,
            configurations=configurations,
        )

        raw = raw.copy()
        summary = summary.copy()
        summary["outer_fold"] = int(outer_fold_number)
        all_shared_inner_runs.append(raw)
        all_shared_outer_summaries.append(summary)

        selected_row = summary.iloc[0]
        shared_outer_selection_rows.append(
            {
                "outer_fold": int(outer_fold_number),
                "config_id": shared_config_id,
                "n_estimators": int(shared_params["n_estimators"]),
                "max_features": shared_params["max_features"],
                "pr_auc_mean_across_datasets": float(
                    selected_row["pr_auc_mean_across_datasets"]
                ),
                "pr_auc_min_across_datasets": float(
                    selected_row["pr_auc_min_across_datasets"]
                ),
            }
        )

        print(
            f"\n  Selected SHARED config for outer fold {outer_fold_number}: "
            f"{shared_config_id} | n_estimators={shared_params['n_estimators']}, "
            f"max_features={shared_params['max_features']}"
        )

        for dataset_name in BENCHMARK_DATASETS:
            record = dataset_records[dataset_name]
            train_positions, test_positions = record["outer_folds"][
                outer_fold_number - 1
            ]

            print(f"    -> train/evaluate {dataset_name}")
            fold_result = train_outer_fold_with_shared_params(
                dataset_name=dataset_name,
                outer_fold_number=outer_fold_number,
                full_df=record["df"],
                outer_train_positions=train_positions,
                outer_test_positions=test_positions,
                selected_features=selected_features,
                shared_config_id=shared_config_id,
                shared_params=shared_params,
            )

            record["fold_results"].append(fold_result)
            record["outer_metric_rows"].append(fold_result["metric_row"])
            record["oof_parts"].append(fold_result["oof_rows"])
            record["importance_parts"].append(fold_result["importance"])

    shared_inner_runs = pd.concat(all_shared_inner_runs, ignore_index=True)
    shared_outer_summaries = pd.concat(
        all_shared_outer_summaries, ignore_index=True
    )
    shared_outer_selections = pd.DataFrame(shared_outer_selection_rows)

    # For the final deployment configuration, aggregate all nested-CV inner
    # validation evidence. This happens after unbiased OOF predictions have
    # already been generated, matching the original deployment-model logic.
    final_shared_summary = summarize_shared_inner_results(shared_inner_runs)
    final_shared_config_id = str(final_shared_summary.iloc[0]["config_id"])
    final_shared_params = recover_shared_config(
        configurations,
        final_shared_config_id,
    )

    print("\n" + "=" * 90)
    print("FINAL SHARED RANDOM FOREST CONFIGURATION")
    print("=" * 90)
    print(f"config_id: {final_shared_config_id}")
    print(f"n_estimators: {final_shared_params['n_estimators']}")
    print(f"max_depth: {final_shared_params['max_depth']}")
    print(f"min_samples_leaf: {final_shared_params['min_samples_leaf']}")
    print(f"max_features: {final_shared_params['max_features']}")
    print(f"bootstrap: {final_shared_params['bootstrap']}")

    save_shared_selection_outputs(
        shared_inner_runs=shared_inner_runs,
        shared_outer_summaries=shared_outer_summaries,
        shared_outer_selections=shared_outer_selections,
        final_summary=final_shared_summary,
        final_config_id=final_shared_config_id,
        final_params=final_shared_params,
    )

    completion_rows = []

    for dataset_name in BENCHMARK_DATASETS:
        record = dataset_records[dataset_name]
        df = record["df"]

        outer_metrics = (
            pd.DataFrame(record["outer_metric_rows"])
            .sort_values("outer_fold")
            .reset_index(drop=True)
        )
        oof_predictions = (
            pd.concat(record["oof_parts"], ignore_index=True)
            .sort_values(ROW_INDEX_COL)
            .reset_index(drop=True)
        )

        if len(oof_predictions) != len(df):
            raise RuntimeError(
                f"{dataset_name}: OOF predictions do not cover every row exactly once."
            )
        if oof_predictions[ROW_INDEX_COL].nunique() != len(df):
            raise RuntimeError(
                f"{dataset_name}: duplicate/missing OOF row indices detected."
            )

        importance_by_fold = pd.concat(
            record["importance_parts"], ignore_index=True
        )

        final_model, final_imputer, final_class_weights = (
            train_final_deployment_model(
                dataset_name=dataset_name,
                df=df,
                selected_features=selected_features,
                shared_config_id=final_shared_config_id,
                shared_params=final_shared_params,
            )
        )

        save_dataset_artifacts(
            dataset_name=dataset_name,
            fold_manifest=record["fold_manifest"],
            outer_metrics=outer_metrics,
            oof_predictions=oof_predictions,
            importance_by_fold=importance_by_fold,
            fold_results=record["fold_results"],
            final_model=final_model,
            final_median_imputer=final_imputer,
            selected_features=selected_features,
            final_shared_config_id=final_shared_config_id,
            final_shared_params=final_shared_params,
            final_class_weights=final_class_weights,
        )

        completion_rows.append(
            {
                "dataset": dataset_name,
                "outer_folds": common_outer_folds,
                "rows": len(df),
                "shared_final_config_id": final_shared_config_id,
                "shared_n_estimators": final_shared_params["n_estimators"],
                "shared_max_features": final_shared_params["max_features"],
            }
        )

    return pd.DataFrame(completion_rows)
