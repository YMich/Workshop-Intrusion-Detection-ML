from __future__ import annotations

import json
import sys
import joblib
import numpy as np
import pandas as pd

from random_forest_config import (
    BASE_RF_PARAMS,
    BENCHMARK_DATASETS,
    CV_RANDOM_STATE,
    DATASET_PATHS,
    LABEL_COL,
    MODEL_ARTIFACT_ROOT,
    MODEL_RANDOM_STATE,
    OUTER_CV_FOLDS,
    PREDICTION_THRESHOLD,
    PROJECT_ROOT,
    RESULT_ROOT,
    ROW_INDEX_COL,
    SHARED_HYPERPARAMETER_PATH,
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


def load_ingested_dataset(dataset_name: str) -> pd.DataFrame:
    path = DATASET_PATHS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name}: ingested dataset not found:\n{path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {dataset_name}: {path} | shape={df.shape}")
    return df


def load_selected_features() -> list[str]:
    """Derive the fixed model-agnostic Step-3 feature schema from CSV headers."""
    reference_features = None
    reference_dataset = None
    for dataset_name in BENCHMARK_DATASETS:
        path = DATASET_PATHS[dataset_name]
        if not path.exists():
            raise FileNotFoundError(f"{dataset_name}: ingested dataset not found:\n{path}")
        header = clean_column_names(pd.read_csv(path, nrows=0))
        validate_required_columns(header, f"{dataset_name}_rf_schema")
        features, _dropped, _manifest = select_feature_columns(header)
        if len(features) != EXPECTED_FINAL_FEATURE_COUNT:
            raise ValueError(
                f"{dataset_name}: expected {EXPECTED_FINAL_FEATURE_COUNT} Step-3 "
                f"features, found {len(features)}."
            )
        if len(features) != len(set(features)):
            raise ValueError(f"{dataset_name}: duplicate Step-3 feature names.")
        if reference_features is None:
            reference_features = list(features)
            reference_dataset = dataset_name
        elif features != reference_features:
            raise ValueError(
                f"{dataset_name}: Step-3 feature schema differs from {reference_dataset}."
            )
    if reference_features is None:
        raise RuntimeError("No benchmark datasets are configured.")
    return reference_features


def prepare_model_input(
    df: pd.DataFrame,
    selected_features: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    required = [SOURCE_FILE_COL, LABEL_COL] + selected_features
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing model-input columns: {missing}")
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


def train_outer_fold(
    dataset_name: str,
    outer_fold_number: int,
    full_df: pd.DataFrame,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    selected_features: list[str],
) -> dict:
    """Fit the frozen RF configuration on one source-aware outer training fold."""
    train_df = full_df.iloc[train_positions]
    test_df = full_df.iloc[test_positions]

    train_sources = set(train_df[SOURCE_FILE_COL].astype(str).unique())
    test_sources = set(test_df[SOURCE_FILE_COL].astype(str).unique())
    if train_sources & test_sources:
        raise RuntimeError(
            f"{dataset_name}: outer fold {outer_fold_number} has SourceFile leakage."
        )

    y_train = train_df[LABEL_COL].astype(int)
    y_test = test_df[LABEL_COL].astype(int)
    imputer = FoldMedianImputer(f"{dataset_name}_outer{outer_fold_number}")
    X_train = imputer.fit_transform(train_df[selected_features])
    X_test = imputer.transform(test_df[selected_features])
    class_weights = compute_training_class_weights(y_train)

    model = build_random_forest(
        params=BASE_RF_PARAMS,
        class_weights=class_weights,
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)
    metrics = calculate_binary_metrics(
        y_test,
        probabilities,
        threshold=PREDICTION_THRESHOLD,
    )

    metric_row = {
        "outer_fold": int(outer_fold_number),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_sources": int(len(train_sources)),
        "test_sources": int(len(test_sources)),
        "train_benign": int((y_train == 0).sum()),
        "train_malicious": int((y_train == 1).sum()),
        "test_benign": int((y_test == 0).sum()),
        "test_malicious": int((y_test == 1).sum()),
        "class_weight_benign": float(class_weights[0]),
        "class_weight_malicious": float(class_weights[1]),
        **metrics,
    }

    oof_rows = pd.DataFrame(
        {
            ROW_INDEX_COL: test_positions.astype(np.int64),
            "Outer_Fold": int(outer_fold_number),
            SOURCE_FILE_COL: test_df[SOURCE_FILE_COL].astype(str).to_numpy(),
            "Actual_Label": y_test.to_numpy(dtype=int),
            "Malicious_Probability": probabilities,
            "Predicted_Label": predictions,
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
        "median_imputer": imputer,
        "class_weights": class_weights,
        "metric_row": metric_row,
        "oof_rows": oof_rows,
        "importance": importance,
    }


def train_final_deployment_model(
    dataset_name: str,
    df: pd.DataFrame,
    selected_features: list[str],
):
    """Fit a deployment artifact on all rows after OOF evaluation is complete."""
    y = df[LABEL_COL].astype(int)
    imputer = FoldMedianImputer(f"{dataset_name}_final_deployment")
    X = imputer.fit_transform(df[selected_features])
    class_weights = compute_training_class_weights(y)
    model = build_random_forest(
        params=BASE_RF_PARAMS,
        class_weights=class_weights,
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(X, y)
    return model, imputer, class_weights


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
        joblib.dump(fold_result["model"], fold_dir / "random_forest.joblib")
        fold_result["median_imputer"].save(fold_dir / "median_imputer.joblib")
        with (fold_dir / "model_config.json").open("w", encoding="utf-8") as handle:
            json.dump(BASE_RF_PARAMS, handle, indent=2)
        with (fold_dir / "class_weights.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {str(k): float(v) for k, v in fold_result["class_weights"].items()},
                handle,
                indent=2,
            )

    joblib.dump(final_model, artifact_dir / "random_forest_final.joblib")
    final_median_imputer.save(artifact_dir / "median_imputer_final.joblib")
    with (artifact_dir / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scope": "shared_dataset2_dataset3_fixed_baseline",
                "parameters": BASE_RF_PARAMS,
            },
            handle,
            indent=2,
        )
    with (artifact_dir / "final_class_weights.json").open("w", encoding="utf-8") as handle:
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


def train_all_datasets() -> pd.DataFrame:
    """Run fixed-parameter source-aware OOF evaluation on D2 and D3."""
    print("\n" + "=" * 90)
    print("RANDOM FOREST - FIXED D2/D3 BASELINE GROUP OOF")
    print("=" * 90)
    print(f"Frozen RF parameters: {BASE_RF_PARAMS}")

    MODEL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with SHARED_HYPERPARAMETER_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scope": "shared_dataset2_dataset3_fixed_baseline",
                "configuration_mode": "fixed_baseline_no_hyperparameter_search",
                "parameters": BASE_RF_PARAMS,
            },
            handle,
            indent=2,
        )

    selected_features = load_selected_features()
    completion_rows = []

    for dataset_name in BENCHMARK_DATASETS:
        raw_df = load_ingested_dataset(dataset_name)
        df = prepare_model_input(raw_df, selected_features, dataset_name)
        group_table = validate_source_groups(df, dataset_name)
        print(f"\n{dataset_name.upper()}")
        print(f"Fixed Step-3 features: {len(selected_features)}")
        print(
            group_table.groupby(LABEL_COL)
            .agg(SourceFiles=(SOURCE_FILE_COL, "size"), Rows=("Rows", "sum"))
            .to_string()
        )

        fold_count = feasible_group_fold_count(
            df,
            requested_folds=OUTER_CV_FOLDS,
            dataset_name=f"{dataset_name}_outer_feasibility",
        )
        if fold_count < 2:
            raise RuntimeError(f"{dataset_name}: not enough SourceFile groups for OOF CV.")

        outer_folds = make_stratified_group_folds(
            df,
            requested_folds=fold_count,
            dataset_name=f"{dataset_name}_outer",
            random_state=CV_RANDOM_STATE,
        )
        fold_manifest = build_fold_manifest(df, outer_folds, ROW_INDEX_COL)

        fold_results = []
        metric_rows = []
        oof_parts = []
        importance_parts = []
        for fold_number, (train_positions, test_positions) in enumerate(
            outer_folds, start=1
        ):
            print(f"  Outer fold {fold_number}/{len(outer_folds)}")
            result = train_outer_fold(
                dataset_name,
                fold_number,
                df,
                train_positions,
                test_positions,
                selected_features,
            )
            fold_results.append(result)
            metric_rows.append(result["metric_row"])
            oof_parts.append(result["oof_rows"])
            importance_parts.append(result["importance"])

        outer_metrics = pd.DataFrame(metric_rows).sort_values("outer_fold")
        oof_predictions = (
            pd.concat(oof_parts, ignore_index=True)
            .sort_values(ROW_INDEX_COL)
            .reset_index(drop=True)
        )
        if len(oof_predictions) != len(df):
            raise RuntimeError(f"{dataset_name}: OOF predictions do not cover every row.")
        if oof_predictions[ROW_INDEX_COL].nunique() != len(df):
            raise RuntimeError(f"{dataset_name}: duplicate/missing OOF row indices.")

        importance_by_fold = pd.concat(importance_parts, ignore_index=True)
        final_model, final_imputer, final_class_weights = train_final_deployment_model(
            dataset_name,
            df,
            selected_features,
        )
        save_dataset_artifacts(
            dataset_name,
            fold_manifest,
            outer_metrics,
            oof_predictions,
            importance_by_fold,
            fold_results,
            final_model,
            final_imputer,
            selected_features,
            final_class_weights,
        )
        completion_rows.append(
            {"dataset": dataset_name, "outer_folds": len(outer_folds), "rows": len(df)}
        )

    return pd.DataFrame(completion_rows)
