from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from random_forest_config import (
    ARTIFACT_ROOT,
    BENCHMARK_DATASETS,
    CV_RANDOM_STATE,
    DATASET_PATHS,
    FINAL_RF_PARAMS,
    LABEL_COL,
    MODEL_RANDOM_STATE,
    OUTER_CV_FOLDS,
    PREDICTION_THRESHOLD,
    RESULT_ROOT,
    ROW_INDEX_COL,
    SOURCE_FILE_COL,
)
from random_forest_preprocessing import FoldMedianImputer
from random_forest_temporal import final_feature_names, load_temporal_dataset
from random_forest_utils import (
    build_final_random_forest,
    calculate_binary_metrics,
    compute_training_class_weights,
    make_source_aware_folds,
    validate_source_groups,
)


def _model_frame(temporal_df: pd.DataFrame) -> pd.DataFrame:
    features = final_feature_names()
    required = [ROW_INDEX_COL, SOURCE_FILE_COL, LABEL_COL] + features
    missing = [c for c in required if c not in temporal_df.columns]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")
    frame = temporal_df[required].copy()
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    frame[LABEL_COL] = pd.to_numeric(frame[LABEL_COL], errors="raise").astype(int)
    return frame


def evaluate_oof(dataset_name: str, temporal_df: pd.DataFrame):
    features = final_feature_names()
    frame = _model_frame(temporal_df)
    validate_source_groups(frame, dataset_name)
    folds = make_source_aware_folds(
        frame,
        requested_folds=OUTER_CV_FOLDS,
        dataset_name=f"{dataset_name}_final_rf",
        random_state=CV_RANDOM_STATE,
    )

    oof_parts = []
    fold_rows = []
    importance_parts = []

    for fold_number, (train_pos, test_pos) in enumerate(folds, start=1):
        train_df = frame.iloc[train_pos]
        test_df = frame.iloc[test_pos]
        y_train = train_df[LABEL_COL].astype(int)
        y_test = test_df[LABEL_COL].astype(int)

        imputer = FoldMedianImputer(f"{dataset_name}_final_rf_fold{fold_number}")
        X_train = imputer.fit_transform(train_df[features])
        X_test = imputer.transform(test_df[features])
        class_weights = compute_training_class_weights(y_train)
        model = build_final_random_forest(class_weights, MODEL_RANDOM_STATE)
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= PREDICTION_THRESHOLD).astype(int)
        metrics = calculate_binary_metrics(y_test, prob)
        fold_rows.append(
            {
                "Dataset": dataset_name,
                "Outer_Fold": fold_number,
                "Feature_Count": len(features),
                "Train_Rows": len(train_df),
                "Test_Rows": len(test_df),
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
    if len(oof) != len(frame) or oof[ROW_INDEX_COL].nunique() != len(frame):
        raise RuntimeError(f"{dataset_name}: OOF predictions do not cover every row exactly once.")

    overall = calculate_binary_metrics(
        oof["Actual_Label"].to_numpy(dtype=int),
        oof["Malicious_Probability"].to_numpy(dtype=float),
    )
    summary = {
        "Dataset": dataset_name,
        "Feature_Count": len(features),
        "Hyperparameters": "shared_final_256_None_2_sqrt",
        **overall,
    }
    importance = pd.concat(importance_parts, ignore_index=True)
    return summary, oof, pd.DataFrame(fold_rows), importance


def fit_full_model(dataset_name: str, temporal_df: pd.DataFrame):
    features = final_feature_names()
    frame = _model_frame(temporal_df)
    y = frame[LABEL_COL].astype(int)
    imputer = FoldMedianImputer(f"{dataset_name}_final_deployment")
    X = imputer.fit_transform(frame[features])
    class_weights = compute_training_class_weights(y)
    model = build_final_random_forest(class_weights, MODEL_RANDOM_STATE)
    model.fit(X, y)
    return model, imputer, class_weights


def cross_dataset_evaluation(records: dict[str, pd.DataFrame]):
    features = final_feature_names()
    rows = []
    prediction_parts = []

    directions = (("dataset2", "dataset3"), ("dataset3", "dataset2"))
    for train_name, test_name in directions:
        train_df = _model_frame(records[train_name])
        test_df = _model_frame(records[test_name])

        y_train = train_df[LABEL_COL].astype(int)
        y_test = test_df[LABEL_COL].astype(int)
        imputer = FoldMedianImputer(f"cross_{train_name}_to_{test_name}")
        X_train = imputer.fit_transform(train_df[features])
        X_test = imputer.transform(test_df[features])
        weights = compute_training_class_weights(y_train)
        model = build_final_random_forest(weights, MODEL_RANDOM_STATE)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= PREDICTION_THRESHOLD).astype(int)
        metrics = calculate_binary_metrics(y_test, prob)

        rows.append(
            {
                "Train_Dataset": train_name,
                "Test_Dataset": test_name,
                "Feature_Count": len(features),
                "Hyperparameters": "shared_final_256_None_2_sqrt",
                "Interpretation": (
                    "cross_dataset_transfer_descriptive; target labels used only for evaluation"
                ),
                **metrics,
            }
        )
        prediction_parts.append(
            pd.DataFrame(
                {
                    "Train_Dataset": train_name,
                    "Test_Dataset": test_name,
                    ROW_INDEX_COL: test_df[ROW_INDEX_COL].to_numpy(dtype=np.int64),
                    SOURCE_FILE_COL: test_df[SOURCE_FILE_COL].astype(str).to_numpy(),
                    "Actual_Label": y_test.to_numpy(dtype=int),
                    "Malicious_Probability": prob,
                    "Predicted_Label": pred,
                }
            )
        )

    return pd.DataFrame(rows), pd.concat(prediction_parts, ignore_index=True)


def save_final_model_artifacts(
    dataset_name: str,
    model,
    imputer: FoldMedianImputer,
    class_weights: dict,
):
    features = final_feature_names()
    out = ARTIFACT_ROOT / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "random_forest_final.joblib")
    imputer.save(out / "median_imputer_final.joblib")
    pd.DataFrame({"Feature": features}).to_csv(out / "selected_features.csv", index=False)

    with (out / "final_model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "RandomForestClassifier",
                "representation": "endpoint_aware_RROLL5_32_features",
                "sequence_key": ["SourceFile", "Src IP", "Dst IP"],
                "rolling_window": 5,
                "causal_history": "previous rows only via shift(1)",
                "rolling_std": "min_periods=1, ddof=0",
                "includes_interflow_delta_prev5_cv": True,
                "feature_count": len(features),
                "hyperparameters": FINAL_RF_PARAMS,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "class_weights": {str(k): float(v) for k, v in class_weights.items()},
                "hyperparameter_scope": "frozen_shared_D2_D3_applied_to_D1",
            },
            handle,
            indent=2,
        )


def run_final_rf_pipeline() -> pd.DataFrame:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FINAL RANDOM FOREST - ENDPOINT-AWARE RROLL5")
    print("=" * 100)
    print("Final feature count: 32")
    print("Sequence key: (SourceFile, Src IP, Dst IP)")
    print("Rolling window: previous 5 flows, causal shift(1)")
    print("Rolling std: min_periods=1, ddof=0")
    print("Includes: InterFlow Delta Prev5 CV")
    print("Frozen final D2/D3 RF hyperparameters applied unchanged to Dataset 1:")
    print("  n_estimators=256")
    print("  max_depth=None")
    print("  min_samples_leaf=2")
    print("  max_features='sqrt'")
    print("  bootstrap=True")
    print("  threshold=0.50")

    temporal_records = {
        name: load_temporal_dataset(DATASET_PATHS[name], name)
        for name in BENCHMARK_DATASETS
    }

    summary_rows = []
    for dataset_name in BENCHMARK_DATASETS:
        print("\n" + "-" * 100)
        print(f"SOURCE-AWARE OOF EVALUATION: {dataset_name.upper()}")
        print("-" * 100)
        summary, oof, fold_metrics, importance = evaluate_oof(
            dataset_name, temporal_records[dataset_name]
        )
        summary_rows.append(summary)

        result_dir = RESULT_ROOT / dataset_name
        result_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(result_dir / "oof_predictions.csv", index=False)
        fold_metrics.to_csv(result_dir / "outer_fold_metrics.csv", index=False)
        importance.to_csv(result_dir / "feature_importance_by_fold.csv", index=False)
        (
            importance.groupby("Feature")["MDI_Importance"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "MDI_Importance_Mean", "std": "MDI_Importance_Std"})
            .sort_values("MDI_Importance_Mean", ascending=False)
            .to_csv(result_dir / "feature_importance_mean.csv", index=False)
        )
        with (result_dir / "oof_overall_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print(
            f"Recall={summary['recall_tpr']:.4f} | FPR={summary['fpr']:.6f} | "
            f"F1={summary['f1']:.4f} | PR-AUC={summary['pr_auc']:.4f}"
        )

        model, imputer, weights = fit_full_model(
            dataset_name, temporal_records[dataset_name]
        )
        save_final_model_artifacts(dataset_name, model, imputer, weights)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULT_ROOT / "final_rf_oof_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("CROSS-DATASET TRANSFER")
    print("=" * 100)
    print("Skipped in the Dataset-1 local runner. Use cross_dataset_evaluation.py for all pairwise transfers.")

    return summary_df
