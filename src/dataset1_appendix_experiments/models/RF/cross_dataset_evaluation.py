from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import random_forest_config as cfg
from random_forest_preprocessing import FoldMedianImputer
from random_forest_temporal import final_feature_names, load_temporal_dataset
from random_forest_utils import (
    build_final_random_forest,
    calculate_binary_metrics,
    compute_training_class_weights,
)

PROJECT_ROOT = cfg.PROJECT_ROOT
INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"
DATASET_PATHS = {name: INGESTED_ROOT / f"{name}_ingested.csv" for name in ("dataset1", "dataset2", "dataset3")}
DIRECTIONS = tuple((s, t) for s in DATASET_PATHS for t in DATASET_PATHS if s != t)
RESULT_ROOT = PROJECT_ROOT / "results" / "dataset1_appendix" / "cross_dataset" / "RF"


def _model_frame(temporal_df: pd.DataFrame) -> pd.DataFrame:
    features = final_feature_names()
    required = [cfg.ROW_INDEX_COL, cfg.SOURCE_FILE_COL, cfg.LABEL_COL] + features
    missing = [column for column in required if column not in temporal_df.columns]
    if missing:
        raise ValueError(f"Missing RF columns: {missing}")
    frame = temporal_df[required].copy()
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame[cfg.LABEL_COL] = pd.to_numeric(frame[cfg.LABEL_COL], errors="raise").astype(int)
    return frame


def evaluate_transfer(source_dataset: str, target_dataset: str) -> dict:
    features = final_feature_names()
    source = _model_frame(load_temporal_dataset(DATASET_PATHS[source_dataset], source_dataset))
    target = _model_frame(load_temporal_dataset(DATASET_PATHS[target_dataset], target_dataset))

    y_source = source[cfg.LABEL_COL].astype(int)
    y_target = target[cfg.LABEL_COL].astype(int)
    imputer = FoldMedianImputer(f"cross_{source_dataset}_to_{target_dataset}")
    X_source = imputer.fit_transform(source[features])
    X_target = imputer.transform(target[features])
    weights = compute_training_class_weights(y_source)
    model = build_final_random_forest(weights, cfg.MODEL_RANDOM_STATE)
    model.fit(X_source, y_source)

    probability = model.predict_proba(X_target)[:, 1]
    prediction = (probability >= cfg.PREDICTION_THRESHOLD).astype(int)
    metrics = calculate_binary_metrics(y_target, probability)

    output_dir = RESULT_ROOT / f"{source_dataset}_to_{target_dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.DataFrame({
        cfg.ROW_INDEX_COL: target[cfg.ROW_INDEX_COL].to_numpy(dtype=np.int64),
        cfg.SOURCE_FILE_COL: target[cfg.SOURCE_FILE_COL].astype(str).to_numpy(),
        "Actual_Label": y_target.to_numpy(dtype=int),
        "Malicious_Probability": probability,
        "Predicted_Label": prediction,
    })
    prediction_table.to_csv(output_dir / "cross_dataset_predictions.csv", index=False)
    (
        prediction_table.groupby(cfg.SOURCE_FILE_COL, dropna=False)
        .agg(
            Rows=("Actual_Label", "size"),
            ActualLabel=("Actual_Label", "first"),
            PredictedMalicious=("Predicted_Label", "sum"),
            MeanMaliciousProbability=("Malicious_Probability", "mean"),
            MedianMaliciousProbability=("Malicious_Probability", "median"),
        )
        .reset_index()
        .to_csv(output_dir / "cross_dataset_by_source.csv", index=False)
    )

    result = {
        "Model": "RandomForest",
        "SourceDataset": source_dataset,
        "TargetDataset": target_dataset,
        "EvaluationPopulation": "full_target",
        "FeatureCount": len(features),
        "TargetUsedForTraining": False,
        "TargetUsedForPreprocessingFit": False,
        "TargetUsedForThresholdCalibration": False,
        "Threshold": float(cfg.PREDICTION_THRESHOLD),
        **metrics,
    }
    pd.DataFrame([result]).to_csv(output_dir / "cross_dataset_metrics.csv", index=False)
    print(
        f"RF {source_dataset} -> {target_dataset}: "
        f"Recall={metrics['recall_tpr']:.4f}, FPR={metrics['fpr']:.4f}, "
        f"F1={metrics['f1']:.4f}, PR-AUC={metrics['pr_auc']:.4f}"
    )
    return result


def main() -> None:
    rows = [evaluate_transfer(source, target) for source, target in DIRECTIONS]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "cross_dataset_summary.csv", index=False)


if __name__ == "__main__":
    main()
