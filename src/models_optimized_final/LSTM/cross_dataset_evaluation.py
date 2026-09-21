from __future__ import annotations

import json

import numpy as np
import pandas as pd
from tensorflow import keras

from lstm import (
    ARTIFACT_ROOT,
    DATASET_PATHS,
    EVAL_SEQUENCE_STRIDE,
    LABEL_COL,
    PROJECT_ROOT,
    SOURCE_FILE_COL,
    DatasetPreprocessor,
    build_sequences,
    calculate_metrics,
    predict_probabilities,
    predictions_from_probabilities,
    transform_and_select,
)

CROSS_RESULT_ROOT = PROJECT_ROOT / "results" / "cross_dataset_evaluation" / "LSTM"
DIRECTIONS = (("dataset2", "dataset3"), ("dataset3", "dataset2"))


def _source_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_file, group in predictions.groupby(SOURCE_FILE_COL, dropna=False, sort=True):
        y = group[LABEL_COL].astype(int).to_numpy()
        p = group["Predicted_Label"].astype(int).to_numpy()
        rows.append({
            SOURCE_FILE_COL: source_file,
            "Rows": int(len(group)),
            "ActualLabel": int(y[0]) if len(set(y)) == 1 else -1,
            "PredictedMalicious": int((p == 1).sum()),
            "Errors": int((p != y).sum()),
            "MeanMaliciousProbability": float(group["MaliciousProbability"].mean()),
            "MedianMaliciousProbability": float(group["MaliciousProbability"].median()),
        })
    return pd.DataFrame(rows)


def evaluate_transfer(source_dataset: str, target_dataset: str) -> dict:
    source_artifact = ARTIFACT_ROOT / source_dataset
    output_dir = CROSS_RESULT_ROOT / f"{source_dataset}_to_{target_dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(source_artifact / "lstm_model.keras")
    preprocessor = DatasetPreprocessor.load(source_artifact / "preprocessor.joblib")
    selected_features = pd.read_csv(source_artifact / "selected_features.csv")["Feature"].tolist()

    with (source_artifact / "best_hyperparameters.json").open("r", encoding="utf-8") as handle:
        params = json.load(handle)
    with (source_artifact / "decision_rule.json").open("r", encoding="utf-8") as handle:
        decision_rule = json.load(handle)

    threshold = float(decision_rule["threshold"])
    sequence_length = int(params["sequence_length"])
    batch_size = int(params["batch_size"])

    target_df = pd.read_csv(DATASET_PATHS[target_dataset], low_memory=False)
    selected_target, actual_features = transform_and_select(
        preprocessor=preprocessor,
        df=target_df,
        dataset_name=f"cross_{source_dataset}_to_{target_dataset}_lstm",
        expected_features=selected_features,
    )
    if actual_features != selected_features:
        raise RuntimeError("Cross-dataset LSTM feature schema mismatch.")

    X_target, y_target, metadata = build_sequences(
        selected_target,
        selected_features,
        sequence_length,
        EVAL_SEQUENCE_STRIDE,
    )
    if len(y_target) != len(target_df):
        raise RuntimeError(
            f"Expected one LSTM decision per target flow ({len(target_df)}), got {len(y_target)}."
        )

    probabilities = predict_probabilities(model, X_target, batch_size)
    predictions = predictions_from_probabilities(probabilities, threshold)
    metrics = calculate_metrics(y_target, probabilities, threshold)

    result = {
        "Model": "LSTM",
        "SourceDataset": source_dataset,
        "TargetDataset": target_dataset,
        "EvaluationRows": int(len(y_target)),
        "EvaluationSourceFiles": int(target_df[SOURCE_FILE_COL].nunique()),
        "TargetUsedForTraining": False,
        "TargetUsedForPreprocessingFit": False,
        "TargetUsedForThresholdCalibration": False,
        "ThresholdOrigin": source_dataset,
        "PreprocessingOrigin": source_dataset,
        "ModelOrigin": source_dataset,
        "SequenceLength": sequence_length,
        **metrics,
    }

    pd.DataFrame([result]).to_csv(output_dir / "cross_dataset_metrics.csv", index=False)
    prediction_table = metadata.copy()
    prediction_table["MaliciousProbability"] = probabilities
    prediction_table["SourceThreshold"] = threshold
    prediction_table["Predicted_Label"] = predictions
    prediction_table.to_csv(output_dir / "cross_dataset_predictions.csv", index=False)
    _source_summary(prediction_table).to_csv(output_dir / "cross_dataset_by_source.csv", index=False)

    print(f"LSTM {source_dataset} -> {target_dataset}: "
          f"Recall={metrics['recall_tpr']:.4f}, FPR={metrics['fpr']:.4f}, "
          f"F1={metrics['f1']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")
    return result


def main() -> None:
    rows = [evaluate_transfer(source, target) for source, target in DIRECTIONS]
    CROSS_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CROSS_RESULT_ROOT / "cross_dataset_summary.csv", index=False)


if __name__ == "__main__":
    main()
