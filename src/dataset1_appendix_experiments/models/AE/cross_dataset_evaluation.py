from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from autoencoder import (
    LABEL_COL,
    PROJECT_ROOT,
    SOURCE_FILE_COL,
    DatasetPreprocessor,
    ORIGINAL_ROW_COL,
    get_or_create_shared_split,
    calculate_metrics,
    reconstruct_matrix,
    score_from_reconstruction,
    transform_and_select,
)

APPENDIX_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset1_appendix" / "models" / "AE"
FINAL_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models_optimized_final" / "AE"
CROSS_RESULT_ROOT = PROJECT_ROOT / "results" / "dataset1_appendix" / "cross_dataset" / "AE"
INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"
DATASET_PATHS = {name: INGESTED_ROOT / f"{name}_ingested.csv" for name in ("dataset1", "dataset2", "dataset3")}
DIRECTIONS = tuple((s, t) for s in DATASET_PATHS for t in DATASET_PATHS if s != t)


def source_artifact_root(dataset_name: str) -> Path:
    return (APPENDIX_ARTIFACT_ROOT if dataset_name == "dataset1" else FINAL_ARTIFACT_ROOT) / dataset_name




def _load_or_rebuild_source_preprocessor(source_dataset: str, source_artifact: Path) -> DatasetPreprocessor:
    """AE preprocessing was fitted on benign SOURCE TRAIN only; reproduce that exactly on pickle mismatch."""
    path = source_artifact / "preprocessor.joblib"
    try:
        return DatasetPreprocessor.load(path)
    except (NotImplementedError, TypeError, ValueError, AttributeError) as exc:
        print(f"[PREPROCESSOR FALLBACK] Could not load {path} ({type(exc).__name__}: {exc}).")
        print(f"[PREPROCESSOR FALLBACK] Rebuilding {source_dataset} AE preprocessing from benign SOURCE TRAIN only.")
        raw = pd.read_csv(DATASET_PATHS[source_dataset], low_memory=False)
        split_root = (
            PROJECT_ROOT / "artifacts" / "dataset1_appendix" / "splits"
            if source_dataset == "dataset1"
            else PROJECT_ROOT / "artifacts" / "splits"
        )
        splits, _manifest, _metadata = get_or_create_shared_split(
            dataset_name=source_dataset, df=raw, split_root=split_root, force_rebuild=False
        )
        train_df = splits["train"].drop(columns=[ORIGINAL_ROW_COL], errors="ignore").copy()
        benign_train = train_df[train_df[LABEL_COL].astype(int) == 0].copy()
        if benign_train.empty:
            raise RuntimeError(f"{source_dataset}: no benign source-training rows available for AE preprocessing rebuild.")
        preprocessor = DatasetPreprocessor(f"{source_dataset}_ae_cross_rebuilt")
        preprocessor.fit(benign_train)
        return preprocessor


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
            "MeanAnomalyScore": float(group["AnomalyScore"].mean()),
            "MedianAnomalyScore": float(group["AnomalyScore"].median()),
        })
    return pd.DataFrame(rows)


def evaluate_transfer(source_dataset: str, target_dataset: str) -> dict:
    source_artifact = source_artifact_root(source_dataset)
    output_dir = CROSS_RESULT_ROOT / f"{source_dataset}_to_{target_dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(source_artifact / "autoencoder.keras")
    preprocessor = _load_or_rebuild_source_preprocessor(source_dataset, source_artifact)
    selected_features = pd.read_csv(source_artifact / "selected_features.csv")["Feature"].tolist()
    with (source_artifact / "best_hyperparameters.json").open("r", encoding="utf-8") as handle:
        params = json.load(handle)
    with (source_artifact / "decision_rule.json").open("r", encoding="utf-8") as handle:
        decision_rule = json.load(handle)

    threshold = float(decision_rule["threshold"])
    score_mode = str(decision_rule["score_mode"])
    target_df = pd.read_csv(DATASET_PATHS[target_dataset], low_memory=False)
    selected_target, X_target, y_target, actual_features, _ = transform_and_select(
        preprocessor=preprocessor,
        df=target_df,
        dataset_name=f"cross_{source_dataset}_to_{target_dataset}_ae",
        expected_features=selected_features,
    )
    if actual_features != selected_features:
        raise RuntimeError("Cross-dataset AE feature schema mismatch.")

    reconstructed = reconstruct_matrix(model, X_target, batch_size=int(params["batch_size"]))
    scores = score_from_reconstruction(X_target, reconstructed, score_mode)
    predictions = (scores > threshold).astype(int)
    metrics = calculate_metrics(y_target, scores, threshold)
    result = {
        "Model": "Autoencoder",
        "SourceDataset": source_dataset,
        "TargetDataset": target_dataset,
        "EvaluationPopulation": "full_target",
        "EvaluationRows": int(len(y_target)),
        "EvaluationSourceFiles": int(target_df[SOURCE_FILE_COL].nunique()),
        "TargetUsedForTraining": False,
        "TargetUsedForPreprocessingFit": False,
        "TargetUsedForThresholdCalibration": False,
        "ThresholdOrigin": source_dataset,
        "PreprocessingOrigin": source_dataset,
        "ModelOrigin": source_dataset,
        "score_mode": score_mode,
        **metrics,
    }
    pd.DataFrame([result]).to_csv(output_dir / "cross_dataset_metrics.csv", index=False)
    keep = [c for c in [SOURCE_FILE_COL, "Timestamp", LABEL_COL] if c in selected_target.columns]
    prediction_table = selected_target[keep].copy()
    prediction_table["AnomalyScore"] = scores
    prediction_table["SourceThreshold"] = threshold
    prediction_table["Predicted_Label"] = predictions
    prediction_table.to_csv(output_dir / "cross_dataset_predictions.csv", index=False)
    _source_summary(prediction_table).to_csv(output_dir / "cross_dataset_by_source.csv", index=False)
    print(
        f"AE {source_dataset} -> {target_dataset}: "
        f"Recall={metrics['recall_tpr']:.4f}, FPR={metrics['fpr']:.4f}, "
        f"F1={metrics['f1']:.4f}, PR-AUC={metrics['pr_auc']:.4f}"
    )
    return result


def main() -> None:
    rows = [evaluate_transfer(source, target) for source, target in DIRECTIONS]
    CROSS_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CROSS_RESULT_ROOT / "cross_dataset_summary.csv", index=False)


if __name__ == "__main__":
    main()
