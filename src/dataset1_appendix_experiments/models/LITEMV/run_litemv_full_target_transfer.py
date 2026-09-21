from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from litemv import (
    ORIGINAL_ROW_COL,
    get_or_create_shared_split,
    PROJECT_ROOT,
    SOURCE_FILE_COL,
    EVAL_SEQUENCE_STRIDE,
    build_sequences,
    calculate_metrics,
    predict_probabilities,
    predictions_from_probabilities,
    transform_and_select,
)
from preprocessing import DatasetPreprocessor

APPENDIX_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset1_appendix" / "models" / "LITEMV"
FINAL_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models_optimized_final" / "LITEMV"
RESULT_ROOT = PROJECT_ROOT / "results" / "dataset1_appendix" / "cross_dataset" / "LITEMV"
INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"
DATASET_PATHS = {name: INGESTED_ROOT / f"{name}_ingested.csv" for name in ("dataset1", "dataset2", "dataset3")}
DIRECTIONS = tuple((s, t) for s in DATASET_PATHS for t in DATASET_PATHS if s != t)


def _load_model(path: Path) -> keras.Model:
    try:
        return keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return keras.models.load_model(path, compile=False)


def source_artifact_dir(dataset_name: str) -> Path:
    root = APPENDIX_ARTIFACT_ROOT if dataset_name == "dataset1" else FINAL_ARTIFACT_ROOT
    return root / dataset_name




def _load_or_rebuild_source_preprocessor(dataset_name: str, artifact_dir: Path) -> DatasetPreprocessor:
    """Load source preprocessing; rebuild deterministically from SOURCE TRAIN if old pandas pickle is incompatible."""
    path = artifact_dir / "preprocessor.joblib"
    try:
        return DatasetPreprocessor.load(path)
    except (NotImplementedError, TypeError, ValueError, AttributeError) as exc:
        print(f"[PREPROCESSOR FALLBACK] Could not load {path} ({type(exc).__name__}: {exc}).")
        print(f"[PREPROCESSOR FALLBACK] Rebuilding {dataset_name} preprocessing from persisted SOURCE TRAIN only.")
        raw = pd.read_csv(DATASET_PATHS[dataset_name], low_memory=False)
        split_root = (
            PROJECT_ROOT / "artifacts" / "dataset1_appendix" / "splits"
            if dataset_name == "dataset1"
            else PROJECT_ROOT / "artifacts" / "splits"
        )
        splits, _manifest, _metadata = get_or_create_shared_split(
            dataset_name=dataset_name, df=raw, split_root=split_root, force_rebuild=False
        )
        train_df = splits["train"].drop(columns=[ORIGINAL_ROW_COL], errors="ignore").copy()
        preprocessor = DatasetPreprocessor(f"{dataset_name}_litemv_cross_rebuilt")
        preprocessor.fit(train_df)
        return preprocessor


def load_source_bundle(dataset_name: str) -> dict:
    artifact_dir = source_artifact_dir(dataset_name)
    required = [
        artifact_dir / "litemv.keras",
        artifact_dir / "preprocessor.joblib",
        artifact_dir / "selected_features.json",
        artifact_dir / "decision_rule.json",
        artifact_dir / "shared_hyperparameters.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing LITEMV source artifacts:\n  " + "\n  ".join(missing))

    with (artifact_dir / "selected_features.json").open("r", encoding="utf-8") as handle:
        selected_features = json.load(handle)
    with (artifact_dir / "decision_rule.json").open("r", encoding="utf-8") as handle:
        decision_rule = json.load(handle)
    with (artifact_dir / "shared_hyperparameters.json").open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    return {
        "model": _load_model(artifact_dir / "litemv.keras"),
        "preprocessor": _load_or_rebuild_source_preprocessor(dataset_name, artifact_dir),
        "selected_features": selected_features,
        "threshold": float(decision_rule["threshold"]),
        "threshold_method": decision_rule["method"],
        "params": params,
    }


def evaluate_full_target(source_dataset: str, target_dataset: str) -> dict:
    print("\n" + "=" * 100)
    print(f"LITEMV FULL-TARGET TRANSFER: {source_dataset.upper()} -> {target_dataset.upper()}")
    print("=" * 100)

    bundle = load_source_bundle(source_dataset)
    target_raw = pd.read_csv(DATASET_PATHS[target_dataset], low_memory=False)
    target_raw = target_raw.drop(columns=[ORIGINAL_ROW_COL], errors="ignore")

    selected_target, actual_features, _dropped, _manifest = transform_and_select(
        bundle["preprocessor"],
        target_raw,
        dataset_name=f"{source_dataset}_to_{target_dataset}_full_target",
        expected_features=bundle["selected_features"],
    )
    if actual_features != bundle["selected_features"]:
        raise RuntimeError("Target feature schema/order differs from frozen source schema.")

    all_y, all_probabilities, all_metadata = [], [], []
    groups = list(selected_target.groupby(SOURCE_FILE_COL, sort=True))
    for index, (source_file, source_df) in enumerate(groups, start=1):
        print(f"[{index}/{len(groups)}] {source_file}: {len(source_df):,} flows")
        X, y, metadata = build_sequences(
            selected_df=source_df,
            selected_features=bundle["selected_features"],
            sequence_length=int(bundle["params"]["sequence_length"]),
            stride=EVAL_SEQUENCE_STRIDE,
        )
        probabilities = predict_probabilities(
            bundle["model"], X, batch_size=int(bundle["params"]["batch_size"])
        )
        all_y.append(y)
        all_probabilities.append(probabilities)
        all_metadata.append(metadata)
        del X

    y_true = np.concatenate(all_y)
    probabilities = np.concatenate(all_probabilities)
    metadata = pd.concat(all_metadata, ignore_index=True)
    if len(y_true) != len(selected_target):
        raise RuntimeError(
            f"Expected one prediction per target flow ({len(selected_target):,}); got {len(y_true):,}."
        )

    predictions = predictions_from_probabilities(probabilities, bundle["threshold"])
    metrics = calculate_metrics(y_true, probabilities, bundle["threshold"])
    direction = f"{source_dataset}_to_{target_dataset}"
    output_dir = RESULT_ROOT / direction
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_table = metadata.copy()
    prediction_table["MaliciousProbability"] = probabilities
    prediction_table["Predicted_Label"] = predictions
    prediction_table["SourceValidationThreshold"] = bundle["threshold"]
    prediction_table["SourceDataset"] = source_dataset
    prediction_table["TargetDataset"] = target_dataset
    prediction_table["EvaluationPopulation"] = "full_target"
    prediction_table.to_csv(output_dir / "full_target_predictions.csv", index=False)

    result = {
        "Model": "LITEMV",
        "SourceDataset": source_dataset,
        "TargetDataset": target_dataset,
        "EvaluationPopulation": "full_target",
        "Rows": int(len(y_true)),
        "SourceFiles": int(metadata[SOURCE_FILE_COL].nunique()),
        "FeatureCount": int(len(bundle["selected_features"])),
        "ThresholdMethod": bundle["threshold_method"],
        "TargetUsedForTraining": False,
        "TargetUsedForPreprocessingFit": False,
        "TargetUsedForThresholdCalibration": False,
        **metrics,
    }
    pd.DataFrame([result]).to_csv(output_dir / "full_target_metrics.csv", index=False)
    print(
        f"Recall={metrics['recall_tpr']:.4f}, FPR={metrics['fpr']:.4f}, "
        f"F1={metrics['f1']:.4f}, PR-AUC={metrics['pr_auc']:.4f}"
    )
    tf.keras.backend.clear_session()
    return result


def main() -> None:
    rows = [evaluate_full_target(source, target) for source, target in DIRECTIONS]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "full_target_transfer_summary.csv", index=False)


if __name__ == "__main__":
    main()
