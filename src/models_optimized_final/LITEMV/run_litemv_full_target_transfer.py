from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from litemv import (
    ARTIFACT_ROOT,
    RESULT_ROOT,
    DATASET_PATHS,
    SHARED_PARAMS,
    ORIGINAL_ROW_COL,
    SOURCE_FILE_COL,
    EVAL_SEQUENCE_STRIDE,
    build_sequences,
    calculate_metrics,
    predict_probabilities,
    predictions_from_probabilities,
    transform_and_select,
)

from preprocessing import DatasetPreprocessor


def load_model(path: Path) -> keras.Model:
    """Load locally-created LITEMV Keras model."""
    try:
        return keras.models.load_model(
            path,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        return keras.models.load_model(
            path,
            compile=False,
        )


def load_source_bundle(dataset_name: str) -> dict:
    """
    Load the FINAL trained O1 model for the source dataset.

    Nothing is retrained or recalibrated.
    """
    artifact_dir = ARTIFACT_ROOT / dataset_name

    model_path = artifact_dir / "litemv.keras"
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    feature_path = artifact_dir / "selected_features.json"
    decision_path = artifact_dir / "decision_rule.json"
    params_path = artifact_dir / "shared_hyperparameters.json"

    required = [
        model_path,
        preprocessor_path,
        feature_path,
        decision_path,
        params_path,
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing final LITEMV artifacts:\n  "
            + "\n  ".join(missing)
        )

    with feature_path.open("r", encoding="utf-8") as handle:
        selected_features = json.load(handle)

    with decision_path.open("r", encoding="utf-8") as handle:
        decision_rule = json.load(handle)

    with params_path.open("r", encoding="utf-8") as handle:
        stored_params = json.load(handle)

    return {
        "model": load_model(model_path),
        "preprocessor": DatasetPreprocessor.load(preprocessor_path),
        "selected_features": selected_features,
        "threshold": float(decision_rule["threshold"]),
        "threshold_method": decision_rule["method"],
        "params": stored_params,
    }


def evaluate_full_target(
    source_dataset: str,
    target_dataset: str,
) -> dict:

    print("\n" + "=" * 100)
    print(
        f"LITEMV FINAL O1 FULL-TARGET TRANSFER: "
        f"{source_dataset.upper()} -> {target_dataset.upper()}"
    )
    print("=" * 100)

    # -----------------------------------------------------------------
    # 1. Load frozen SOURCE model / preprocessor / features / threshold
    # -----------------------------------------------------------------

    bundle = load_source_bundle(source_dataset)

    print(
        f"Frozen source threshold: "
        f"{bundle['threshold']:.8f} "
        f"({bundle['threshold_method']})"
    )
    print(
        f"Frozen source features: "
        f"{len(bundle['selected_features'])}"
    )

    # -----------------------------------------------------------------
    # 2. Load EVERY row from the target dataset
    # -----------------------------------------------------------------

    target_path = DATASET_PATHS[target_dataset]

    print(f"Loading complete target dataset: {target_path}")

    target_raw = pd.read_csv(
        target_path,
        low_memory=False,
    )

    target_raw = target_raw.drop(
        columns=[ORIGINAL_ROW_COL],
        errors="ignore",
    )

    print(f"Full target rows: {len(target_raw):,}")
    print(
        f"Full target SourceFiles: "
        f"{target_raw[SOURCE_FILE_COL].nunique():,}"
    )

    # -----------------------------------------------------------------
    # 3. Apply SOURCE preprocessing.
    #
    # IMPORTANT:
    #   preprocessor.fit() is NEVER called on target data.
    # -----------------------------------------------------------------

    (
        selected_target,
        actual_features,
        _dropped,
        _manifest,
    ) = transform_and_select(
        bundle["preprocessor"],
        target_raw,
        dataset_name=(
            f"{source_dataset}_to_"
            f"{target_dataset}_full_target"
        ),
        expected_features=bundle["selected_features"],
    )

    if actual_features != bundle["selected_features"]:
        raise RuntimeError(
            "Target feature schema/order differs from "
            "the frozen source feature schema."
        )

    del target_raw

    # -----------------------------------------------------------------
    # 4. Evaluate SourceFile-by-SourceFile.
    #
    # This keeps chronological context inside the same SourceFile,
    # never crosses SourceFile boundaries, and avoids constructing
    # one enormous sequence tensor for the complete dataset.
    # -----------------------------------------------------------------

    all_y = []
    all_probabilities = []
    all_metadata = []

    source_files = list(
        selected_target.groupby(
            SOURCE_FILE_COL,
            sort=True,
        )
    )

    total_sources = len(source_files)

    for index, (source_file, source_df) in enumerate(
        source_files,
        start=1,
    ):
        print(
            f"[{index}/{total_sources}] "
            f"{source_file}: {len(source_df):,} flows"
        )

        X, y, metadata = build_sequences(
            selected_df=source_df,
            selected_features=bundle["selected_features"],
            sequence_length=int(
                bundle["params"]["sequence_length"]
            ),
            stride=EVAL_SEQUENCE_STRIDE,
        )

        probabilities = predict_probabilities(
            bundle["model"],
            X,
            batch_size=int(
                bundle["params"]["batch_size"]
            ),
        )

        all_y.append(y)
        all_probabilities.append(probabilities)
        all_metadata.append(metadata)

        del X

    # -----------------------------------------------------------------
    # 5. Combine full-target predictions
    # -----------------------------------------------------------------

    y_true = np.concatenate(all_y)
    probabilities = np.concatenate(all_probabilities)

    metadata = pd.concat(
        all_metadata,
        ignore_index=True,
    )

    if len(y_true) != len(selected_target):
        raise RuntimeError(
            f"Expected one prediction per full-target flow: "
            f"{len(selected_target):,}; "
            f"got {len(y_true):,}."
        )

    predictions = predictions_from_probabilities(
        probabilities,
        bundle["threshold"],
    )

    metrics = calculate_metrics(
        y_true,
        probabilities,
        bundle["threshold"],
    )

    # -----------------------------------------------------------------
    # 6. Save independently from ordinary held-out test results
    # -----------------------------------------------------------------

    direction = f"{source_dataset}_to_{target_dataset}"

    output_dir = (
        RESULT_ROOT
        / "cross_dataset_full_target"
        / direction
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_table = metadata.copy()

    prediction_table["MaliciousProbability"] = probabilities
    prediction_table["Predicted_Label"] = predictions
    prediction_table["SourceValidationThreshold"] = (
        bundle["threshold"]
    )
    prediction_table["SourceDataset"] = source_dataset
    prediction_table["TargetDataset"] = target_dataset
    prediction_table["EvaluationPopulation"] = "full_target"

    prediction_table.to_csv(
        output_dir / "full_target_predictions.csv",
        index=False,
    )

    result = {
        "SourceDataset": source_dataset,
        "TargetDataset": target_dataset,
        "EvaluationPopulation": "full_target",
        "Rows": int(len(y_true)),
        "SourceFiles": int(
            metadata[SOURCE_FILE_COL].nunique()
        ),
        "FeatureCount": int(
            len(bundle["selected_features"])
        ),
        "ThresholdMethod": bundle["threshold_method"],
        **metrics,
    }

    pd.DataFrame([result]).to_csv(
        output_dir / "full_target_metrics.csv",
        index=False,
    )

    print("\nRESULT")
    print("-" * 70)
    print(f"Rows      : {len(y_true):,}")
    print(f"Accuracy  : {metrics['accuracy']:.6f}")
    print(f"Precision : {metrics['precision']:.6f}")
    print(f"Recall    : {metrics['recall_tpr']:.6f}")
    print(f"FPR       : {metrics['fpr']:.6f}")
    print(f"F1        : {metrics['f1']:.6f}")
    print(f"F2        : {metrics['f2']:.6f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.6f}")
    print(f"PR-AUC    : {metrics['pr_auc']:.6f}")
    print(f"MCC       : {metrics['mcc']:.6f}")
    print(
        f"TN={metrics['tn']:,} | "
        f"FP={metrics['fp']:,} | "
        f"FN={metrics['fn']:,} | "
        f"TP={metrics['tp']:,}"
    )

    tf.keras.backend.clear_session()

    return result


def main():
    results = []

    # D2-trained final LITEMV -> every D3 flow
    results.append(
        evaluate_full_target(
            source_dataset="dataset2",
            target_dataset="dataset3",
        )
    )

    # D3-trained final LITEMV -> every D2 flow
    results.append(
        evaluate_full_target(
            source_dataset="dataset3",
            target_dataset="dataset2",
        )
    )

    summary = pd.DataFrame(results)

    output_root = (
        RESULT_ROOT
        / "cross_dataset_full_target"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_root
        / "full_target_transfer_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 100)
    print("FINAL LITEMV O1 FULL-TARGET CROSS-DATASET RESULTS")
    print("=" * 100)

    print(
        summary[
            [
                "SourceDataset",
                "TargetDataset",
                "Rows",
                "accuracy",
                "precision",
                "recall_tpr",
                "fpr",
                "f1",
                "f2",
                "roc_auc",
                "pr_auc",
                "fp",
                "fn",
                "tp",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved summary:\n{summary_path}")


if __name__ == "__main__":
    main()