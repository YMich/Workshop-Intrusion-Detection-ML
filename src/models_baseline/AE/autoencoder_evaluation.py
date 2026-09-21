from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

MODEL_DIR = Path(__file__).resolve().parent
SRC_ROOT = Path(__file__).resolve().parents[2]

for path in [
    MODEL_DIR,
    SRC_ROOT / "preprocessing",
    SRC_ROOT / "feature_engineering",
]:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from autoencoder import (  # noqa: E402
    ARTIFACT_ROOT,
    BENCHMARK_DATASETS,
    DATASET_PATHS,
    LABEL_COL,
    RESULT_ROOT,
    SHARED_CONFIG_PATH,
    SOURCE_FILE_COL,
    calculate_metrics,
    reconstruct_matrix,
    score_from_reconstruction,
)
from preprocessing import DatasetPreprocessor  # noqa: E402
from feature_selection import select_and_engineer_dataset  # noqa: E402
from source_aware_split import (  # noqa: E402
    ORIGINAL_ROW_COL,
    apply_split_manifest,
)


def save_plots(
    y_true,
    scores,
    predictions,
    threshold,
    score_mode,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malicious"],
    ).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Autoencoder - Shared Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        scores,
        ax=ax,
        name="Autoencoder",
    )
    ax.set_title("Autoencoder - Shared Test ROC")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        scores,
        ax=ax,
        name="Autoencoder",
    )
    ax.set_title("Autoencoder - Shared Test Precision-Recall")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=300)
    plt.close(fig)

    score_table = pd.DataFrame(
        {
            "Label": np.asarray(y_true, dtype=int),
            "AnomalyScore": np.asarray(scores, dtype=float),
        }
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        score_table.loc[score_table["Label"] == 0, "AnomalyScore"],
        bins=50,
        alpha=0.6,
        label="Benign",
    )
    ax.hist(
        score_table.loc[score_table["Label"] == 1, "AnomalyScore"],
        bins=50,
        alpha=0.6,
        label="Malicious",
    )
    ax.axvline(
        threshold,
        linestyle="--",
        label="Validation-calibrated threshold",
    )
    ax.set_xlabel(f"Anomaly Score ({score_mode})")
    ax.set_ylabel("Flow Count")
    ax.set_title("Autoencoder - Shared Test Anomaly Scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "anomaly_score_distribution.png", dpi=300)
    plt.close(fig)


def evaluate_dataset(dataset_name: str) -> dict:
    print("\n" + "=" * 90)
    print(f"AUTOENCODER SHARED-SPLIT EVALUATION - {dataset_name.upper()}")
    print("=" * 90)

    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name

    required = [
        artifact_dir / "autoencoder.keras",
        artifact_dir / "preprocessor.joblib",
        artifact_dir / "split_manifest.csv",
        artifact_dir / "selected_features.csv",
        artifact_dir / "model_config.json",
        artifact_dir / "decision_rule.json",
        SHARED_CONFIG_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation artifact: {path}")

    model = tf.keras.models.load_model(artifact_dir / "autoencoder.keras")
    preprocessor = DatasetPreprocessor.load(artifact_dir / "preprocessor.joblib")
    split_manifest = pd.read_csv(artifact_dir / "split_manifest.csv")
    selected_features = pd.read_csv(
        artifact_dir / "selected_features.csv"
    )["Feature"].tolist()

    with (artifact_dir / "model_config.json").open(
        "r", encoding="utf-8"
    ) as handle:
        local_params = json.load(handle)

    with SHARED_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        shared_config = json.load(handle)

    if local_params != shared_config.get("parameters"):
        raise RuntimeError(
            f"{dataset_name}: saved Autoencoder model configuration do not match "
            "the shared D2/D3 configuration."
        )

    with (artifact_dir / "decision_rule.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        decision_rule = json.load(handle)

    threshold = float(decision_rule["threshold"])
    score_mode = str(decision_rule["score_mode"])
    if score_mode != str(shared_config.get("score_mode")):
        raise RuntimeError(
            f"{dataset_name}: saved anomaly score mode does not match the "
            "shared D2/D3 configuration."
        )

    batch_size = int(local_params["batch_size"])

    df = pd.read_csv(DATASET_PATHS[dataset_name], low_memory=False)
    splits = apply_split_manifest(df, split_manifest)
    test_df = splits["test"].drop(
        columns=[ORIGINAL_ROW_COL],
        errors="ignore",
    )

    normalized_test = preprocessor.transform(test_df)

    (
        selected_test,
        actual_features,
        _dropped,
        _manifest,
    ) = select_and_engineer_dataset(
        dataset_name=f"{dataset_name}_autoencoder_shared_test",
        df=normalized_test,
        expected_selected_features=selected_features,
    )

    if actual_features != selected_features:
        raise ValueError(
            f"{dataset_name}: test feature schema differs from training."
        )

    X_test = selected_test[selected_features].to_numpy(dtype=np.float32)
    y_test = selected_test[LABEL_COL].astype(int).to_numpy()

    reconstruction = reconstruct_matrix(
        model,
        X_test,
        batch_size=batch_size,
    )
    scores = score_from_reconstruction(
        X_test,
        reconstruction,
        score_mode,
    )
    predictions = (scores > threshold).astype(int)

    metrics = calculate_metrics(y_test, scores, threshold)

    result = {
        "Dataset": dataset_name,
        "HyperparameterScope": "shared_dataset2_dataset3",
        "SharedHyperparametersVerified": True,
        "TestRows": int(len(y_test)),
        "TestSourceFiles": int(test_df[SOURCE_FILE_COL].nunique()),
        "score_mode": score_mode,
        "threshold_method": decision_rule["threshold_method"],
        **metrics,
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(
        result_dir / "test_metrics.csv",
        index=False,
    )

    with (result_dir / "test_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)

    prediction_table = selected_test[[SOURCE_FILE_COL, LABEL_COL]].copy()
    prediction_table["Anomaly_Score"] = scores
    prediction_table["Threshold"] = threshold
    prediction_table["Score_Mode"] = score_mode
    prediction_table["Predicted_Label"] = predictions
    prediction_table.to_csv(
        result_dir / "test_predictions.csv",
        index=False,
    )

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual_Benign", "Actual_Malicious"],
        columns=["Predicted_Benign", "Predicted_Malicious"],
    ).to_csv(result_dir / "confusion_matrix.csv")

    save_plots(
        y_true=y_test,
        scores=scores,
        predictions=predictions,
        threshold=threshold,
        score_mode=score_mode,
        output_dir=result_dir,
    )

    print(f"Test rows: {len(y_test):,}")
    print(f"Recall={metrics['recall_tpr']:.4f}")
    print(f"FPR={metrics['fpr']:.4f}")
    print(f"F1={metrics['f1']:.4f}")
    print(f"PR-AUC={metrics['pr_auc']:.4f}")

    return result


def evaluate_all_datasets(
    only_dataset: str | None = None,
) -> pd.DataFrame:
    if only_dataset is not None and only_dataset not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark dataset: {only_dataset}")

    dataset_names = (
        [only_dataset]
        if only_dataset is not None
        else list(BENCHMARK_DATASETS)
    )

    summary = pd.DataFrame(
        [evaluate_dataset(dataset_name) for dataset_name in dataset_names]
    )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        RESULT_ROOT / "autoencoder_shared_test_summary.csv",
        index=False,
    )
    return summary
