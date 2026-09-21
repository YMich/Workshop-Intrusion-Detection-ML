from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from random_forest_config import (  # noqa: E402
    BASE_RF_PARAMS,
    BENCHMARK_DATASETS,
    MODEL_ARTIFACT_ROOT,
    PREDICTION_THRESHOLD,
    RESULT_ROOT,
    SHARED_HYPERPARAMETER_PATH,
)
from random_forest_utils import METRIC_COLUMNS, calculate_binary_metrics  # noqa: E402


def save_oof_plots(y_true, probabilities, predictions, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malicious"],
    ).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Random Forest - Group-OOF Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "oof_confusion_matrix.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax, name="Random Forest OOF")
    ax.set_title("Random Forest - Group-OOF ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "oof_roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true, probabilities, ax=ax, name="Random Forest OOF"
    )
    ax.set_title("Random Forest - Group-OOF Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "oof_precision_recall_curve.png", dpi=300)
    plt.close(fig)


def build_fold_stability_summary(outer_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(outer_metrics[metric], errors="raise")
        rows.append(
            {
                "Metric": metric,
                "Fold_Mean": float(values.mean()),
                "Fold_Std": float(values.std(ddof=0)),
                "Fold_Min": float(values.min()),
                "Fold_Max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _load_shared_configuration() -> dict:
    if not SHARED_HYPERPARAMETER_PATH.exists():
        raise FileNotFoundError(
            f"Missing shared RF configuration artifact: {SHARED_HYPERPARAMETER_PATH}"
        )
    with SHARED_HYPERPARAMETER_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("parameters") != BASE_RF_PARAMS:
        raise RuntimeError("Saved RF configuration does not match the frozen baseline.")
    return config


def verify_dataset_artifact(dataset_name: str) -> None:
    path = MODEL_ARTIFACT_ROOT / dataset_name / "model_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing RF model configuration: {path}")
    with path.open("r", encoding="utf-8") as handle:
        local = json.load(handle)
    if local.get("parameters") != BASE_RF_PARAMS:
        raise RuntimeError(
            f"{dataset_name}: saved RF model configuration differs from the frozen baseline."
        )


def evaluate_dataset(dataset_name: str) -> dict:
    print("\n" + "=" * 90)
    print(f"RANDOM FOREST GROUP-OOF EVALUATION - {dataset_name.upper()}")
    print("=" * 90)

    result_dir = RESULT_ROOT / dataset_name
    oof_path = result_dir / "oof_predictions.csv"
    outer_metrics_path = result_dir / "outer_fold_metrics.csv"
    for path in [oof_path, outer_metrics_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required OOF result not found: {path}")

    _load_shared_configuration()
    verify_dataset_artifact(dataset_name)
    oof = pd.read_csv(oof_path, low_memory=False)
    outer_metrics = pd.read_csv(outer_metrics_path, low_memory=False)

    y_true = pd.to_numeric(oof["Actual_Label"], errors="raise").astype(int).to_numpy()
    probabilities = pd.to_numeric(
        oof["Malicious_Probability"], errors="raise"
    ).to_numpy(dtype=float)
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)

    overall_metrics = calculate_binary_metrics(
        y_true,
        probabilities,
        threshold=PREDICTION_THRESHOLD,
    )
    stability = build_fold_stability_summary(outer_metrics)
    stability.to_csv(result_dir / "outer_fold_stability_summary.csv", index=False)

    metric_summary_rows = []
    for metric in METRIC_COLUMNS:
        row = stability[stability["Metric"] == metric].iloc[0]
        metric_summary_rows.append(
            {
                "Metric": metric,
                "OOF_Overall": overall_metrics[metric],
                "Outer_Fold_Mean": row["Fold_Mean"],
                "Outer_Fold_Std": row["Fold_Std"],
                "Outer_Fold_Min": row["Fold_Min"],
                "Outer_Fold_Max": row["Fold_Max"],
            }
        )
    pd.DataFrame(metric_summary_rows).to_csv(
        result_dir / "oof_metric_summary.csv", index=False
    )

    save_oof_plots(
        y_true,
        probabilities,
        predictions,
        result_dir / "plots",
    )

    return {"Dataset": dataset_name, **overall_metrics}


def evaluate_all_datasets() -> pd.DataFrame:
    rows = [evaluate_dataset(name) for name in BENCHMARK_DATASETS]
    summary = pd.DataFrame(rows)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_ROOT / "random_forest_oof_summary.csv", index=False)
    return summary
