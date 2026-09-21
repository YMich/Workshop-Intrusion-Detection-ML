from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from tensorflow import keras

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from litemv import (  # noqa: E402
    ARTIFACT_ROOT,
    BENCHMARK_DATASETS,
    DATASET_PATHS,
    EVAL_SEQUENCE_STRIDE,
    LABEL_COL,
    RESULT_ROOT,
    SHARED_PARAMS,
    SOURCE_FILE_COL,
    build_sequences,
    calculate_metrics,
    load_dataset_and_split,
    predict_probabilities,
    predictions_from_probabilities,
    transform_and_select,
)
from preprocessing import DatasetPreprocessor  # noqa: E402


def _load_model(path: Path) -> keras.Model:
    # Some versions of the aeon LITE network may serialize Lambda-based custom
    # filters. safe_mode=False permits loading our own locally-created artifact.
    try:
        return keras.models.load_model(
            path,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        # Compatibility with TensorFlow/Keras versions that predate safe_mode.
        return keras.models.load_model(
            path,
            compile=False,
        )


def save_test_plots(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malicious"],
    ).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("LITEMV Final O1 - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
        name="LITEMV Final O1",
    )
    ax.set_title("LITEMV Final O1 - Test ROC")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
        name="LITEMV Final O1",
    )
    ax.set_title("LITEMV Final O1 - Test Precision-Recall")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=300)
    plt.close(fig)

    probability_table = pd.DataFrame(
        {
            "Label": np.asarray(y_true, dtype=int),
            "MaliciousProbability": np.asarray(probabilities, dtype=float),
        }
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        probability_table.loc[
            probability_table["Label"] == 0,
            "MaliciousProbability",
        ],
        bins=50,
        alpha=0.6,
        label="Benign",
    )
    ax.hist(
        probability_table.loc[
            probability_table["Label"] == 1,
            "MaliciousProbability",
        ],
        bins=50,
        alpha=0.6,
        label="Malicious",
    )
    ax.axvline(
        threshold,
        linestyle="--",
        label=f"Validation threshold={threshold:.4f}",
    )
    ax.set_xlabel("Predicted Malicious Probability")
    ax.set_ylabel("Count")
    ax.set_title("LITEMV Final O1 - Test Probability Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "probability_distribution.png", dpi=300)
    plt.close(fig)


def evaluate_dataset(dataset_name: str) -> dict:
    if dataset_name not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark dataset: {dataset_name}")

    print("\n" + "=" * 100)
    print(f"LITEMV FINAL TEST - {dataset_name.upper()}")
    print("=" * 100)

    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name

    model_path = artifact_dir / "litemv.keras"
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    feature_path = artifact_dir / "selected_features.json"
    decision_path = artifact_dir / "decision_rule.json"
    params_path = artifact_dir / "shared_hyperparameters.json"

    required_paths = [
        model_path,
        preprocessor_path,
        feature_path,
        decision_path,
        params_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trained LITEMV artifacts:\n  " + "\n  ".join(missing)
        )

    with feature_path.open("r", encoding="utf-8") as handle:
        selected_features = json.load(handle)

    with decision_path.open("r", encoding="utf-8") as handle:
        decision_rule = json.load(handle)

    with params_path.open("r", encoding="utf-8") as handle:
        stored_params = json.load(handle)

    # Guard against evaluating an artifact trained with a different architecture.
    expected_params = json.loads(json.dumps(SHARED_PARAMS))
    if stored_params != expected_params:
        raise RuntimeError(
            f"{dataset_name}: stored LITEMV hyperparameters do not match "
            "the current shared configuration."
        )

    threshold = float(decision_rule["threshold"])
    sequence_length = int(stored_params["sequence_length"])
    batch_size = int(stored_params["batch_size"])

    model = _load_model(model_path)
    preprocessor = DatasetPreprocessor.load(preprocessor_path)

    # Reuse the exact same persisted shared split; never resplit at evaluation.
    splits, _manifest, split_metadata = load_dataset_and_split(
        dataset_name,
        force_resplit=False,
    )
    test_df = splits["test"]

    (
        selected_test,
        actual_features,
        _dropped,
        _manifest,
    ) = transform_and_select(
        preprocessor,
        test_df,
        dataset_name=f"{dataset_name}_litemv_test",
        expected_features=selected_features,
    )

    if actual_features != selected_features:
        raise ValueError(
            f"{dataset_name}: test feature schema differs from training."
        )

    X_test, y_test, metadata = build_sequences(
        selected_test,
        selected_features,
        sequence_length,
        EVAL_SEQUENCE_STRIDE,
    )

    if len(y_test) != len(test_df):
        raise RuntimeError(
            f"{dataset_name}: expected one prediction per test flow "
            f"({len(test_df):,}); got {len(y_test):,}."
        )

    probabilities = predict_probabilities(
        model,
        X_test,
        batch_size=batch_size,
    )
    predictions = predictions_from_probabilities(
        probabilities,
        threshold,
    )
    metrics = calculate_metrics(
        y_test,
        probabilities,
        threshold,
    )

    result_record = {
        "Dataset": dataset_name,
        "SplitMode": split_metadata.get("mode"),
        "TestRows": int(len(test_df)),
        "TestSourceFiles": int(test_df[SOURCE_FILE_COL].nunique()),
        "FeatureCount": int(len(selected_features)),
        "SequenceLength": int(sequence_length),
        "HyperparameterScope": "shared_dataset2_dataset3",
        "ThresholdMethod": decision_rule["method"],
        **metrics,
    }

    result_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result_record]).to_csv(
        result_dir / "test_metrics.csv",
        index=False,
    )
    with (result_dir / "test_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result_record, handle, indent=2)

    prediction_table = metadata.copy()
    prediction_table["MaliciousProbability"] = probabilities
    prediction_table["Predicted_Label"] = predictions
    prediction_table["ValidationCalibratedThreshold"] = threshold
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

    save_test_plots(
        y_true=y_test,
        probabilities=probabilities,
        predictions=predictions,
        threshold=threshold,
        output_dir=result_dir,
    )

    print(f"Test flows/sequences: {len(y_test):,}")
    print(
        f"Threshold: {threshold:.6f} "
        f"({decision_rule['method']}, validation only)"
    )
    print(f"Recall / TPR: {metrics['recall_tpr']:.4f}")
    print(f"FPR: {metrics['fpr']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"F2: {metrics['f2']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")

    return result_record


def evaluate_all_datasets(
    only_dataset: str | None = None,
) -> pd.DataFrame:
    dataset_names = (
        [only_dataset]
        if only_dataset is not None
        else list(DATASET_PATHS.keys())
    )

    rows = [evaluate_dataset(dataset_name) for dataset_name in dataset_names]

    summary = pd.DataFrame(rows)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        RESULT_ROOT / "litemv_test_summary.csv",
        index=False,
    )

    return summary
