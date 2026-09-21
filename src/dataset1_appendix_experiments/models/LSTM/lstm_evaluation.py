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
from tensorflow import keras


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from lstm import (  # noqa: E402
    ARTIFACT_ROOT,
    DATASET_PATHS,
    EVAL_SEQUENCE_STRIDE,
    LABEL_COL,
    RESULT_ROOT,
    SHARED_HYPERPARAMETER_PATH,
    SOURCE_FILE_COL,
    build_sequences,
    calculate_metrics,
    load_dataset_and_split,
    predict_probabilities,
    predictions_from_probabilities,
    transform_and_select,
)
from preprocessing import DatasetPreprocessor  # noqa: E402


# ============================================================
# PLOTS
# ============================================================


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
    ax.set_title("LSTM - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
        name="LSTM",
    )
    ax.set_title("LSTM - Test ROC")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
        name="LSTM",
    )
    ax.set_title("LSTM - Test Precision-Recall")
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
    ax.set_ylabel("Flow Count")
    ax.set_title("LSTM - Test Probability Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "probability_distribution.png", dpi=300)
    plt.close(fig)


# ============================================================
# DATASET EVALUATION
# ============================================================


def evaluate_dataset(dataset_name: str) -> dict:
    print("\n" + "=" * 90)
    print(f"LSTM FINAL EVALUATION - {dataset_name.upper()}")
    print("=" * 90)

    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name

    model_path = artifact_dir / "lstm_model.keras"
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    feature_path = artifact_dir / "selected_features.csv"
    params_path = artifact_dir / "best_hyperparameters.json"
    decision_path = artifact_dir / "decision_rule.json"

    for path in [
        model_path,
        preprocessor_path,
        feature_path,
        params_path,
        decision_path,
        SHARED_HYPERPARAMETER_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing LSTM evaluation artifact: {path}")

    model = keras.models.load_model(model_path)
    preprocessor = DatasetPreprocessor.load(preprocessor_path)
    selected_features = pd.read_csv(feature_path)["Feature"].tolist()

    with params_path.open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    with decision_path.open("r", encoding="utf-8") as handle:
        decision_rule = json.load(handle)

    with SHARED_HYPERPARAMETER_PATH.open("r", encoding="utf-8") as handle:
        shared_params = json.load(handle)

    if params != shared_params:
        raise RuntimeError(
            f"{dataset_name}: per-dataset best_hyperparameters.json does not match "
            "the shared Dataset-2/Dataset-3 hyperparameter artifact."
        )

    threshold = float(decision_rule["threshold"])
    sequence_length = int(params["sequence_length"])
    batch_size = int(params["batch_size"])

    # Reuse the exact same shared row-level split manifest produced during
    # training. No new split is generated here.
    splits, _manifest, split_metadata = load_dataset_and_split(
        dataset_name,
        force_resplit=False,
    )
    test_df = splits["test"]

    # Reuse the exact final training transformation path: training-fitted preprocessing,
    # feature selection, and the accepted brittle-feature pruning rules.
    selected_test, actual_features = transform_and_select(
        preprocessor=preprocessor,
        df=test_df,
        dataset_name=f"{dataset_name}_lstm_final_test",
        expected_features=selected_features,
    )

    if actual_features != selected_features:
        raise ValueError(
            f"{dataset_name}: final test feature schema differs from training."
        )

    X_test, y_test, metadata = build_sequences(
        selected_test,
        selected_features,
        sequence_length,
        EVAL_SEQUENCE_STRIDE,
    )

    # Because each split is sequenced independently and evaluation stride is 1,
    # every test flow must receive exactly one decision.
    if len(y_test) != len(test_df):
        raise RuntimeError(
            f"{dataset_name}: expected one prediction per test flow "
            f"({len(test_df):,}); got {len(y_test):,}."
        )

    probabilities = predict_probabilities(
        model,
        X_test,
        batch_size,
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
        "SplitMode": split_metadata["mode"],
        "TestRows": int(len(test_df)),
        "TestSourceFiles": int(test_df[SOURCE_FILE_COL].nunique()),
        "SequenceLength": int(sequence_length),
        "HyperparameterScope": "frozen_shared_D2_D3_applied_to_D1",
        "SharedHyperparametersVerified": True,
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

    # For Section 2.2, retain source-level evidence showing which campaigns
    # improved or regressed after the feature-pruning intervention.
    fp_table = prediction_table[
        (prediction_table[LABEL_COL].astype(int) == 0)
        & (prediction_table["Predicted_Label"].astype(int) == 1)
    ]
    fn_table = prediction_table[
        (prediction_table[LABEL_COL].astype(int) == 1)
        & (prediction_table["Predicted_Label"].astype(int) == 0)
    ]

    for error_name, error_table in (("false_positives", fp_table), ("false_negatives", fn_table)):
        error_table.to_csv(result_dir / f"{error_name}.csv", index=False)
        if error_table.empty:
            source_summary = pd.DataFrame(columns=[SOURCE_FILE_COL, "ErrorCount"])
        else:
            source_summary = (
                error_table.groupby(SOURCE_FILE_COL, dropna=False)
                .size()
                .reset_index(name="ErrorCount")
                .sort_values("ErrorCount", ascending=False, kind="mergesort")
            )
        source_summary.to_csv(
            result_dir / f"{error_name}_by_source.csv",
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
    summary.to_csv(RESULT_ROOT / "lstm_test_summary.csv", index=False)

    # Final package: historical O1/O2 comparisons are intentionally omitted.

    return summary
