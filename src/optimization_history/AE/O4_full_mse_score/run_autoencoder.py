from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _find_project_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "data" / "ingested").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate Workshop_final project root. Expected a parent containing "
        "both src/ and data/ingested/."
    )


PROJECT_ROOT = _find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_DIR = Path(__file__).resolve().parent

for path in [
    MODEL_DIR,
    SRC_ROOT / "preprocessing",
    SRC_ROOT / "feature_engineering",
]:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from preprocessing import DatasetPreprocessor  # noqa: E402
from feature_selection import select_and_engineer_dataset  # noqa: E402
from source_aware_split import ORIGINAL_ROW_COL, apply_split_manifest  # noqa: E402


DATASET_PATHS = {
    "dataset2": PROJECT_ROOT / "data" / "ingested" / "dataset2_ingested.csv",
    "dataset3": PROJECT_ROOT / "data" / "ingested" / "dataset3_ingested.csv",
}

O3_ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts" / "models_same_hyper" / "Autoencoder"
    / "optimization_3_mse_loss"
)
O3_RESULT_ROOT = (
    PROJECT_ROOT / "results" / "models_same_hyper" / "Autoencoder"
    / "optimization_3_mse_loss"
)
O4_ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts" / "models_same_hyper" / "Autoencoder"
    / "optimization_4_full_mse_score"
)
O4_RESULT_ROOT = (
    PROJECT_ROOT / "results" / "models_same_hyper" / "Autoencoder"
    / "optimization_4_full_mse_score"
)

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1
SCORE_MODE = "mse"
BATCH_SIZE = 256

O1_EXPLICIT_DROP_FEATURES = (
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Total Connection Flow Time",
    "Fwd Header Length",
    "Bwd Header Length",
)
O1_FLAG_KEEP = "URG Flag Count"
EXPECTED_BASELINE_FEATURE_COUNT = 57
EXPECTED_O1_FEATURE_COUNT = 43

FINAL_THRESHOLD_FPR_CANDIDATES = [0.01, 0.02, 0.05]
FINAL_THRESHOLD_BENIGN_QUANTILES = [0.95, 0.975, 0.98, 0.99, 0.995]


def apply_o1_feature_pruning(selected_df, selected_features):
    explicit = set(O1_EXPLICIT_DROP_FEATURES)
    dropped, kept = [], []
    for feature in selected_features:
        drop_explicit = feature in explicit
        drop_flag = ("Flag" in feature) and feature != O1_FLAG_KEEP
        (dropped if (drop_explicit or drop_flag) else kept).append(feature)

    if len(selected_features) == EXPECTED_BASELINE_FEATURE_COUNT and len(kept) != EXPECTED_O1_FEATURE_COUNT:
        raise RuntimeError(
            "O1 pruning expected 43 features from the 57-feature baseline, "
            f"but produced {len(kept)}. Dropped={dropped}"
        )
    return selected_df, kept, dropped


def reconstruct_matrix(model, X: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    outputs = []
    for start in range(0, len(X), int(batch_size)):
        batch = X[start:start + int(batch_size)]
        outputs.append(model(batch, training=False).numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def full_mse_score(X: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    return np.mean(np.square(X - reconstructed), axis=1)


def calculate_metrics(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[BENIGN_LABEL, MALICIOUS_LABEL]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_tnr": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def _best_fbeta_threshold(y_true, scores, beta: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        raise ValueError("Unable to calibrate reconstruction threshold.")
    precision, recall = precision[:-1], recall[:-1]
    beta_sq = beta ** 2
    denominator = beta_sq * precision + recall
    fbeta = np.divide(
        (1.0 + beta_sq) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_value = np.nanmax(fbeta)
    candidates = np.flatnonzero(np.isclose(fbeta, best_value))
    index = int(candidates[np.argmax(recall[candidates])]) if len(candidates) > 1 else int(candidates[0])
    return float(thresholds[index])


def _benign_quantile_threshold(y_true, scores, quantile: float) -> float:
    benign_scores = np.asarray(scores, dtype=float)[np.asarray(y_true, dtype=int) == BENIGN_LABEL]
    if len(benign_scores) == 0:
        raise ValueError("Validation split contains no benign rows.")
    return float(np.quantile(benign_scores, quantile))


def _recall_at_fpr_cap_threshold(y_true, scores, fpr_cap: float) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eligible = np.flatnonzero(fpr <= float(fpr_cap))
    if len(eligible) == 0:
        return float(np.nextafter(np.max(scores), np.inf))
    finite = eligible[np.isfinite(thresholds[eligible])]
    eligible = finite if len(finite) else eligible
    best_tpr = np.max(tpr[eligible])
    best = eligible[np.isclose(tpr[eligible], best_tpr)]
    # Prefer the lowest threshold among equal-recall valid candidates.
    return float(np.min(thresholds[best]))


def calibrate_threshold(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Threshold calibration requires benign and malicious validation rows.")

    primary = _best_fbeta_threshold(y_true, scores, beta=1.0)
    specs = [("validation_f1", primary)]
    for q in FINAL_THRESHOLD_BENIGN_QUANTILES:
        specs.append((f"benign_q{q:g}", _benign_quantile_threshold(y_true, scores, q)))
    specs.append(("validation_f2", _best_fbeta_threshold(y_true, scores, beta=2.0)))
    for cap in FINAL_THRESHOLD_FPR_CANDIDATES:
        specs.append((f"recall_at_fpr_{cap:g}", _recall_at_fpr_cap_threshold(y_true, scores, cap)))

    rows, seen = [], set()
    for method, threshold in specs:
        key = round(float(threshold), 12)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "threshold_method": method,
            "threshold": float(threshold),
            "selected_for_final": method == "validation_f1",
            **calculate_metrics(y_true, scores, float(threshold)),
        })
    return float(primary), "validation_f1", pd.DataFrame(rows)


def transform_split(preprocessor, df, dataset_name: str, expected_features: list[str]):
    normalized = preprocessor.transform(df)
    selected, baseline_features, _dropped, _manifest = select_and_engineer_dataset(
        dataset_name=dataset_name,
        df=normalized,
        expected_selected_features=None,
    )
    selected, actual_features, _o1_dropped = apply_o1_feature_pruning(selected, list(baseline_features))
    if actual_features != expected_features:
        raise ValueError(
            f"{dataset_name}: O4 feature schema differs from O3 training schema. "
            f"Expected {len(expected_features)}, got {len(actual_features)}."
        )
    X = selected[expected_features].to_numpy(dtype=np.float32)
    y = selected[LABEL_COL].astype(int).to_numpy()
    return selected, X, y


def save_plots(y_true, scores, predictions, threshold, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malicious"]).plot(
        ax=ax, values_format="d", colorbar=False
    )
    ax.set_title("Autoencoder O4 - Test Confusion Matrix")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax, name="Autoencoder O4")
    ax.set_title("Autoencoder O4 - Test ROC")
    fig.tight_layout(); fig.savefig(output_dir / "roc_curve.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax, name="Autoencoder O4")
    ax.set_title("Autoencoder O4 - Test Precision-Recall")
    fig.tight_layout(); fig.savefig(output_dir / "precision_recall_curve.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    benign = scores[np.asarray(y_true) == 0]
    malicious = scores[np.asarray(y_true) == 1]
    ax.hist(benign, bins=80, alpha=0.55, density=True, label="Benign")
    ax.hist(malicious, bins=80, alpha=0.55, density=True, label="Malicious")
    ax.axvline(threshold, linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_xlabel("Anomaly Score (full-feature MSE)"); ax.set_ylabel("Density")
    ax.set_title("Autoencoder O4 - Test Score Distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "anomaly_score_distribution.png", dpi=300); plt.close(fig)


def evaluate_dataset(dataset_name: str) -> dict:
    print("\n" + "=" * 90)
    print(f"AUTOENCODER O4 - FULL-FEATURE MSE SCORE - {dataset_name.upper()}")
    print("=" * 90)

    input_path = DATASET_PATHS[dataset_name]
    o3_artifact = O3_ARTIFACT_ROOT / dataset_name
    o4_artifact = O4_ARTIFACT_ROOT / dataset_name
    result_dir = O4_RESULT_ROOT / dataset_name

    required = [
        input_path,
        o3_artifact / "autoencoder.keras",
        o3_artifact / "preprocessor.joblib",
        o3_artifact / "split_manifest.csv",
        o3_artifact / "selected_features.csv",
        o3_artifact / "decision_rule.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required O3 artifact/input: {path}")

    with (o3_artifact / "decision_rule.json").open("r", encoding="utf-8") as handle:
        o3_rule = json.load(handle)
    if str(o3_rule.get("score_mode")) != "top5_mse":
        raise RuntimeError(
            f"Expected O3 score_mode='top5_mse', found {o3_rule.get('score_mode')!r}. "
            "O4 is defined as a score-only Top-5 MSE -> full MSE ablation."
        )

    model = tf.keras.models.load_model(o3_artifact / "autoencoder.keras")
    preprocessor = DatasetPreprocessor.load(o3_artifact / "preprocessor.joblib")
    manifest = pd.read_csv(o3_artifact / "split_manifest.csv")
    selected_features = pd.read_csv(o3_artifact / "selected_features.csv")["Feature"].tolist()
    if len(selected_features) != 43:
        raise RuntimeError(f"Expected O3 43-feature schema, found {len(selected_features)} features.")

    df = pd.read_csv(input_path, low_memory=False)
    splits = apply_split_manifest(df, manifest)
    validation_df = splits["validation"].drop(columns=[ORIGINAL_ROW_COL], errors="ignore")
    test_df = splits["test"].drop(columns=[ORIGINAL_ROW_COL], errors="ignore")

    selected_val, X_val, y_val = transform_split(
        preprocessor, validation_df, f"{dataset_name}_o4_validation", selected_features
    )
    selected_test, X_test, y_test = transform_split(
        preprocessor, test_df, f"{dataset_name}_o4_test", selected_features
    )

    val_recon = reconstruct_matrix(model, X_val)
    test_recon = reconstruct_matrix(model, X_test)
    val_scores = full_mse_score(X_val, val_recon)
    test_scores = full_mse_score(X_test, test_recon)

    threshold, threshold_method, candidates = calibrate_threshold(y_val, val_scores)
    test_metrics = calculate_metrics(y_test, test_scores, threshold)
    predictions = (test_scores > threshold).astype(int)

    result = {
        "Dataset": dataset_name,
        "TestRows": int(len(y_test)),
        "TestSourceFiles": int(test_df[SOURCE_FILE_COL].nunique()),
        "FeatureCount": int(len(selected_features)),
        "Optimization": "O4_full_mse_score",
        "score_mode": SCORE_MODE,
        "threshold_method": threshold_method,
        **test_metrics,
    }

    o4_artifact.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(o4_artifact / "validation_threshold_candidates.csv", index=False)
    pd.DataFrame([result]).to_csv(result_dir / "test_metrics.csv", index=False)
    with (result_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    pd.DataFrame({
        "Feature": selected_features,
    }).to_csv(o4_artifact / "selected_features.csv", index=False)

    with (o4_artifact / "decision_rule.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "optimization_stage": "O4_full_mse_score",
            "reused_model_from": str(o3_artifact / "autoencoder.keras"),
            "reused_preprocessor_from": str(o3_artifact / "preprocessor.joblib"),
            "reused_split_manifest_from": str(o3_artifact / "split_manifest.csv"),
            "training_loss": "mse",
            "bottleneck_dim": 4,
            "feature_count": 43,
            "previous_score_mode": "top5_mse",
            "score_mode": "mse",
            "threshold_method": threshold_method,
            "threshold": float(threshold),
            "threshold_calibration_scope": "O4 validation split only",
            "test_used_for_selection": False,
            "model_retrained_for_o4": False,
        }, handle, indent=2)

    prediction_table = selected_test[[SOURCE_FILE_COL, LABEL_COL]].copy()
    prediction_table["Anomaly_Score"] = test_scores
    prediction_table["Threshold"] = threshold
    prediction_table["Score_Mode"] = SCORE_MODE
    prediction_table["Predicted_Label"] = predictions
    prediction_table.to_csv(result_dir / "test_predictions.csv", index=False)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual_Benign", "Actual_Malicious"],
        columns=["Predicted_Benign", "Predicted_Malicious"],
    ).to_csv(result_dir / "confusion_matrix.csv")
    save_plots(y_test, test_scores, predictions, threshold, result_dir)

    print("Model training: REUSED exact O3 weights (no retraining)")
    print("Features: 43 (same O1/O2/O3 schema)")
    print("Bottleneck: 4 (same O3 model)")
    print("Training loss: MSE (same O3 model)")
    print("O4-only change: score_mode top5_mse -> mse")
    print(f"Validation threshold: {threshold:.8f}")
    print(f"Test rows: {len(y_test):,}")
    print(f"Recall={test_metrics['recall_tpr']:.4f}")
    print(f"FPR={test_metrics['fpr']:.4f}")
    print(f"F1={test_metrics['f1']:.4f}")
    print(f"ROC-AUC={test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC={test_metrics['pr_auc']:.4f}")
    return result


def write_comparison(summary: pd.DataFrame):
    metrics = [
        "accuracy", "balanced_accuracy", "precision", "recall_tpr",
        "specificity_tnr", "fpr", "f1", "f2", "roc_auc", "pr_auc",
        "mcc", "tn", "fp", "fn", "tp", "threshold",
    ]
    rows = []
    for _, o4 in summary.iterrows():
        dataset = str(o4["Dataset"])
        o3_path = O3_RESULT_ROOT / dataset / "test_metrics.csv"
        if not o3_path.exists():
            print(f"[WARN] O3 metrics not found for comparison: {o3_path}")
            continue
        o3 = pd.read_csv(o3_path).iloc[0]
        record = {
            "Dataset": dataset,
            "Optimization": "O4_full_mse_score",
            "O3FeatureCount": int(o3["FeatureCount"]),
            "O4FeatureCount": int(o4["FeatureCount"]),
            "O3_score_mode": str(o3.get("score_mode", "top5_mse")),
            "O4_score_mode": "mse",
        }
        for metric in metrics:
            if metric in o3.index and metric in o4.index:
                a = float(o3[metric]); b = float(o4[metric])
                record[f"O3_{metric}"] = a
                record[f"O4_{metric}"] = b
                record[f"Delta_{metric}"] = b - a
        rows.append(record)
    if rows:
        O4_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(O4_RESULT_ROOT / "optimization_4_vs_o3.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="O4 score-only Autoencoder experiment: reuse exact O3 model and change Top-5 MSE to full-feature MSE."
    )
    parser.add_argument("--dataset", choices=["dataset2", "dataset3"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 90)
    print("AUTOENCODER O4 - FULL-FEATURE MSE ANOMALY SCORE")
    print("=" * 90)
    print("O4 is a score-only ablation: NO MODEL RETRAINING.")
    print("Reused from O3: 43 features, 64-32 encoder, bottleneck=4, LR=0.002, MSE loss, L2=1e-5, noise=0.02, batch=256.")
    print("Only change: anomaly score Top-5 MSE -> full-feature MSE; threshold recalibrated on validation only.")

    datasets = [args.dataset] if args.dataset else ["dataset2", "dataset3"]
    summary = pd.DataFrame([evaluate_dataset(name) for name in datasets])
    O4_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(O4_RESULT_ROOT / "autoencoder_o4_test_summary.csv", index=False)
    write_comparison(summary)

    print("\n" + "=" * 90)
    print("AUTOENCODER O4 COMPLETE")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
