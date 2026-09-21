from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
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
)


def _find_project_root() -> Path:
    """Locate Workshop_final without assuming a fixed drive letter."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "preprocessing").exists() and (parent / "data" / "ingested").exists():
            return parent
    raise RuntimeError(
        "Could not locate project root. Expected a parent containing "
        "src/preprocessing and data/ingested."
    )


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = _find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"
SHARED_SPLIT_ROOT = PROJECT_ROOT / "artifacts" / "splits"

DATASET_PATHS = {
    "dataset2": INGESTED_ROOT / "dataset2_ingested.csv",
    "dataset3": INGESTED_ROOT / "dataset3_ingested.csv",
}
BENCHMARK_DATASETS = tuple(DATASET_PATHS.keys())

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models_optimized_final" / "IF"
RESULT_ROOT = PROJECT_ROOT / "results" / "models_optimized_final" / "IF"

MODEL_DIR = Path(__file__).resolve().parent
for path in [MODEL_DIR, SRC_ROOT / "preprocessing", SRC_ROOT / "feature_engineering"]:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from preprocessing import DatasetPreprocessor  # noqa: E402
from feature_selection import select_and_engineer_dataset  # noqa: E402
from source_aware_split import (  # noqa: E402
    ORIGINAL_ROW_COL,
    get_or_create_shared_split,
)


# =============================================================================
# FINAL RETAINED CONFIGURATION
# =============================================================================

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

EXPECTED_BASELINE_FEATURES = 57
EXPECTED_FINAL_FEATURES = 43
FINAL_RANDOM_STATE = 42

# Frozen shared D2/D3 Isolation Forest configuration used by the final model.
# No model-hyperparameter search is performed by this file.
FINAL_IF_PARAMS = {
    "n_estimators": 200,
    "max_samples": 512,
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "n_jobs": -1,
    "verbose": 0,
    "warm_start": False,
}

# O1: remove environment-dependent / brittle features.
MANDATORY_DROPS = {
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Total Connection Flow Time",
    "Fwd Header Length",
    "Bwd Header Length",
}
FLAG_KEEP = "URG Flag Count"


# =============================================================================
# FEATURE HARDENING
# =============================================================================

def prune_final_features(features: list[str]) -> tuple[list[str], list[str]]:
    """Apply the retained O1 57 -> 43 feature hardening rule."""
    retained: list[str] = []
    dropped: list[str] = []

    for feature in features:
        should_drop = feature in MANDATORY_DROPS
        if "Flag" in feature and feature != FLAG_KEEP:
            should_drop = True

        if should_drop:
            dropped.append(feature)
        else:
            retained.append(feature)

    missing = sorted(MANDATORY_DROPS - set(features))
    if missing:
        raise ValueError(
            "Final IF schema mismatch: mandatory brittle features are absent: "
            f"{missing}"
        )

    if len(features) != EXPECTED_BASELINE_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_BASELINE_FEATURES} baseline selected features; "
            f"got {len(features)}. Refusing an uncontrolled final run."
        )

    if len(retained) != EXPECTED_FINAL_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FINAL_FEATURES} retained features; got {len(retained)}. "
            f"Dropped={dropped}"
        )

    return retained, dropped


def transform_select_prune(
    preprocessor: DatasetPreprocessor,
    df: pd.DataFrame,
    dataset_name: str,
    expected_final_features: list[str] | None = None,
):
    normalized = preprocessor.transform(df)
    selected_df, baseline_features, _dropped, _manifest = select_and_engineer_dataset(
        dataset_name=dataset_name,
        df=normalized,
    )

    final_features, o1_dropped = prune_final_features(list(baseline_features))

    if expected_final_features is not None and final_features != expected_final_features:
        raise ValueError(
            f"{dataset_name}: final 43-feature schema/order differs from training.\n"
            f"Expected: {expected_final_features}\n"
            f"Actual:   {final_features}"
        )

    X = selected_df[final_features].copy()
    y = selected_df[LABEL_COL].astype(int).copy()
    return selected_df, X, y, final_features, o1_dropped


# =============================================================================
# MODEL + SCORING
# =============================================================================

def build_model() -> IsolationForest:
    return IsolationForest(
        n_estimators=FINAL_IF_PARAMS["n_estimators"],
        max_samples=FINAL_IF_PARAMS["max_samples"],
        contamination=FINAL_IF_PARAMS["contamination"],
        max_features=FINAL_IF_PARAMS["max_features"],
        bootstrap=FINAL_IF_PARAMS["bootstrap"],
        n_jobs=FINAL_IF_PARAMS["n_jobs"],
        random_state=FINAL_RANDOM_STATE,
        verbose=FINAL_IF_PARAMS["verbose"],
        warm_start=FINAL_IF_PARAMS["warm_start"],
    )


def anomaly_scores(model: IsolationForest, X) -> np.ndarray:
    """Higher score means more anomalous."""
    return -model.score_samples(X)


def calculate_metrics(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[BENIGN_LABEL, MALICIOUS_LABEL],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity_tnr": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def choose_threshold_max_f1(y_validation, validation_scores):
    """Choose threshold on mixed VALIDATION only; TEST never participates."""
    y = np.asarray(y_validation, dtype=int)
    scores = np.asarray(validation_scores, dtype=float)

    precision, recall, thresholds = precision_recall_curve(
        y,
        scores,
        pos_label=MALICIOUS_LABEL,
    )
    if len(thresholds) == 0:
        raise ValueError("No validation threshold candidates were produced.")

    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1_values = np.divide(
        2.0 * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    best_f1 = float(np.max(f1_values))
    candidate_indices = np.flatnonzero(
        np.isclose(f1_values, best_f1, rtol=1e-12, atol=1e-12)
    )

    rows = []
    for index in candidate_indices:
        threshold = float(thresholds[index])
        metrics = calculate_metrics(y, scores, threshold)
        rows.append(metrics)

    # Same baseline tie-break philosophy:
    # F1 -> recall -> precision -> lower FPR -> higher threshold.
    rows.sort(
        key=lambda row: (
            row["f1"],
            row["recall_tpr"],
            row["precision"],
            -row["fpr"],
            row["threshold"],
        ),
        reverse=True,
    )

    best = rows[0]
    return float(best["threshold"]), best


# =============================================================================
# SPLIT LOADING
# =============================================================================

def load_splits(dataset_name: str, force_resplit: bool = False):
    path = DATASET_PATHS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing ingested dataset: {path}")

    df = pd.read_csv(path, low_memory=False)
    splits, manifest, metadata = get_or_create_shared_split(
        dataset_name=dataset_name,
        df=df,
        split_root=SHARED_SPLIT_ROOT,
        force_rebuild=force_resplit,
    )

    splits = {
        name: frame.drop(columns=[ORIGINAL_ROW_COL], errors="ignore").copy()
        for name, frame in splits.items()
    }
    return splits, manifest, metadata


# =============================================================================
# TRAINING
# =============================================================================

def train_dataset(dataset_name: str, force_resplit: bool = False) -> dict:
    print("\n" + "=" * 92)
    print(f"FINAL OPTIMIZED ISOLATION FOREST - {dataset_name.upper()}")
    print("=" * 92)

    splits, split_manifest, split_metadata = load_splits(
        dataset_name,
        force_resplit=force_resplit,
    )
    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    print(
        f"Rows | train={len(train_df):,}, validation={len(validation_df):,}, "
        f"test={len(test_df):,}"
    )
    print(f"Shared split mode: {split_metadata.get('mode')}")
    if split_metadata.get("oversized_source") is not None:
        print(
            "Chronologically split oversized SourceFile: "
            f"{split_metadata.get('oversized_source')}"
        )

    benign_train = train_df[
        train_df[LABEL_COL].astype(int) == BENIGN_LABEL
    ].copy()
    if benign_train.empty:
        raise ValueError("No benign training rows are available for Isolation Forest fitting.")

    # Baseline logic: preprocessing fit on benign TRAIN only.
    preprocessor = DatasetPreprocessor(f"{dataset_name}_if_final_optimized")
    preprocessor.fit(benign_train)

    _, X_train, y_train, final_features, dropped_features = transform_select_prune(
        preprocessor,
        benign_train,
        f"{dataset_name}_if_final_train",
    )

    if set(y_train.unique()) != {BENIGN_LABEL}:
        raise RuntimeError("Malicious rows reached Isolation Forest fitting.")

    model = build_model()
    model.fit(X_train)

    # Mixed validation only for the operating threshold.
    _, X_validation, y_validation, _, _ = transform_select_prune(
        preprocessor,
        validation_df,
        f"{dataset_name}_if_final_validation",
        expected_final_features=final_features,
    )
    validation_scores = anomaly_scores(model, X_validation)
    threshold, validation_metrics = choose_threshold_max_f1(
        y_validation,
        validation_scores,
    )

    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_dir / "isolation_forest.joblib")
    preprocessor.save(artifact_dir / "preprocessor.joblib")
    split_manifest.to_csv(artifact_dir / "split_manifest.csv", index=False)
    pd.DataFrame({"Feature": final_features}).to_csv(
        artifact_dir / "selected_features.csv",
        index=False,
    )
    pd.DataFrame({"DroppedFeature": dropped_features}).to_csv(
        artifact_dir / "dropped_brittle_features.csv",
        index=False,
    )

    with (artifact_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(FINAL_IF_PARAMS, handle, indent=2)

    with (artifact_dir / "operating_threshold.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": float(threshold),
                "threshold_rule": "Anomaly_Score >= threshold => Malicious",
                "threshold_objective": "maximize validation F1",
                "score_definition": "-IsolationForest.score_samples(X)",
                "test_used_for_threshold_selection": False,
            },
            handle,
            indent=2,
        )

    with (artifact_dir / "training_strategy.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "final_feature_representation": "O1_43_feature_hardened",
                "baseline_feature_count": EXPECTED_BASELINE_FEATURES,
                "final_feature_count": EXPECTED_FINAL_FEATURES,
                "mandatory_drops": sorted(MANDATORY_DROPS),
                "flag_rule": "drop every feature containing 'Flag' except exact 'URG Flag Count'",
                "hyperparameters_shared_across_dataset2_dataset3": True,
                "hyperparameters": FINAL_IF_PARAMS,
                "random_state": FINAL_RANDOM_STATE,
                "preprocessing_fit_scope": "this dataset's benign TRAIN rows only",
                "detector_fit_scope": "this dataset's benign TRAIN rows only",
                "threshold_calibration_scope": "this dataset's mixed VALIDATION rows only",
                "test_used_for_model_or_threshold_selection": False,
                "split_mode": split_metadata.get("mode"),
                "oversized_source": split_metadata.get("oversized_source"),
            },
            handle,
            indent=2,
        )

    pd.DataFrame([{"Dataset": dataset_name, **validation_metrics}]).to_csv(
        result_dir / "validation_operating_point.csv",
        index=False,
    )

    print(f"Final features: {len(final_features)} (57 -> 43)")
    print(f"Benign training rows: {len(X_train):,}")
    print(f"Validation threshold: {threshold:.8f}")

    # TEST is evaluated only after the validation threshold is frozen.
    return evaluate_dataset(dataset_name, splits=splits)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_dataset(dataset_name: str, splits=None) -> dict:
    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name

    required = [
        artifact_dir / "isolation_forest.joblib",
        artifact_dir / "preprocessor.joblib",
        artifact_dir / "selected_features.csv",
        artifact_dir / "best_hyperparameters.json",
        artifact_dir / "operating_threshold.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing final IF artifact: {path}")

    model = joblib.load(artifact_dir / "isolation_forest.joblib")
    preprocessor = DatasetPreprocessor.load(artifact_dir / "preprocessor.joblib")
    final_features = pd.read_csv(artifact_dir / "selected_features.csv")["Feature"].tolist()

    with (artifact_dir / "best_hyperparameters.json").open("r", encoding="utf-8") as handle:
        saved_params = json.load(handle)
    if saved_params != FINAL_IF_PARAMS:
        raise RuntimeError(
            f"{dataset_name}: saved hyperparameters differ from the final shared configuration."
        )

    with (artifact_dir / "operating_threshold.json").open("r", encoding="utf-8") as handle:
        threshold = float(json.load(handle)["threshold"])

    if splits is None:
        splits, _, _ = load_splits(dataset_name, force_resplit=False)
    test_df = splits["test"]

    selected_test, X_test, y_test, _, _ = transform_select_prune(
        preprocessor,
        test_df,
        f"{dataset_name}_if_final_test",
        expected_final_features=final_features,
    )

    scores = anomaly_scores(model, X_test)
    metrics = calculate_metrics(y_test, scores, threshold)
    predictions = (scores >= threshold).astype(int)

    result = {
        "Dataset": dataset_name,
        "Optimization": "FINAL_O1_feature_pruning",
        "FeatureCount": len(final_features),
        "HyperparameterScope": "shared_dataset2_dataset3",
        "SharedHyperparametersVerified": True,
        "TestRows": int(len(test_df)),
        "TestSourceFiles": int(test_df[SOURCE_FILE_COL].nunique()),
        "ThresholdMethod": "validation_f1",
        **metrics,
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(result_dir / "test_metrics.csv", index=False)
    with (result_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    prediction_table = selected_test[[SOURCE_FILE_COL, LABEL_COL]].copy()
    prediction_table["AnomalyScore"] = scores
    prediction_table["DecisionThreshold"] = threshold
    prediction_table["PredictedLabel"] = predictions
    prediction_table.to_csv(result_dir / "test_predictions.csv", index=False)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual_Benign", "Actual_Malicious"],
        columns=["Predicted_Benign", "Predicted_Malicious"],
    ).to_csv(result_dir / "confusion_matrix.csv")

    print("\n" + "-" * 92)
    print(f"FINAL TEST - {dataset_name.upper()}")
    print("-" * 92)
    print(f"Recall / TPR: {metrics['recall_tpr']:.4f}")
    print(f"FPR: {metrics['fpr']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"F2: {metrics['f2']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(
        f"TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )

    return result


def train_all(force_resplit: bool = False) -> pd.DataFrame:
    print("Final shared Isolation Forest hyperparameters:")
    for key, value in FINAL_IF_PARAMS.items():
        print(f"  {key}: {value}")
    print(f"  random_state: {FINAL_RANDOM_STATE}")

    rows = [
        train_dataset(dataset_name, force_resplit=force_resplit)
        for dataset_name in BENCHMARK_DATASETS
    ]
    summary = pd.DataFrame(rows)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_ROOT / "isolation_forest_final_test_summary.csv", index=False)
    return summary


def evaluate_all() -> pd.DataFrame:
    rows = [evaluate_dataset(dataset_name) for dataset_name in BENCHMARK_DATASETS]
    summary = pd.DataFrame(rows)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_ROOT / "isolation_forest_final_test_summary.csv", index=False)
    return summary
