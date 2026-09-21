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
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "preprocessing").exists() and (parent / "data" / "ingested").exists():
            return parent
    raise RuntimeError(
        "Could not locate Workshop_final project root. Expected a parent containing "
        "src/preprocessing and data/ingested."
    )


PROJECT_ROOT = _find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"
SHARED_SPLIT_ROOT = PROJECT_ROOT / "artifacts" / "splits"

DATASET_PATHS = {
    "dataset2": INGESTED_ROOT / "dataset2_ingested.csv",
    "dataset3": INGESTED_ROOT / "dataset3_ingested.csv",
}

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "models_same_hyper"
    / "IsolationForest"
    / "optimization_1_feature_pruning"
)
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "models_same_hyper"
    / "IsolationForest"
    / "optimization_1_feature_pruning"
)

MODEL_DIR = Path(__file__).resolve().parent
for path in [MODEL_DIR, SRC_ROOT / "preprocessing", SRC_ROOT / "feature_engineering"]:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from preprocessing import DatasetPreprocessor  # noqa: E402
from feature_selection import select_and_engineer_dataset  # noqa: E402
from source_aware_split import ORIGINAL_ROW_COL, get_or_create_shared_split  # noqa: E402

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1
EXPECTED_BASELINE_FEATURES = 57
EXPECTED_O1_FEATURES = 43
FINAL_RANDOM_STATE = 42

MANDATORY_DROPS = {
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Total Connection Flow Time",
    "Fwd Header Length",
    "Bwd Header Length",
}
FLAG_KEEP = "URG Flag Count"

IF_PARAM_KEYS = {
    "n_estimators",
    "max_samples",
    "contamination",
    "max_features",
    "bootstrap",
    "n_jobs",
    "verbose",
    "warm_start",
}


def _is_if_params(obj: dict) -> bool:
    return isinstance(obj, dict) and IF_PARAM_KEYS.issubset(obj.keys())


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def locate_shared_if_hyperparameters() -> tuple[dict, str]:
    """Find the existing D2/D3 shared IF hyperparameters and verify equality."""
    artifacts = PROJECT_ROOT / "artifacts"
    candidates = []

    # First prefer explicit shared artifacts.
    for path in artifacts.rglob("shared_best_hyperparameters.json"):
        try:
            params = _load_json(path)
        except Exception:
            continue
        if _is_if_params(params):
            score = 0
            lower = str(path).lower()
            if "models_same_hyper" in lower:
                score += 10
            if "isolation" in lower or "forest" in lower:
                score += 10
            candidates.append((score, path, params))

    if candidates:
        candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
        _, path, params = candidates[0]
        return params, str(path)

    # Otherwise find matching dataset2/dataset3 best_hyperparameters.json siblings.
    d2_files = list(artifacts.rglob("dataset2/best_hyperparameters.json"))
    pairs = []
    for d2_path in d2_files:
        d3_path = d2_path.parent.parent / "dataset3" / "best_hyperparameters.json"
        if not d3_path.exists():
            continue
        try:
            p2 = _load_json(d2_path)
            p3 = _load_json(d3_path)
        except Exception:
            continue
        if not (_is_if_params(p2) and _is_if_params(p3)):
            continue
        if p2 != p3:
            continue
        lower = str(d2_path.parent.parent).lower()
        score = 0
        if "models_same_hyper" in lower:
            score += 10
        if "isolation" in lower or "forest" in lower:
            score += 10
        # Do not reuse a previous optimization as the baseline source if avoidable.
        if "optimization" in lower or "improved" in lower:
            score -= 5
        pairs.append((score, d2_path.parent.parent, p2))

    if not pairs:
        raise FileNotFoundError(
            "Could not find an existing shared Isolation Forest hyperparameter artifact. "
            "O1 intentionally refuses to retune parameters because this must be a controlled "
            "feature-pruning ablation. Ensure the current models_same_hyper IF baseline has "
            "matching dataset2/dataset3 best_hyperparameters.json files or an IF "
            "shared_best_hyperparameters.json artifact."
        )

    pairs.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    _, root, params = pairs[0]
    return params, str(root)


def prune_o1_features(features: list[str]) -> tuple[list[str], list[str]]:
    dropped = []
    retained = []
    for feature in features:
        drop = feature in MANDATORY_DROPS
        if "Flag" in feature and feature != FLAG_KEEP:
            drop = True
        if drop:
            dropped.append(feature)
        else:
            retained.append(feature)

    missing_mandatory = sorted(MANDATORY_DROPS - set(features))
    if missing_mandatory:
        raise ValueError(
            "O1 schema mismatch: mandatory baseline features are absent: "
            f"{missing_mandatory}"
        )

    if len(features) != EXPECTED_BASELINE_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_BASELINE_FEATURES} baseline features before O1 pruning; "
            f"got {len(features)}. Refusing an uncontrolled comparison."
        )
    if len(retained) != EXPECTED_O1_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_O1_FEATURES} features after O1 pruning; got {len(retained)}. "
            f"Dropped={dropped}"
        )
    return retained, dropped


def transform_select_prune(
    preprocessor: DatasetPreprocessor,
    df: pd.DataFrame,
    dataset_name: str,
    expected_o1_features: list[str] | None = None,
):
    normalized = preprocessor.transform(df)
    selected_df, baseline_features, _dropped, _manifest = select_and_engineer_dataset(
        dataset_name=dataset_name,
        df=normalized,
    )
    retained, o1_dropped = prune_o1_features(list(baseline_features))

    if expected_o1_features is not None and retained != expected_o1_features:
        raise ValueError(
            f"{dataset_name}: O1 feature schema/order differs from training.\n"
            f"Expected: {expected_o1_features}\nActual: {retained}"
        )

    X = selected_df[retained].copy()
    y = selected_df[LABEL_COL].astype(int).copy()
    return selected_df, X, y, retained, o1_dropped


def build_model(params: dict) -> IsolationForest:
    missing = IF_PARAM_KEYS - set(params)
    if missing:
        raise ValueError(f"Missing IF hyperparameters: {sorted(missing)}")
    return IsolationForest(
        n_estimators=int(params["n_estimators"]),
        max_samples=params["max_samples"],
        contamination=params["contamination"],
        max_features=float(params["max_features"]),
        bootstrap=bool(params["bootstrap"]),
        n_jobs=int(params["n_jobs"]),
        random_state=FINAL_RANDOM_STATE,
        verbose=int(params["verbose"]),
        warm_start=bool(params["warm_start"]),
    )


def anomaly_scores(model: IsolationForest, X) -> np.ndarray:
    return -model.score_samples(X)


def calculate_metrics(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, pred, zero_division=0)),
        "specificity_tnr": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def choose_threshold_max_f1(y_validation, scores):
    y = np.asarray(y_validation, dtype=int)
    scores = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, scores, pos_label=1)
    if len(thresholds) == 0:
        raise ValueError("No validation threshold candidates were produced.")
    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    rows = []
    for i, threshold in enumerate(thresholds):
        metrics = calculate_metrics(y, scores, float(threshold))
        rows.append({
            "threshold": float(threshold),
            "precision_curve": float(precision[i]),
            "recall_curve": float(recall[i]),
            "f1_curve": float(f1[i]),
            **metrics,
        })
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["f1", "recall_tpr", "precision", "fpr", "threshold"],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    best = table.iloc[0]
    return float(best["threshold"]), calculate_metrics(y, scores, float(best["threshold"])), table


def load_splits(dataset_name: str, force_resplit: bool = False):
    path = DATASET_PATHS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path, low_memory=False)
    splits, manifest, metadata = get_or_create_shared_split(
        dataset_name=dataset_name,
        df=df,
        split_root=SHARED_SPLIT_ROOT,
        force_rebuild=force_resplit,
    )
    for key in list(splits):
        splits[key] = splits[key].drop(columns=[ORIGINAL_ROW_COL], errors="ignore").copy()
    return splits, manifest, metadata


def train_dataset(dataset_name: str, shared_params: dict, shared_param_source: str, force_resplit: bool = False):
    print("\n" + "=" * 90)
    print(f"ISOLATION FOREST O1 - FEATURE PRUNING - {dataset_name.upper()}")
    print("=" * 90)
    splits, split_manifest, split_metadata = load_splits(dataset_name, force_resplit)
    train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]
    print(f"Rows | train={len(train_df):,}, validation={len(val_df):,}, test={len(test_df):,}")
    print(f"Shared split mode: {split_metadata.get('mode')}")
    if split_metadata.get("oversized_source"):
        print(f"Chronologically split oversized SourceFile: {split_metadata.get('oversized_source')}")

    benign_train = train_df[train_df[LABEL_COL].astype(int) == BENIGN_LABEL].copy()
    if benign_train.empty:
        raise ValueError("No benign training rows available.")

    preprocessor = DatasetPreprocessor(f"{dataset_name}_if_o1")
    preprocessor.fit(benign_train)
    _, X_train, y_train, features, dropped = transform_select_prune(
        preprocessor, benign_train, f"{dataset_name}_if_o1_train"
    )
    if set(y_train.unique()) != {BENIGN_LABEL}:
        raise RuntimeError("Malicious rows reached IF fitting.")

    model = build_model(shared_params)
    model.fit(X_train)

    _, X_val, y_val, _, _ = transform_select_prune(
        preprocessor, val_df, f"{dataset_name}_if_o1_validation", expected_o1_features=features
    )
    val_scores = anomaly_scores(model, X_val)
    threshold, val_metrics, threshold_table = choose_threshold_max_f1(y_val, val_scores)

    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_dir / "isolation_forest.joblib")
    preprocessor.save(artifact_dir / "preprocessor.joblib")
    split_manifest.to_csv(artifact_dir / "split_manifest.csv", index=False)
    pd.DataFrame({"Feature": features}).to_csv(artifact_dir / "selected_features.csv", index=False)
    pd.DataFrame({"DroppedFeature": dropped}).to_csv(artifact_dir / "o1_dropped_features.csv", index=False)
    threshold_table.to_csv(result_dir / "validation_threshold_candidates.csv", index=False)

    with (artifact_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as h:
        json.dump(shared_params, h, indent=2)
    with (artifact_dir / "decision_rule.json").open("w", encoding="utf-8") as h:
        json.dump({
            "threshold": threshold,
            "method": "validation_f1",
            "score_definition": "-IsolationForest.score_samples(X)",
            "shared_hyperparameter_source": shared_param_source,
            "optimization": "O1_feature_pruning",
        }, h, indent=2)
    with (artifact_dir / "training_strategy.json").open("w", encoding="utf-8") as h:
        json.dump({
            "optimization": "O1_feature_pruning",
            "feature_count_before": EXPECTED_BASELINE_FEATURES,
            "feature_count_after": EXPECTED_O1_FEATURES,
            "mandatory_drops": sorted(MANDATORY_DROPS),
            "flag_rule": "drop every feature containing 'Flag' except exact 'URG Flag Count'",
            "preprocessing": "unchanged baseline signed-log1p + RobustScaler",
            "fit_scope": "benign training rows only",
            "threshold": "validation max-F1, unchanged from baseline",
            "shared_if_hyperparameters": shared_params,
            "shared_hyperparameter_source": shared_param_source,
            "test_used_for_selection": False,
        }, h, indent=2)

    pd.DataFrame([{"Dataset": dataset_name, **val_metrics}]).to_csv(
        result_dir / "validation_operating_point.csv", index=False
    )

    print(f"O1 features: {len(features)} (baseline {EXPECTED_BASELINE_FEATURES})")
    print("Dropped O1 features:")
    for feature in dropped:
        print(f"  - {feature}")
    print(f"Validation threshold: {threshold:.8f}")
    print(f"Benign training rows: {len(X_train):,}")
    return evaluate_dataset(dataset_name, splits=splits)


def evaluate_dataset(dataset_name: str, splits=None) -> dict:
    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name
    model = joblib.load(artifact_dir / "isolation_forest.joblib")
    preprocessor = DatasetPreprocessor.load(artifact_dir / "preprocessor.joblib")
    features = pd.read_csv(artifact_dir / "selected_features.csv")["Feature"].tolist()
    with (artifact_dir / "decision_rule.json").open("r", encoding="utf-8") as h:
        rule = json.load(h)
    threshold = float(rule["threshold"])

    if splits is None:
        splits, _, _ = load_splits(dataset_name, force_resplit=False)
    test_df = splits["test"]
    selected_test, X_test, y_test, _, _ = transform_select_prune(
        preprocessor, test_df, f"{dataset_name}_if_o1_test", expected_o1_features=features
    )
    scores = anomaly_scores(model, X_test)
    metrics = calculate_metrics(y_test, scores, threshold)
    pred = (scores >= threshold).astype(int)

    result = {
        "Dataset": dataset_name,
        "Optimization": "O1_feature_pruning",
        "FeatureCount": len(features),
        "TestRows": len(test_df),
        "TestSourceFiles": int(test_df[SOURCE_FILE_COL].nunique()),
        "ThresholdMethod": "validation_f1",
        **metrics,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(result_dir / "test_metrics.csv", index=False)
    prediction_table = selected_test[[SOURCE_FILE_COL, LABEL_COL]].copy()
    prediction_table["AnomalyScore"] = scores
    prediction_table["Threshold"] = threshold
    prediction_table["Predicted_Label"] = pred
    prediction_table.to_csv(result_dir / "test_predictions.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    pd.DataFrame(cm, index=["Actual_Benign", "Actual_Malicious"], columns=["Predicted_Benign", "Predicted_Malicious"]).to_csv(
        result_dir / "confusion_matrix.csv"
    )

    print("\n" + "-" * 90)
    print(f"O1 FINAL TEST - {dataset_name.upper()}")
    print("-" * 90)
    print(f"Recall / TPR: {metrics['recall_tpr']:.4f}")
    print(f"FPR: {metrics['fpr']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"F2: {metrics['f2']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    return result


def train_all(force_resplit: bool = False):
    params, source = locate_shared_if_hyperparameters()
    print("Shared IF hyperparameters verified and frozen for D2/D3:")
    for key in sorted(params):
        print(f"  {key}: {params[key]}")
    print(f"Source: {source}")
    rows = [train_dataset(ds, params, source, force_resplit=force_resplit) for ds in DATASET_PATHS]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_ROOT / "isolation_forest_o1_test_summary.csv", index=False)
    return summary


def evaluate_all():
    rows = [evaluate_dataset(ds) for ds in DATASET_PATHS]
    summary = pd.DataFrame(rows)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_ROOT / "isolation_forest_o1_test_summary.csv", index=False)
    return summary
