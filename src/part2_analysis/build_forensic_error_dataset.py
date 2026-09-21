from __future__ import annotations

"""
Build the frozen FINAL BASELINE Dataset-2/Dataset-3 sample-level forensic snapshot
for report Section 2.1.

This script reads ONLY the final pre-optimization baseline outputs:
  - Random Forest: results/models_baseline/RandomForest/<dataset>/oof_predictions.csv
  - LITEMV: results/models_baseline/LITEMV/<dataset>/test_predictions.csv
  - Autoencoder: results/models_baseline/AE/<dataset>/test_predictions.csv
  - LSTM: results/models_baseline/LSTM/<dataset>/test_predictions.csv

It deliberately does NOT read final_optimized, models_optimized_final,
or any optimization_* experiment directory. Those results belong to Section 2.2.

Evaluation rule
---------------
Section 2.1 uses each model's leakage-safe native baseline evaluation protocol:
  - Random Forest: ALL source-aware outer-fold out-of-fold (OOF) predictions using the frozen shared RF configuration.
  - LITEMV / AE / LSTM: the persisted shared source-aware held-out test split.

For cross-model sample-overlap analysis only, the already-created RF OOF ledger is
restricted to the exact shared test row indices used by LITEMV/AE/LSTM. That restricted
RF view must NOT replace the full RF OOF ledger for RF confusion matrices, metrics,
source-level error analysis, or feature-error contrasts.

No model is retrained or retuned by this script.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# PROJECT CONTRACT
# =============================================================================

DATASET_NAMES = ("dataset2", "dataset3")
MODEL_NAMES = (
    "random_forest",
    "litemv",
    "autoencoder",
    "lstm",
)

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
TIMESTAMP_COL = "Timestamp"
FLOW_ID_COL = "Flow ID"
ORIGINAL_ROW_COL = "Original_Row_Index"
MANIFEST_ORIGINAL_ROW_COL = "__OriginalRowIndex"
MANIFEST_SPLIT_COL = "__Split"

METADATA_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "SourceFile",
]

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "litemv": "LITEMV",
    "autoencoder": "Autoencoder",
    "lstm": "LSTM",
}


# =============================================================================
# PATHS -- FINAL PRE-OPTIMIZATION BASELINE ONLY
# =============================================================================

EXPECTED_BASELINE_FEATURE_COUNT = 57

MODEL_RESULT_ROOTS = {
    "random_forest": Path("results/models_baseline/RandomForest"),
    "litemv": Path("results/models_baseline/LITEMV"),
    "autoencoder": Path("results/models_baseline/AE"),
    "lstm": Path("results/models_baseline/LSTM"),
}

MODEL_ARTIFACT_ROOTS = {
    "random_forest": Path("artifacts/models_baseline/RandomForest"),
    "litemv": Path("artifacts/models_baseline/LITEMV"),
    "autoencoder": Path("artifacts/models_baseline/AE"),
    "lstm": Path("artifacts/models_baseline/LSTM"),
}


def infer_project_root() -> Path:
    """Infer <PROJECT_ROOT> when installed under <PROJECT_ROOT>/src/part2_analysis/."""
    script_path = Path(__file__).resolve()

    if script_path.parent.name.lower() == "part2_analysis":
        candidate = script_path.parents[2]
        if (candidate / "src" / "part2_analysis").exists():
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "data").exists() and (candidate / "results").exists():
            return candidate

    raise RuntimeError(
        "Could not infer PROJECT_ROOT. Place this script under "
        "<PROJECT_ROOT>/src/part2_analysis/ or pass --project-root explicitly."
    )


def dataset_path(project_root: Path, dataset_name: str) -> Path:
    return project_root / "data" / "ingested" / f"{dataset_name}_ingested.csv"


def shared_split_manifest_path(project_root: Path, dataset_name: str) -> Path:
    return (
        project_root
        / "artifacts"
        / "splits"
        / f"{dataset_name}_source_aware_split.csv"
    )


def prediction_path(project_root: Path, model_name: str, dataset_name: str) -> Path:
    """Return the exact final-baseline prediction file; never search optimization folders."""
    if model_name not in MODEL_RESULT_ROOTS:
        raise ValueError(f"Unknown model: {model_name}")

    filename = "oof_predictions.csv" if model_name == "random_forest" else "test_predictions.csv"
    return project_root / MODEL_RESULT_ROOTS[model_name] / dataset_name / filename


def model_selected_features_path(
    project_root: Path,
    model_name: str,
    dataset_name: str,
) -> Path:
    """Return the exact final-baseline selected-feature artifact."""
    if model_name not in MODEL_ARTIFACT_ROOTS:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = (
        project_root
        / MODEL_ARTIFACT_ROOTS[model_name]
        / dataset_name
        / "selected_features.csv"
    )
    if model_path.exists():
        return model_path

    # All final baseline models use the fixed Step-3 57-feature schema. This fallback
    # supports older baseline runs that did not duplicate selected_features.csv inside
    # each model artifact directory.
    fallback = project_root / "artifacts" / "feature_selection" / "selected_features.csv"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Missing final-baseline selected_features.csv for {MODEL_DISPLAY[model_name]} "
        f"on {dataset_name}. Expected:\n{model_path}\n"
        f"Fallback also missing:\n{fallback}"
    )

# =============================================================================
# GENERIC UTILITIES
# =============================================================================


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}:\n{path}")


def normalize_source(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace("\\\\", "/", regex=False)
    )


def numeric_labels(series: pd.Series, description: str) -> pd.Series:
    labels = pd.to_numeric(series, errors="raise").astype(int)
    invalid = sorted(set(labels.unique()) - {0, 1})
    if invalid:
        raise ValueError(f"{description} contains labels outside {{0, 1}}: {invalid}")
    return labels


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, hash_file: bool) -> dict:
    stat = path.stat()
    record = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }
    if hash_file:
        record["sha256"] = sha256_file(path)
    return record


def load_model_selected_features(
    project_root: Path,
    model_name: str,
    dataset_name: str,
) -> list[str]:
    path = model_selected_features_path(project_root, model_name, dataset_name)
    ensure_exists(path, f"{MODEL_DISPLAY[model_name]} selected-feature artifact for {dataset_name}")

    table = pd.read_csv(path)
    if table.empty:
        raise ValueError(f"Selected-feature artifact is empty: {path}")

    feature_col = "Feature" if "Feature" in table.columns else table.columns[0]
    features = table[feature_col].dropna().astype(str).str.strip().tolist()

    if not features:
        raise ValueError(f"No features found in {path}")
    if len(features) != len(set(features)):
        raise ValueError(f"Duplicate feature names found in {path}")
    if len(features) != EXPECTED_BASELINE_FEATURE_COUNT:
        raise ValueError(
            f"{dataset_name}/{model_name}: final baseline must use exactly "
            f"{EXPECTED_BASELINE_FEATURE_COUNT} Step-3 features; found {len(features)} in {path}. "
            "Refusing to mix an optimized feature schema into Section 2.1 baseline analysis."
        )

    return features


# =============================================================================
# BASELINE FEATURE SPACE
# =============================================================================


def prepare_analysis_rows(
    rows: pd.DataFrame,
    model_name: str,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Validate the fixed 57-feature final-baseline schema in raw/interpretable space.

    No optimized feature pruning or O3 engineered features are introduced here.
    """
    out = rows.copy()
    missing = [feature for feature in selected_features if feature not in out.columns]
    if missing:
        raise ValueError(
            f"{model_name}: final-baseline selected features are not present in the "
            f"ingested held-out rows. Missing={missing}"
        )
    return out

# =============================================================================
# SHARED TEST RECONSTRUCTION
# =============================================================================


def reconstruct_shared_test_rows(
    project_root: Path,
    dataset_name: str,
    ingested: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = shared_split_manifest_path(project_root, dataset_name)
    ensure_exists(path, f"shared split manifest for {dataset_name}")

    manifest = pd.read_csv(path, low_memory=False)
    required = {
        MANIFEST_ORIGINAL_ROW_COL,
        SOURCE_FILE_COL,
        LABEL_COL,
        MANIFEST_SPLIT_COL,
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{dataset_name}: shared split manifest is missing {missing}")

    if len(manifest) != len(ingested):
        raise ValueError(
            f"{dataset_name}: split-manifest rows ({len(manifest):,}) do not match "
            f"ingested rows ({len(ingested):,}). Refusing stale split."
        )

    ordered = manifest.sort_values(MANIFEST_ORIGINAL_ROW_COL).reset_index(drop=True)
    indices = pd.to_numeric(
        ordered[MANIFEST_ORIGINAL_ROW_COL],
        errors="raise",
    ).astype(np.int64)
    expected = np.arange(len(ingested), dtype=np.int64)
    if not np.array_equal(indices.to_numpy(), expected):
        raise ValueError(
            f"{dataset_name}: split manifest does not contain a complete 0..N-1 row sequence."
        )

    source_ok = (
        normalize_source(ingested[SOURCE_FILE_COL].reset_index(drop=True))
        == normalize_source(ordered[SOURCE_FILE_COL])
    )
    label_ok = (
        numeric_labels(ingested[LABEL_COL], "ingested labels").reset_index(drop=True)
        == numeric_labels(ordered[LABEL_COL], "manifest labels")
    )
    if not bool(source_ok.all()) or not bool(label_ok.all()):
        raise ValueError(
            f"{dataset_name}: ingested dataset identity does not match the persisted split manifest."
        )

    test_manifest = ordered[
        ordered[MANIFEST_SPLIT_COL].astype(str).str.lower() == "test"
    ].copy()
    test_indices = test_manifest[MANIFEST_ORIGINAL_ROW_COL].astype(np.int64).to_numpy()

    test_rows = ingested.iloc[test_indices].copy().reset_index(drop=True)
    test_rows[ORIGINAL_ROW_COL] = test_indices
    return test_rows, test_manifest.reset_index(drop=True)


# =============================================================================
# LEDGER CONSTRUCTION
# =============================================================================


def error_type(true_label: pd.Series, predicted_label: pd.Series) -> pd.Series:
    y_true = true_label.astype(int).to_numpy()
    y_pred = predicted_label.astype(int).to_numpy()
    values = np.select(
        [
            (y_true == 0) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
            (y_true == 1) & (y_pred == 1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="INVALID",
    )
    if np.any(values == "INVALID"):
        raise ValueError("Could not assign TP/TN/FP/FN to every prediction row.")
    return pd.Series(values, index=true_label.index, dtype="object")


def make_ledger(
    evaluated_rows: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
    true_labels: Iterable,
    predicted_labels: Iterable,
    scores: Iterable,
    thresholds: Iterable | float,
    score_type: str,
    evaluation_protocol: str,
    selected_features: list[str],
    extra_columns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = prepare_analysis_rows(
        evaluated_rows.reset_index(drop=True),
        model_name,
        selected_features,
    )

    y_true = pd.Series(true_labels).reset_index(drop=True)
    y_pred = pd.Series(predicted_labels).reset_index(drop=True)
    score = pd.to_numeric(pd.Series(scores), errors="raise").reset_index(drop=True)

    if np.isscalar(thresholds):
        threshold = pd.Series(np.repeat(float(thresholds), len(rows)), dtype=float)
    else:
        threshold = pd.to_numeric(
            pd.Series(thresholds),
            errors="raise",
        ).reset_index(drop=True)

    lengths = {
        "evaluated_rows": len(rows),
        "true_labels": len(y_true),
        "predicted_labels": len(y_pred),
        "scores": len(score),
        "thresholds": len(threshold),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{dataset_name}/{model_name}: ledger lengths differ: {lengths}")

    y_true = numeric_labels(y_true, f"{dataset_name}/{model_name} true labels")
    y_pred = numeric_labels(y_pred, f"{dataset_name}/{model_name} predictions")

    keep_metadata = [column for column in METADATA_COLUMNS if column in rows.columns]

    ledger = pd.DataFrame(
        {
            "Dataset": dataset_name,
            "Model": model_name,
            "ModelDisplay": MODEL_DISPLAY[model_name],
            "EvaluationProtocol": evaluation_protocol,
            ORIGINAL_ROW_COL: pd.to_numeric(
                rows[ORIGINAL_ROW_COL], errors="raise"
            ).astype(np.int64),
            "TrueLabel": y_true,
            "PredictedLabel": y_pred,
            "ErrorType": error_type(y_true, y_pred),
            "ScoreType": score_type,
            "Score": score.astype(float),
            "Threshold": threshold.astype(float),
        }
    )

    ledger["SignedScoreMargin"] = ledger["Score"] - ledger["Threshold"]
    ledger["AbsoluteScoreMargin"] = ledger["SignedScoreMargin"].abs()

    for column in keep_metadata:
        ledger[column] = rows[column].reset_index(drop=True)

    feature_frame = rows[selected_features].reset_index(drop=True).copy()
    ledger = pd.concat([ledger, feature_frame], axis=1)

    if extra_columns is not None and not extra_columns.empty:
        extra = extra_columns.reset_index(drop=True)
        if len(extra) != len(ledger):
            raise ValueError(
                f"{dataset_name}/{model_name}: extra-column length does not match ledger."
            )
        for column in extra.columns:
            if column not in ledger.columns:
                ledger[column] = extra[column]

    # A common canonical order makes cross-model joins and manual inspection easier.
    ledger = ledger.sort_values(ORIGINAL_ROW_COL, kind="mergesort").reset_index(drop=True)
    return ledger


# =============================================================================
# MODEL-SPECIFIC PREDICTION ALIGNMENT
# =============================================================================


def _verify_source_label_identity(
    predictions: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    pred_source_col: str,
    pred_label_col: str,
    description: str,
) -> None:
    if len(predictions) != len(rows):
        raise ValueError(
            f"{description}: prediction rows ({len(predictions):,}) != held-out rows "
            f"({len(rows):,})."
        )

    source_ok = (
        normalize_source(predictions[pred_source_col]).reset_index(drop=True)
        == normalize_source(rows[SOURCE_FILE_COL]).reset_index(drop=True)
    )
    label_ok = (
        numeric_labels(predictions[pred_label_col], f"{description} labels").reset_index(drop=True)
        == numeric_labels(rows[LABEL_COL], f"{description} held-out labels").reset_index(drop=True)
    )
    if not bool(source_ok.all()) or not bool(label_ok.all()):
        bad = np.flatnonzero((~source_ok | ~label_ok).to_numpy())[:5]
        raise ValueError(
            f"{description}: saved prediction ordering does not match the persisted shared "
            f"test rows. First mismatches: {bad.tolist()}"
        )


def build_random_forest_ledger(
    project_root: Path,
    dataset_name: str,
    ingested: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, Path]:
    """
    Build the official Section-2.1 RF ledger from all source-aware OOF predictions.

    Every dataset row must appear exactly once in the OOF file. The shared
    15% holdout restriction is applied later, only for cross-model overlap.
    """
    path = prediction_path(project_root, "random_forest", dataset_name)
    ensure_exists(path, f"final-baseline RF OOF predictions for {dataset_name}")
    predictions = pd.read_csv(path, low_memory=False)

    required = {
        ORIGINAL_ROW_COL,
        SOURCE_FILE_COL,
        "Actual_Label",
        "Malicious_Probability",
        "Predicted_Label",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{dataset_name}/RF: missing prediction columns: {missing}")

    indices = pd.to_numeric(
        predictions[ORIGINAL_ROW_COL],
        errors="raise",
    ).astype(np.int64)

    if indices.duplicated().any():
        raise ValueError(
            f"{dataset_name}/RF: duplicate {ORIGINAL_ROW_COL} values in OOF file."
        )

    expected = np.arange(len(ingested), dtype=np.int64)
    sorted_indices = np.sort(indices.to_numpy())
    if not np.array_equal(sorted_indices, expected):
        missing_indices = sorted(set(expected.tolist()) - set(sorted_indices.tolist()))
        extra_indices = sorted(set(sorted_indices.tolist()) - set(expected.tolist()))
        raise ValueError(
            f"{dataset_name}/RF: full OOF file must cover every dataset row exactly once. "
            f"Missing={missing_indices[:10]}; Extra={extra_indices[:10]}"
        )

    aligned = (
        predictions.assign(**{ORIGINAL_ROW_COL: indices})
        .sort_values(ORIGINAL_ROW_COL, kind="mergesort")
        .reset_index(drop=True)
    )

    evaluated_rows = ingested.iloc[aligned[ORIGINAL_ROW_COL].to_numpy()].copy()
    evaluated_rows = evaluated_rows.reset_index(drop=True)
    evaluated_rows[ORIGINAL_ROW_COL] = aligned[ORIGINAL_ROW_COL].to_numpy()

    _verify_source_label_identity(
        aligned,
        evaluated_rows,
        pred_source_col=SOURCE_FILE_COL,
        pred_label_col="Actual_Label",
        description=f"{dataset_name}/random_forest_full_oof",
    )

    score = pd.to_numeric(aligned["Malicious_Probability"], errors="raise")
    predicted = numeric_labels(aligned["Predicted_Label"], "RF Predicted_Label")
    implied = (score >= 0.5).astype(int)
    inconsistent = implied != predicted
    if bool(inconsistent.any()):
        bad = np.flatnonzero(inconsistent.to_numpy())[:5]
        raise ValueError(
            f"{dataset_name}/RF: Predicted_Label is inconsistent with threshold 0.5. "
            f"Rows={bad.tolist()}"
        )

    extra_cols = [
        column
        for column in ("Outer_Fold",)
        if column in aligned.columns
    ]

    ledger = make_ledger(
        evaluated_rows,
        dataset_name=dataset_name,
        model_name="random_forest",
        true_labels=aligned["Actual_Label"],
        predicted_labels=aligned["Predicted_Label"],
        scores=score,
        thresholds=0.5,
        score_type="MaliciousProbability",
        evaluation_protocol="source_aware_group_oof_full_dataset",
        selected_features=selected_features,
        extra_columns=aligned[extra_cols] if extra_cols else None,
    )
    return ledger, path


def restrict_rf_ledger_to_shared_test(
    rf_ledger: pd.DataFrame,
    test_rows: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Restrict the already-built full RF OOF ledger to the shared held-out rows.

    This view is ONLY for row-by-row cross-model overlap. It is never used for
    the official RF Section-2.1 confusion matrix or RF forensic summaries.
    """
    requested = np.sort(
        test_rows[ORIGINAL_ROW_COL].astype(np.int64).to_numpy()
    )
    restricted = (
        rf_ledger[
            rf_ledger[ORIGINAL_ROW_COL].astype(np.int64).isin(requested)
        ]
        .copy()
        .sort_values(ORIGINAL_ROW_COL, kind="mergesort")
        .reset_index(drop=True)
    )

    actual = restricted[ORIGINAL_ROW_COL].astype(np.int64).to_numpy()
    if not np.array_equal(actual, requested):
        missing = sorted(set(requested.tolist()) - set(actual.tolist()))
        raise RuntimeError(
            f"{dataset_name}/RF: could not restrict full OOF ledger to every shared "
            f"test row. Missing={missing[:10]}"
        )

    restricted["EvaluationProtocol"] = (
        "source_aware_group_oof_restricted_to_shared_test_for_overlap_only"
    )
    return restricted


def build_autoencoder_ledger(
    project_root: Path,
    dataset_name: str,
    test_rows: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, Path]:
    path = prediction_path(project_root, "autoencoder", dataset_name)
    ensure_exists(path, f"final-baseline Autoencoder test predictions for {dataset_name}")
    predictions = pd.read_csv(path, low_memory=False)

    required = {SOURCE_FILE_COL, LABEL_COL, "Anomaly_Score", "Threshold", "Predicted_Label"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{dataset_name}/AE: missing prediction columns: {missing}")

    _verify_source_label_identity(
        predictions,
        test_rows,
        pred_source_col=SOURCE_FILE_COL,
        pred_label_col=LABEL_COL,
        description=f"{dataset_name}/autoencoder",
    )

    extra_cols = [column for column in ("Score_Mode",) if column in predictions.columns]
    ledger = make_ledger(
        test_rows,
        dataset_name=dataset_name,
        model_name="autoencoder",
        true_labels=predictions[LABEL_COL],
        predicted_labels=predictions["Predicted_Label"],
        scores=predictions["Anomaly_Score"],
        thresholds=predictions["Threshold"],
        score_type="ReconstructionAnomalyScore",
        evaluation_protocol="shared_source_aware_test",
        selected_features=selected_features,
        extra_columns=predictions[extra_cols] if extra_cols else None,
    )
    return ledger, path


def _canonical_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        return series.astype(str).str.strip()
    return parsed.astype("int64").astype(str)


def _add_sequence_match_keys(
    df: pd.DataFrame,
    use_flow_id: bool,
    model_name: str,
) -> pd.DataFrame:
    work = df.copy()
    work["__key_source"] = normalize_source(work[SOURCE_FILE_COL])
    work["__key_time"] = _canonical_timestamp(work[TIMESTAMP_COL])
    work["__key_label"] = numeric_labels(work[LABEL_COL], f"{model_name} alignment labels").astype(str)

    keys = ["__key_source", "__key_time", "__key_label"]
    if use_flow_id:
        work["__key_flow"] = work[FLOW_ID_COL].astype(str).str.strip()
        keys.insert(1, "__key_flow")

    work["__key_occurrence"] = work.groupby(keys, dropna=False).cumcount()
    return work


def align_sequence_predictions_to_test_rows(
    predictions: pd.DataFrame,
    test_rows: pd.DataFrame,
    dataset_name: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sequence-model construction sorts each SourceFile chronologically, so prediction
    order is not assumed to equal raw test-row order. Match by SourceFile + Timestamp
    + Label and, when available, Flow ID.  A within-key occurrence counter safely
    resolves duplicate keys.
    """
    required = {SOURCE_FILE_COL, LABEL_COL, TIMESTAMP_COL}
    missing_pred = sorted(required - set(predictions.columns))
    missing_rows = sorted(required - set(test_rows.columns))
    if missing_pred or missing_rows:
        raise ValueError(
            f"{dataset_name}/{model_name}: alignment keys missing. "
            f"pred_missing={missing_pred}, test_missing={missing_rows}"
        )

    use_flow_id = FLOW_ID_COL in predictions.columns and FLOW_ID_COL in test_rows.columns
    pred_keyed = _add_sequence_match_keys(predictions.reset_index(drop=True), use_flow_id, model_name)
    row_keyed = _add_sequence_match_keys(test_rows.reset_index(drop=True), use_flow_id, model_name)

    key_cols = ["__key_source"]
    if use_flow_id:
        key_cols.append("__key_flow")
    key_cols.extend(["__key_time", "__key_label", "__key_occurrence"])

    row_lookup = row_keyed[key_cols + [ORIGINAL_ROW_COL]].copy()
    merged = pred_keyed.merge(
        row_lookup,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )
    if merged[ORIGINAL_ROW_COL].isna().any():
        bad = merged.loc[merged[ORIGINAL_ROW_COL].isna(), key_cols].head().to_dict("records")
        raise ValueError(
            f"{dataset_name}/{model_name}: could not map all predictions to shared test rows. "
            f"Examples={bad}"
        )

    merged[ORIGINAL_ROW_COL] = pd.to_numeric(
        merged[ORIGINAL_ROW_COL], errors="raise"
    ).astype(np.int64)
    if merged[ORIGINAL_ROW_COL].duplicated().any():
        raise ValueError(f"{dataset_name}/{model_name}: duplicate mapped original row indices.")

    merged = merged.sort_values(ORIGINAL_ROW_COL, kind="mergesort").reset_index(drop=True)
    indexed_rows = test_rows.set_index(ORIGINAL_ROW_COL, drop=False)
    aligned_rows = indexed_rows.loc[merged[ORIGINAL_ROW_COL].to_numpy()].reset_index(drop=True)

    # Remove internal matching helpers before saving any extra columns.
    merged = merged.drop(columns=[column for column in merged.columns if column.startswith("__key_")])
    return merged, aligned_rows


def build_litemv_ledger(
    project_root: Path,
    dataset_name: str,
    test_rows: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, Path]:
    path = prediction_path(project_root, "litemv", dataset_name)
    ensure_exists(path, f"final-baseline LITEMV test predictions for {dataset_name}")
    predictions = pd.read_csv(path, low_memory=False)

    required = {
        SOURCE_FILE_COL,
        LABEL_COL,
        TIMESTAMP_COL,
        "MaliciousProbability",
        "ValidationCalibratedThreshold",
        "Predicted_Label",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{dataset_name}/LITEMV: missing prediction columns: {missing}")

    aligned_predictions, aligned_rows = align_sequence_predictions_to_test_rows(
        predictions,
        test_rows,
        dataset_name,
        "litemv",
    )

    extra_cols = [
        column
        for column in ("TargetRowInSplitSourceSegment", "SequenceLength")
        if column in aligned_predictions.columns
    ]

    ledger = make_ledger(
        aligned_rows,
        dataset_name=dataset_name,
        model_name="litemv",
        true_labels=aligned_predictions[LABEL_COL],
        predicted_labels=aligned_predictions["Predicted_Label"],
        scores=aligned_predictions["MaliciousProbability"],
        thresholds=aligned_predictions["ValidationCalibratedThreshold"],
        score_type="MaliciousProbability",
        evaluation_protocol="shared_source_aware_test",
        selected_features=selected_features,
        extra_columns=aligned_predictions[extra_cols] if extra_cols else None,
    )
    return ledger, path


def build_lstm_ledger(
    project_root: Path,
    dataset_name: str,
    test_rows: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, Path]:
    path = prediction_path(project_root, "lstm", dataset_name)
    ensure_exists(path, f"final-baseline LSTM test predictions for {dataset_name}")
    predictions = pd.read_csv(path, low_memory=False)

    required = {
        SOURCE_FILE_COL,
        LABEL_COL,
        "MaliciousProbability",
        "ValidationCalibratedThreshold",
        "Predicted_Label",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{dataset_name}/LSTM: missing prediction columns: {missing}")

    aligned_predictions, aligned_rows = align_sequence_predictions_to_test_rows(
        predictions,
        test_rows,
        dataset_name,
        "lstm",
    )

    extra_cols = [
        column
        for column in ("TargetRowInSplitSourceSegment", "SequenceLength")
        if column in aligned_predictions.columns
    ]

    ledger = make_ledger(
        aligned_rows,
        dataset_name=dataset_name,
        model_name="lstm",
        true_labels=aligned_predictions[LABEL_COL],
        predicted_labels=aligned_predictions["Predicted_Label"],
        scores=aligned_predictions["MaliciousProbability"],
        thresholds=aligned_predictions["ValidationCalibratedThreshold"],
        score_type="MaliciousProbability",
        evaluation_protocol="shared_source_aware_test",
        selected_features=selected_features,
        extra_columns=aligned_predictions[extra_cols] if extra_cols else None,
    )
    return ledger, path


# =============================================================================
# FORENSIC SUMMARIES
# =============================================================================


def summarize_metrics(ledger: pd.DataFrame) -> dict:
    y_true = ledger["TrueLabel"].astype(int).to_numpy()
    y_pred = ledger["PredictedLabel"].astype(int).to_numpy()
    score = ledger["Score"].astype(float).to_numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    fpr = fp / (fp + tn) if (tn + fp) else np.nan

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, score)
        pr_auc = average_precision_score(y_true, score)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "Dataset": ledger["Dataset"].iloc[0],
        "Model": ledger["Model"].iloc[0],
        "ModelDisplay": ledger["ModelDisplay"].iloc[0],
        "EvaluationProtocol": ledger["EvaluationProtocol"].iloc[0],
        "Rows": int(len(ledger)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall_TPR": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity_TNR": float(specificity),
        "FPR": float(fpr),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "ThresholdMin": float(ledger["Threshold"].min()),
        "ThresholdMax": float(ledger["Threshold"].max()),
    }


def summarize_error_counts(ledger: pd.DataFrame) -> pd.DataFrame:
    counts = ledger["ErrorType"].value_counts().reindex(["TN", "FP", "FN", "TP"], fill_value=0)
    return pd.DataFrame(
        [
            {
                "Dataset": ledger["Dataset"].iloc[0],
                "Model": ledger["Model"].iloc[0],
                "ModelDisplay": ledger["ModelDisplay"].iloc[0],
                "TN": int(counts["TN"]),
                "FP": int(counts["FP"]),
                "FN": int(counts["FN"]),
                "TP": int(counts["TP"]),
                "TotalErrors": int(counts["FP"] + counts["FN"]),
            }
        ]
    )


def summarize_errors_by_source(ledger: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        ledger.groupby(SOURCE_FILE_COL, dropna=False)["ErrorType"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["TN", "FP", "FN", "TP"], fill_value=0)
        .reset_index()
    )

    grouped.insert(0, "ModelDisplay", ledger["ModelDisplay"].iloc[0])
    grouped.insert(0, "Model", ledger["Model"].iloc[0])
    grouped.insert(0, "Dataset", ledger["Dataset"].iloc[0])

    grouped["RowsEvaluated"] = grouped[["TN", "FP", "FN", "TP"]].sum(axis=1)
    grouped["BenignRows"] = grouped["TN"] + grouped["FP"]
    grouped["MaliciousRows"] = grouped["FN"] + grouped["TP"]

    grouped["SourceFPR"] = np.divide(
        grouped["FP"],
        grouped["BenignRows"],
        out=np.full(len(grouped), np.nan, dtype=float),
        where=grouped["BenignRows"].to_numpy() > 0,
    )
    grouped["SourceFNR"] = np.divide(
        grouped["FN"],
        grouped["MaliciousRows"],
        out=np.full(len(grouped), np.nan, dtype=float),
        where=grouped["MaliciousRows"].to_numpy() > 0,
    )
    grouped["SourceErrorRate"] = (
        (grouped["FP"] + grouped["FN"]) / grouped["RowsEvaluated"]
    )

    total_fp = int((ledger["ErrorType"] == "FP").sum())
    total_fn = int((ledger["ErrorType"] == "FN").sum())
    grouped["FractionOfModelFP"] = grouped["FP"] / total_fp if total_fp else 0.0
    grouped["FractionOfModelFN"] = grouped["FN"] / total_fn if total_fn else 0.0

    return grouped.sort_values(
        ["FP", "FN", "RowsEvaluated"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def representative_errors(ledger: pd.DataFrame) -> pd.DataFrame:
    selections: list[pd.Series] = []
    for kind in ("FP", "FN"):
        subset = ledger[ledger["ErrorType"] == kind]
        if subset.empty:
            continue

        borderline_idx = subset["AbsoluteScoreMargin"].idxmin()
        confident_idx = subset["AbsoluteScoreMargin"].idxmax()

        if borderline_idx == confident_idx:
            row = ledger.loc[borderline_idx].copy()
            row["RepresentativeKind"] = "borderline_and_high_confidence"
            selections.append(row)
        else:
            row = ledger.loc[borderline_idx].copy()
            row["RepresentativeKind"] = "borderline"
            selections.append(row)

            row = ledger.loc[confident_idx].copy()
            row["RepresentativeKind"] = "high_confidence"
            selections.append(row)

    if not selections:
        return pd.DataFrame(columns=[*ledger.columns, "RepresentativeKind"])
    return pd.DataFrame(selections).reset_index(drop=True)


def build_cross_model_overlap(
    ledgers: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ledgers["lstm"][[ORIGINAL_ROW_COL, SOURCE_FILE_COL, "TrueLabel"]].copy()
    base = base.rename(columns={"TrueLabel": "Label"})

    for model_name in MODEL_NAMES:
        model = ledgers[model_name][
            [ORIGINAL_ROW_COL, "PredictedLabel", "ErrorType", "Score", "Threshold"]
        ].copy()
        model = model.rename(
            columns={
                "PredictedLabel": f"{model_name}_PredictedLabel",
                "ErrorType": f"{model_name}_ErrorType",
                "Score": f"{model_name}_Score",
                "Threshold": f"{model_name}_Threshold",
            }
        )
        base = base.merge(model, on=ORIGINAL_ROW_COL, how="left", validate="one_to_one")

    prediction_columns = [f"{model}_PredictedLabel" for model in MODEL_NAMES]
    if base[prediction_columns].isna().any().any():
        raise ValueError("Cross-model overlap is missing at least one model prediction.")

    wrong_matrix = pd.DataFrame(
        {
            model: base[f"{model}_ErrorType"].isin(["FP", "FN"])
            for model in MODEL_NAMES
        }
    )
    base["ModelsWrong"] = wrong_matrix.sum(axis=1).astype(int)
    base["WrongModels"] = wrong_matrix.apply(
        lambda row: "|".join([model for model in MODEL_NAMES if bool(row[model])])
        if bool(row.any())
        else "none",
        axis=1,
    )

    summary = (
        base.groupby(["Label", "WrongModels", "ModelsWrong"], dropna=False)
        .size()
        .reset_index(name="Rows")
        .sort_values(["Label", "Rows"], ascending=[True, False], kind="mergesort")
        .reset_index(drop=True)
    )
    return base, summary


# =============================================================================
# FEATURE-LEVEL ERROR CONTRASTS
# =============================================================================


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return result

    order = valid.sort_values().index
    ordered = valid.loc[order].to_numpy(dtype=float)
    m = len(ordered)
    adjusted = ordered * m / np.arange(1, m + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result.loc[order] = adjusted
    return result


def feature_error_contrasts(
    ledger: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    FP vs TN and FN vs TP feature contrasts in raw/interpretable feature space.

    Rank-biserial sign:
      positive -> error samples tend to have larger values than correct reference
      negative -> error samples tend to have smaller values than correct reference
    """
    records = []
    comparisons = (("FP", "TN"), ("FN", "TP"))

    for error_group, reference_group in comparisons:
        error_rows = ledger[ledger["ErrorType"] == error_group]
        reference_rows = ledger[ledger["ErrorType"] == reference_group]

        for feature in selected_features:
            error_values = (
                pd.to_numeric(error_rows[feature], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            reference_values = (
                pd.to_numeric(reference_rows[feature], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            record = {
                "Dataset": ledger["Dataset"].iloc[0],
                "Model": ledger["Model"].iloc[0],
                "ModelDisplay": ledger["ModelDisplay"].iloc[0],
                "Comparison": f"{error_group}_vs_{reference_group}",
                "ErrorGroup": error_group,
                "ReferenceGroup": reference_group,
                "Feature": feature,
                "ErrorN": int(len(error_values)),
                "ReferenceN": int(len(reference_values)),
                "ErrorMedian": np.nan,
                "ReferenceMedian": np.nan,
                "ErrorQ1": np.nan,
                "ErrorQ3": np.nan,
                "ReferenceQ1": np.nan,
                "ReferenceQ3": np.nan,
                "MedianDifference": np.nan,
                "RobustMedianShiftVsReferenceIQR": np.nan,
                "RankBiserialEffect": np.nan,
                "AbsRankBiserialEffect": np.nan,
                "MannWhitneyP": np.nan,
            }

            if len(error_values) and len(reference_values):
                error_q1 = float(error_values.quantile(0.25))
                error_median = float(error_values.median())
                error_q3 = float(error_values.quantile(0.75))
                reference_q1 = float(reference_values.quantile(0.25))
                reference_median = float(reference_values.median())
                reference_q3 = float(reference_values.quantile(0.75))
                difference = error_median - reference_median
                reference_iqr = reference_q3 - reference_q1

                record.update(
                    {
                        "ErrorMedian": error_median,
                        "ReferenceMedian": reference_median,
                        "ErrorQ1": error_q1,
                        "ErrorQ3": error_q3,
                        "ReferenceQ1": reference_q1,
                        "ReferenceQ3": reference_q3,
                        "MedianDifference": difference,
                        "RobustMedianShiftVsReferenceIQR": (
                            difference / reference_iqr if reference_iqr > 0 else np.nan
                        ),
                    }
                )

                u_result = mannwhitneyu(
                    error_values.to_numpy(dtype=float),
                    reference_values.to_numpy(dtype=float),
                    alternative="two-sided",
                    method="asymptotic",
                )
                effect = (
                    2.0 * float(u_result.statistic)
                    / (len(error_values) * len(reference_values))
                    - 1.0
                )
                record["RankBiserialEffect"] = effect
                record["AbsRankBiserialEffect"] = abs(effect)
                record["MannWhitneyP"] = float(u_result.pvalue)

            records.append(record)

    result = pd.DataFrame(records)
    result["FDR_Q"] = np.nan
    for comparison in result["Comparison"].unique():
        mask = result["Comparison"] == comparison
        result.loc[mask, "FDR_Q"] = benjamini_hochberg(
            result.loc[mask, "MannWhitneyP"]
        )

    result["FDR_Significant_0_05"] = result["FDR_Q"] < 0.05
    result["Direction"] = np.select(
        [
            result["RankBiserialEffect"] > 0,
            result["RankBiserialEffect"] < 0,
        ],
        ["ErrorHigher", "ErrorLower"],
        default="NoDirectionalEffect",
    )
    return result.sort_values(
        ["Comparison", "AbsRankBiserialEffect"],
        ascending=[True, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


# =============================================================================
# OUTPUTS
# =============================================================================


def save_ledger_outputs(
    ledger: pd.DataFrame,
    output_dir: Path,
    selected_features: list[str],
) -> None:
    model_name = ledger["Model"].iloc[0]
    compact_columns = [column for column in ledger.columns if column not in selected_features]
    ledger[compact_columns].to_csv(
        output_dir / f"{model_name}_sample_ledger.csv",
        index=False,
    )
    ledger[ledger["ErrorType"].isin(["FP", "FN"])].to_csv(
        output_dir / f"{model_name}_errors_only.csv",
        index=False,
    )


def analyze_dataset(
    project_root: Path,
    snapshot_root: Path,
    dataset_name: str,
    hash_inputs: bool,
) -> dict:
    print("\n" + "=" * 94)
    print(f"FINAL BASELINE FORENSIC SNAPSHOT - {dataset_name.upper()}")
    print("=" * 94)

    input_path = dataset_path(project_root, dataset_name)
    ensure_exists(input_path, f"ingested dataset {dataset_name}")
    ingested = pd.read_csv(input_path, low_memory=False).reset_index(drop=True)

    required_ingested = {LABEL_COL, SOURCE_FILE_COL, TIMESTAMP_COL}
    missing = sorted(required_ingested - set(ingested.columns))
    if missing:
        raise ValueError(f"{dataset_name}: ingested dataset is missing {missing}")
    numeric_labels(ingested[LABEL_COL], f"{dataset_name} ingested Label")

    test_rows, _ = reconstruct_shared_test_rows(project_root, dataset_name, ingested)

    feature_schemas = {
        model: load_model_selected_features(project_root, model, dataset_name)
        for model in MODEL_NAMES
    }

    # Verify the baseline feature schema for shared held-out rows used by
    # LITEMV/AE/LSTM. RF is additionally checked against all ingested rows.
    for model, features in feature_schemas.items():
        prepare_analysis_rows(test_rows, model, features)
    prepare_analysis_rows(
        ingested,
        "random_forest",
        feature_schemas["random_forest"],
    )

    # Main Section-2.1 ledgers use each model's native leakage-safe evaluation
    # protocol. RF therefore remains FULL OOF; LITEMV/AE/LSTM remain shared-test.
    ledgers: dict[str, pd.DataFrame] = {}
    prediction_files: dict[str, Path] = {}

    ledgers["random_forest"], prediction_files["random_forest"] = build_random_forest_ledger(
        project_root, dataset_name, ingested, feature_schemas["random_forest"]
    )
    ledgers["litemv"], prediction_files["litemv"] = build_litemv_ledger(
        project_root, dataset_name, test_rows, feature_schemas["litemv"]
    )
    ledgers["autoencoder"], prediction_files["autoencoder"] = build_autoencoder_ledger(
        project_root, dataset_name, test_rows, feature_schemas["autoencoder"]
    )
    ledgers["lstm"], prediction_files["lstm"] = build_lstm_ledger(
        project_root, dataset_name, test_rows, feature_schemas["lstm"]
    )

    # LITEMV/AE/LSTM must cover exactly the persisted shared test rows.
    expected_test_indices = np.sort(
        test_rows[ORIGINAL_ROW_COL].astype(np.int64).to_numpy()
    )
    for model_name in ("litemv", "autoencoder", "lstm"):
        actual = np.sort(
            ledgers[model_name][ORIGINAL_ROW_COL].astype(np.int64).to_numpy()
        )
        if not np.array_equal(actual, expected_test_indices):
            raise RuntimeError(
                f"{dataset_name}/{model_name}: ledger does not cover the exact "
                "persisted shared test rows."
            )

    # RF must cover the complete dataset exactly once in its full OOF ledger.
    rf_indices = np.sort(
        ledgers["random_forest"][ORIGINAL_ROW_COL].astype(np.int64).to_numpy()
    )
    expected_rf_indices = np.arange(len(ingested), dtype=np.int64)
    if not np.array_equal(rf_indices, expected_rf_indices):
        raise RuntimeError(
            f"{dataset_name}/random_forest: official RF forensic ledger is not "
            "the complete source-aware OOF dataset."
        )

    # Separate RF view for cross-model overlap only.
    rf_overlap_ledger = restrict_rf_ledger_to_shared_test(
        ledgers["random_forest"],
        test_rows,
        dataset_name,
    )
    overlap_ledgers = dict(ledgers)
    overlap_ledgers["random_forest"] = rf_overlap_ledger

    dataset_output = snapshot_root / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    # Save the restricted RF view under an explicit name so it cannot be
    # mistaken for the official full-OOF RF ledger.
    rf_overlap_compact_columns = [
        column
        for column in rf_overlap_ledger.columns
        if column not in feature_schemas["random_forest"]
    ]
    rf_overlap_ledger[rf_overlap_compact_columns].to_csv(
        dataset_output / "random_forest_shared_test_overlap_only.csv",
        index=False,
    )

    metrics_tables = []
    count_tables = []
    source_tables = []
    representative_tables = []
    contrast_tables = []
    schema_rows = []

    for model_name in MODEL_NAMES:
        ledger = ledgers[model_name]
        selected_features = feature_schemas[model_name]
        save_ledger_outputs(ledger, dataset_output, selected_features)

        metrics_tables.append(pd.DataFrame([summarize_metrics(ledger)]))
        count_tables.append(summarize_error_counts(ledger))
        source_tables.append(summarize_errors_by_source(ledger))
        representative_tables.append(representative_errors(ledger))
        contrast_tables.append(feature_error_contrasts(ledger, selected_features))

        for position, feature in enumerate(selected_features, start=1):
            schema_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "ModelDisplay": MODEL_DISPLAY[model_name],
                    "FeaturePosition": position,
                    "Feature": feature,
                }
            )

        fp = int((ledger["ErrorType"] == "FP").sum())
        fn = int((ledger["ErrorType"] == "FN").sum())
        print(
            f"{MODEL_DISPLAY[model_name]:18s} rows={len(ledger):7,d}  "
            f"features={len(selected_features):2d}  FP={fp:6,d}  FN={fn:6,d}"
        )

    metrics = pd.concat(metrics_tables, ignore_index=True)
    counts = pd.concat(count_tables, ignore_index=True)
    by_source = pd.concat(source_tables, ignore_index=True)
    representatives = pd.concat(representative_tables, ignore_index=True, sort=False)
    contrasts = pd.concat(contrast_tables, ignore_index=True, sort=False)
    schemas = pd.DataFrame(schema_rows)

    metrics.to_csv(dataset_output / "forensic_metrics.csv", index=False)
    counts.to_csv(dataset_output / "error_counts.csv", index=False)
    by_source.to_csv(dataset_output / "error_by_source.csv", index=False)
    representatives.to_csv(dataset_output / "representative_errors.csv", index=False)
    contrasts.to_csv(dataset_output / "feature_error_contrasts.csv", index=False)
    schemas.to_csv(dataset_output / "model_feature_schemas.csv", index=False)

    (
        contrasts.sort_values(
            ["Model", "Comparison", "AbsRankBiserialEffect"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .groupby(["Model", "Comparison"], group_keys=False)
        .head(10)
        .to_csv(dataset_output / "top10_feature_error_contrasts.csv", index=False)
    )

    # Cross-model overlap is the ONLY place where RF is restricted to the same
    # shared held-out rows as LITEMV/AE/LSTM.
    overlap, overlap_summary = build_cross_model_overlap(overlap_ledgers)
    overlap.to_csv(
        dataset_output / "cross_model_error_overlap.csv.gz",
        index=False,
        compression="gzip",
    )
    overlap_summary.to_csv(
        dataset_output / "cross_model_error_overlap_summary.csv",
        index=False,
    )

    overlap_count_tables = [
        summarize_error_counts(overlap_ledgers[model_name])
        for model_name in MODEL_NAMES
    ]
    overlap_counts = pd.concat(overlap_count_tables, ignore_index=True)
    overlap_counts["Purpose"] = "cross_model_overlap_same_shared_test_rows_only"
    overlap_counts.to_csv(
        dataset_output / "cross_model_overlap_error_counts.csv",
        index=False,
    )

    rf_overlap_fp = int((rf_overlap_ledger["ErrorType"] == "FP").sum())
    rf_overlap_fn = int((rf_overlap_ledger["ErrorType"] == "FN").sum())
    print(
        f"RF overlap-only view  rows={len(rf_overlap_ledger):7,d}  "
        f"FP={rf_overlap_fp:6,d}  FN={rf_overlap_fn:6,d}  "
        "(NOT used for official RF Section-2.1 metrics)"
    )

    # Provenance diagnostic: record the confusion counts stored in the CURRENT
    # baseline prediction files. If AE/LSTM counts no longer match an older
    # report table, this output proves the saved prediction input itself changed.
    native_prediction_rows = []
    for model_name in MODEL_NAMES:
        path = prediction_files[model_name]
        table = pd.read_csv(path, low_memory=False)
        true_col = "Actual_Label" if model_name == "random_forest" else LABEL_COL
        pred_col = "Predicted_Label"

        if true_col not in table.columns or pred_col not in table.columns:
            raise ValueError(
                f"{dataset_name}/{model_name}: cannot create prediction provenance "
                f"summary; missing {true_col!r} or {pred_col!r} in {path}"
            )

        y_true_native = numeric_labels(
            table[true_col],
            f"{dataset_name}/{model_name} native prediction labels",
        )
        y_pred_native = numeric_labels(
            table[pred_col],
            f"{dataset_name}/{model_name} native predicted labels",
        )
        tn, fp, fn, tp = confusion_matrix(
            y_true_native,
            y_pred_native,
            labels=[0, 1],
        ).ravel()

        native_prediction_rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "ModelDisplay": MODEL_DISPLAY[model_name],
                "Path": str(path),
                "Rows": int(len(table)),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "SHA256": sha256_file(path) if hash_inputs else "",
                "NativeProtocol": (
                    "full_source_aware_oof"
                    if model_name == "random_forest"
                    else "shared_source_aware_test"
                ),
            }
        )

    prediction_provenance = pd.DataFrame(native_prediction_rows)
    prediction_provenance.to_csv(
        dataset_output / "prediction_file_provenance_and_counts.csv",
        index=False,
    )

    input_records = {
        "ingested_dataset": file_record(input_path, hash_inputs),
        "shared_split_manifest": file_record(
            shared_split_manifest_path(project_root, dataset_name),
            hash_inputs,
        ),
        "prediction_files": {
            model: file_record(path, hash_inputs)
            for model, path in prediction_files.items()
        },
        "selected_feature_artifacts": {
            model: file_record(
                model_selected_features_path(project_root, model, dataset_name),
                hash_inputs,
            )
            for model in MODEL_NAMES
        },
    }

    with (dataset_output / "input_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(input_records, handle, indent=2)

    return {
        "metrics": metrics,
        "counts": counts,
        "overlap_counts": overlap_counts,
        "prediction_provenance": prediction_provenance,
        "inputs": input_records,
        "feature_schemas": feature_schemas,
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the final PRE-OPTIMIZATION BASELINE D2/D3 four-model sample-level "
            "forensic ledgers for Milestone 3 Section 2.1 using each model's native leakage-safe evaluation protocol."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Workshop project root; normally inferred when script is under src/part2_analysis/.",
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_NAMES, "all"],
        default="all",
        help="Analyze dataset2, dataset3, or both (default: all).",
    )
    parser.add_argument(
        "--snapshot-name",
        default="final_baseline_v1",
        help="Output snapshot name under results/forensic_analysis/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of an existing forensic snapshot.",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip SHA-256 hashes for input provenance metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else infer_project_root()
    )

    snapshot_root = (
        project_root / "results" / "forensic_analysis" / args.snapshot_name
    )
    marker_path = snapshot_root / "FORENSIC_SNAPSHOT.json"
    if marker_path.exists() and not args.force:
        raise FileExistsError(
            "Forensic snapshot already exists and is frozen:\n"
            f"{snapshot_root}\n"
            "Use a new --snapshot-name, or pass --force only if replacement is intentional."
        )

    snapshot_root.mkdir(parents=True, exist_ok=True)
    datasets = DATASET_NAMES if args.dataset == "all" else (args.dataset,)

    all_metrics = []
    all_counts = []
    all_overlap_counts = []
    all_prediction_provenance = []
    dataset_records = {}
    schema_summary = {}

    for dataset_name in datasets:
        result = analyze_dataset(
            project_root=project_root,
            snapshot_root=snapshot_root,
            dataset_name=dataset_name,
            hash_inputs=not args.skip_hashes,
        )
        all_metrics.append(result["metrics"])
        all_counts.append(result["counts"])
        all_overlap_counts.append(result["overlap_counts"])
        all_prediction_provenance.append(result["prediction_provenance"])
        dataset_records[dataset_name] = result["inputs"]
        schema_summary[dataset_name] = {
            model: len(features)
            for model, features in result["feature_schemas"].items()
        }

    metrics = pd.concat(all_metrics, ignore_index=True)
    counts = pd.concat(all_counts, ignore_index=True)
    overlap_counts = pd.concat(all_overlap_counts, ignore_index=True)
    prediction_provenance = pd.concat(all_prediction_provenance, ignore_index=True)

    metrics.to_csv(snapshot_root / "forensic_metrics_all_datasets.csv", index=False)
    counts.to_csv(snapshot_root / "error_counts_all_datasets.csv", index=False)
    overlap_counts.to_csv(
        snapshot_root / "cross_model_overlap_error_counts_all_datasets.csv",
        index=False,
    )
    prediction_provenance.to_csv(
        snapshot_root / "prediction_file_provenance_and_counts_all_datasets.csv",
        index=False,
    )

    snapshot_payload = {
        "snapshot_name": args.snapshot_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Frozen final pre-optimization D2/D3 baseline for Section 2.1 forensic error analysis and Section 2.2 incremental optimization comparison."
        ),
        "snapshot_type": "final_pre_optimization_baseline",
        "result_roots": {
            "random_forest": "results/models_baseline/RandomForest",
            "litemv": "results/models_baseline/LITEMV",
            "autoencoder": "results/models_baseline/AE",
            "lstm": "results/models_baseline/LSTM",
        },
        "project_root": str(project_root),
        "datasets": list(datasets),
        "models": list(MODEL_NAMES),
        "feature_count_by_dataset_and_model": schema_summary,
        "dataset_inputs": dataset_records,
        "evaluation_rule": (
            "Official Section 2.1 per-model results use each model's native leakage-safe "
            "baseline protocol: Random Forest uses ALL source-aware outer-fold OOF predictions with the frozen shared RF configuration; "
            "LITEMV/AE/LSTM use the persisted shared source-aware held-out test split."
        ),
        "cross_model_comparability_rule": (
            "Only the cross-model sample-overlap analysis restricts the already-created RF "
            "OOF ledger to the exact shared held-out row indices used by LITEMV/AE/LSTM. "
            "That restricted RF view is never used for the official RF confusion matrix, "
            "metrics, source-level error analysis, or feature-error contrasts."
        ),
        "feature_analysis_rule": (
            "FP-vs-TN and FN-vs-TP contrasts use the fixed final-baseline Step-3 57-feature "
            "schema in raw/interpretable units. No optimized feature pruning, augmentation, "
            "normalization experiment, or RF O3 engineered feature is introduced."
        ),
    }
    with marker_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot_payload, handle, indent=2)

    print("\n" + "=" * 94)
    print("FINAL BASELINE FORENSIC SNAPSHOT COMPLETE")
    print("=" * 94)
    print(f"Output: {snapshot_root}")
    print("Baseline only: no final_optimized/improved/optimization_* result is read.")
    print("Section 2.1 official protocol: RF=full source-aware OOF with the frozen shared configuration; LITEMV/AE/LSTM=shared held-out test.")
    print("RF is restricted to shared test rows only for cross-model overlap outputs.")
    print("See prediction_file_provenance_and_counts_all_datasets.csv to trace stale/wrong prediction files.")


if __name__ == "__main__":
    main()
