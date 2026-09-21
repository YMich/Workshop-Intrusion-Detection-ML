from __future__ import annotations

"""
Final hybrid intrusion-detection pipeline
=========================================

Architecture:
    LSTM -> LITEMV -> RF

The architecture is FIXED. This file does not compare/select alternative
architectures and never uses TEST for model/rule selection.

Per dataset (dataset2 / dataset3):
1. Reuse the exact persisted source-aware TRAIN / VALIDATION / TEST split.
2. Load the frozen Part-3 LSTM and LITEMV fitted model snapshot used for the report.
3. Reconstruct each sequence model's preprocessing from TRAIN only. This avoids
   relying on old pandas/joblib pickles while preserving the original fit scope.
4. Fit the RF arbiter on TRAIN only using the exact final endpoint-aware
   RROLL5 32-feature representation and exact final RF hyperparameters.
5. Calibrate ONLY the routing thresholds of the fixed three-stage cascade on
   VALIDATION.
6. Freeze the hybrid rule.
7. Evaluate the frozen hybrid exactly once on TEST.

Cascade logic
-------------
Positive path:
    LSTM says malicious.
      - If LSTM is confidently positive -> MALICIOUS directly.
      - Otherwise route to LITEMV:
          * LITEMV confirms -> MALICIOUS.
          * LITEMV does not confirm -> RF arbitrates:
              RF confirms -> MALICIOUS
              RF rejects  -> BENIGN

Negative path:
    LSTM says benign.
      - Normally BENIGN.
      - Rescue to MALICIOUS only when BOTH LITEMV and RF exceed very strict
        validation-derived rescue thresholds.

The rescue thresholds are set strictly above the largest secondary-model score
observed on benign VALIDATION rows that LSTM classified benign, so each
secondary rescue condition alone has zero validation false positives before
the joint requirement is applied.

Outputs
-------
Artifacts:
    artifacts/hybrid_pipeline/<dataset>/

Results:
    results/hybrid_pipeline/<dataset>/
    results/hybrid_pipeline/hybrid_test_summary.csv

Run:
    python hybrid_pipeline.py
    python hybrid_pipeline.py --dataset dataset2
    python hybrid_pipeline.py --dataset dataset3
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from tensorflow import keras


# =============================================================================
# PROJECT / EXACT FINAL MODEL DISCOVERY
# =============================================================================


def discover_project_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (
            (candidate / "src").is_dir()
            and (candidate / "data" / "ingested").is_dir()
        ):
            return candidate
    raise RuntimeError(
        "Could not locate project root. Expected a parent containing "
        "src/ and data/ingested/."
    )


def require_file(project_root: Path, relative: str) -> Path:
    path = project_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing required final source file: {path}")
    return path.resolve()


PROJECT_ROOT = discover_project_root()

LSTM_FILE = require_file(
    PROJECT_ROOT,
    "src/models_optimized_final/LSTM/lstm.py",
)
LITEMV_FILE = require_file(
    PROJECT_ROOT,
    "src/models_optimized_final/LITEMV/litemv.py",
)
RF_TRAINING_FILE = require_file(
    PROJECT_ROOT,
    "src/models_optimized_final/RF/random_forest_training.py",
)
RF_CONFIG_FILE = require_file(
    PROJECT_ROOT,
    "src/models_optimized_final/RF/random_forest_config.py",
)
RF_TEMPORAL_FILE = require_file(
    PROJECT_ROOT,
    "src/models_optimized_final/RF/random_forest_temporal.py",
)
RF_UTILS_FILE = require_file(
    PROJECT_ROOT,
    "src/models_optimized_final/RF/random_forest_utils.py",
)
RF_PREPROCESSING_FILE = require_file(
    PROJECT_ROOT,
    "src/models_optimized_final/RF/random_forest_preprocessing.py",
)

LSTM_DIR = LSTM_FILE.parent
LITEMV_DIR = LITEMV_FILE.parent
RF_DIR = RF_TRAINING_FILE.parent

# Put the exact final implementation folders first.
for path in (
    PROJECT_ROOT / "src" / "feature_engineering",
    PROJECT_ROOT / "src" / "preprocessing",
    LITEMV_DIR,
    RF_DIR,
    LSTM_DIR,
):
    text = str(path)
    while text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

import lstm  # noqa: E402
import litemv  # noqa: E402
import random_forest_config as rf_config  # noqa: E402
import random_forest_temporal as rf_temporal  # noqa: E402
import random_forest_utils as rf_utils  # noqa: E402
from random_forest_preprocessing import FoldMedianImputer  # noqa: E402


def assert_exact_import(module, expected: Path, label: str) -> None:
    loaded = Path(module.__file__).resolve()
    if loaded != expected.resolve():
        raise RuntimeError(
            f"Imported wrong {label} module.\n"
            f"Expected: {expected.resolve()}\n"
            f"Loaded:   {loaded}"
        )


assert_exact_import(lstm, LSTM_FILE, "LSTM")
assert_exact_import(litemv, LITEMV_FILE, "LITEMV")
assert_exact_import(rf_config, RF_CONFIG_FILE, "RF config")
assert_exact_import(rf_temporal, RF_TEMPORAL_FILE, "RF temporal")
assert_exact_import(rf_utils, RF_UTILS_FILE, "RF utils")


# =============================================================================
# FINAL IMPLEMENTATION GUARDS
# =============================================================================


if getattr(lstm, "FINAL_NAMESPACE", None) != "models_optimized_final":
    raise RuntimeError(
        "The loaded LSTM is not the accepted models_optimized_final version."
    )

if int(
    getattr(lstm, "AUGMENTATION_COPIES_PER_MALICIOUS_SOURCE", -1)
) != 2:
    raise RuntimeError(
        "The loaded LSTM is not the accepted O1+O2 final implementation."
    )

if int(getattr(litemv, "BASELINE_EXPECTED_FEATURE_COUNT", -1)) != 57:
    raise RuntimeError("Expected final LITEMV baseline feature count = 57.")

if int(getattr(litemv, "EXPECTED_FEATURE_COUNT", -1)) != 43:
    raise RuntimeError("Expected final LITEMV feature count = 43.")

if int(getattr(litemv, "O1_EXPECTED_DROP_COUNT", -1)) != 14:
    raise RuntimeError("Expected final LITEMV to remove exactly 14 O1 features.")

for api_name, module in (
    ("FINAL_RF_PARAMS", rf_config),
    ("MODEL_RANDOM_STATE", rf_config),
    ("PREDICTION_THRESHOLD", rf_config),
    ("ROW_INDEX_COL", rf_config),
    ("build_temporal_features", rf_temporal),
    ("final_feature_names", rf_temporal),
    ("compute_training_class_weights", rf_utils),
    ("build_final_random_forest", rf_utils),
):
    if not hasattr(module, api_name):
        raise RuntimeError(
            f"Final RF implementation is missing required API: {api_name}"
        )


# =============================================================================
# CONSTANTS / PATHS
# =============================================================================


DATASETS = ("dataset2", "dataset3")

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
TIMESTAMP_COL = "Timestamp"
ROW_KEY_COL = "TargetRowInSplitSourceSegment"

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "hybrid_pipeline"
RESULT_ROOT = PROJECT_ROOT / "results" / "hybrid_pipeline"

# Part 3 intentionally uses the frozen fitted model snapshot that generated the
# report-facing hybrid results.  These weights are kept separate from the
# independently retrained Part-2 models so neither stage can overwrite the
# other.  The model IMPLEMENTATIONS still come from src/models_optimized_final.
PART3_BASE_MODEL_ROOT = ARTIFACT_ROOT / "base_models"

# Validation-only search grids. These match the accepted Part-3 design.
LSTM_POSITIVE_BAND_QUANTILES = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.65,
    0.80,
    1.00,
)

SECONDARY_CONFIRM_QUANTILES = (
    0.00,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
    1.00,
)

EPS = 1e-12


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_from_predictions(y_true, predictions) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_tpr": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "f2": float(
            fbeta_score(
                y_true,
                predictions,
                beta=2.0,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix_csv(
    y_true: np.ndarray,
    predictions: np.ndarray,
    path: Path,
) -> None:
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )
    pd.DataFrame(
        matrix,
        index=["Actual_Benign", "Actual_Malicious"],
        columns=["Predicted_Benign", "Predicted_Malicious"],
    ).to_csv(path)


# =============================================================================
# SHARED SOURCE-AWARE SPLIT
# =============================================================================


def load_shared_splits(dataset_name: str):
    """
    The final LSTM is the canonical owner of the persisted shared split.

    force_resplit is intentionally hard-coded False. The final hybrid must not
    silently change the benchmark population.
    """
    splits, manifest, metadata = lstm.load_dataset_and_split(
        dataset_name,
        force_resplit=False,
    )
    return splits, manifest, metadata


# =============================================================================
# LSTM — LOAD TRAINED WEIGHTS, RECONSTRUCT TRAIN-FITTED PREPROCESSING
# =============================================================================


def load_lstm_bundle(
    dataset_name: str,
    train_df: pd.DataFrame,
) -> dict:
    artifact_dir = PART3_BASE_MODEL_ROOT / dataset_name / "LSTM"

    model_path = artifact_dir / "lstm_model.keras"
    feature_path = artifact_dir / "selected_features.csv"
    params_path = artifact_dir / "best_hyperparameters.json"
    decision_path = artifact_dir / "decision_rule.json"

    for path in (
        model_path,
        feature_path,
        params_path,
        decision_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing frozen Part-3 LSTM artifact: {path}"
            )

    selected_features = pd.read_csv(feature_path)["Feature"].tolist()
    params = load_json(params_path)
    decision_rule = load_json(decision_path)

    model = keras.models.load_model(
        model_path,
        compile=False,
    )

    # Reconstruct exactly from TRAIN rather than unpickling the historical
    # pandas-containing preprocessor artifact. Fit scope is unchanged.
    preprocessor = lstm.DatasetPreprocessor(
        f"{dataset_name}_hybrid_lstm"
    )
    preprocessor.fit(train_df)

    train_selected, actual_features = lstm.transform_and_select(
        preprocessor=preprocessor,
        df=train_df,
        dataset_name=f"{dataset_name}_hybrid_lstm_train",
        expected_features=selected_features,
    )

    if actual_features != selected_features:
        raise RuntimeError(
            f"{dataset_name}: reconstructed LSTM feature schema/order "
            "does not match the trained artifact."
        )

    hybrid_artifact_dir = ARTIFACT_ROOT / dataset_name
    hybrid_artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(
        hybrid_artifact_dir / "lstm_preprocessor_reconstructed.joblib"
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "params": params,
        "threshold": float(decision_rule["threshold"]),
        "threshold_method": decision_rule.get("method"),
    }


def score_lstm_split(
    dataset_name: str,
    split_name: str,
    split_df: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    selected, actual_features = lstm.transform_and_select(
        preprocessor=bundle["preprocessor"],
        df=split_df,
        dataset_name=f"{dataset_name}_hybrid_lstm_{split_name}",
        expected_features=bundle["selected_features"],
    )

    if actual_features != bundle["selected_features"]:
        raise RuntimeError(
            f"{dataset_name}/{split_name}: LSTM feature schema mismatch."
        )

    X, y, metadata = lstm.build_sequences(
        selected,
        bundle["selected_features"],
        int(bundle["params"]["sequence_length"]),
        lstm.EVAL_SEQUENCE_STRIDE,
    )

    if len(y) != len(split_df):
        raise RuntimeError(
            f"{dataset_name}/{split_name}: expected one LSTM prediction "
            f"per flow ({len(split_df):,}); got {len(y):,}."
        )

    probability = lstm.predict_probabilities(
        bundle["model"],
        X,
        batch_size=int(bundle["params"]["batch_size"]),
    )

    output = metadata[
        [
            SOURCE_FILE_COL,
            TIMESTAMP_COL,
            LABEL_COL,
            ROW_KEY_COL,
        ]
    ].copy()
    output["LSTM_Probability"] = probability
    return output


# =============================================================================
# LITEMV — LOAD TRAINED WEIGHTS, RECONSTRUCT TRAIN-FITTED PREPROCESSING
# =============================================================================


def load_litemv_model(path: Path):
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


def load_litemv_bundle(
    dataset_name: str,
    train_df: pd.DataFrame,
) -> dict:
    artifact_dir = PART3_BASE_MODEL_ROOT / dataset_name / "LITEMV"

    model_path = artifact_dir / "litemv.keras"
    feature_path = artifact_dir / "selected_features.json"
    params_path = artifact_dir / "shared_hyperparameters.json"
    decision_path = artifact_dir / "decision_rule.json"

    for path in (
        model_path,
        feature_path,
        params_path,
        decision_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing frozen Part-3 LITEMV artifact: {path}"
            )

    with feature_path.open("r", encoding="utf-8") as handle:
        selected_features = json.load(handle)

    params = load_json(params_path)
    decision_rule = load_json(decision_path)

    expected_params = json.loads(json.dumps(litemv.SHARED_PARAMS))
    if params != expected_params:
        raise RuntimeError(
            f"{dataset_name}: stored LITEMV hyperparameters do not match "
            "the current final implementation."
        )

    model = load_litemv_model(model_path)

    preprocessor = litemv.DatasetPreprocessor(dataset_name)
    preprocessor.fit(train_df)

    (
        _train_selected,
        actual_features,
        _dropped,
        _manifest,
    ) = litemv.transform_and_select(
        preprocessor=preprocessor,
        df=train_df,
        dataset_name=f"{dataset_name}_hybrid_litemv_train",
        expected_features=selected_features,
    )

    if actual_features != selected_features:
        raise RuntimeError(
            f"{dataset_name}: reconstructed LITEMV feature schema/order "
            "does not match the trained artifact."
        )

    hybrid_artifact_dir = ARTIFACT_ROOT / dataset_name
    hybrid_artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(
        hybrid_artifact_dir / "litemv_preprocessor_reconstructed.joblib"
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "params": params,
        "threshold": float(decision_rule["threshold"]),
        "threshold_method": decision_rule.get("method"),
    }


def score_litemv_split(
    dataset_name: str,
    split_name: str,
    split_df: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    (
        selected,
        actual_features,
        _dropped,
        _manifest,
    ) = litemv.transform_and_select(
        preprocessor=bundle["preprocessor"],
        df=split_df,
        dataset_name=f"{dataset_name}_hybrid_litemv_{split_name}",
        expected_features=bundle["selected_features"],
    )

    if actual_features != bundle["selected_features"]:
        raise RuntimeError(
            f"{dataset_name}/{split_name}: LITEMV feature schema mismatch."
        )

    X, y, metadata = litemv.build_sequences(
        selected,
        bundle["selected_features"],
        int(bundle["params"]["sequence_length"]),
        litemv.EVAL_SEQUENCE_STRIDE,
    )

    if len(y) != len(split_df):
        raise RuntimeError(
            f"{dataset_name}/{split_name}: expected one LITEMV prediction "
            f"per flow ({len(split_df):,}); got {len(y):,}."
        )

    probability = litemv.predict_probabilities(
        bundle["model"],
        X,
        batch_size=int(bundle["params"]["batch_size"]),
    )

    output = metadata[
        [
            SOURCE_FILE_COL,
            TIMESTAMP_COL,
            LABEL_COL,
            ROW_KEY_COL,
        ]
    ].copy()
    output["LITEMV_Probability"] = probability
    return output


# =============================================================================
# RF ARBITER — EXACT FINAL ENDPOINT-AWARE RROLL5 REPRESENTATION
# =============================================================================


def prepare_rf_split(
    dataset_name: str,
    split_name: str,
    split_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Rebuild the final RF temporal representation independently inside one shared
    split so rolling history never crosses TRAIN/VALIDATION/TEST boundaries.
    """
    temporal = rf_temporal.build_temporal_features(
        split_df,
        dataset_name=f"{dataset_name}_hybrid_{split_name}_rf",
        require_label=True,
    )

    features = list(rf_temporal.final_feature_names())
    if len(features) != 32:
        raise RuntimeError(
            f"Final RF must expose exactly 32 features; got {len(features)}."
        )

    required = [
        rf_config.ROW_INDEX_COL,
        rf_config.SOURCE_FILE_COL,
        rf_config.LABEL_COL,
        *features,
    ]
    missing = [
        column
        for column in required
        if column not in temporal.columns
    ]
    if missing:
        raise RuntimeError(
            f"{dataset_name}/{split_name}: RF temporal frame missing {missing}."
        )

    frame = temporal[required].copy()

    for feature in features:
        frame[feature] = (
            pd.to_numeric(
                frame[feature],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    frame[rf_config.LABEL_COL] = pd.to_numeric(
        frame[rf_config.LABEL_COL],
        errors="raise",
    ).astype(int)

    expected = np.arange(
        len(split_df),
        dtype=np.int64,
    )
    actual = frame[
        rf_config.ROW_INDEX_COL
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(actual, expected):
        raise RuntimeError(
            f"{dataset_name}/{split_name}: RF temporal builder did not "
            "restore raw split row order."
        )

    return frame, features


def train_rf_arbiter(
    dataset_name: str,
    train_df: pd.DataFrame,
) -> dict:
    frame, features = prepare_rf_split(
        dataset_name,
        "train",
        train_df,
    )

    y_train = frame[
        rf_config.LABEL_COL
    ].astype(int)

    imputer = FoldMedianImputer(
        f"{dataset_name}_hybrid_rf"
    )
    X_train = imputer.fit_transform(
        frame[features]
    )

    class_weights = (
        rf_utils.compute_training_class_weights(
            y_train
        )
    )

    model = rf_utils.build_final_random_forest(
        class_weights=class_weights,
        random_state=rf_config.MODEL_RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    artifact_dir = ARTIFACT_ROOT / dataset_name
    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        artifact_dir / "rf_arbiter.joblib",
    )
    imputer.save(
        artifact_dir / "rf_arbiter_median_imputer.joblib"
    )
    pd.DataFrame(
        {"Feature": features}
    ).to_csv(
        artifact_dir / "rf_arbiter_selected_features.csv",
        index=False,
    )

    with (
        artifact_dir / "rf_arbiter_metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(
                {
                    "representation": "endpoint_aware_RROLL5_32_features",
                    "fit_scope": "shared_train_only",
                    "temporal_history_scope": (
                        "built independently per shared split"
                    ),
                    "hyperparameters": dict(
                        rf_config.FINAL_RF_PARAMS
                    ),
                    "random_state": int(
                        rf_config.MODEL_RANDOM_STATE
                    ),
                    "standalone_threshold": float(
                        rf_config.PREDICTION_THRESHOLD
                    ),
                    "class_weights": class_weights,
                    "validation_used_for_fit": False,
                    "test_used_for_fit": False,
                }
            ),
            handle,
            indent=2,
        )

    return {
        "model": model,
        "imputer": imputer,
        "features": features,
        "threshold": float(
            rf_config.PREDICTION_THRESHOLD
        ),
    }


def raw_split_sequence_keys(
    split_df: pd.DataFrame,
    rf_probability: np.ndarray,
) -> pd.DataFrame:
    """
    Convert row-order RF scores into the SourceFile-local chronological target key
    used by both sequence models.
    """
    if len(split_df) != len(rf_probability):
        raise RuntimeError(
            "RF probability count differs from raw split row count."
        )

    table = split_df[
        [
            SOURCE_FILE_COL,
            TIMESTAMP_COL,
            LABEL_COL,
        ]
    ].copy()

    table["RF_Probability"] = np.asarray(
        rf_probability,
        dtype=float,
    )
    table["__RawOrder"] = np.arange(
        len(table),
        dtype=np.int64,
    )
    table["__ParsedTimestamp"] = pd.to_datetime(
        table[TIMESTAMP_COL],
        errors="coerce",
        utc=True,
    )

    if table["__ParsedTimestamp"].isna().any():
        raise RuntimeError(
            "At least one Timestamp could not be parsed during RF alignment."
        )

    blocks = []

    for _source, group in table.groupby(
        SOURCE_FILE_COL,
        sort=True,
    ):
        group = group.sort_values(
            [
                "__ParsedTimestamp",
                "__RawOrder",
            ],
            kind="mergesort",
        ).copy()

        group[ROW_KEY_COL] = np.arange(
            len(group),
            dtype=np.int64,
        )
        blocks.append(group)

    return pd.concat(
        blocks,
        ignore_index=True,
    ).drop(
        columns=[
            "__RawOrder",
            "__ParsedTimestamp",
        ]
    )


def score_rf_split(
    dataset_name: str,
    split_name: str,
    split_df: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    frame, actual_features = prepare_rf_split(
        dataset_name,
        split_name,
        split_df,
    )

    if actual_features != bundle["features"]:
        raise RuntimeError(
            f"{dataset_name}/{split_name}: RF feature schema/order "
            "differs from TRAIN."
        )

    X = bundle["imputer"].transform(
        frame[bundle["features"]]
    )

    probability = bundle["model"].predict_proba(X)[:, 1]

    return raw_split_sequence_keys(
        split_df,
        probability,
    )


# =============================================================================
# ALIGN THREE MODEL OUTPUTS
# =============================================================================


def align_outputs(
    lstm_scores: pd.DataFrame,
    litemv_scores: pd.DataFrame,
    rf_scores: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        SOURCE_FILE_COL,
        ROW_KEY_COL,
    ]

    left = lstm_scores[
        keys
        + [
            LABEL_COL,
            TIMESTAMP_COL,
            "LSTM_Probability",
        ]
    ].copy()

    middle = litemv_scores[
        keys
        + [
            LABEL_COL,
            "LITEMV_Probability",
        ]
    ].rename(
        columns={
            LABEL_COL: "__LITEMV_Label"
        }
    )

    right = rf_scores[
        keys
        + [
            LABEL_COL,
            "RF_Probability",
        ]
    ].rename(
        columns={
            LABEL_COL: "__RF_Label"
        }
    )

    merged = (
        left.merge(
            middle,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            right,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    )

    expected = len(lstm_scores)
    if not (
        len(merged)
        == expected
        == len(litemv_scores)
        == len(rf_scores)
    ):
        raise RuntimeError(
            "Three-model alignment failed: "
            f"LSTM={len(lstm_scores)}, "
            f"LITEMV={len(litemv_scores)}, "
            f"RF={len(rf_scores)}, "
            f"merged={len(merged)}."
        )

    labels = merged[
        LABEL_COL
    ].to_numpy(dtype=int)

    if not np.array_equal(
        labels,
        merged["__LITEMV_Label"].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "LSTM and LITEMV labels disagree after alignment."
        )

    if not np.array_equal(
        labels,
        merged["__RF_Label"].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "LSTM and RF labels disagree after alignment."
        )

    return merged.drop(
        columns=[
            "__LITEMV_Label",
            "__RF_Label",
        ]
    )


# =============================================================================
# VALIDATION-ONLY ROUTING THRESHOLD SEARCH
# =============================================================================


def unique_finite(values) -> list[float]:
    return sorted(
        {
            float(value)
            for value in values
            if np.isfinite(value)
        }
    )


def positive_band_candidates(
    lstm_probability: np.ndarray,
    lstm_threshold: float,
) -> list[float]:
    probability = np.asarray(
        lstm_probability,
        dtype=float,
    )

    positive = probability[
        probability >= float(lstm_threshold)
    ]

    values = [float(lstm_threshold)]

    if len(positive):
        values.extend(
            np.quantile(
                positive,
                LSTM_POSITIVE_BAND_QUANTILES,
            ).tolist()
        )

    return unique_finite(
        value
        for value in values
        if value >= float(lstm_threshold) - EPS
    )


def confirm_candidates(
    probability: np.ndarray,
    routed_mask: np.ndarray,
    model_threshold: float,
) -> list[float]:
    probability = np.asarray(
        probability,
        dtype=float,
    )
    routed_mask = np.asarray(
        routed_mask,
        dtype=bool,
    )

    values = [
        0.0,
        float(model_threshold),
        1.0,
    ]

    routed = probability[routed_mask]

    if len(routed):
        values.extend(
            np.quantile(
                routed,
                SECONDARY_CONFIRM_QUANTILES,
            ).tolist()
        )

    return unique_finite(
        value
        for value in values
        if -EPS <= value <= 1.0 + EPS
    )


def zero_fp_rescue_threshold(
    y_true: np.ndarray,
    lstm_probability: np.ndarray,
    secondary_probability: np.ndarray,
    lstm_threshold: float,
    lower_bound: float,
) -> float:
    """
    Strictly above every benign-validation secondary score among LSTM-negative
    rows. A value > 1 is allowed and simply disables that side of rescue.
    """
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    lstm_probability = np.asarray(
        lstm_probability,
        dtype=float,
    )
    secondary_probability = np.asarray(
        secondary_probability,
        dtype=float,
    )

    lstm_negative = (
        lstm_probability
        < float(lstm_threshold)
    )

    benign_scores = secondary_probability[
        lstm_negative
        & (y_true == 0)
    ]

    if len(benign_scores) == 0:
        return float(lower_bound)

    strict_above = float(
        np.nextafter(
            np.max(benign_scores),
            np.inf,
        )
    )

    return float(
        max(
            float(lower_bound),
            strict_above,
        )
    )


# =============================================================================
# FIXED THREE-STAGE CASCADE
# =============================================================================


def apply_hybrid_cascade(
    lstm_probability: np.ndarray,
    litemv_probability: np.ndarray,
    rf_probability: np.ndarray,
    lstm_threshold: float,
    positive_filter_cutoff: float,
    litemv_confirm_threshold: float,
    rf_confirm_threshold: float,
    litemv_rescue_threshold: float,
    rf_rescue_threshold: float,
) -> dict:
    """
    Apply the fixed LSTM -> LITEMV -> RF architecture.
    """
    lp = np.asarray(
        lstm_probability,
        dtype=float,
    )
    vp = np.asarray(
        litemv_probability,
        dtype=float,
    )
    rp = np.asarray(
        rf_probability,
        dtype=float,
    )

    if not (
        len(lp)
        == len(vp)
        == len(rp)
    ):
        raise ValueError(
            "Component probability arrays have different lengths."
        )

    baseline = (
        lp >= float(lstm_threshold)
    ).astype(int)

    final = baseline.copy()

    # Only borderline / low-confidence LSTM positives are reconsidered.
    low_positive_route = (
        (baseline == 1)
        & (
            lp
            <= float(positive_filter_cutoff) + EPS
        )
    )

    # LITEMV is the first confirmation model.
    litemv_confirms = (
        low_positive_route
        & (
            vp
            >= float(litemv_confirm_threshold)
        )
    )

    # RF is consulted only when LITEMV does not confirm.
    rf_arbiter_route = (
        low_positive_route
        & (~litemv_confirms)
    )

    final[low_positive_route] = 0
    final[litemv_confirms] = 1

    rf_keeps_malicious = (
        rf_arbiter_route
        & (
            rp >= float(rf_confirm_threshold)
        )
    )
    final[rf_keeps_malicious] = 1

    # Very conservative LSTM-negative rescue.
    rescue_route = (
        (baseline == 0)
        & (
            vp >= float(litemv_rescue_threshold)
        )
        & (
            rp >= float(rf_rescue_threshold)
        )
    )
    final[rescue_route] = 1

    return {
        "prediction": final,
        "baseline": baseline,
        "low_positive_route": low_positive_route,
        "litemv_confirms": litemv_confirms,
        "rf_arbiter_route": rf_arbiter_route,
        "rf_keeps_malicious": rf_keeps_malicious,
        "rescue_route": rescue_route,
    }


def route_labels(applied: dict) -> np.ndarray:
    n = len(applied["prediction"])

    final = applied["prediction"]
    baseline = applied["baseline"]

    labels = np.full(
        n,
        "LSTM_DIRECT",
        dtype=object,
    )

    low = applied["low_positive_route"]
    litemv_confirms = applied[
        "litemv_confirms"
    ]
    rf_route = applied[
        "rf_arbiter_route"
    ]
    rescue = applied[
        "rescue_route"
    ]

    labels[
        low & litemv_confirms
    ] = "LITEMV_CONFIRMED_LSTM_POSITIVE"

    labels[
        rf_route
        & (final == 1)
    ] = "RF_ARBITRATED_KEEP_MALICIOUS"

    labels[
        rf_route
        & (final == 0)
    ] = "LITEMV_RF_FILTERED_LSTM_POSITIVE"

    labels[
        rescue
        & (baseline == 0)
        & (final == 1)
    ] = "JOINT_LITEMV_RF_RESCUE"

    return labels


def search_hybrid_rule(
    validation: pd.DataFrame,
    lstm_threshold: float,
    litemv_threshold: float,
    rf_threshold: float,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Architecture is fixed. Search ONLY its routing thresholds on VALIDATION.

    Accepted rule constraint:
      - strictly higher validation F1 than LSTM baseline
      - validation FPR no worse than LSTM baseline
      - validation FNR no worse than LSTM baseline
    """
    y = validation[
        LABEL_COL
    ].astype(int).to_numpy()

    lp = validation[
        "LSTM_Probability"
    ].to_numpy(dtype=float)

    vp = validation[
        "LITEMV_Probability"
    ].to_numpy(dtype=float)

    rp = validation[
        "RF_Probability"
    ].to_numpy(dtype=float)

    lstm_baseline_pred = (
        lp >= float(lstm_threshold)
    ).astype(int)

    baseline_metrics = metrics_from_predictions(
        y,
        lstm_baseline_pred,
    )

    positive_cutoffs = positive_band_candidates(
        lp,
        lstm_threshold,
    )

    baseline_positive = (
        lp >= float(lstm_threshold)
    )

    litemv_confirm_candidates = confirm_candidates(
        vp,
        baseline_positive,
        litemv_threshold,
    )

    rf_confirm_candidates = confirm_candidates(
        rp,
        baseline_positive,
        rf_threshold,
    )

    litemv_rescue_threshold = (
        zero_fp_rescue_threshold(
            y_true=y,
            lstm_probability=lp,
            secondary_probability=vp,
            lstm_threshold=lstm_threshold,
            lower_bound=litemv_threshold,
        )
    )

    rf_rescue_threshold = (
        zero_fp_rescue_threshold(
            y_true=y,
            lstm_probability=lp,
            secondary_probability=rp,
            lstm_threshold=lstm_threshold,
            lower_bound=rf_threshold,
        )
    )

    rows = []

    for positive_cutoff in positive_cutoffs:
        for litemv_confirm in litemv_confirm_candidates:
            for rf_confirm in rf_confirm_candidates:
                applied = apply_hybrid_cascade(
                    lstm_probability=lp,
                    litemv_probability=vp,
                    rf_probability=rp,
                    lstm_threshold=lstm_threshold,
                    positive_filter_cutoff=positive_cutoff,
                    litemv_confirm_threshold=litemv_confirm,
                    rf_confirm_threshold=rf_confirm,
                    litemv_rescue_threshold=litemv_rescue_threshold,
                    rf_rescue_threshold=rf_rescue_threshold,
                )

                prediction = applied["prediction"]

                metrics = metrics_from_predictions(
                    y,
                    prediction,
                )

                rows.append(
                    {
                        "positive_filter_cutoff": float(
                            positive_cutoff
                        ),
                        "litemv_confirm_threshold": float(
                            litemv_confirm
                        ),
                        "rf_confirm_threshold": float(
                            rf_confirm
                        ),
                        "litemv_rescue_threshold": float(
                            litemv_rescue_threshold
                        ),
                        "rf_rescue_threshold": float(
                            rf_rescue_threshold
                        ),
                        "low_positive_routed_rows": int(
                            applied[
                                "low_positive_route"
                            ].sum()
                        ),
                        "litemv_confirmed_rows": int(
                            applied[
                                "litemv_confirms"
                            ].sum()
                        ),
                        "rf_arbiter_rows": int(
                            applied[
                                "rf_arbiter_route"
                            ].sum()
                        ),
                        "rescue_routed_rows": int(
                            applied[
                                "rescue_route"
                            ].sum()
                        ),
                        "total_changed_rows": int(
                            np.sum(
                                prediction
                                != lstm_baseline_pred
                            )
                        ),
                        **metrics,
                    }
                )

    search = pd.DataFrame(rows)

    safe = search[
        (search["fpr"] <= baseline_metrics["fpr"] + EPS)
        & (
            search["fnr"]
            <= baseline_metrics["fnr"] + EPS
        )
        & (
            search["f1"]
            > baseline_metrics["f1"] + EPS
        )
    ].copy()

    if safe.empty:
        raise RuntimeError(
            "The fixed LSTM -> LITEMV -> RF cascade did not produce a "
            "validation-safe strict F1 improvement. Refusing to use TEST "
            "to choose or repair the final hybrid rule."
        )

    selected = (
        safe.sort_values(
            [
                "f1",
                "fpr",
                "fnr",
                "total_changed_rows",
                "rf_arbiter_rows",
            ],
            ascending=[
                False,
                True,
                True,
                True,
                True,
            ],
            kind="mergesort",
        )
        .iloc[0]
        .to_dict()
    )

    selected["selection_scope"] = "validation_only"
    selected["test_used_for_selection"] = False
    selected["architecture"] = "LSTM_LITEMV_RF_FIXED_CASCADE"

    return search, selected, baseline_metrics


# =============================================================================
# DATASET PIPELINE
# =============================================================================


def run_dataset(
    dataset_name: str,
) -> dict:
    print("\n" + "=" * 110)
    print(
        f"FINAL HYBRID PIPELINE | {dataset_name.upper()} | "
        "LSTM -> LITEMV -> RF"
    )
    print("=" * 110)

    result_dir = RESULT_ROOT / dataset_name
    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. Shared persisted split
    # -------------------------------------------------------------------------
    splits, _manifest, split_metadata = (
        load_shared_splits(dataset_name)
    )

    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    print(
        f"Rows | train={len(train_df):,} | "
        f"validation={len(validation_df):,} | "
        f"test={len(test_df):,} | "
        f"split_mode={split_metadata.get('mode')}"
    )

    # -------------------------------------------------------------------------
    # 2. Component models
    # -------------------------------------------------------------------------
    print("[1/3] Loading final LSTM and reconstructing TRAIN preprocessing...")
    lstm_bundle = load_lstm_bundle(
        dataset_name,
        train_df,
    )

    print("[2/3] Loading final LITEMV and reconstructing TRAIN preprocessing...")
    litemv_bundle = load_litemv_bundle(
        dataset_name,
        train_df,
    )

    print("[3/3] Fitting final RF arbiter on shared TRAIN only...")
    rf_bundle = train_rf_arbiter(
        dataset_name,
        train_df,
    )

    # -------------------------------------------------------------------------
    # 3. VALIDATION component probabilities
    # -------------------------------------------------------------------------
    print("Scoring VALIDATION...")

    validation = align_outputs(
        score_lstm_split(
            dataset_name,
            "validation",
            validation_df,
            lstm_bundle,
        ),
        score_litemv_split(
            dataset_name,
            "validation",
            validation_df,
            litemv_bundle,
        ),
        score_rf_split(
            dataset_name,
            "validation",
            validation_df,
            rf_bundle,
        ),
    )

    validation.to_csv(
        result_dir / "validation_component_probabilities.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 4. VALIDATION-only hybrid threshold calibration
    # -------------------------------------------------------------------------
    print("Calibrating fixed cascade on VALIDATION only...")

    search, selected_rule, baseline_validation_metrics = (
        search_hybrid_rule(
            validation=validation,
            lstm_threshold=lstm_bundle["threshold"],
            litemv_threshold=litemv_bundle["threshold"],
            rf_threshold=rf_bundle["threshold"],
        )
    )

    search.to_csv(
        result_dir / "validation_hybrid_rule_search.csv",
        index=False,
    )

    selected_rule.update(
        {
            "dataset": dataset_name,
            "split_mode": split_metadata.get("mode"),
            "lstm_threshold": float(
                lstm_bundle["threshold"]
            ),
            "litemv_standalone_threshold": float(
                litemv_bundle["threshold"]
            ),
            "rf_standalone_threshold": float(
                rf_bundle["threshold"]
            ),
            "lstm_threshold_method": (
                lstm_bundle["threshold_method"]
            ),
            "litemv_threshold_method": (
                litemv_bundle["threshold_method"]
            ),
            "validation_baseline_lstm_metrics": (
                baseline_validation_metrics
            ),
            "validation_rows": int(
                len(validation)
            ),
            "test_used_for_rule_selection": False,
            "architecture_selection_performed": False,
            "architecture_fixed_before_test": True,
        }
    )

    # Freeze rule BEFORE any test scoring.
    with (
        artifact_dir / "hybrid_rule.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(selected_rule),
            handle,
            indent=2,
        )

    validation_applied = apply_hybrid_cascade(
        lstm_probability=validation[
            "LSTM_Probability"
        ].to_numpy(dtype=float),
        litemv_probability=validation[
            "LITEMV_Probability"
        ].to_numpy(dtype=float),
        rf_probability=validation[
            "RF_Probability"
        ].to_numpy(dtype=float),
        lstm_threshold=float(
            selected_rule["lstm_threshold"]
        ),
        positive_filter_cutoff=float(
            selected_rule[
                "positive_filter_cutoff"
            ]
        ),
        litemv_confirm_threshold=float(
            selected_rule[
                "litemv_confirm_threshold"
            ]
        ),
        rf_confirm_threshold=float(
            selected_rule[
                "rf_confirm_threshold"
            ]
        ),
        litemv_rescue_threshold=float(
            selected_rule[
                "litemv_rescue_threshold"
            ]
        ),
        rf_rescue_threshold=float(
            selected_rule[
                "rf_rescue_threshold"
            ]
        ),
    )

    y_validation = validation[
        LABEL_COL
    ].astype(int).to_numpy()

    validation_metrics = metrics_from_predictions(
        y_validation,
        validation_applied["prediction"],
    )

    validation_record = {
        "Dataset": dataset_name,
        "Model": "LSTM_LITEMV_RF",
        "Rows": int(len(validation)),
        **validation_metrics,
    }

    pd.DataFrame(
        [validation_record]
    ).to_csv(
        result_dir / "validation_metrics.csv",
        index=False,
    )

    print(
        "VALIDATION | "
        f"F1={validation_metrics['f1']:.6f} | "
        f"FPR={validation_metrics['fpr']:.6f} | "
        f"FNR={validation_metrics['fnr']:.6f} | "
        f"FP={validation_metrics['fp']} | "
        f"FN={validation_metrics['fn']}"
    )

    # -------------------------------------------------------------------------
    # 5. TEST — reporting only after rule is frozen
    # -------------------------------------------------------------------------
    print("Scoring TEST with frozen hybrid rule...")

    test = align_outputs(
        score_lstm_split(
            dataset_name,
            "test",
            test_df,
            lstm_bundle,
        ),
        score_litemv_split(
            dataset_name,
            "test",
            test_df,
            litemv_bundle,
        ),
        score_rf_split(
            dataset_name,
            "test",
            test_df,
            rf_bundle,
        ),
    )

    applied = apply_hybrid_cascade(
        lstm_probability=test[
            "LSTM_Probability"
        ].to_numpy(dtype=float),
        litemv_probability=test[
            "LITEMV_Probability"
        ].to_numpy(dtype=float),
        rf_probability=test[
            "RF_Probability"
        ].to_numpy(dtype=float),
        lstm_threshold=float(
            selected_rule["lstm_threshold"]
        ),
        positive_filter_cutoff=float(
            selected_rule[
                "positive_filter_cutoff"
            ]
        ),
        litemv_confirm_threshold=float(
            selected_rule[
                "litemv_confirm_threshold"
            ]
        ),
        rf_confirm_threshold=float(
            selected_rule[
                "rf_confirm_threshold"
            ]
        ),
        litemv_rescue_threshold=float(
            selected_rule[
                "litemv_rescue_threshold"
            ]
        ),
        rf_rescue_threshold=float(
            selected_rule[
                "rf_rescue_threshold"
            ]
        ),
    )

    y_test = test[
        LABEL_COL
    ].astype(int).to_numpy()

    hybrid_prediction = applied[
        "prediction"
    ]

    test_metrics = metrics_from_predictions(
        y_test,
        hybrid_prediction,
    )

    lstm_baseline_prediction = (
        test["LSTM_Probability"].to_numpy(dtype=float)
        >= float(lstm_bundle["threshold"])
    ).astype(int)

    lstm_test_metrics = metrics_from_predictions(
        y_test,
        lstm_baseline_prediction,
    )

    prediction_table = test.copy()
    prediction_table["LSTM_Baseline_Label"] = (
        lstm_baseline_prediction
    )
    prediction_table["Predicted_Label"] = (
        hybrid_prediction
    )
    prediction_table["DecisionRoute"] = (
        route_labels(applied)
    )

    prediction_table.to_csv(
        result_dir / "test_predictions.csv",
        index=False,
    )

    route_summary = (
        prediction_table[
            "DecisionRoute"
        ]
        .value_counts(dropna=False)
        .rename_axis("DecisionRoute")
        .reset_index(name="Rows")
    )
    route_summary.to_csv(
        result_dir / "test_route_summary.csv",
        index=False,
    )

    hybrid_test_record = {
        "Dataset": dataset_name,
        "Model": "LSTM_LITEMV_RF",
        "Rows": int(len(test)),
        **test_metrics,
    }

    lstm_test_record = {
        "Dataset": dataset_name,
        "Model": "LSTM",
        "Rows": int(len(test)),
        **lstm_test_metrics,
    }

    comparison_record = {
        "Dataset": dataset_name,
        "Baseline_F1": float(
            lstm_test_metrics["f1"]
        ),
        "Hybrid_F1": float(
            test_metrics["f1"]
        ),
        "Delta_F1": float(
            test_metrics["f1"]
            - lstm_test_metrics["f1"]
        ),
        "Baseline_FPR": float(
            lstm_test_metrics["fpr"]
        ),
        "Hybrid_FPR": float(
            test_metrics["fpr"]
        ),
        "Delta_FPR": float(
            test_metrics["fpr"]
            - lstm_test_metrics["fpr"]
        ),
        "Baseline_FNR": float(
            lstm_test_metrics["fnr"]
        ),
        "Hybrid_FNR": float(
            test_metrics["fnr"]
        ),
        "Delta_FNR": float(
            test_metrics["fnr"]
            - lstm_test_metrics["fnr"]
        ),
        "Baseline_FP": int(
            lstm_test_metrics["fp"]
        ),
        "Hybrid_FP": int(
            test_metrics["fp"]
        ),
        "Baseline_FN": int(
            lstm_test_metrics["fn"]
        ),
        "Hybrid_FN": int(
            test_metrics["fn"]
        ),
        "Corrected_LSTM_False_Positives": int(
            np.sum(
                (y_test == 0)
                & (lstm_baseline_prediction == 1)
                & (hybrid_prediction == 0)
            )
        ),
        "Introduced_New_False_Positives": int(
            np.sum(
                (y_test == 0)
                & (lstm_baseline_prediction == 0)
                & (hybrid_prediction == 1)
            )
        ),
        "Rescued_LSTM_False_Negatives": int(
            np.sum(
                (y_test == 1)
                & (lstm_baseline_prediction == 0)
                & (hybrid_prediction == 1)
            )
        ),
        "Introduced_New_False_Negatives": int(
            np.sum(
                (y_test == 1)
                & (lstm_baseline_prediction == 1)
                & (hybrid_prediction == 0)
            )
        ),
        "Total_Changed_Decisions": int(
            np.sum(
                hybrid_prediction
                != lstm_baseline_prediction
            )
        ),
    }

    pd.DataFrame(
        [
            lstm_test_record,
            hybrid_test_record,
        ]
    ).to_csv(
        result_dir / "test_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        [comparison_record]
    ).to_csv(
        result_dir / "hybrid_vs_lstm.csv",
        index=False,
    )

    save_confusion_matrix_csv(
        y_true=y_test,
        predictions=hybrid_prediction,
        path=result_dir / "confusion_matrix.csv",
    )

    # -------------------------------------------------------------------------
    # 6. Final manifest
    # -------------------------------------------------------------------------
    manifest = {
        "model": "LSTM -> LITEMV -> RF gated cascade",
        "architecture": "fixed_three_stage_cascade",
        "primary_detector": "LSTM",
        "secondary_confirmation": "LITEMV",
        "disagreement_arbiter": "Random Forest",
        "negative_rescue": (
            "requires joint LITEMV + RF strong evidence"
        ),
        "dataset": dataset_name,
        "split_mode": split_metadata.get("mode"),
        "base_model_sources": {
            "lstm": str(LSTM_FILE),
            "litemv": str(LITEMV_FILE),
            "rf_config": str(RF_CONFIG_FILE),
            "rf_temporal": str(RF_TEMPORAL_FILE),
            "rf_utils": str(RF_UTILS_FILE),
        },
        "base_model_artifacts": {
            "lstm": str(
                PART3_BASE_MODEL_ROOT
                / dataset_name
                / "LSTM"
            ),
            "litemv": str(
                PART3_BASE_MODEL_ROOT
                / dataset_name
                / "LITEMV"
            ),
        },
        "rf_fit_scope": "shared_train_only",
        "hybrid_rule_scope": "validation_only",
        "architecture_selection_scope": (
            "none; architecture fixed before test"
        ),
        "test_role": "final_reporting_only",
        "test_used_for_training": False,
        "test_used_for_routing_threshold_selection": False,
        "test_used_for_architecture_selection": False,
        "simple_majority_vote": False,
        "selected_rule_file": str(
            artifact_dir / "hybrid_rule.json"
        ),
    }

    with (
        artifact_dir / "HYBRID_MODEL_MANIFEST.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(manifest),
            handle,
            indent=2,
        )

    print(
        "TEST       | "
        f"F1={test_metrics['f1']:.6f} | "
        f"FPR={test_metrics['fpr']:.6f} | "
        f"FNR={test_metrics['fnr']:.6f} | "
        f"FP={test_metrics['fp']} | "
        f"FN={test_metrics['fn']}"
    )
    print(
        "vs LSTM    | "
        f"DeltaF1={comparison_record['Delta_F1']:+.6f} | "
        f"DeltaFPR={comparison_record['Delta_FPR']:+.6f} | "
        f"DeltaFNR={comparison_record['Delta_FNR']:+.6f}"
    )

    # Explicitly release TensorFlow graphs before the next dataset.
    keras.backend.clear_session()

    return {
        **hybrid_test_record,
        **{
            "Baseline_LSTM_F1": comparison_record["Baseline_F1"],
            "Delta_F1": comparison_record["Delta_F1"],
            "Baseline_LSTM_FPR": comparison_record["Baseline_FPR"],
            "Delta_FPR": comparison_record["Delta_FPR"],
            "Baseline_LSTM_FNR": comparison_record["Baseline_FNR"],
            "Delta_FNR": comparison_record["Delta_FNR"],
        },
    }


# =============================================================================
# MAIN
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Final fixed LSTM -> LITEMV -> RF gated cascade. "
            "Routing thresholds are calibrated on validation only."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help=(
            "Run one benchmark dataset. Default: dataset2 and dataset3."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = (
        [args.dataset]
        if args.dataset is not None
        else list(DATASETS)
    )

    rows = [
        run_dataset(dataset_name)
        for dataset_name in datasets
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(
        RESULT_ROOT / "hybrid_test_summary.csv",
        index=False,
    )

    methodology = {
        "final_model": "LSTM -> LITEMV -> RF gated cascade",
        "architecture_fixed_before_test": True,
        "architecture_compared_on_test": False,
        "routing_threshold_calibration": "validation_only_per_dataset",
        "rf_fit_scope": "shared_train_only",
        "shared_split_reused": True,
        "test_used_for_selection": False,
        "simple_majority_vote": False,
    }

    with (
        RESULT_ROOT / "methodology.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            methodology,
            handle,
            indent=2,
        )

    print("\n" + "=" * 110)
    print("FINAL HYBRID SUMMARY")
    print("=" * 110)
    print(summary.to_string(index=False))
    print(f"\nResults:   {RESULT_ROOT}")
    print(f"Artifacts: {ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()
