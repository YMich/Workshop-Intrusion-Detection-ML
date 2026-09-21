from __future__ import annotations

"""
Part 3 hybrid ablation tests:
    1) LSTM baseline
    2) LITEMV baseline
    3) RF baseline (Part-3 train-only fit)
    4) LSTM + RF
    5) LSTM + LITEMV
    6) LSTM + LITEMV + RF

Design goals
------------
- Use only the exact final implementations under src/models_optimized_final:
    * LSTM: src/models_optimized_final/LSTM/lstm.py
      (O1 brittle-feature pruning + O2 malicious temporal/rate augmentation)
    * RF: src/models_optimized_final/RF/random_forest_training.py
    * LITEMV: src/models_optimized_final/LITEMV/litemv.py
      (final accepted O1 57 -> 43 environment-hardened model)
- Reuse the exact persisted source-aware train/validation/test split.
- Consume the frozen final LSTM/LITEMV validation/test prediction CSVs directly;
  never re-unpickle their preprocessors or recompute their predictions in Part 3.
- Never use TEST to select routing rules or thresholds.
- Tune hybrid routing on VALIDATION only.
- Retain a hybrid rule only when validation F1 strictly improves AND
  validation FPR does not worsen AND validation FNR does not worsen.
- Select the final architecture using VALIDATION only.
- Evaluate the frozen candidates/final architecture on TEST for reporting.

The triple model is intentionally asymmetric:
- LSTM is the primary detector.
- Only low-confidence LSTM positives enter the positive-correction path.
- LITEMV is the first temporal second opinion.
- If LITEMV rejects a low-confidence LSTM positive, RF acts as arbiter.
- An LSTM-negative row can be rescued only when BOTH LITEMV and RF give
  exceptionally strong malicious evidence. Rescue cutoffs are calibrated so
  they introduce zero new validation false positives before rule selection.

This tests the error-analysis hypothesis rather than simple majority voting.
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
# PROJECT / MODULE DISCOVERY
# =============================================================================


def discover_project_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "data" / "ingested").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate project root. Expected a parent containing src/ and data/ingested/."
    )


def require_exact_file(project_root: Path, relative: str) -> Path:
    path = project_root / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required current model source: {path}\n"
            "Part 3 must run against the exact accepted Part-2 implementations."
        )
    return path


PROJECT_ROOT = discover_project_root()

LSTM_FILE = require_exact_file(
    PROJECT_ROOT,
    "src/dataset1_appendix_experiments/models/LSTM/lstm.py",
)
RF_FILE = require_exact_file(
    PROJECT_ROOT,
    "src/dataset1_appendix_experiments/models/RF/random_forest_training.py",
)
RF_CONFIG_FILE = require_exact_file(
    PROJECT_ROOT,
    "src/dataset1_appendix_experiments/models/RF/random_forest_config.py",
)
RF_PREPROCESSING_FILE = require_exact_file(
    PROJECT_ROOT,
    "src/dataset1_appendix_experiments/models/RF/random_forest_preprocessing.py",
)
RF_TEMPORAL_FILE = require_exact_file(
    PROJECT_ROOT,
    "src/dataset1_appendix_experiments/models/RF/random_forest_temporal.py",
)
RF_UTILS_FILE = require_exact_file(
    PROJECT_ROOT,
    "src/dataset1_appendix_experiments/models/RF/random_forest_utils.py",
)

LITEMV_FILE = require_exact_file(
    PROJECT_ROOT,
    "src/dataset1_appendix_experiments/models/LITEMV/litemv.py",
)

LSTM_DIR = LSTM_FILE.parent
RF_DIR = RF_FILE.parent
LITEMV_DIR = LITEMV_FILE.parent

# Exact current model directories first.  The LITEMV module itself also resolves
# the shared splitter, but the accepted final LSTM splitter should remain visible.
for path in [
    PROJECT_ROOT / "src" / "feature_engineering",
    PROJECT_ROOT / "src" / "preprocessing",
    LITEMV_DIR,
    RF_DIR,
    LSTM_DIR,
]:
    text = str(path)
    while text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

import lstm  # noqa: E402
import litemv  # noqa: E402
import random_forest_config as rf_config  # noqa: E402
import random_forest_training as rf_training  # noqa: E402
import random_forest_temporal as rf_temporal  # noqa: E402
import random_forest_utils as rf_utils  # noqa: E402
from random_forest_preprocessing import FoldMedianImputer  # noqa: E402


def _assert_exact_import(module, expected: Path, label: str) -> None:
    loaded = Path(module.__file__).resolve()
    expected = expected.resolve()
    if loaded != expected:
        raise RuntimeError(
            f"Python imported the wrong {label} module.\n"
            f"Expected: {expected}\n"
            f"Loaded:   {loaded}"
        )


_assert_exact_import(lstm, LSTM_FILE, "LSTM")
_assert_exact_import(litemv, LITEMV_FILE, "LITEMV")
_assert_exact_import(rf_training, RF_FILE, "RF training")
_assert_exact_import(rf_config, RF_CONFIG_FILE, "RF config")
_assert_exact_import(rf_temporal, RF_TEMPORAL_FILE, "RF temporal")
_assert_exact_import(rf_utils, RF_UTILS_FILE, "RF utils")

for _required_rf_api, _module in (
    ("FINAL_RF_PARAMS", rf_config),
    ("MODEL_RANDOM_STATE", rf_config),
    ("PREDICTION_THRESHOLD", rf_config),
    ("LABEL_COL", rf_config),
    ("SOURCE_FILE_COL", rf_config),
    ("ROW_INDEX_COL", rf_config),
    ("build_temporal_features", rf_temporal),
    ("final_feature_names", rf_temporal),
    ("build_final_random_forest", rf_utils),
    ("compute_training_class_weights", rf_utils),
):
    if not hasattr(_module, _required_rf_api):
        raise RuntimeError(
            f"Final RF module is missing required API: {_required_rf_api}"
        )

# Fail fast if the LSTM is not the accepted O1+O2 final version.
if getattr(lstm, "FINAL_NAMESPACE", None) != "dataset1_appendix":
    raise RuntimeError(
        "Loaded Dataset-1 appendix LSTM does not declare FINAL_NAMESPACE='dataset1_appendix'."
    )
if int(getattr(lstm, "AUGMENTATION_COPIES_PER_MALICIOUS_SOURCE", -1)) != 2:
    raise RuntimeError(
        "Loaded LSTM is not the accepted O2 temporal/rate augmentation implementation."
    )

# Fail fast if LITEMV is not the accepted final O1 hardening implementation.
if int(getattr(litemv, "BASELINE_EXPECTED_FEATURE_COUNT", -1)) != 57:
    raise RuntimeError("Expected final LITEMV baseline feature count = 57.")
if int(getattr(litemv, "EXPECTED_FEATURE_COUNT", -1)) != 43:
    raise RuntimeError("Expected final LITEMV feature count = 43.")
if int(getattr(litemv, "O1_EXPECTED_DROP_COUNT", -1)) != 14:
    raise RuntimeError("Expected final LITEMV O1 drop count = 14.")


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

DATASETS = ("dataset1",)
LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
TIMESTAMP_COL = "Timestamp"
ROW_KEY_COL = "TargetRowInSplitSourceSegment"

FINAL_LSTM_ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts" / "dataset1_appendix" / "models" / "LSTM"
)
FINAL_LSTM_RESULT_ROOT = (
    PROJECT_ROOT / "results" / "dataset1_appendix" / "models" / "LSTM"
)

FINAL_LITEMV_ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts" / "dataset1_appendix" / "models" / "LITEMV"
)
FINAL_LITEMV_RESULT_ROOT = (
    PROJECT_ROOT / "results" / "dataset1_appendix" / "models" / "LITEMV"
)

if Path(lstm.ARTIFACT_ROOT).resolve() != FINAL_LSTM_ARTIFACT_ROOT.resolve():
    raise RuntimeError(
        f"Final LSTM artifact root mismatch: {lstm.ARTIFACT_ROOT}"
    )
if Path(lstm.RESULT_ROOT).resolve() != FINAL_LSTM_RESULT_ROOT.resolve():
    raise RuntimeError(
        f"Final LSTM result root mismatch: {lstm.RESULT_ROOT}"
    )
if Path(litemv.ARTIFACT_ROOT).resolve() != FINAL_LITEMV_ARTIFACT_ROOT.resolve():
    raise RuntimeError(
        "Final LITEMV artifact root mismatch.\n"
        f"Expected: {FINAL_LITEMV_ARTIFACT_ROOT}\n"
        f"Loaded:   {litemv.ARTIFACT_ROOT}"
    )
if Path(litemv.RESULT_ROOT).resolve() != FINAL_LITEMV_RESULT_ROOT.resolve():
    raise RuntimeError(
        "Final LITEMV result root mismatch.\n"
        f"Expected: {FINAL_LITEMV_RESULT_ROOT}\n"
        f"Loaded:   {litemv.RESULT_ROOT}"
    )

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "dataset1_appendix"
    / "part3_hybrid"
)
ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "dataset1_appendix"
    / "part3_hybrid"
)

# Quantile grids are derived only from VALIDATION scores.
LSTM_POSITIVE_BAND_QUANTILES = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00
)
SECONDARY_CONFIRM_QUANTILES = (
    0.00, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 1.00
)

EPS = 1e-12


# =============================================================================
# METRICS
# =============================================================================


def metrics_from_predictions(y_true, predictions) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
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
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(
            fbeta_score(y_true, predictions, beta=2, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def correction_summary(
    y_true: np.ndarray,
    lstm_pred: np.ndarray,
    hybrid_pred: np.ndarray,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(lstm_pred, dtype=int)
    hybrid = np.asarray(hybrid_pred, dtype=int)

    return {
        "Corrected_LSTM_False_Positives": int(
            np.sum((y == 0) & (base == 1) & (hybrid == 0))
        ),
        "Introduced_New_False_Positives": int(
            np.sum((y == 0) & (base == 0) & (hybrid == 1))
        ),
        "Rescued_LSTM_False_Negatives": int(
            np.sum((y == 1) & (base == 0) & (hybrid == 1))
        ),
        "Introduced_New_False_Negatives": int(
            np.sum((y == 1) & (base == 1) & (hybrid == 0))
        ),
        "Total_Changed_Decisions": int(np.sum(base != hybrid)),
    }


# =============================================================================
# FROZEN LSTM / LITEMV PREDICTION EVIDENCE
# =============================================================================


def _load_decision_rule(
    artifact_root: Path,
    dataset_name: str,
    model_label: str,
) -> dict:
    path = Path(artifact_root) / dataset_name / "decision_rule.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing final {model_label} decision rule: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        rule = json.load(handle)
    if "threshold" not in rule:
        raise RuntimeError(
            f"{model_label} decision rule has no threshold: {path}"
        )
    return rule


def _load_saved_prediction_table(
    result_root: Path,
    dataset_name: str,
    split_name: str,
    probability_output_name: str,
    model_label: str,
) -> pd.DataFrame:
    path = Path(result_root) / dataset_name / f"{split_name}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen {model_label} {split_name} predictions: {path}\n"
            f"Run the final {model_label} training/evaluation first."
        )

    table = pd.read_csv(path, low_memory=False)

    required = {
        SOURCE_FILE_COL,
        TIMESTAMP_COL,
        LABEL_COL,
        ROW_KEY_COL,
        "MaliciousProbability",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(
            f"{model_label} {split_name} prediction file is missing "
            f"required columns: {missing}\nFile: {path}"
        )

    out = table[
        [
            SOURCE_FILE_COL,
            TIMESTAMP_COL,
            LABEL_COL,
            ROW_KEY_COL,
            "MaliciousProbability",
        ]
    ].copy()

    out[LABEL_COL] = pd.to_numeric(
        out[LABEL_COL], errors="raise"
    ).astype(int)
    out[ROW_KEY_COL] = pd.to_numeric(
        out[ROW_KEY_COL], errors="raise"
    ).astype(int)
    out[probability_output_name] = pd.to_numeric(
        out.pop("MaliciousProbability"),
        errors="raise",
    ).astype(float)

    if out[[SOURCE_FILE_COL, ROW_KEY_COL]].duplicated().any():
        duplicates = out.loc[
            out[[SOURCE_FILE_COL, ROW_KEY_COL]].duplicated(keep=False),
            [SOURCE_FILE_COL, ROW_KEY_COL],
        ].head(10)
        raise RuntimeError(
            f"{model_label} {split_name} predictions contain duplicate "
            "alignment keys.\n"
            + duplicates.to_string(index=False)
        )

    return out


def _verify_prediction_threshold_column(
    result_root: Path,
    dataset_name: str,
    split_name: str,
    threshold: float,
    model_label: str,
) -> None:
    path = Path(result_root) / dataset_name / f"{split_name}_predictions.csv"
    table = pd.read_csv(path, low_memory=False, nrows=10)
    column = "ValidationCalibratedThreshold"
    if column not in table.columns or table.empty:
        return

    values = pd.to_numeric(table[column], errors="coerce").dropna().unique()
    if len(values) == 0:
        return
    if len(values) != 1 or not np.isclose(
        float(values[0]),
        float(threshold),
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"{model_label} {split_name} persisted threshold does not match "
            f"decision_rule.json. rule={threshold}, file_values={values.tolist()}"
        )


def load_lstm_prediction_evidence(
    dataset_name: str,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    rule = _load_decision_rule(
        FINAL_LSTM_ARTIFACT_ROOT,
        dataset_name,
        "LSTM",
    )
    threshold = float(rule["threshold"])

    validation = _load_saved_prediction_table(
        FINAL_LSTM_RESULT_ROOT,
        dataset_name,
        "validation",
        "LSTM_Probability",
        "LSTM",
    )
    test = _load_saved_prediction_table(
        FINAL_LSTM_RESULT_ROOT,
        dataset_name,
        "test",
        "LSTM_Probability",
        "LSTM",
    )

    _verify_prediction_threshold_column(
        FINAL_LSTM_RESULT_ROOT,
        dataset_name,
        "validation",
        threshold,
        "LSTM",
    )
    _verify_prediction_threshold_column(
        FINAL_LSTM_RESULT_ROOT,
        dataset_name,
        "test",
        threshold,
        "LSTM",
    )

    return threshold, validation, test


def load_litemv_prediction_evidence(
    dataset_name: str,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    rule = _load_decision_rule(
        FINAL_LITEMV_ARTIFACT_ROOT,
        dataset_name,
        "LITEMV",
    )
    threshold = float(rule["threshold"])

    validation = _load_saved_prediction_table(
        FINAL_LITEMV_RESULT_ROOT,
        dataset_name,
        "validation",
        "LITEMV_Probability",
        "LITEMV",
    )
    test = _load_saved_prediction_table(
        FINAL_LITEMV_RESULT_ROOT,
        dataset_name,
        "test",
        "LITEMV_Probability",
        "LITEMV",
    )

    _verify_prediction_threshold_column(
        FINAL_LITEMV_RESULT_ROOT,
        dataset_name,
        "validation",
        threshold,
        "LITEMV",
    )
    _verify_prediction_threshold_column(
        FINAL_LITEMV_RESULT_ROOT,
        dataset_name,
        "test",
        threshold,
        "LITEMV",
    )

    return threshold, validation, test


def verify_final_lstm_consistency(
    dataset_name: str,
    threshold: float,
    test_scores: pd.DataFrame,
) -> dict:
    path = FINAL_LSTM_RESULT_ROOT / dataset_name / "test_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing final reported LSTM metrics: {path}"
        )

    table = pd.read_csv(path)
    if len(table) != 1:
        raise RuntimeError(
            f"Expected one row in {path}, found {len(table)}."
        )
    reported = table.iloc[0].to_dict()

    if "threshold" in reported and not pd.isna(reported["threshold"]):
        reported_threshold = float(reported["threshold"])
        if not np.isclose(
            threshold, reported_threshold, rtol=0.0, atol=1e-10
        ):
            raise RuntimeError(
                f"{dataset_name}: LSTM threshold mismatch. "
                f"artifact={threshold:.12f}, result={reported_threshold:.12f}"
            )

    y = test_scores[LABEL_COL].astype(int).to_numpy()
    p = test_scores["LSTM_Probability"].to_numpy(dtype=float)
    pred = (p >= threshold).astype(int)
    reproduced = metrics_from_predictions(y, pred)

    if "TestRows" in reported and not pd.isna(reported["TestRows"]):
        if len(y) != int(reported["TestRows"]):
            raise RuntimeError(
                f"{dataset_name}: LSTM test row mismatch."
            )

    for key in ("tn", "fp", "fn", "tp"):
        if key in reported and not pd.isna(reported[key]):
            if reproduced[key] != int(reported[key]):
                raise RuntimeError(
                    f"{dataset_name}: LSTM {key} mismatch: "
                    f"reproduced={reproduced[key]}, "
                    f"reported={int(reported[key])}"
                )

    for key in ("accuracy", "precision", "recall_tpr", "fpr", "f1"):
        if key in reported and not pd.isna(reported[key]):
            if not np.isclose(
                reproduced[key],
                float(reported[key]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"{dataset_name}: LSTM {key} mismatch: "
                    f"reproduced={reproduced[key]:.12f}, "
                    f"reported={float(reported[key]):.12f}"
                )

    print(
        f"[{dataset_name}] frozen final LSTM predictions verified | "
        f"F1={reproduced['f1']:.6f} | FP={reproduced['fp']} | "
        f"FN={reproduced['fn']}"
    )
    return reproduced


def verify_final_litemv_consistency(
    dataset_name: str,
    threshold: float,
    test_scores: pd.DataFrame,
) -> dict:
    path = FINAL_LITEMV_RESULT_ROOT / dataset_name / "test_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing final reported LITEMV metrics: {path}"
        )

    table = pd.read_csv(path)
    if len(table) != 1:
        raise RuntimeError(
            f"Expected one row in {path}, found {len(table)}."
        )
    reported = table.iloc[0].to_dict()

    if "threshold" in reported and not pd.isna(reported["threshold"]):
        reported_threshold = float(reported["threshold"])
        if not np.isclose(
            threshold,
            reported_threshold,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"{dataset_name}: LITEMV threshold mismatch. "
                f"artifact={threshold:.12f}, result={reported_threshold:.12f}"
            )

    y = test_scores[LABEL_COL].astype(int).to_numpy()
    p = test_scores["LITEMV_Probability"].to_numpy(dtype=float)
    pred = (p >= threshold).astype(int)
    reproduced = metrics_from_predictions(y, pred)

    for key in ("tn", "fp", "fn", "tp"):
        if key in reported and not pd.isna(reported[key]):
            if reproduced[key] != int(reported[key]):
                raise RuntimeError(
                    f"{dataset_name}: LITEMV {key} mismatch: "
                    f"reproduced={reproduced[key]}, "
                    f"reported={int(reported[key])}"
                )

    for key in ("precision", "recall_tpr", "fpr", "f1"):
        if key in reported and not pd.isna(reported[key]):
            if not np.isclose(
                reproduced[key],
                float(reported[key]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"{dataset_name}: LITEMV {key} mismatch: "
                    f"reproduced={reproduced[key]:.12f}, "
                    f"reported={float(reported[key]):.12f}"
                )

    print(
        f"[{dataset_name}] frozen final LITEMV predictions verified | "
        f"F1={reproduced['f1']:.6f} | FP={reproduced['fp']} | "
        f"FN={reproduced['fn']}"
    )
    return reproduced


# =============================================================================
# RF STAGE — EXACT FINAL ENDPOINT-AWARE RROLL5 REPRESENTATION
# =============================================================================


def _prepare_rf_split(
    dataset_name: str,
    split_name: str,
    split_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the final RF's exact causal endpoint-aware representation on THIS split.

    Building each train/validation/test split independently is deliberate: no
    rolling history is allowed to cross a shared split boundary. Within the
    split, the final RF implementation still uses its exact sequence key
    (SourceFile, Src IP, Dst IP), Timestamp ordering, shift(1), and rolling-5
    history.
    """
    temporal = rf_temporal.build_temporal_features(
        split_df,
        dataset_name=f"{dataset_name}_part3_{split_name}_rf",
        require_label=True,
    )
    features = list(rf_temporal.final_feature_names())

    if len(features) != 32:
        raise RuntimeError(
            f"{dataset_name}/{split_name}: final RF must expose 32 features; "
            f"got {len(features)}."
        )

    required = [
        rf_config.ROW_INDEX_COL,
        rf_config.SOURCE_FILE_COL,
        rf_config.LABEL_COL,
        *features,
    ]
    missing = [column for column in required if column not in temporal.columns]
    if missing:
        raise RuntimeError(
            f"{dataset_name}/{split_name}: RF temporal frame is missing {missing}."
        )

    frame = temporal[required].copy()
    for feature in features:
        frame[feature] = (
            pd.to_numeric(frame[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
    frame[rf_config.LABEL_COL] = pd.to_numeric(
        frame[rf_config.LABEL_COL],
        errors="raise",
    ).astype(int)

    # build_temporal_features restores Original_Row_Index order before returning.
    expected = np.arange(len(split_df), dtype=np.int64)
    actual = frame[rf_config.ROW_INDEX_COL].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise RuntimeError(
            f"{dataset_name}/{split_name}: RF temporal rows do not map 1:1 "
            "to the shared split's original row order."
        )

    return frame, features


def train_stage_rf(dataset_name: str, train_df: pd.DataFrame):
    train_model_df, selected_features = _prepare_rf_split(
        dataset_name=dataset_name,
        split_name="train",
        split_df=train_df,
    )

    X_raw = train_model_df[selected_features]
    y_train = train_model_df[rf_config.LABEL_COL].astype(int)

    imputer = FoldMedianImputer(f"{dataset_name}_part3_rf")
    X_train = imputer.fit_transform(X_raw)

    class_weights = rf_utils.compute_training_class_weights(y_train)
    model = rf_utils.build_final_random_forest(
        class_weights=class_weights,
        random_state=rf_config.MODEL_RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    artifact_dir = ARTIFACT_ROOT / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_dir / "stage_rf.joblib")
    imputer.save(artifact_dir / "stage_rf_median_imputer.joblib")
    pd.DataFrame({"Feature": selected_features}).to_csv(
        artifact_dir / "stage_rf_selected_features.csv",
        index=False,
    )

    with (artifact_dir / "stage_rf_configuration.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "model": "RandomForestClassifier",
                "representation": "endpoint_aware_RROLL5_32_features",
                "fit_scope": "shared TRAIN partition only",
                "temporal_history_scope": (
                    "built independently inside each shared split; "
                    "no rolling history crosses train/validation/test boundaries"
                ),
                "hyperparameters": dict(rf_config.FINAL_RF_PARAMS),
                "prediction_threshold": float(rf_config.PREDICTION_THRESHOLD),
                "class_weights": {
                    str(k): float(v)
                    for k, v in class_weights.items()
                },
                "validation_used_for_rf_fit": False,
                "test_used_for_rf_fit": False,
            },
            handle,
            indent=2,
        )

    return model, imputer, selected_features


def _raw_split_sequence_keys(
    split_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Convert row-aligned RF probabilities to the same SourceFile-local target key
    used by LSTM/LITEMV: chronological position inside this shared split segment.
    """
    if len(split_df) != len(probabilities):
        raise ValueError(
            "RF probability count differs from raw split row count."
        )

    key = split_df[
        [SOURCE_FILE_COL, TIMESTAMP_COL, LABEL_COL]
    ].copy()
    key["RF_Probability"] = np.asarray(
        probabilities, dtype=float
    )
    key["__OriginalSplitOrder"] = np.arange(len(key), dtype=np.int64)
    key["__ParsedTimestamp"] = pd.to_datetime(
        key[TIMESTAMP_COL],
        errors="coerce",
        utc=True,
    )

    if key["__ParsedTimestamp"].isna().any():
        raise ValueError(
            "At least one Timestamp could not be parsed for RF alignment."
        )

    blocks = []
    for source_file, group in key.groupby(
        SOURCE_FILE_COL, sort=True
    ):
        # LSTM/LITEMV both use stable Timestamp ordering. Original split order
        # therefore resolves timestamp ties identically.
        group = group.sort_values(
            ["__ParsedTimestamp", "__OriginalSplitOrder"],
            kind="mergesort",
        ).copy()
        group[ROW_KEY_COL] = np.arange(
            len(group), dtype=int
        )
        blocks.append(group)

    return pd.concat(
        blocks, ignore_index=True
    ).drop(
        columns=["__ParsedTimestamp", "__OriginalSplitOrder"]
    )


def score_rf_split(
    dataset_name: str,
    split_name: str,
    split_df: pd.DataFrame,
    model,
    imputer: FoldMedianImputer,
    selected_features: list[str],
) -> pd.DataFrame:
    model_df, actual_features = _prepare_rf_split(
        dataset_name=dataset_name,
        split_name=split_name,
        split_df=split_df,
    )

    if actual_features != selected_features:
        raise RuntimeError(
            f"{dataset_name}/{split_name}: final RF feature schema differs "
            "from the TRAIN schema."
        )

    X = imputer.transform(model_df[selected_features])
    probabilities = model.predict_proba(X)[:, 1]

    # model_df is restored to the raw split's row order by the temporal builder.
    return _raw_split_sequence_keys(split_df, probabilities)


# =============================================================================
# ALIGNMENT
# =============================================================================


def align_model_outputs(
    lstm_scores: pd.DataFrame,
    litemv_scores: pd.DataFrame,
    rf_scores: pd.DataFrame,
) -> pd.DataFrame:
    keys = [SOURCE_FILE_COL, ROW_KEY_COL]

    left = lstm_scores[
        keys + [LABEL_COL, TIMESTAMP_COL, "LSTM_Probability"]
    ].copy()

    middle = litemv_scores[
        keys + [LABEL_COL, "LITEMV_Probability"]
    ].rename(columns={LABEL_COL: "__LITEMV_Label"})

    right = rf_scores[
        keys + [LABEL_COL, "RF_Probability"]
    ].rename(columns={LABEL_COL: "__RF_Label"})

    merged = left.merge(
        middle,
        on=keys,
        how="inner",
        validate="one_to_one",
    ).merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    expected = len(lstm_scores)
    if not (
        len(merged)
        == expected
        == len(litemv_scores)
        == len(rf_scores)
    ):
        raise RuntimeError(
            "Model row alignment failed: "
            f"LSTM={len(lstm_scores)}, "
            f"LITEMV={len(litemv_scores)}, "
            f"RF={len(rf_scores)}, merged={len(merged)}"
        )

    if not np.array_equal(
        merged[LABEL_COL].to_numpy(dtype=int),
        merged["__LITEMV_Label"].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "LSTM/LITEMV labels differ after row alignment."
        )
    if not np.array_equal(
        merged[LABEL_COL].to_numpy(dtype=int),
        merged["__RF_Label"].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "LSTM/RF labels differ after row alignment."
        )

    return merged.drop(
        columns=["__LITEMV_Label", "__RF_Label"]
    )


# =============================================================================
# VALIDATION CANDIDATES / SAFETY HELPERS
# =============================================================================


def _unique_finite(values) -> list[float]:
    return sorted(
        {
            float(v)
            for v in values
            if np.isfinite(v)
        }
    )


def _positive_band_candidates(
    lstm_probability: np.ndarray,
    lstm_threshold: float,
) -> list[float]:
    lp = np.asarray(lstm_probability, dtype=float)
    positive = lp[lp >= float(lstm_threshold)]

    values = [float(lstm_threshold)]
    if len(positive):
        values.extend(
            np.quantile(
                positive,
                LSTM_POSITIVE_BAND_QUANTILES,
            ).tolist()
        )

    return _unique_finite(
        v
        for v in values
        if v >= float(lstm_threshold) - EPS
    )


def _confirm_candidates(
    probability: np.ndarray,
    routed_mask: np.ndarray,
    model_threshold: float,
) -> list[float]:
    p = np.asarray(probability, dtype=float)
    mask = np.asarray(routed_mask, dtype=bool)
    values = [0.0, float(model_threshold), 1.0]

    routed = p[mask]
    if len(routed):
        values.extend(
            np.quantile(
                routed,
                SECONDARY_CONFIRM_QUANTILES,
            ).tolist()
        )

    return _unique_finite(
        v
        for v in values
        if -EPS <= v <= 1.0 + EPS
    )


def _zero_fp_rescue_threshold(
    y_true: np.ndarray,
    lstm_probability: np.ndarray,
    secondary_probability: np.ndarray,
    lstm_threshold: float,
    lower_bound: float,
) -> float:
    """
    Return a threshold strictly above the largest secondary-model score
    assigned to any benign validation row that the LSTM calls benign.

    This means the rescue branch cannot create a new validation FP.
    A threshold slightly above 1.0 is allowed; it simply disables rescue.
    """
    y = np.asarray(y_true, dtype=int)
    lp = np.asarray(lstm_probability, dtype=float)
    sp = np.asarray(secondary_probability, dtype=float)

    lstm_negative = lp < float(lstm_threshold)
    benign_scores = sp[lstm_negative & (y == 0)]

    if len(benign_scores) == 0:
        return float(lower_bound)

    strict_above = float(
        np.nextafter(np.max(benign_scores), np.inf)
    )
    return float(max(float(lower_bound), strict_above))


def _validation_safe(
    candidate: dict,
    baseline: dict,
) -> bool:
    return (
        candidate["fpr"] <= baseline["fpr"] + EPS
        and candidate["fnr"] <= baseline["fnr"] + EPS
    )


def _strictly_improves_f1(
    candidate: dict,
    baseline: dict,
) -> bool:
    return candidate["f1"] > baseline["f1"] + EPS


# =============================================================================
# TWO-MODEL CASCADE: LSTM + SECONDARY
# =============================================================================


def apply_two_model_cascade(
    lstm_probability: np.ndarray,
    secondary_probability: np.ndarray,
    lstm_threshold: float,
    positive_filter_cutoff: float,
    secondary_confirm_threshold: float,
    secondary_rescue_threshold: float,
    enabled: bool,
):
    lp = np.asarray(lstm_probability, dtype=float)
    sp = np.asarray(secondary_probability, dtype=float)

    baseline = (
        lp >= float(lstm_threshold)
    ).astype(int)
    final = baseline.copy()

    filter_route = np.zeros(len(lp), dtype=bool)
    rescue_route = np.zeros(len(lp), dtype=bool)

    if not enabled:
        return final, filter_route, rescue_route, baseline

    filter_route = (
        (baseline == 1)
        & (
            lp
            <= float(positive_filter_cutoff) + EPS
        )
    )
    final[filter_route] = (
        sp[filter_route]
        >= float(secondary_confirm_threshold)
    ).astype(int)

    rescue_route = (
        (baseline == 0)
        & (
            sp
            >= float(secondary_rescue_threshold)
        )
    )
    final[rescue_route] = 1

    return final, filter_route, rescue_route, baseline


def search_two_model_rule(
    validation: pd.DataFrame,
    lstm_threshold: float,
    secondary_column: str,
    secondary_standalone_threshold: float,
    architecture_name: str,
):
    y = validation[LABEL_COL].astype(int).to_numpy()
    lp = validation["LSTM_Probability"].to_numpy(dtype=float)
    sp = validation[secondary_column].to_numpy(dtype=float)

    baseline_pred = (
        lp >= float(lstm_threshold)
    ).astype(int)
    baseline_metrics = metrics_from_predictions(
        y, baseline_pred
    )

    positive_cutoffs = _positive_band_candidates(
        lp, lstm_threshold
    )
    baseline_positive = (
        lp >= float(lstm_threshold)
    )
    confirm_thresholds = _confirm_candidates(
        sp,
        baseline_positive,
        secondary_standalone_threshold,
    )
    rescue_threshold = _zero_fp_rescue_threshold(
        y_true=y,
        lstm_probability=lp,
        secondary_probability=sp,
        lstm_threshold=lstm_threshold,
        lower_bound=secondary_standalone_threshold,
    )

    rows = []

    # Explicit disabled baseline candidate.
    rows.append(
        {
            "Architecture": architecture_name,
            "enabled": False,
            "positive_filter_cutoff": float(
                lstm_threshold
            ),
            "secondary_confirm_threshold": 0.0,
            "secondary_rescue_threshold": float(
                rescue_threshold
            ),
            "filter_routed_rows": 0,
            "rescue_routed_rows": 0,
            "total_changed_rows": 0,
            **baseline_metrics,
        }
    )

    for positive_cutoff in positive_cutoffs:
        for confirm_threshold in confirm_thresholds:
            (
                pred,
                filter_route,
                rescue_route,
                _baseline,
            ) = apply_two_model_cascade(
                lstm_probability=lp,
                secondary_probability=sp,
                lstm_threshold=lstm_threshold,
                positive_filter_cutoff=positive_cutoff,
                secondary_confirm_threshold=confirm_threshold,
                secondary_rescue_threshold=rescue_threshold,
                enabled=True,
            )

            metrics = metrics_from_predictions(
                y, pred
            )
            rows.append(
                {
                    "Architecture": architecture_name,
                    "enabled": True,
                    "positive_filter_cutoff": float(
                        positive_cutoff
                    ),
                    "secondary_confirm_threshold": float(
                        confirm_threshold
                    ),
                    "secondary_rescue_threshold": float(
                        rescue_threshold
                    ),
                    "filter_routed_rows": int(
                        filter_route.sum()
                    ),
                    "rescue_routed_rows": int(
                        rescue_route.sum()
                    ),
                    "total_changed_rows": int(
                        np.sum(pred != baseline_pred)
                    ),
                    **metrics,
                }
            )

    search = pd.DataFrame(rows)

    safe = search[
        (search["fpr"] <= baseline_metrics["fpr"] + EPS)
        & (search["fnr"] <= baseline_metrics["fnr"] + EPS)
    ].copy()
    strict = safe[
        search.loc[safe.index, "f1"]
        > baseline_metrics["f1"] + EPS
    ].copy()

    if strict.empty:
        selected = search.iloc[0].to_dict()
        status = "disabled_no_validation_safe_f1_improvement"
    else:
        selected = (
            strict.sort_values(
                [
                    "f1",
                    "fpr",
                    "fnr",
                    "total_changed_rows",
                    "secondary_confirm_threshold",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                    False,
                ],
                kind="mergesort",
            )
            .iloc[0]
            .to_dict()
        )
        status = (
            "enabled_validation_f1_improvement_"
            "without_fpr_or_fnr_worsening"
        )

    selected["selection_status"] = status
    return search, selected, baseline_metrics


# =============================================================================
# THREE-MODEL CASCADE: LSTM -> LITEMV -> RF ARBITRATION
# =============================================================================


def apply_three_model_cascade(
    lstm_probability: np.ndarray,
    litemv_probability: np.ndarray,
    rf_probability: np.ndarray,
    lstm_threshold: float,
    positive_filter_cutoff: float,
    litemv_confirm_threshold: float,
    rf_confirm_threshold: float,
    litemv_rescue_threshold: float,
    rf_rescue_threshold: float,
    enabled: bool,
):
    """
    Positive path:
        confident LSTM positive -> malicious directly.
        low-confidence LSTM positive:
            LITEMV supports malicious -> malicious.
            LITEMV rejects -> RF arbitrates.
            only if both secondary models reject -> benign.

    Negative path:
        LSTM negative -> benign unless BOTH LITEMV and RF exceed their
        zero-validation-FP rescue cutoffs.
    """
    lp = np.asarray(lstm_probability, dtype=float)
    vp = np.asarray(litemv_probability, dtype=float)
    rp = np.asarray(rf_probability, dtype=float)

    baseline = (
        lp >= float(lstm_threshold)
    ).astype(int)
    final = baseline.copy()

    low_positive_route = np.zeros(
        len(lp), dtype=bool
    )
    litemv_confirms = np.zeros(
        len(lp), dtype=bool
    )
    rf_arbiter_route = np.zeros(
        len(lp), dtype=bool
    )
    rescue_route = np.zeros(
        len(lp), dtype=bool
    )

    if not enabled:
        return {
            "prediction": final,
            "baseline": baseline,
            "low_positive_route": low_positive_route,
            "litemv_confirms": litemv_confirms,
            "rf_arbiter_route": rf_arbiter_route,
            "rescue_route": rescue_route,
        }

    low_positive_route = (
        (baseline == 1)
        & (
            lp
            <= float(positive_filter_cutoff) + EPS
        )
    )

    litemv_confirms = (
        low_positive_route
        & (
            vp
            >= float(litemv_confirm_threshold)
        )
    )

    # RF is consulted only when a routed low-confidence LSTM positive
    # is NOT confirmed by LITEMV.
    rf_arbiter_route = (
        low_positive_route
        & (~litemv_confirms)
    )

    # Start by filtering the routed positives, then restore positives
    # supported by LITEMV or the RF arbiter.
    final[low_positive_route] = 0
    final[litemv_confirms] = 1
    final[
        rf_arbiter_route
        & (
            rp >= float(rf_confirm_threshold)
        )
    ] = 1

    # Conservative false-negative rescue: both secondary models must agree.
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
        "rescue_route": rescue_route,
    }


def search_three_model_rule(
    validation: pd.DataFrame,
    lstm_threshold: float,
    litemv_threshold: float,
    rf_threshold: float,
):
    y = validation[LABEL_COL].astype(int).to_numpy()
    lp = validation["LSTM_Probability"].to_numpy(dtype=float)
    vp = validation["LITEMV_Probability"].to_numpy(dtype=float)
    rp = validation["RF_Probability"].to_numpy(dtype=float)

    baseline_pred = (
        lp >= float(lstm_threshold)
    ).astype(int)
    baseline_metrics = metrics_from_predictions(
        y, baseline_pred
    )

    positive_cutoffs = _positive_band_candidates(
        lp, lstm_threshold
    )
    baseline_positive = (
        lp >= float(lstm_threshold)
    )

    litemv_confirms = _confirm_candidates(
        vp,
        baseline_positive,
        litemv_threshold,
    )
    rf_confirms = _confirm_candidates(
        rp,
        baseline_positive,
        rf_threshold,
    )

    litemv_rescue = _zero_fp_rescue_threshold(
        y_true=y,
        lstm_probability=lp,
        secondary_probability=vp,
        lstm_threshold=lstm_threshold,
        lower_bound=litemv_threshold,
    )
    rf_rescue = _zero_fp_rescue_threshold(
        y_true=y,
        lstm_probability=lp,
        secondary_probability=rp,
        lstm_threshold=lstm_threshold,
        lower_bound=rf_threshold,
    )

    rows = []

    rows.append(
        {
            "Architecture": "LSTM_LITEMV_RF",
            "enabled": False,
            "positive_filter_cutoff": float(
                lstm_threshold
            ),
            "litemv_confirm_threshold": 0.0,
            "rf_confirm_threshold": 0.0,
            "litemv_rescue_threshold": float(
                litemv_rescue
            ),
            "rf_rescue_threshold": float(
                rf_rescue
            ),
            "low_positive_routed_rows": 0,
            "litemv_confirmed_rows": 0,
            "rf_arbiter_rows": 0,
            "rescue_routed_rows": 0,
            "total_changed_rows": 0,
            **baseline_metrics,
        }
    )

    for positive_cutoff in positive_cutoffs:
        for litemv_confirm in litemv_confirms:
            for rf_confirm in rf_confirms:
                applied = apply_three_model_cascade(
                    lstm_probability=lp,
                    litemv_probability=vp,
                    rf_probability=rp,
                    lstm_threshold=lstm_threshold,
                    positive_filter_cutoff=positive_cutoff,
                    litemv_confirm_threshold=litemv_confirm,
                    rf_confirm_threshold=rf_confirm,
                    litemv_rescue_threshold=litemv_rescue,
                    rf_rescue_threshold=rf_rescue,
                    enabled=True,
                )

                pred = applied["prediction"]
                metrics = metrics_from_predictions(
                    y, pred
                )

                rows.append(
                    {
                        "Architecture": "LSTM_LITEMV_RF",
                        "enabled": True,
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
                            litemv_rescue
                        ),
                        "rf_rescue_threshold": float(
                            rf_rescue
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
                                pred
                                != baseline_pred
                            )
                        ),
                        **metrics,
                    }
                )

    search = pd.DataFrame(rows)

    safe = search[
        (search["fpr"] <= baseline_metrics["fpr"] + EPS)
        & (search["fnr"] <= baseline_metrics["fnr"] + EPS)
    ].copy()
    strict = safe[
        safe["f1"]
        > baseline_metrics["f1"] + EPS
    ].copy()

    if strict.empty:
        selected = search.iloc[0].to_dict()
        status = "disabled_no_validation_safe_f1_improvement"
    else:
        selected = (
            strict.sort_values(
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
        status = (
            "enabled_validation_f1_improvement_"
            "without_fpr_or_fnr_worsening"
        )

    selected["selection_status"] = status
    return search, selected, baseline_metrics


# =============================================================================
# RULE APPLICATION / ROUTE LABELS
# =============================================================================


def apply_selected_two_model(
    frame: pd.DataFrame,
    selected: dict,
    lstm_threshold: float,
    secondary_column: str,
):
    return apply_two_model_cascade(
        lstm_probability=frame[
            "LSTM_Probability"
        ].to_numpy(dtype=float),
        secondary_probability=frame[
            secondary_column
        ].to_numpy(dtype=float),
        lstm_threshold=lstm_threshold,
        positive_filter_cutoff=float(
            selected["positive_filter_cutoff"]
        ),
        secondary_confirm_threshold=float(
            selected["secondary_confirm_threshold"]
        ),
        secondary_rescue_threshold=float(
            selected["secondary_rescue_threshold"]
        ),
        enabled=bool(selected["enabled"]),
    )


def apply_selected_three_model(
    frame: pd.DataFrame,
    selected: dict,
    lstm_threshold: float,
):
    return apply_three_model_cascade(
        lstm_probability=frame[
            "LSTM_Probability"
        ].to_numpy(dtype=float),
        litemv_probability=frame[
            "LITEMV_Probability"
        ].to_numpy(dtype=float),
        rf_probability=frame[
            "RF_Probability"
        ].to_numpy(dtype=float),
        lstm_threshold=lstm_threshold,
        positive_filter_cutoff=float(
            selected["positive_filter_cutoff"]
        ),
        litemv_confirm_threshold=float(
            selected["litemv_confirm_threshold"]
        ),
        rf_confirm_threshold=float(
            selected["rf_confirm_threshold"]
        ),
        litemv_rescue_threshold=float(
            selected["litemv_rescue_threshold"]
        ),
        rf_rescue_threshold=float(
            selected["rf_rescue_threshold"]
        ),
        enabled=bool(selected["enabled"]),
    )


def three_model_route_labels(
    applied: dict,
) -> np.ndarray:
    n = len(applied["prediction"])
    final = applied["prediction"]
    base = applied["baseline"]

    route = np.full(
        n, "LSTM_DIRECT", dtype=object
    )

    low = applied["low_positive_route"]
    li = applied["litemv_confirms"]
    rf_route = applied["rf_arbiter_route"]
    rescue = applied["rescue_route"]

    route[low & li] = (
        "LITEMV_CONFIRMED_LSTM_POSITIVE"
    )
    route[
        rf_route & (final == 1)
    ] = "RF_ARBITRATED_KEEP_MALICIOUS"
    route[
        rf_route & (final == 0)
    ] = "LITEMV_RF_FILTERED_LSTM_POSITIVE"
    route[
        rescue & (base == 0) & (final == 1)
    ] = "JOINT_LITEMV_RF_RESCUE"

    return route


# =============================================================================
# PART-4 COMPATIBILITY API (appendix copy only)
# =============================================================================

def load_shared_splits(dataset_name: str):
    """Expose the exact persisted Dataset-1 split to the Part-4 appendix runner."""
    return lstm.load_dataset_and_split(dataset_name, force_resplit=False)


def apply_hybrid_cascade(
    lstm_probability,
    litemv_probability,
    rf_probability,
    lstm_threshold: float,
    positive_filter_cutoff: float,
    litemv_confirm_threshold: float,
    rf_confirm_threshold: float,
    litemv_rescue_threshold: float,
    rf_rescue_threshold: float,
    enabled: bool = True,
):
    return apply_three_model_cascade(
        lstm_probability=lstm_probability,
        litemv_probability=litemv_probability,
        rf_probability=rf_probability,
        lstm_threshold=lstm_threshold,
        positive_filter_cutoff=positive_filter_cutoff,
        litemv_confirm_threshold=litemv_confirm_threshold,
        rf_confirm_threshold=rf_confirm_threshold,
        litemv_rescue_threshold=litemv_rescue_threshold,
        rf_rescue_threshold=rf_rescue_threshold,
        enabled=enabled,
    )


def route_labels(applied: dict) -> np.ndarray:
    return three_model_route_labels(applied)


# =============================================================================
# FINAL ARCHITECTURE SELECTION ON VALIDATION ONLY
# =============================================================================


def choose_final_architecture(
    validation_metrics: pd.DataFrame,
) -> str:
    """
    Choose only among the LSTM baseline and validation-selected hybrids.

    TEST is never consulted.  Ties prefer the simpler architecture.
    """
    complexity = {
        "LSTM": 1,
        "LSTM_RF": 2,
        "LSTM_LITEMV": 2,
        "LSTM_LITEMV_RF": 3,
    }

    eligible = validation_metrics[
        validation_metrics["Model"].isin(
            complexity
        )
    ].copy()
    eligible["Complexity"] = eligible[
        "Model"
    ].map(complexity)

    selected = (
        eligible.sort_values(
            [
                "f1",
                "fpr",
                "fnr",
                "Complexity",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
            kind="mergesort",
        )
        .iloc[0]
    )
    return str(selected["Model"])


# =============================================================================
# ONE DATASET END-TO-END
# =============================================================================


def evaluate_dataset(
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 112)
    print(
        f"PART 3 HYBRID ABLATION | {dataset_name.upper()} | "
        "LSTM / LITEMV / RF"
    )
    print("=" * 112)

    print(f"LSTM source:    {Path(lstm.__file__).resolve()}")
    print(f"LITEMV source:  {Path(litemv.__file__).resolve()}")
    print(f"RF source:      {Path(rf_training.__file__).resolve()}")

    # Shared persisted source-aware split.
    splits, _manifest, split_metadata = (
        lstm.load_dataset_and_split(
            dataset_name,
            force_resplit=False,
        )
    )
    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    print(
        f"Shared rows | train={len(train_df):,} | "
        f"validation={len(validation_df):,} | "
        f"test={len(test_df):,} | "
        f"split_mode={split_metadata.get('mode')}"
    )

    # -------------------------------------------------------------------------
    # Load the exact frozen prediction evidence already produced by the final
    # LSTM/LITEMV runs. Do not re-unpickle their preprocessors or recompute
    # predictions in this environment.
    # -------------------------------------------------------------------------
    (
        lstm_threshold,
        val_lstm,
        test_lstm,
    ) = load_lstm_prediction_evidence(dataset_name)

    (
        litemv_threshold,
        val_litemv,
        test_litemv,
    ) = load_litemv_prediction_evidence(dataset_name)

    if len(val_lstm) != len(validation_df) or len(test_lstm) != len(test_df):
        raise RuntimeError(
            f"{dataset_name}: saved LSTM prediction counts do not match "
            "the persisted shared split."
        )
    if len(val_litemv) != len(validation_df) or len(test_litemv) != len(test_df):
        raise RuntimeError(
            f"{dataset_name}: saved LITEMV prediction counts do not match "
            "the persisted shared split."
        )

    # RF is refit on TRAIN only so the common VALIDATION split remains clean
    # for cascade selection.
    (
        rf_model,
        rf_imputer,
        rf_features,
    ) = train_stage_rf(
        dataset_name, train_df
    )
    rf_threshold = float(
        rf_config.PREDICTION_THRESHOLD
    )

    # -------------------------------------------------------------------------
    # Score RF on VALIDATION and TEST independently.
    # -------------------------------------------------------------------------
    val_rf = score_rf_split(
        dataset_name,
        "validation",
        validation_df,
        rf_model,
        rf_imputer,
        rf_features,
    )
    test_rf = score_rf_split(
        dataset_name,
        "test",
        test_df,
        rf_model,
        rf_imputer,
        rf_features,
    )

    # Guardrails: exact accepted Part-2 baselines must be reproduced.
    verify_final_lstm_consistency(
        dataset_name,
        lstm_threshold,
        test_lstm,
    )
    verify_final_litemv_consistency(
        dataset_name,
        litemv_threshold,
        test_litemv,
    )

    validation = align_model_outputs(
        val_lstm, val_litemv, val_rf
    )
    test = align_model_outputs(
        test_lstm, test_litemv, test_rf
    )

    result_dir = RESULT_ROOT / dataset_name
    result_dir.mkdir(
        parents=True, exist_ok=True
    )

    # Part 4 consumes the exact frozen component probabilities.
    validation.to_csv(
        result_dir / "validation_component_probabilities.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # VALIDATION-ONLY rule search.
    # -------------------------------------------------------------------------
    (
        search_lstm_rf,
        selected_lstm_rf,
        _,
    ) = search_two_model_rule(
        validation=validation,
        lstm_threshold=lstm_threshold,
        secondary_column="RF_Probability",
        secondary_standalone_threshold=rf_threshold,
        architecture_name="LSTM_RF",
    )
    search_lstm_rf.to_csv(
        result_dir
        / "validation_search_lstm_rf.csv",
        index=False,
    )

    (
        search_lstm_litemv,
        selected_lstm_litemv,
        _,
    ) = search_two_model_rule(
        validation=validation,
        lstm_threshold=lstm_threshold,
        secondary_column="LITEMV_Probability",
        secondary_standalone_threshold=litemv_threshold,
        architecture_name="LSTM_LITEMV",
    )
    search_lstm_litemv.to_csv(
        result_dir
        / "validation_search_lstm_litemv.csv",
        index=False,
    )

    (
        search_triple,
        selected_triple,
        _,
    ) = search_three_model_rule(
        validation=validation,
        lstm_threshold=lstm_threshold,
        litemv_threshold=litemv_threshold,
        rf_threshold=rf_threshold,
    )
    search_triple.to_csv(
        result_dir
        / "validation_search_lstm_litemv_rf.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Apply selected rules to VALIDATION so architecture selection itself
    # remains validation-only.
    # -------------------------------------------------------------------------
    y_val = validation[
        LABEL_COL
    ].astype(int).to_numpy()

    val_lstm_pred = (
        validation["LSTM_Probability"].to_numpy(
            dtype=float
        )
        >= lstm_threshold
    ).astype(int)

    val_litemv_pred = (
        validation["LITEMV_Probability"].to_numpy(
            dtype=float
        )
        >= litemv_threshold
    ).astype(int)

    val_rf_pred = (
        validation["RF_Probability"].to_numpy(
            dtype=float
        )
        >= rf_threshold
    ).astype(int)

    (
        val_lstm_rf_pred,
        _,
        _,
        _,
    ) = apply_selected_two_model(
        validation,
        selected_lstm_rf,
        lstm_threshold,
        "RF_Probability",
    )

    (
        val_lstm_litemv_pred,
        _,
        _,
        _,
    ) = apply_selected_two_model(
        validation,
        selected_lstm_litemv,
        lstm_threshold,
        "LITEMV_Probability",
    )

    val_triple_applied = (
        apply_selected_three_model(
            validation,
            selected_triple,
            lstm_threshold,
        )
    )
    val_triple_pred = val_triple_applied[
        "prediction"
    ]

    validation_rows = []
    for model_name, pred in [
        ("LSTM", val_lstm_pred),
        ("LITEMV", val_litemv_pred),
        ("RF", val_rf_pred),
        ("LSTM_RF", val_lstm_rf_pred),
        (
            "LSTM_LITEMV",
            val_lstm_litemv_pred,
        ),
        (
            "LSTM_LITEMV_RF",
            val_triple_pred,
        ),
    ]:
        validation_rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "Rows": int(len(y_val)),
                **metrics_from_predictions(
                    y_val, pred
                ),
            }
        )

    validation_metrics = pd.DataFrame(
        validation_rows
    )
    validation_metrics.to_csv(
        result_dir
        / "validation_ablation_metrics.csv",
        index=False,
    )

    final_architecture = choose_final_architecture(
        validation_metrics
    )

    # Freeze every rule + the validation-selected final architecture before
    # calculating TEST metrics.
    selected_rules = {
        "dataset": dataset_name,
        "split_mode": split_metadata.get("mode"),
        "selection_scope": "validation_only",
        "test_used_for_rule_selection": False,
        "test_used_for_architecture_selection": False,
        "lstm_threshold": lstm_threshold,
        "litemv_threshold": litemv_threshold,
        "rf_standalone_threshold": rf_threshold,
        "LSTM_RF": selected_lstm_rf,
        "LSTM_LITEMV": selected_lstm_litemv,
        "LSTM_LITEMV_RF": selected_triple,
        "validation_selected_final_architecture": (
            final_architecture
        ),
    }

    with (
        result_dir / "selected_rules.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(selected_rules),
            handle,
            indent=2,
        )

    # Part-4 expects the frozen three-model rule in a flat artifact.
    triple_rule = dict(selected_triple)
    hybrid_rule = {
        "dataset": dataset_name,
        "selection_scope": "dataset1_validation_only",
        "enabled": bool(triple_rule["enabled"]),
        "lstm_threshold": float(lstm_threshold),
        "litemv_standalone_threshold": float(litemv_threshold),
        "rf_standalone_threshold": float(rf_threshold),
        "positive_filter_cutoff": float(triple_rule["positive_filter_cutoff"]),
        "litemv_confirm_threshold": float(triple_rule["litemv_confirm_threshold"]),
        "rf_confirm_threshold": float(triple_rule["rf_confirm_threshold"]),
        "litemv_rescue_threshold": float(triple_rule["litemv_rescue_threshold"]),
        "rf_rescue_threshold": float(triple_rule["rf_rescue_threshold"]),
        "validation_selected_final_architecture": final_architecture,
        "triple_selection_status": triple_rule.get("selection_status"),
        "test_used_for_rule_selection": False,
    }
    appendix_artifact_dir = ARTIFACT_ROOT / dataset_name
    appendix_artifact_dir.mkdir(parents=True, exist_ok=True)
    with (appendix_artifact_dir / "hybrid_rule.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(hybrid_rule), handle, indent=2)

    # -------------------------------------------------------------------------
    # TEST reporting after all selection is frozen.
    # -------------------------------------------------------------------------
    y_test = test[
        LABEL_COL
    ].astype(int).to_numpy()

    lstm_pred = (
        test["LSTM_Probability"].to_numpy(
            dtype=float
        )
        >= lstm_threshold
    ).astype(int)

    litemv_pred = (
        test["LITEMV_Probability"].to_numpy(
            dtype=float
        )
        >= litemv_threshold
    ).astype(int)

    rf_pred = (
        test["RF_Probability"].to_numpy(
            dtype=float
        )
        >= rf_threshold
    ).astype(int)

    (
        lstm_rf_pred,
        lstm_rf_filter,
        lstm_rf_rescue,
        _,
    ) = apply_selected_two_model(
        test,
        selected_lstm_rf,
        lstm_threshold,
        "RF_Probability",
    )

    (
        lstm_litemv_pred,
        lstm_litemv_filter,
        lstm_litemv_rescue,
        _,
    ) = apply_selected_two_model(
        test,
        selected_lstm_litemv,
        lstm_threshold,
        "LITEMV_Probability",
    )

    triple_applied = apply_selected_three_model(
        test,
        selected_triple,
        lstm_threshold,
    )
    triple_pred = triple_applied["prediction"]

    test_predictions = test.copy()
    test_predictions["LSTM_Threshold"] = (
        lstm_threshold
    )
    test_predictions["LITEMV_Threshold"] = (
        litemv_threshold
    )
    test_predictions["RF_Threshold"] = (
        rf_threshold
    )

    test_predictions["LSTM_Prediction"] = (
        lstm_pred
    )
    test_predictions["LITEMV_Prediction"] = (
        litemv_pred
    )
    test_predictions["RF_Prediction"] = rf_pred
    test_predictions["LSTM_RF_Prediction"] = (
        lstm_rf_pred
    )
    test_predictions[
        "LSTM_LITEMV_Prediction"
    ] = lstm_litemv_pred
    test_predictions[
        "LSTM_LITEMV_RF_Prediction"
    ] = triple_pred

    test_predictions[
        "LSTM_RF_Filter_Route"
    ] = lstm_rf_filter.astype(int)
    test_predictions[
        "LSTM_RF_Rescue_Route"
    ] = lstm_rf_rescue.astype(int)
    test_predictions[
        "LSTM_LITEMV_Filter_Route"
    ] = lstm_litemv_filter.astype(int)
    test_predictions[
        "LSTM_LITEMV_Rescue_Route"
    ] = lstm_litemv_rescue.astype(int)

    test_predictions[
        "Triple_Low_Positive_Route"
    ] = triple_applied[
        "low_positive_route"
    ].astype(int)
    test_predictions[
        "Triple_LITEMV_Confirmed"
    ] = triple_applied[
        "litemv_confirms"
    ].astype(int)
    test_predictions[
        "Triple_RF_Arbiter_Route"
    ] = triple_applied[
        "rf_arbiter_route"
    ].astype(int)
    test_predictions[
        "Triple_Rescue_Route"
    ] = triple_applied[
        "rescue_route"
    ].astype(int)
    test_predictions[
        "Triple_Decision_Route"
    ] = three_model_route_labels(
        triple_applied
    )
    # Stable aliases consumed by the Part-4 appendix adapter.
    test_predictions["Predicted_Label"] = triple_pred
    test_predictions["DecisionRoute"] = test_predictions["Triple_Decision_Route"]
    test_predictions.to_csv(
        result_dir / "test_predictions.csv",
        index=False,
    )

    test_rows = []
    test_prediction_map = {
        "LSTM": lstm_pred,
        "LITEMV": litemv_pred,
        "RF": rf_pred,
        "LSTM_RF": lstm_rf_pred,
        "LSTM_LITEMV": lstm_litemv_pred,
        "LSTM_LITEMV_RF": triple_pred,
    }

    for model_name, pred in (
        test_prediction_map.items()
    ):
        test_rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "ValidationSelectedFinalArchitecture": (
                    model_name == final_architecture
                ),
                "Rows": int(len(y_test)),
                **metrics_from_predictions(
                    y_test, pred
                ),
            }
        )

    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(
        result_dir / "test_ablation_metrics.csv",
        index=False,
    )

    # Deltas + exact error corrections for every hybrid.
    lstm_metrics = metrics_from_predictions(
        y_test, lstm_pred
    )
    comparisons = []
    for model_name in (
        "LSTM_RF",
        "LSTM_LITEMV",
        "LSTM_LITEMV_RF",
    ):
        pred = test_prediction_map[model_name]
        metrics = metrics_from_predictions(
            y_test, pred
        )
        comparisons.append(
            {
                "Dataset": dataset_name,
                "Hybrid": model_name,
                "ValidationSelectedFinalArchitecture": (
                    model_name
                    == final_architecture
                ),
                "Baseline_F1": (
                    lstm_metrics["f1"]
                ),
                "Hybrid_F1": metrics["f1"],
                "Delta_F1": (
                    metrics["f1"]
                    - lstm_metrics["f1"]
                ),
                "Baseline_FPR": (
                    lstm_metrics["fpr"]
                ),
                "Hybrid_FPR": metrics["fpr"],
                "Delta_FPR": (
                    metrics["fpr"]
                    - lstm_metrics["fpr"]
                ),
                "Baseline_FNR": (
                    lstm_metrics["fnr"]
                ),
                "Hybrid_FNR": metrics["fnr"],
                "Delta_FNR": (
                    metrics["fnr"]
                    - lstm_metrics["fnr"]
                ),
                **correction_summary(
                    y_true=y_test,
                    lstm_pred=lstm_pred,
                    hybrid_pred=pred,
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparisons
    )
    comparison_df.to_csv(
        result_dir
        / "test_hybrids_vs_lstm.csv",
        index=False,
    )

    # Confusion matrices.
    for model_name, pred in (
        test_prediction_map.items()
    ):
        cm = confusion_matrix(
            y_test, pred, labels=[0, 1]
        )
        pd.DataFrame(
            cm,
            index=[
                "Actual_Benign",
                "Actual_Malicious",
            ],
            columns=[
                "Predicted_Benign",
                "Predicted_Malicious",
            ],
        ).to_csv(
            result_dir
            / f"confusion_matrix_{model_name.lower()}.csv"
        )

    print("\nVALIDATION ablation:")
    print(
        validation_metrics.to_string(
            index=False
        )
    )
    print(
        "\nValidation-selected final architecture: "
        f"{final_architecture}"
    )

    print("\nTEST ablation (reporting only):")
    print(test_metrics.to_string(index=False))

    print("\nHybrid deltas vs final LSTM:")
    print(comparison_df.to_string(index=False))

    return test_metrics, comparison_df


# =============================================================================
# JSON HELPERS
# =============================================================================


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


# =============================================================================
# DRIVER
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Part-3 leakage-safe hybrid ablation: "
            "LSTM, LITEMV, RF, LSTM+RF, "
            "LSTM+LITEMV, and LSTM+LITEMV+RF."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help=(
            "Evaluate one dataset only. "
            "Default: Dataset 1 appendix experiment."
        ),
    )
    args = parser.parse_args()

    names = (
        [args.dataset]
        if args.dataset
        else list(DATASETS)
    )

    all_test = []
    all_comparisons = []

    for dataset_name in names:
        test_metrics, comparisons = (
            evaluate_dataset(dataset_name)
        )
        all_test.append(test_metrics)
        all_comparisons.append(comparisons)

    test_summary = pd.concat(
        all_test, ignore_index=True
    )
    comparison_summary = pd.concat(
        all_comparisons, ignore_index=True
    )

    RESULT_ROOT.mkdir(
        parents=True, exist_ok=True
    )
    test_summary.to_csv(
        RESULT_ROOT / "test_ablation_summary.csv",
        index=False,
    )
    comparison_summary.to_csv(
        RESULT_ROOT / "hybrid_vs_lstm_summary.csv",
        index=False,
    )

    methodology = {
        "part": "Part 3 - Advanced Hybrid Pipeline",
        "candidate_architectures": [
            "LSTM",
            "LSTM_RF",
            "LSTM_LITEMV",
            "LSTM_LITEMV_RF",
        ],
        "standalone_diagnostic_models": [
            "LITEMV",
            "RF",
        ],
        "primary_detector": (
            "final optimized O1+O2 LSTM"
        ),
        "litemv_role": (
            "first temporal second opinion on "
            "low-confidence LSTM positives; "
            "joint high-confidence rescue evidence "
            "for LSTM negatives"
        ),
        "rf_role": (
            "arbiter when routed LSTM and LITEMV "
            "do not support the same malicious decision; "
            "joint rescue confirmation"
        ),
        "litemv_source": (
            "src/models_optimized_final/LITEMV/litemv.py; "
            "accepted final O1 43-feature model"
        ),
        "rf_source": (
            "src/models_optimized_final/RF/random_forest_training.py "
            "using endpoint-aware RROLL5 temporal features from random_forest_temporal.py "
            "and FINAL_RF_PARAMS from random_forest_config.py"
        ),
        "rf_fit_scope": (
            "shared TRAIN split only"
        ),
        "hybrid_rule_selection": (
            "VALIDATION only"
        ),
        "final_architecture_selection": (
            "VALIDATION only; rank by F1, then FPR, "
            "then FNR, then prefer lower complexity"
        ),
        "hybrid_retention_constraint": (
            "strict validation F1 improvement with "
            "no worsening of validation FPR or FNR"
        ),
        "test_used_for_rule_selection": False,
        "test_used_for_architecture_selection": False,
        "test_role": (
            "final reporting only after rules and "
            "architecture are frozen"
        ),
        "simple_majority_vote_used": False,
    }

    with (
        RESULT_ROOT / "methodology.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            methodology,
            handle,
            indent=2,
        )

    print("\n" + "=" * 112)
    print("PART 3 FINAL ABLATION SUMMARY")
    print("=" * 112)
    print(test_summary.to_string(index=False))

    print("\n" + "=" * 112)
    print("HYBRID DELTAS VS LSTM")
    print("=" * 112)
    print(
        comparison_summary.to_string(
            index=False
        )
    )

    print("\nResults:")
    print(f"  {RESULT_ROOT}")
    print("Artifacts:")
    print(f"  {ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()
