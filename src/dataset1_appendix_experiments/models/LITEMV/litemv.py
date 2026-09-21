from __future__ import annotations

"""
Final optimized LITEMV for the Cobalt Strike HTTPS C2 project (O1, 43 features).

Design contract
---------------
- Dataset 2 and Dataset 3 only.
- Reuses the exact persisted source-aware train/validation/test split manifests.
- Fits preprocessing on TRAIN only.
- Starts from the common model-agnostic 57-feature schema and applies the accepted
  O1 environment-hardening rule, retaining 43 predictive features.
- Builds causal flow windows independently inside each SourceFile and split.
- Uses one shared LITEMV architecture/training configuration for D2 and D3.
- Class weights, fitted preprocessing, early-stopping epoch, and decision threshold
  are dataset-specific learned quantities.
- Threshold calibration uses VALIDATION only; TEST remains untouched until evaluation.
- This is the final selected LITEMV configuration. Rejected source-balancing and
  temporal-augmentation experiments are intentionally not included.

LITEMV backbone
---------------
The backbone is aeon's implementation of the published multivariate LITE network:
    LITENetwork(use_litemv=True)

Install if needed:
    pip install aeon
"""

import json
import os
import random
import sys
from pathlib import Path

# TensorFlow environment setup must happen before TensorFlow import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
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
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

tf.get_logger().setLevel("ERROR")

try:
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass


# =============================================================================
# PROJECT PATHS
# =============================================================================

def _find_project_root(start: Path) -> Path:
    start = Path(start).resolve()
    for candidate in (start.parent, *start.parents):
        if (
            (candidate / "data" / "ingested").exists()
            and (candidate / "src" / "preprocessing").exists()
        ):
            return candidate
    raise RuntimeError(
        "Could not locate Workshop_final. Expected a parent containing "
        "data/ingested and src/preprocessing."
    )


PROJECT_ROOT = _find_project_root(Path(__file__))
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_DIR = Path(__file__).resolve().parent
INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"

DATASET_PATHS = {
    "dataset1": INGESTED_ROOT / "dataset1_ingested.csv",
}
BENCHMARK_DATASETS = tuple(DATASET_PATHS.keys())

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset1_appendix" / "models" / "LITEMV"
RESULT_ROOT = PROJECT_ROOT / "results" / "dataset1_appendix" / "models" / "LITEMV"
SHARED_SPLIT_ROOT = PROJECT_ROOT / "artifacts" / "dataset1_appendix" / "splits"

# source_aware_split.py has historically lived with the LSTM implementation.
_SPLITTER_CANDIDATES = [
    MODEL_DIR,
    SRC_ROOT / "models_baseline" / "LSTM",
    SRC_ROOT / "models_optimized_final" / "LSTM",
]

for _path in [
    *_SPLITTER_CANDIDATES,
    SRC_ROOT / "preprocessing",
    SRC_ROOT / "feature_engineering",
]:
    if _path.exists():
        _text = str(_path)
        if _text not in sys.path:
            sys.path.insert(0, _text)

from preprocessing import DatasetPreprocessor  # noqa: E402
from feature_selection import select_and_engineer_dataset  # noqa: E402
from source_aware_split import (  # noqa: E402
    ORIGINAL_ROW_COL,
    get_or_create_shared_split,
)


# =============================================================================
# COLUMNS / LABELS / REPRESENTATION
# =============================================================================

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
TIMESTAMP_COL = "Timestamp"
FLOW_ID_COL = "Flow ID"

BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

# Common Step-3 schema before LITEMV-specific hardening.
BASELINE_EXPECTED_FEATURE_COUNT = 57

# Final accepted LITEMV representation.
EXPECTED_FEATURE_COUNT = 43

# O1 environment-hardening rule.
O1_MANDATORY_DROPS = {
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Total Connection Flow Time",
    "Fwd Header Length",
    "Bwd Header Length",
}
O1_ALLOWED_FLAG_FEATURE = "URG Flag Count"
O1_EXPECTED_DROP_COUNT = 14

# Train stride matches the existing sequence-model protocol: reduce heavily
# overlapping windows during fitting, while validation/test score every flow.
TRAIN_SEQUENCE_STRIDE = 5
EVAL_SEQUENCE_STRIDE = 1

# Conv1D does not honor Keras Masking. Prefixes are LEFT padded with zeros.
# After the project's training-fitted signed-log1p + RobustScaler, zero is the
# neutral/central value for a typical feature, making it a much safer padding
# value than the LSTM's very large sentinel.
PADDING_VALUE = 0.0


# =============================================================================
# SHARED LITEMV HYPERPARAMETERS
# =============================================================================

# The LITEMV architecture values n_filters=32 and kernel_size=40 follow the
# published LITE/LITEMV setup and aeon defaults. Training controls are fixed
# identically for Dataset 2 and Dataset 3.
SHARED_PARAMS = {
    "sequence_length": 40,
    "n_filters": 32,
    "kernel_size": 40,
    "strides": 1,
    "activation": "relu",
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 500,
    "early_stopping_patience": 40,
    "early_stopping_min_delta": 1e-5,
    "reduce_lr_patience": 12,
    "reduce_lr_factor": 0.5,
    "min_learning_rate": 1e-6,
    "shuffle": True,
    "fit_verbose": 2,
}

RANDOM_STATE = 42

# Exact operating-point protocol used by the sequence-model benchmark.
THRESHOLD_METHODS = (
    "validation_f1",
    "validation_f2",
    "recall_at_fpr_cap",
)
VALIDATION_FPR_CAP = 0.02


# =============================================================================
# REPRODUCIBILITY / DEVICE
# =============================================================================

def set_random_seed(seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def describe_compute_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return "GPU detected: " + ", ".join(device.name for device in gpus)
    return "No GPU detected; TensorFlow will use CPU."


# =============================================================================
# DATA + SHARED SOURCE-AWARE SPLIT
# =============================================================================

def load_dataset_and_split(
    dataset_name: str,
    force_resplit: bool = False,
):
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_path = DATASET_PATHS[dataset_name]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing ingested dataset: {dataset_path}")

    df = pd.read_csv(dataset_path, low_memory=False)

    splits, manifest, metadata = get_or_create_shared_split(
        dataset_name=dataset_name,
        df=df,
        split_root=SHARED_SPLIT_ROOT,
        force_rebuild=force_resplit,
    )

    # The shared splitter adds __OriginalRowIndex only so every model can reuse
    # the exact same persisted rows. It is bookkeeping metadata, NOT a
    # predictive feature. This mirrors the existing LSTM pipeline exactly.
    for split_name in list(splits):
        splits[split_name] = splits[split_name].drop(
            columns=[ORIGINAL_ROW_COL],
            errors="ignore",
        )

    return splits, manifest, metadata


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    total = sum(len(frame) for frame in splits.values())
    rows = []

    for split_name in ("train", "validation", "test"):
        frame = splits[split_name]
        labels = frame[LABEL_COL].astype(int)
        benign = int((labels == BENIGN_LABEL).sum())
        malicious = int((labels == MALICIOUS_LABEL).sum())

        rows.append(
            {
                "Split": split_name,
                "Rows": int(len(frame)),
                "Row_Fraction": float(len(frame) / total),
                "Benign_Rows": benign,
                "Malicious_Rows": malicious,
                "Malicious_Percent": (
                    float(100.0 * malicious / len(frame)) if len(frame) else 0.0
                ),
                "SourceFiles": int(frame[SOURCE_FILE_COL].nunique()),
            }
        )

    return pd.DataFrame(rows)


def _is_o1_flag_drop(feature_name: str) -> bool:
    normalized = " ".join(str(feature_name).strip().split()).casefold()
    allowed = O1_ALLOWED_FLAG_FEATURE.casefold()
    return ("flag" in normalized) and (normalized != allowed)


def resolve_o1_features(
    baseline_features: list[str],
) -> tuple[list[str], list[str]]:
    """Apply the accepted 57 -> 43 environment-hardening rule."""
    if len(baseline_features) != BASELINE_EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {BASELINE_EXPECTED_FEATURE_COUNT} baseline features, "
            f"got {len(baseline_features)}."
        )

    missing_mandatory = sorted(
        feature
        for feature in O1_MANDATORY_DROPS
        if feature not in baseline_features
    )
    if missing_mandatory:
        raise RuntimeError(
            f"O1 mandatory hardening features are missing: {missing_mandatory}"
        )

    dropped = [
        feature
        for feature in baseline_features
        if (
            feature in O1_MANDATORY_DROPS
            or _is_o1_flag_drop(feature)
        )
    ]
    kept = [
        feature
        for feature in baseline_features
        if feature not in set(dropped)
    ]

    # URG Flag Count is an allowed exception IF PRESENT in the 57-feature input;
    # it is not required to exist because upstream constant-feature removal may
    # already have eliminated it.
    bad_flags = [
        feature
        for feature in kept
        if _is_o1_flag_drop(feature)
    ]
    if bad_flags:
        raise RuntimeError(
            f"O1 retained forbidden flag features: {bad_flags}"
        )

    if len(dropped) != O1_EXPECTED_DROP_COUNT:
        raise RuntimeError(
            f"O1 must drop exactly {O1_EXPECTED_DROP_COUNT} features; "
            f"got {len(dropped)}: {dropped}"
        )
    if len(kept) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"O1 must retain exactly {EXPECTED_FEATURE_COUNT} features; "
            f"got {len(kept)}."
        )

    return kept, dropped


def transform_and_select(
    preprocessor: DatasetPreprocessor,
    df: pd.DataFrame,
    dataset_name: str,
    expected_features: list[str] | None = None,
):
    """
    Run common preprocessing/feature selection, then apply final O1 hardening.

    The common selector still produces the canonical 57-feature schema. The
    final model then removes the accepted 14 brittle/environment-dependent
    fields and feeds the remaining 43 features to LITEMV.
    """
    normalized = preprocessor.transform(df)

    selected_df, baseline_features, dropped, manifest = select_and_engineer_dataset(
        dataset_name=dataset_name,
        df=normalized,
        expected_selected_features=None,
    )

    if len(baseline_features) != BASELINE_EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"{dataset_name}: expected {BASELINE_EXPECTED_FEATURE_COUNT} common "
            f"features before O1 hardening, got {len(baseline_features)}."
        )

    selected_features, o1_dropped = resolve_o1_features(baseline_features)

    if expected_features is not None and selected_features != expected_features:
        missing = [f for f in expected_features if f not in selected_features]
        extra = [f for f in selected_features if f not in expected_features]
        raise RuntimeError(
            f"{dataset_name}: final O1 feature schema/order differs from training. "
            f"Missing={missing}; Extra={extra}; "
            f"OrderMatches={selected_features == expected_features}"
        )

    combined_dropped = list(dropped) + [
        feature for feature in o1_dropped if feature not in dropped
    ]

    return selected_df, selected_features, combined_dropped, manifest


# =============================================================================
# CAUSAL MULTIVARIATE WINDOWS
# =============================================================================

def _full_sliding_windows(values: np.ndarray, sequence_length: int) -> np.ndarray:
    """Return windows ending at rows L-1...N-1, shape=(N-L+1, L, F)."""
    values = np.asarray(values, dtype=np.float32)

    if len(values) < sequence_length:
        return np.empty(
            (0, sequence_length, values.shape[1]),
            dtype=np.float32,
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        values,
        window_shape=sequence_length,
        axis=0,
    )
    # NumPy returns (n_windows, features, length) for axis=0.
    windows = np.transpose(windows, (0, 2, 1))
    return np.asarray(windows, dtype=np.float32)


def _source_sequences(
    values: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
    stride: int,
):
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if values.ndim != 2:
        raise ValueError("values must be a 2D feature matrix.")
    if len(values) != len(labels):
        raise ValueError("Feature and label lengths differ.")
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be >= 1.")

    n_rows, n_features = values.shape
    if n_rows == 0:
        return (
            np.empty((0, sequence_length, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    # Every target gets only its own past/current flows. Short prefixes are
    # left-padded, so the current flow always occupies the final timestep.
    sequences = np.full(
        (n_rows, sequence_length, n_features),
        PADDING_VALUE,
        dtype=np.float32,
    )

    prefix_count = min(sequence_length - 1, n_rows)
    for target_index in range(prefix_count):
        valid_count = target_index + 1
        sequences[
            target_index,
            sequence_length - valid_count :,
            :,
        ] = values[:valid_count, :]

    if n_rows >= sequence_length:
        sequences[sequence_length - 1 :, :, :] = _full_sliding_windows(
            values,
            sequence_length,
        )

    target_indices = np.arange(n_rows, dtype=np.int64)[::stride]

    return (
        sequences[target_indices],
        labels[target_indices],
        target_indices,
    )


def build_sequences(
    selected_df: pd.DataFrame,
    selected_features: list[str],
    sequence_length: int,
    stride: int,
):
    """
    Build causal LITEMV windows independently inside each SourceFile in THIS split.

    This guarantees:
    - no window crosses SourceFile boundaries;
    - no window crosses train/validation/test boundaries;
    - eval stride 1 yields exactly one decision per held-out flow.
    """
    required = {SOURCE_FILE_COL, TIMESTAMP_COL, LABEL_COL}
    missing = required - set(selected_df.columns)
    if missing:
        raise ValueError(
            f"Missing LITEMV sequence metadata columns: {sorted(missing)}"
        )

    sequence_blocks = []
    label_blocks = []
    metadata_blocks = []

    for source_file, source_df in selected_df.groupby(SOURCE_FILE_COL, sort=True):
        source_df = source_df.copy()
        source_df["__ParsedTimestamp"] = pd.to_datetime(
            source_df[TIMESTAMP_COL],
            errors="coerce",
            utc=True,
        )

        if source_df["__ParsedTimestamp"].isna().any():
            bad = int(source_df["__ParsedTimestamp"].isna().sum())
            raise ValueError(
                f"{source_file}: {bad} Timestamp values could not be parsed."
            )

        source_df = source_df.sort_values(
            "__ParsedTimestamp",
            kind="mergesort",
        ).reset_index(drop=True)

        source_labels = source_df[LABEL_COL].astype(int).to_numpy()
        if len(np.unique(source_labels)) != 1:
            raise ValueError(
                f"{source_file}: source segment contains mixed labels."
            )

        values = source_df[selected_features].to_numpy(dtype=np.float32)

        sequences, targets, target_indices = _source_sequences(
            values=values,
            labels=source_labels,
            sequence_length=sequence_length,
            stride=stride,
        )

        if len(sequences) == 0:
            continue

        metadata_columns = [SOURCE_FILE_COL, TIMESTAMP_COL, LABEL_COL]
        if FLOW_ID_COL in source_df.columns:
            metadata_columns.insert(1, FLOW_ID_COL)

        metadata = (
            source_df.iloc[target_indices][metadata_columns]
            .copy()
            .reset_index(drop=True)
        )
        metadata["TargetRowInSplitSourceSegment"] = target_indices
        metadata["SequenceLength"] = int(sequence_length)

        sequence_blocks.append(sequences)
        label_blocks.append(targets)
        metadata_blocks.append(metadata)

    if not sequence_blocks:
        raise ValueError("No LITEMV sequences could be constructed.")

    X = np.concatenate(sequence_blocks, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(label_blocks, axis=0).astype(np.int64, copy=False)
    metadata = pd.concat(metadata_blocks, ignore_index=True)

    if not (len(X) == len(y) == len(metadata)):
        raise RuntimeError("Sequence, target, and metadata counts differ.")

    return X, y, metadata


# =============================================================================
# CLASS IMBALANCE
# =============================================================================

def calculate_class_weights(y_train: np.ndarray) -> dict[int, float]:
    y_train = np.asarray(y_train, dtype=int)
    classes = np.unique(y_train)

    if set(classes.tolist()) != {BENIGN_LABEL, MALICIOUS_LABEL}:
        raise ValueError(
            "Supervised LITEMV training requires both classes. "
            f"Found: {classes.tolist()}"
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }


# =============================================================================
# LITEMV MODEL
# =============================================================================

def _import_lite_network():
    try:
        from aeon.networks import LITENetwork
    except ImportError as exc:
        raise ImportError(
            "LITEMV requires aeon. Install it in the active environment with:\n"
            "    pip install aeon\n"
            "Then rerun the experiment."
        ) from exc
    return LITENetwork


def build_litemv_model(feature_count: int, params: dict) -> keras.Model:
    """
    Build the published multivariate LITE backbone and add a binary sigmoid head.
    """
    LITENetwork = _import_lite_network()

    sequence_length = int(params["sequence_length"])

    backbone = LITENetwork(
        use_litemv=True,
        n_filters=int(params["n_filters"]),
        kernel_size=int(params["kernel_size"]),
        strides=int(params["strides"]),
        activation=str(params["activation"]),
    )

    inputs, representation = backbone.build_network(
        input_shape=(sequence_length, feature_count)
    )

    outputs = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="malicious_probability",
    )(representation)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="c2_litemv_classifier",
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=float(params["learning_rate"]),
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )

    return model


def fit_litemv(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    params: dict,
    class_weights: dict[int, float],
):
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=float(params["reduce_lr_factor"]),
            patience=int(params["reduce_lr_patience"]),
            min_lr=float(params["min_learning_rate"]),
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(params["early_stopping_patience"]),
            min_delta=float(params["early_stopping_min_delta"]),
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        class_weight=class_weights,
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        shuffle=bool(params["shuffle"]),
        verbose=int(params["fit_verbose"]),
        callbacks=callbacks,
    )

    val_losses = np.asarray(history.history.get("val_loss", []), dtype=float)
    best_epoch = (
        int(np.nanargmin(val_losses) + 1)
        if len(val_losses)
        else int(len(history.history.get("loss", [])))
    )

    return history, max(best_epoch, 1)


def predict_probabilities(
    model: keras.Model,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    batch_size = max(int(batch_size), 1)

    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        batch = tf.convert_to_tensor(X[start:stop], dtype=tf.float32)
        output = model(batch, training=False).numpy().reshape(-1)
        outputs.append(np.asarray(output, dtype=np.float32))

    if not outputs:
        return np.empty((0,), dtype=float)

    return np.concatenate(outputs, axis=0).astype(float, copy=False)


# =============================================================================
# METRICS + VALIDATION-ONLY THRESHOLD CALIBRATION
# =============================================================================

def predictions_from_probabilities(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (
        np.asarray(probabilities, dtype=float) >= float(threshold)
    ).astype(int)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = predictions_from_probabilities(probabilities, threshold)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[BENIGN_LABEL, MALICIOUS_LABEL],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    def _safe_roc_auc():
        return (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else float("nan")
        )

    def _safe_pr_auc():
        return (
            float(average_precision_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else float("nan")
        )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_tnr": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "roc_auc": _safe_roc_auc(),
        "pr_auc": _safe_pr_auc(),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def _best_fbeta_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    beta: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to calibrate a validation threshold.")

    precision = precision[:-1]
    recall = recall[:-1]
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall

    scores = np.divide(
        (1.0 + beta_sq) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_value = np.nanmax(scores)
    candidates = np.flatnonzero(np.isclose(scores, best_value))

    if len(candidates) > 1:
        best_index = int(candidates[np.argmax(recall[candidates])])
    else:
        best_index = int(candidates[0])

    return float(thresholds[best_index])


def _recall_at_fpr_cap_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fpr_cap: float,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)

    eligible = np.flatnonzero(
        (fpr <= float(fpr_cap)) & np.isfinite(thresholds)
    )

    if len(eligible) == 0:
        return float(np.nextafter(np.max(probabilities), np.inf))

    best_tpr = np.max(tpr[eligible])
    candidates = eligible[np.isclose(tpr[eligible], best_tpr)]

    min_fpr = np.min(fpr[candidates])
    candidates = candidates[np.isclose(fpr[candidates], min_fpr)]

    best_index = int(candidates[np.argmax(thresholds[candidates])])
    return float(thresholds[best_index])


def calibrate_validation_threshold(
    y_validation: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    thresholds = {
        "validation_f1": _best_fbeta_threshold(
            y_validation,
            probabilities,
            beta=1.0,
        ),
        "validation_f2": _best_fbeta_threshold(
            y_validation,
            probabilities,
            beta=2.0,
        ),
        "recall_at_fpr_cap": _recall_at_fpr_cap_threshold(
            y_validation,
            probabilities,
            fpr_cap=VALIDATION_FPR_CAP,
        ),
    }

    candidates = []
    for method in THRESHOLD_METHODS:
        threshold = float(thresholds[method])
        metrics = calculate_metrics(
            y_validation,
            probabilities,
            threshold,
        )
        candidates.append(
            {
                "method": method,
                "threshold": threshold,
                **metrics,
            }
        )

    # Same operational ranking as the LSTM pipeline.
    candidates.sort(
        key=lambda row: (
            -row["f2"],
            -row["f1"],
            -row["recall_tpr"],
            row["fpr"],
            -row["precision"],
        )
    )

    best = candidates[0]

    return {
        "method": best["method"],
        "threshold": float(best["threshold"]),
        "metrics": {
            key: value
            for key, value in best.items()
            if key != "method"
        },
        "all_candidates": candidates,
    }


# =============================================================================
# ARTIFACT HELPERS
# =============================================================================

def _jsonable_params(params: dict) -> dict:
    output = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            output[key] = list(value)
        elif isinstance(value, (np.integer,)):
            output[key] = int(value)
        elif isinstance(value, (np.floating,)):
            output[key] = float(value)
        else:
            output[key] = value
    return output


def _save_history(history, path: Path) -> None:
    pd.DataFrame(history.history).to_csv(path, index=False)


def _save_model_summary(model: keras.Model, path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# TRAIN ONE DATASET
# =============================================================================

def train_dataset(
    dataset_name: str,
    force_resplit: bool = False,
) -> dict:
    if dataset_name not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark dataset: {dataset_name}")

    print("\n" + "=" * 100)
    print(f"LITEMV TRAINING - {dataset_name.upper()}")
    print("=" * 100)

    set_random_seed(RANDOM_STATE)
    tf.keras.backend.clear_session()

    splits, _manifest, split_metadata = load_dataset_and_split(
        dataset_name,
        force_resplit=force_resplit,
    )

    train_df = splits["train"]
    validation_df = splits["validation"]

    print(split_summary(splits).to_string(index=False))

    # Fit preprocessing on TRAIN only.
    preprocessor = DatasetPreprocessor(dataset_name)
    preprocessor.fit(train_df)

    (
        train_selected,
        selected_features,
        dropped_features,
        feature_manifest,
    ) = transform_and_select(
        preprocessor,
        train_df,
        dataset_name=f"{dataset_name}_litemv_train",
    )

    (
        validation_selected,
        validation_features,
        _validation_dropped,
        _validation_manifest,
    ) = transform_and_select(
        preprocessor,
        validation_df,
        dataset_name=f"{dataset_name}_litemv_validation",
        expected_features=selected_features,
    )

    if validation_features != selected_features:
        raise RuntimeError("Validation feature schema differs from training.")

    if len(selected_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{dataset_name}: expected {EXPECTED_FEATURE_COUNT} final hardened features, "
            f"got {len(selected_features)}."
        )

    sequence_length = int(SHARED_PARAMS["sequence_length"])

    X_train, y_train, train_meta = build_sequences(
        train_selected,
        selected_features,
        sequence_length,
        TRAIN_SEQUENCE_STRIDE,
    )
    X_validation, y_validation, validation_meta = build_sequences(
        validation_selected,
        selected_features,
        sequence_length,
        EVAL_SEQUENCE_STRIDE,
    )

    if len(y_validation) != len(validation_df):
        raise RuntimeError(
            f"{dataset_name}: expected one validation prediction per flow "
            f"({len(validation_df):,}); got {len(y_validation):,}."
        )

    class_weights = calculate_class_weights(y_train)

    print(f"\nSelected features: {len(selected_features)}")
    print(f"Train sequences: {len(y_train):,} | shape={X_train.shape}")
    print(
        f"Validation sequences: {len(y_validation):,} | "
        f"shape={X_validation.shape}"
    )
    print(f"Class weights: {class_weights}")
    print(f"Shared LITEMV params: {SHARED_PARAMS}")

    model = build_litemv_model(
        feature_count=len(selected_features),
        params=SHARED_PARAMS,
    )

    history, best_epoch = fit_litemv(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        params=SHARED_PARAMS,
        class_weights=class_weights,
    )

    validation_probabilities = predict_probabilities(
        model,
        X_validation,
        batch_size=int(SHARED_PARAMS["batch_size"]),
    )

    calibration = calibrate_validation_threshold(
        y_validation,
        validation_probabilities,
    )
    threshold = float(calibration["threshold"])
    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        threshold,
    )
    validation_predictions = predictions_from_probabilities(
        validation_probabilities,
        threshold,
    )

    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Persist artifacts.
    model.save(artifact_dir / "litemv.keras")
    preprocessor.save(artifact_dir / "preprocessor.joblib")

    with (artifact_dir / "selected_features.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(selected_features, handle, indent=2)

    with (artifact_dir / "shared_hyperparameters.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(_jsonable_params(SHARED_PARAMS), handle, indent=2)

    with (artifact_dir / "training_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "dataset": dataset_name,
                "random_state": RANDOM_STATE,
                "split_mode": split_metadata.get("mode"),
                "feature_count": len(selected_features),
                "expected_feature_count": EXPECTED_FEATURE_COUNT,
                "baseline_feature_count_before_hardening": BASELINE_EXPECTED_FEATURE_COUNT,
                "optimization_stage": "O1_43_HARDENED",
                "o1_mandatory_drops": sorted(O1_MANDATORY_DROPS),
                "o1_allowed_flag_feature": O1_ALLOWED_FLAG_FEATURE,
                "train_sequence_stride": TRAIN_SEQUENCE_STRIDE,
                "eval_sequence_stride": EVAL_SEQUENCE_STRIDE,
                "padding_value": PADDING_VALUE,
                "best_epoch": int(best_epoch),
                "class_weights": {
                    str(key): float(value)
                    for key, value in class_weights.items()
                },
                "architecture": "LITEMV",
                "backbone": "aeon.networks.LITENetwork(use_litemv=True)",
                "hyperparameter_scope": "frozen_shared_D2_D3_applied_to_D1",
                "dataset_specific_learned_values": [
                    "preprocessing_parameters",
                    "class_weights",
                    "early_stopping_epoch",
                    "decision_threshold",
                ],
                "test_used_for_training_or_calibration": False,
            },
            handle,
            indent=2,
        )

    with (artifact_dir / "decision_rule.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "calibration_scope": "validation_only_per_dataset",
                "method": calibration["method"],
                "threshold": threshold,
                "validation_fpr_cap": VALIDATION_FPR_CAP,
                "test_used_for_threshold_selection": False,
                "all_validation_candidates": calibration["all_candidates"],
            },
            handle,
            indent=2,
        )

    feature_manifest.to_csv(
        artifact_dir / "feature_drop_manifest.csv",
        index=False,
    )
    pd.DataFrame(
        {"Feature": selected_features}
    ).to_csv(
        artifact_dir / "selected_features.csv",
        index=False,
    )
    pd.DataFrame(
        {"DroppedFeature": dropped_features}
    ).to_csv(
        artifact_dir / "step3_dropped_features.csv",
        index=False,
    )
    split_summary(splits).to_csv(
        artifact_dir / "split_summary.csv",
        index=False,
    )
    _save_history(history, result_dir / "training_history.csv")
    _save_model_summary(model, artifact_dir / "model_summary.txt")

    # Save validation-only evidence.
    validation_table = validation_meta.copy()
    validation_table["MaliciousProbability"] = validation_probabilities
    validation_table["Predicted_Label"] = validation_predictions
    validation_table["ValidationCalibratedThreshold"] = threshold
    validation_table.to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
    )

    with (result_dir / "validation_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(validation_metrics, handle, indent=2)

    pd.DataFrame([validation_metrics]).to_csv(
        result_dir / "validation_metrics.csv",
        index=False,
    )
    pd.DataFrame(calibration["all_candidates"]).to_csv(
        result_dir / "threshold_candidates.csv",
        index=False,
    )

    print("\nValidation-only calibration:")
    print(
        f"  method={calibration['method']} | threshold={threshold:.6f} | "
        f"F2={validation_metrics['f2']:.4f} | "
        f"Recall={validation_metrics['recall_tpr']:.4f} | "
        f"FPR={validation_metrics['fpr']:.4f} | "
        f"PR-AUC={validation_metrics['pr_auc']:.4f}"
    )
    print("  TEST has not been used.")

    # Explicitly release large arrays before the next dataset.
    del X_train, X_validation
    tf.keras.backend.clear_session()

    return {
        "Dataset": dataset_name,
        "SplitMode": split_metadata.get("mode"),
        "FeatureCount": len(selected_features),
        "SequenceLength": sequence_length,
        "BestEpoch": int(best_epoch),
        "ThresholdMethod": calibration["method"],
        **validation_metrics,
    }


def train_all_datasets(
    only_dataset: str | None = None,
    force_resplit: bool = False,
) -> pd.DataFrame:
    dataset_names = (
        [only_dataset]
        if only_dataset is not None
        else list(BENCHMARK_DATASETS)
    )

    rows = [
        train_dataset(
            dataset_name,
            force_resplit=force_resplit,
        )
        for dataset_name in dataset_names
    ]

    summary = pd.DataFrame(rows)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        RESULT_ROOT / "litemv_validation_summary.csv",
        index=False,
    )

    # A single shared-configuration file makes the D2/D3 contract explicit.
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_ROOT / "shared_hyperparameters.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(_jsonable_params(SHARED_PARAMS), handle, indent=2)

    return summary
