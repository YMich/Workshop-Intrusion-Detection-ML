from __future__ import annotations

import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

# TensorFlow environment setup must happen before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from tensorflow.keras import layers, regularizers


tf.get_logger().setLevel("ERROR")

try:
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

GPU_DEVICES = tf.config.list_physical_devices("GPU")
for _gpu in GPU_DEVICES:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass


# ============================================================
# PROJECT PATHS
# ============================================================

# Expected location:
#   <PROJECT_ROOT>/src/models_baseline/LSTM/lstm.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]

INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"

# Baseline benchmark uses Dataset 2 and Dataset 3 only.
# PROJECT_ROOT is derived from this file location.
DATASET_PATHS = {
    "dataset2": INGESTED_ROOT / "dataset2_ingested.csv",
    "dataset3": INGESTED_ROOT / "dataset3_ingested.csv",
}
BENCHMARK_DATASETS = tuple(DATASET_PATHS.keys())

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models_baseline" / "LSTM"
RESULT_ROOT = PROJECT_ROOT / "results" / "models_baseline" / "LSTM"
SHARED_SPLIT_ROOT = PROJECT_ROOT / "artifacts" / "splits"
SHARED_HYPERPARAMETER_PATH = ARTIFACT_ROOT / "shared_configuration.json"


# ============================================================
# REUSE EXISTING PIPELINE + LOCAL SPLITTER
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent

for _path in [
    MODEL_DIR,
    SRC_ROOT / "preprocessing",
    SRC_ROOT / "feature_engineering",
]:
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from preprocessing import DatasetPreprocessor  # noqa: E402
from feature_selection import select_and_engineer_dataset  # noqa: E402
from source_aware_split import (  # noqa: E402
    ORIGINAL_ROW_COL,
    get_or_create_shared_split,
)


# ============================================================
# COLUMNS / LABELS
# ============================================================

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
TIMESTAMP_COL = "Timestamp"
FLOW_ID_COL = "Flow ID"

BENIGN_LABEL = 0
MALICIOUS_LABEL = 1
EXPECTED_FEATURE_COUNT = 57


# ============================================================
# TEMPORAL REPRESENTATION
# ============================================================

# Padding uses an impossible post-scaling sentinel. Right padding is compatible
# with Keras Masking and the fast LSTM path on supported GPUs.
MASK_VALUE = -1_000_000.0

# Training stride reduces nearly identical overlapping windows. Validation/test
# use stride 1 so every held-out flow receives one prediction.
TRAIN_SEQUENCE_STRIDE = 5
EVAL_SEQUENCE_STRIDE = 1


# ============================================================
# FIXED SHARED LSTM CONFIGURATION
# ============================================================

# Frozen baseline configuration used identically for Dataset 2 and Dataset 3.
# No hyperparameter search, batch-size sweep, or stability-based selection is
# performed by the submitted baseline code.
BASE_PARAMS = {
    "sequence_length": 20,
    "lstm_units": (32,),
    "dense_units": 32,
    "dropout_rate": 0.20,
    "recurrent_dropout": 0.0,
    "activation": "tanh",
    "recurrent_activation": "sigmoid",
    "dense_activation": "relu",
    "output_activation": "sigmoid",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "recurrent_initializer": "orthogonal",
    "bias_initializer": "zeros",
    "l2_regularization": 1e-5,
    "learning_rate": 1e-3,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-7,
    "adam_amsgrad": False,
    "batch_size": 128,
    "epochs": 25,
    "shuffle": True,
    "fit_verbose": 0,
    "early_stopping_patience": 4,
    "early_stopping_min_delta": 1e-4,
}

FINAL_RANDOM_STATE = 42

# ============================================================
# VALIDATION-ONLY THRESHOLD CALIBRATION
# ============================================================

# All threshold choices are computed strictly from VALIDATION probabilities.
THRESHOLD_METHODS = (
    "validation_f1",
    "validation_f2",
    "recall_at_fpr_cap",
)

VALIDATION_FPR_CAP = 0.02


# ============================================================
# REPRODUCIBILITY / DEVICE
# ============================================================


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def describe_compute_device() -> str:
    if GPU_DEVICES:
        return "GPU detected: " + ", ".join(device.name for device in GPU_DEVICES)
    return "No TensorFlow GPU detected; using CPU."


# ============================================================
# SHARED SOURCE-AWARE 70/15/15 SPLIT
# ============================================================


def load_dataset_and_split(
    dataset_name: str,
    force_resplit: bool = False,
):
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

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

    # __OriginalRowIndex exists only to make the shared manifest reusable. It is
    # not a predictive feature and must never reach preprocessing/model input.
    for split_name in list(splits):
        splits[split_name] = splits[split_name].drop(
            columns=[ORIGINAL_ROW_COL],
            errors="ignore",
        )

    return splits, manifest, metadata


# ============================================================
# REUSE STEP 2 + STEP 3
# ============================================================


def transform_and_select(
    preprocessor: DatasetPreprocessor,
    df: pd.DataFrame,
    dataset_name: str,
    expected_features: list[str] | None = None,
):
    normalized = preprocessor.transform(df)

    (
        selected_df,
        selected_features,
        _dropped,
        _manifest,
    ) = select_and_engineer_dataset(
        dataset_name=dataset_name,
        df=normalized,
        expected_selected_features=expected_features,
    )

    if len(selected_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{dataset_name}: expected {EXPECTED_FEATURE_COUNT} selected "
            f"features, got {len(selected_features)}."
        )

    return selected_df, selected_features


# ============================================================
# SEQUENCE CONSTRUCTION
# ============================================================


def _full_sliding_windows(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) < length:
        return np.empty((0, length, values.shape[1]), dtype=np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(
        values,
        window_shape=length,
        axis=0,
    )

    # sliding_window_view(axis=0) returns (n_windows, features, length).
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

    if len(values) == 0:
        return (
            np.empty((0, sequence_length, values.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    n_rows, n_features = values.shape

    sequences = np.full(
        (n_rows, sequence_length, n_features),
        MASK_VALUE,
        dtype=np.float32,
    )

    prefix_count = min(sequence_length - 1, n_rows)
    for target_index in range(prefix_count):
        valid_count = target_index + 1
        sequences[target_index, :valid_count, :] = values[:valid_count, :]

    if n_rows >= sequence_length:
        full_windows = _full_sliding_windows(values, sequence_length)
        sequences[sequence_length - 1 :, :, :] = full_windows

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
    Build causal sequences independently inside each SourceFile PRESENT IN THIS
    SPLIT. Therefore the oversized-source exception never creates a sequence
    that crosses train -> validation -> test boundaries.
    """
    required = {SOURCE_FILE_COL, TIMESTAMP_COL, LABEL_COL}
    missing = required - set(selected_df.columns)

    if missing:
        raise ValueError(
            f"Missing LSTM sequence metadata columns: {sorted(missing)}"
        )

    sequence_blocks = []
    label_blocks = []
    metadata_blocks = []

    for source_file, source_df in selected_df.groupby(
        SOURCE_FILE_COL,
        sort=True,
    ):
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
            ["__ParsedTimestamp"],
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
        raise ValueError("No LSTM sequences could be constructed.")

    X = np.concatenate(sequence_blocks, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(label_blocks, axis=0).astype(np.int64, copy=False)
    metadata = pd.concat(metadata_blocks, ignore_index=True)

    if not (len(X) == len(y) == len(metadata)):
        raise RuntimeError("Sequence, target, and metadata counts differ.")

    return X, y, metadata


# ============================================================
# CLASS IMBALANCE
# ============================================================


def calculate_class_weights(y_train: np.ndarray) -> dict[int, float]:
    y_train = np.asarray(y_train, dtype=int)
    classes = np.unique(y_train)

    if set(classes.tolist()) != {BENIGN_LABEL, MALICIOUS_LABEL}:
        raise ValueError(
            "Supervised LSTM training requires both classes. "
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


# ============================================================
# MODEL
# ============================================================


def build_lstm_model(feature_count: int, params: dict) -> keras.Model:
    sequence_length = int(params["sequence_length"])
    lstm_units = tuple(int(value) for value in params["lstm_units"])

    if not lstm_units:
        raise ValueError("lstm_units cannot be empty.")

    regularizer = regularizers.l2(float(params["l2_regularization"]))

    inputs = keras.Input(
        shape=(sequence_length, feature_count),
        name="flow_sequence",
    )

    x = layers.Masking(mask_value=MASK_VALUE, name="mask_padding")(inputs)

    for layer_index, units in enumerate(lstm_units):
        return_sequences = layer_index < len(lstm_units) - 1

        x = layers.LSTM(
            units=units,
            activation=params["activation"],
            recurrent_activation=params["recurrent_activation"],
            use_bias=bool(params["use_bias"]),
            kernel_initializer=params["kernel_initializer"],
            recurrent_initializer=params["recurrent_initializer"],
            bias_initializer=params["bias_initializer"],
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=0.0,
            recurrent_dropout=float(params["recurrent_dropout"]),
            return_sequences=return_sequences,
            name=f"lstm_{layer_index + 1}_{units}",
        )(x)

        dropout_rate = float(params["dropout_rate"])
        if dropout_rate > 0:
            x = layers.Dropout(
                dropout_rate,
                name=f"dropout_after_lstm_{layer_index + 1}",
            )(x)

    dense_units = int(params["dense_units"])
    if dense_units > 0:
        x = layers.Dense(
            dense_units,
            activation=params["dense_activation"],
            kernel_initializer=params["kernel_initializer"],
            bias_initializer=params["bias_initializer"],
            kernel_regularizer=regularizer,
            name="dense",
        )(x)

        if float(params["dropout_rate"]) > 0:
            x = layers.Dropout(
                float(params["dropout_rate"]),
                name="dropout_after_dense",
            )(x)

    outputs = layers.Dense(
        1,
        activation=params["output_activation"],
        kernel_initializer=params["kernel_initializer"],
        bias_initializer=params["bias_initializer"],
        name="malicious_probability",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="c2_lstm_classifier")

    optimizer = keras.optimizers.Adam(
        learning_rate=float(params["learning_rate"]),
        beta_1=float(params["adam_beta_1"]),
        beta_2=float(params["adam_beta_2"]),
        epsilon=float(params["adam_epsilon"]),
        amsgrad=bool(params["adam_amsgrad"]),
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


def fit_lstm(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    params: dict,
    class_weights: dict[int, float],
):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(params["early_stopping_patience"]),
        min_delta=float(params["early_stopping_min_delta"]),
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        class_weight=class_weights,
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        shuffle=bool(params["shuffle"]),
        verbose=int(params["fit_verbose"]),
        callbacks=[early_stopping],
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


# ============================================================
# VALIDATION-ONLY THRESHOLD CALIBRATION
# ============================================================


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

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_tnr": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
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

    # Tie-break toward higher recall.
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

    # Same recall -> lower FPR -> more conservative threshold.
    min_fpr = np.min(fpr[candidates])
    candidates = candidates[np.isclose(fpr[candidates], min_fpr)]
    best_index = int(candidates[np.argmax(thresholds[candidates])])

    return float(thresholds[best_index])


def calibrate_validation_threshold(
    y_validation: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Compare three decision rules using VALIDATION ONLY and freeze the best one.
    TEST labels/probabilities are never used here.
    """
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

    # Operational ranking for intrusion detection. F2 gives recall additional
    # weight, then F1, recall, lower FPR, and precision.
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
            if key not in {"method"}
        },
        "all_candidates": candidates,
    }


def _jsonable_params(params: dict) -> dict:
    return {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in params.items()
    }
















# ============================================================
# FINAL MODEL + FINAL VALIDATION-ONLY THRESHOLD
# ============================================================


def train_final_model(
    dataset_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    best_params: dict,
    expected_features: list[str],
):
    """
    Final model is trained on TRAIN only and uses VALIDATION only for:
      * early stopping;
      * final threshold calibration.

    It is NOT retrained on TRAIN+VALIDATION afterwards, because that would
    invalidate the calibrated threshold's relationship to the fitted model.
    TEST remains completely untouched.
    """
    preprocessor = DatasetPreprocessor(f"{dataset_name}_lstm_final")
    preprocessor.fit(train_df)

    train_selected, selected_features = transform_and_select(
        preprocessor,
        train_df,
        f"{dataset_name}_lstm_final_train",
        expected_features=expected_features,
    )

    validation_selected, validation_features = transform_and_select(
        preprocessor,
        validation_df,
        f"{dataset_name}_lstm_final_validation",
        expected_features=selected_features,
    )

    if validation_features != selected_features:
        raise RuntimeError("Final validation feature order differs from training.")

    sequence_length = int(best_params["sequence_length"])

    X_train, y_train, _ = build_sequences(
        train_selected,
        selected_features,
        sequence_length,
        TRAIN_SEQUENCE_STRIDE,
    )

    X_validation, y_validation, validation_metadata = build_sequences(
        validation_selected,
        selected_features,
        sequence_length,
        EVAL_SEQUENCE_STRIDE,
    )

    class_weights = calculate_class_weights(y_train)

    set_random_seed(FINAL_RANDOM_STATE)
    tf.keras.backend.clear_session()

    model = build_lstm_model(len(selected_features), best_params)

    history, best_epoch = fit_lstm(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        best_params,
        class_weights,
    )

    validation_probabilities = predict_probabilities(
        model,
        X_validation,
        int(best_params["batch_size"]),
    )

    calibration = calibrate_validation_threshold(
        y_validation,
        validation_probabilities,
    )

    training_info = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "train_sequences": int(len(y_train)),
        "validation_sequences": int(len(y_validation)),
        "train_benign_sequences": int((y_train == BENIGN_LABEL).sum()),
        "train_malicious_sequences": int((y_train == MALICIOUS_LABEL).sum()),
        "class_weight_benign": float(class_weights[BENIGN_LABEL]),
        "class_weight_malicious": float(class_weights[MALICIOUS_LABEL]),
        "best_epoch": int(best_epoch),
    }

    validation_predictions = validation_metadata.copy()
    validation_predictions["MaliciousProbability"] = validation_probabilities
    validation_predictions["Predicted_Label"] = predictions_from_probabilities(
        validation_probabilities,
        calibration["threshold"],
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "history": history,
        "calibration": calibration,
        "training_info": training_info,
        "validation_predictions": validation_predictions,
    }


# ============================================================
# SAVE ARTIFACTS
# ============================================================



# ============================================================
# PER-DATASET FINAL ARTIFACTS USING THE SHARED HYPERPARAMETERS
# ============================================================


def save_training_outputs(
    dataset_name: str,
    split_manifest: pd.DataFrame,
    split_metadata: dict,
    shared_best_params: dict,
    final: dict,
) -> None:
    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name

    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    final["model"].save(artifact_dir / "lstm_model.keras")
    final["preprocessor"].save(artifact_dir / "preprocessor.joblib")

    pd.DataFrame({"Feature": final["selected_features"]}).to_csv(
        artifact_dir / "selected_features.csv",
        index=False,
    )

    # Both dataset directories receive the exact same frozen configuration.
    with (artifact_dir / "model_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_jsonable_params(shared_best_params), handle, indent=2)

    calibration = final["calibration"]

    with (artifact_dir / "decision_rule.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "calibration_scope": "validation_only_per_dataset",
                "method": calibration["method"],
                "threshold": float(calibration["threshold"]),
                "validation_fpr_cap": float(VALIDATION_FPR_CAP),
                "test_used_for_threshold_selection": False,
                "all_validation_candidates": calibration["all_candidates"],
            },
            handle,
            indent=2,
        )

    with (artifact_dir / "training_strategy.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "model": "supervised LSTM sequence classifier",
                "benchmark_datasets": list(BENCHMARK_DATASETS),
                "hyperparameter_scope": "shared across Dataset 2 and Dataset 3",
                "configuration_mode": (
                    "fixed shared baseline configuration; no hyperparameter search "
                    "is performed during this run"
                ),
                "shared_hyperparameter_artifact": str(SHARED_HYPERPARAMETER_PATH),
                "split_strategy": split_metadata,
                "shared_split_manifest": str(
                    SHARED_SPLIT_ROOT / f"{dataset_name}_source_aware_split.csv"
                ),
                "imbalance_strategy": "balanced class weights from this dataset's training sequences only",
                "preprocessing_fit_scope": "this dataset's training split only",
                "early_stopping_scope": "this dataset's validation split only",
                "threshold_calibration_scope": "this dataset's validation split only",
                "final_model_retrained_on_validation": False,
                "test_used_for_model_selection": False,
                "sequence_boundary_rule": "never cross SourceFile or split boundary",
                "sequence_ordering": "chronological Timestamp order",
                "training_sequence_stride": int(TRAIN_SEQUENCE_STRIDE),
                "evaluation_sequence_stride": int(EVAL_SEQUENCE_STRIDE),
                "mask_value": float(MASK_VALUE),
                **final["training_info"],
            },
            handle,
            indent=2,
        )

    split_manifest.to_csv(artifact_dir / "split_manifest.csv", index=False)

    history_df = pd.DataFrame(final["history"].history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = "Epoch"
    history_df.to_csv(result_dir / "final_training_history.csv")

    pd.DataFrame([final["calibration"]["metrics"]]).to_csv(
        result_dir / "validation_metrics_at_selected_threshold.csv",
        index=False,
    )

    final["validation_predictions"].to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
    )


# ============================================================
# JOINT BENCHMARK TRAINING
# ============================================================


def _print_split_summary(
    dataset_name: str,
    splits: dict[str, pd.DataFrame],
    split_metadata: dict,
) -> None:
    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]
    total = len(train_df) + len(validation_df) + len(test_df)

    print("\n" + "-" * 90)
    print(f"{dataset_name.upper()} SPLIT")
    print("-" * 90)
    print(
        f"Rows | train={len(train_df):,} ({len(train_df)/total:.2%}), "
        f"validation={len(validation_df):,} ({len(validation_df)/total:.2%}), "
        f"test={len(test_df):,} ({len(test_df)/total:.2%})"
    )
    print(f"Split mode: {split_metadata['mode']}")
    if split_metadata.get("oversized_source") is not None:
        print(
            "Oversized SourceFile chronologically split: "
            f"{split_metadata['oversized_source']} "
            f"({split_metadata['oversized_rows']:,} rows)"
        )


def train_benchmark_datasets(force_resplit: bool = False) -> None:
    """Train D2/D3 LSTMs with one frozen shared configuration."""
    print("\n" + "=" * 90)
    print("LSTM BASELINE TRAINING - FIXED CONFIGURATION FOR DATASET 2 + DATASET 3")
    print("=" * 90)
    print(describe_compute_device())

    shared_params = dict(BASE_PARAMS)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with SHARED_HYPERPARAMETER_PATH.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable_params(shared_params), handle, indent=2)

    print("\nFrozen shared LSTM configuration:")
    print(f"  sequence_length: {shared_params['sequence_length']}")
    print(f"  lstm_units: {shared_params['lstm_units']}")
    print(f"  dense_units: {shared_params['dense_units']}")
    print(f"  dropout_rate: {shared_params['dropout_rate']}")
    print(f"  learning_rate: {shared_params['learning_rate']}")
    print(f"  batch_size: {shared_params['batch_size']}")

    for dataset_name in BENCHMARK_DATASETS:
        splits, manifest, split_metadata = load_dataset_and_split(
            dataset_name,
            force_resplit=force_resplit,
        )
        _print_split_summary(dataset_name, splits, split_metadata)

        print("\n" + "-" * 90)
        print(f"TRAIN FIXED BASELINE - {dataset_name.upper()}")
        print("-" * 90)

        final = train_final_model(
            dataset_name=dataset_name,
            train_df=splits["train"],
            validation_df=splits["validation"],
            best_params=shared_params,
            expected_features=None,
        )

        print(
            f"Selected threshold: {final['calibration']['threshold']:.6f} "
            f"({final['calibration']['method']}, validation only for {dataset_name})"
        )

        save_training_outputs(
            dataset_name=dataset_name,
            split_manifest=manifest,
            split_metadata=split_metadata,
            shared_best_params=shared_params,
            final=final,
        )

        print(f"Artifacts: {ARTIFACT_ROOT / dataset_name}")
        print(f"Results: {RESULT_ROOT / dataset_name}")



# Backward-compatible entry point used by run_lstm.py.
def train_all_datasets(force_resplit: bool = False) -> None:
    train_benchmark_datasets(force_resplit=force_resplit)

