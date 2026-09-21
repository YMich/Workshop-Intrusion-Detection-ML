from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

# TensorFlow defaults. Set before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
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
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers


tf.get_logger().setLevel("ERROR")
try:
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass


# ============================================================
# PROJECT PATHS
# ============================================================


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
INGESTED_ROOT = PROJECT_ROOT / "data" / "ingested"

DATASET_PATHS = {
    "dataset2": INGESTED_ROOT / "dataset2_ingested.csv",
    "dataset3": INGESTED_ROOT / "dataset3_ingested.csv",
}
BENCHMARK_DATASETS = tuple(DATASET_PATHS.keys())

# Final model outputs are isolated under the models_optimized_final namespace.
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models_optimized_final" / "AE"
RESULT_ROOT = PROJECT_ROOT / "results" / "models_optimized_final" / "AE"
SHARED_SPLIT_ROOT = PROJECT_ROOT / "artifacts" / "splits"
FINAL_CONFIG_PATH = ARTIFACT_ROOT / "final_configuration.json"


# ============================================================
# PROJECT MODULES
# ============================================================

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
from source_aware_split import (  # noqa: E402
    ORIGINAL_ROW_COL,
    get_or_create_shared_split,
)


# ============================================================
# LABELS / FINAL HARDENED FEATURE SCHEMA
# ============================================================

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

# Final O1-hardened schema: remove environment-dependent TCP/capture artifacts.
FINAL_EXPLICIT_DROP_FEATURES = (
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Total Connection Flow Time",
    "Fwd Header Length",
    "Bwd Header Length",
)
FINAL_FLAG_KEEP = "URG Flag Count"
EXPECTED_BASELINE_FEATURE_COUNT = 57
EXPECTED_FINAL_FEATURE_COUNT = 43


def apply_final_feature_pruning(
    selected_df: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Apply the final 43-feature hardening rule."""
    explicit = set(FINAL_EXPLICIT_DROP_FEATURES)
    dropped: list[str] = []
    kept: list[str] = []

    for feature in selected_features:
        drop_explicit = feature in explicit
        drop_flag = ("Flag" in feature) and feature != FINAL_FLAG_KEEP
        if drop_explicit or drop_flag:
            dropped.append(feature)
        else:
            kept.append(feature)

    if len(selected_features) == EXPECTED_BASELINE_FEATURE_COUNT:
        if len(kept) != EXPECTED_FINAL_FEATURE_COUNT:
            raise RuntimeError(
                "Final feature pruning expected 43 features from the 57-feature "
                f"baseline, but produced {len(kept)}. Dropped={dropped}"
            )

    return selected_df, kept, dropped


# ============================================================
# FINAL FIXED SHARED AUTOENCODER CONFIGURATION (O5)
# ============================================================

# Same model hyperparameters are used for Dataset 2 and Dataset 3.
FINAL_PARAMS = {
    "encoder_hidden_dims": (64, 32),
    "bottleneck_dim": 4,
    "hidden_activation": "relu",
    "output_activation": "linear",
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "use_bias": True,
    "l2_regularization": 1e-5,
    "learning_rate": 2e-3,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-7,
    "adam_amsgrad": False,
    "loss": "mse",
    "huber_delta": 1.0,
    "batch_size": 256,
    "epochs": 20,
    "shuffle": True,
    "fit_verbose": 0,
    "internal_validation_fraction": 0.10,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-5,
    "denoising_noise_std": 0.0,
    "steps_per_execution": 16,
}
FINAL_SCORE_MODE = "top5_mse"
FINAL_RANDOM_STATE = 42

# Final threshold is still learned separately from each dataset's validation set.
FINAL_THRESHOLD_FPR_CANDIDATES = [0.01, 0.02, 0.05]
FINAL_THRESHOLD_BENIGN_QUANTILES = [0.95, 0.975, 0.98, 0.99, 0.995]
THRESHOLD_CALIBRATION_VERSION = "validation_f1_primary_v2"


def get_final_configuration() -> tuple[dict, str]:
    return deepcopy(FINAL_PARAMS), FINAL_SCORE_MODE


def _jsonable_params(params: dict) -> dict:
    return {
        key: (
            list(value)
            if isinstance(value, tuple)
            else value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in params.items()
    }


def save_final_configuration() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "shared_dataset2_dataset3",
        "status": "final_optimized",
        "parameters": _jsonable_params(FINAL_PARAMS),
        "score_mode": FINAL_SCORE_MODE,
        "feature_count": EXPECTED_FINAL_FEATURE_COUNT,
        "feature_hardening_rule": {
            "explicit_drops": list(FINAL_EXPLICIT_DROP_FEATURES),
            "drop_all_flag_features_except": FINAL_FLAG_KEEP,
        },
        "threshold_scope": "per_dataset_validation_only",
    }
    with FINAL_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# ============================================================
# REPRODUCIBILITY / DATA HELPERS
# ============================================================


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def benign_only(df: pd.DataFrame) -> pd.DataFrame:
    result = df[df[LABEL_COL].astype(int) == BENIGN_LABEL].copy()
    if result.empty:
        raise ValueError("No benign rows are available for Autoencoder training.")
    return result


def transform_and_select(
    preprocessor: DatasetPreprocessor,
    df: pd.DataFrame,
    dataset_name: str,
    expected_features: list[str] | None = None,
):
    normalized = preprocessor.transform(df)

    selected_df, baseline_features, _dropped, _manifest = select_and_engineer_dataset(
        dataset_name=dataset_name,
        df=normalized,
        expected_selected_features=None,
    )

    selected_df, selected_features, final_dropped = apply_final_feature_pruning(
        selected_df,
        list(baseline_features),
    )

    if expected_features is not None and list(selected_features) != list(expected_features):
        missing = [f for f in expected_features if f not in selected_features]
        extra = [f for f in selected_features if f not in expected_features]
        raise ValueError(
            f"{dataset_name}: final hardened feature schema differs from training. "
            f"Missing={missing}; Extra={extra}"
        )

    X = selected_df[selected_features].to_numpy(dtype=np.float32)
    y = selected_df[LABEL_COL].astype(int).to_numpy()
    return selected_df, X, y, selected_features, final_dropped


# ============================================================
# AUTOENCODER
# ============================================================


def _keras_loss(params: dict):
    loss_name = str(params["loss"]).lower()
    if loss_name == "mse":
        return keras.losses.MeanSquaredError()
    if loss_name == "mae":
        return keras.losses.MeanAbsoluteError()
    if loss_name == "huber":
        return keras.losses.Huber(delta=float(params["huber_delta"]))
    raise ValueError(f"Unsupported Autoencoder loss: {params['loss']}")


def build_autoencoder(input_dim: int, params: dict) -> keras.Model:
    regularizer = (
        regularizers.l2(float(params["l2_regularization"]))
        if float(params["l2_regularization"]) > 0
        else None
    )

    inputs = keras.Input(shape=(input_dim,), dtype="float32", name="features")
    x = inputs

    for index, units in enumerate(params["encoder_hidden_dims"]):
        x = layers.Dense(
            units=int(units),
            activation=params["hidden_activation"],
            use_bias=bool(params["use_bias"]),
            kernel_initializer=params["kernel_initializer"],
            bias_initializer=params["bias_initializer"],
            kernel_regularizer=regularizer,
            name=f"encoder_dense_{index + 1}",
        )(x)

    x = layers.Dense(
        units=int(params["bottleneck_dim"]),
        activation=params["hidden_activation"],
        use_bias=bool(params["use_bias"]),
        kernel_initializer=params["kernel_initializer"],
        bias_initializer=params["bias_initializer"],
        kernel_regularizer=regularizer,
        name="bottleneck",
    )(x)

    for index, units in enumerate(reversed(params["encoder_hidden_dims"])):
        x = layers.Dense(
            units=int(units),
            activation=params["hidden_activation"],
            use_bias=bool(params["use_bias"]),
            kernel_initializer=params["kernel_initializer"],
            bias_initializer=params["bias_initializer"],
            kernel_regularizer=regularizer,
            name=f"decoder_dense_{index + 1}",
        )(x)

    outputs = layers.Dense(
        units=int(input_dim),
        activation=params["output_activation"],
        use_bias=bool(params["use_bias"]),
        kernel_initializer=params["kernel_initializer"],
        bias_initializer=params["bias_initializer"],
        name="reconstruction",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="flow_autoencoder")
    optimizer = keras.optimizers.Adam(
        learning_rate=float(params["learning_rate"]),
        beta_1=float(params["adam_beta_1"]),
        beta_2=float(params["adam_beta_2"]),
        epsilon=float(params["adam_epsilon"]),
        amsgrad=bool(params["adam_amsgrad"]),
    )
    model.compile(
        optimizer=optimizer,
        loss=_keras_loss(params),
        steps_per_execution=int(params.get("steps_per_execution", 1)),
    )
    return model


def _corrupt_with_noise(
    X: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_std <= 0:
        return X.copy()
    noise = rng.normal(0.0, float(noise_std), size=X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


def fit_autoencoder(
    model: keras.Model,
    X_benign_train: np.ndarray,
    params: dict,
    seed: int,
):
    if len(X_benign_train) < 20:
        raise ValueError(
            "Too few benign training rows for Autoencoder fitting: "
            f"{len(X_benign_train)}"
        )

    X_fit, X_early_stop = train_test_split(
        X_benign_train,
        test_size=float(params["internal_validation_fraction"]),
        random_state=seed,
        shuffle=True,
    )

    rng = np.random.default_rng(seed)
    X_fit_input = _corrupt_with_noise(
        X_fit, float(params["denoising_noise_std"]), rng
    )
    X_early_input = _corrupt_with_noise(
        X_early_stop, float(params["denoising_noise_std"]), rng
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=float(params["early_stopping_min_delta"]),
        patience=int(params["early_stopping_patience"]),
        mode="min",
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_fit_input,
        X_fit,
        validation_data=(X_early_input, X_early_stop),
        batch_size=int(params["batch_size"]),
        epochs=int(params["epochs"]),
        shuffle=bool(params["shuffle"]),
        verbose=int(params["fit_verbose"]),
        callbacks=[early_stopping],
    )

    val_loss = history.history.get("val_loss", [])
    if val_loss:
        best_epoch = int(np.argmin(val_loss) + 1)
        best_val_loss = float(np.min(val_loss))
    else:
        best_epoch = len(history.history.get("loss", []))
        best_val_loss = float("nan")

    return history, best_epoch, best_val_loss


def reconstruct_matrix(
    model: keras.Model,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(X), int(batch_size)):
        batch = X[start:start + int(batch_size)]
        outputs.append(model(batch, training=False).numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def score_from_reconstruction(
    X: np.ndarray,
    reconstructed: np.ndarray,
    score_mode: str,
) -> np.ndarray:
    absolute_error = np.abs(X - reconstructed)
    squared_error = np.square(X - reconstructed)

    if score_mode == "mse":
        return np.mean(squared_error, axis=1)
    if score_mode == "mae":
        return np.mean(absolute_error, axis=1)
    if score_mode in {"top5_mse", "top10_mse"}:
        k = 5 if score_mode == "top5_mse" else 10
        k = min(k, squared_error.shape[1])
        top_k = np.partition(
            squared_error,
            kth=squared_error.shape[1] - k,
            axis=1,
        )[:, -k:]
        return np.mean(top_k, axis=1)

    raise ValueError(f"Unknown reconstruction score mode: {score_mode}")


# ============================================================
# VALIDATION-ONLY THRESHOLD CALIBRATION
# ============================================================


def _best_fbeta_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    beta: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        raise ValueError("Unable to calibrate reconstruction threshold.")

    precision = precision[:-1]
    recall = recall[:-1]
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
    if len(candidates) > 1:
        index = int(candidates[np.argmax(recall[candidates])])
    else:
        index = int(candidates[0])
    return float(thresholds[index])


def _recall_at_fpr_cap_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    fpr_cap: float,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eligible = np.flatnonzero(fpr <= float(fpr_cap))
    if len(eligible) == 0:
        return float(np.nextafter(np.max(scores), np.inf))

    finite = eligible[np.isfinite(thresholds[eligible])]
    if len(finite) > 0:
        eligible = finite

    best_tpr = np.max(tpr[eligible])
    candidates = eligible[np.isclose(tpr[eligible], best_tpr)]
    min_fpr = np.min(fpr[candidates])
    candidates = candidates[np.isclose(fpr[candidates], min_fpr)]
    index = int(candidates[np.argmax(thresholds[candidates])])
    return float(thresholds[index])


def _benign_quantile_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    quantile: float,
) -> float:
    benign_scores = np.asarray(scores, dtype=float)[np.asarray(y_true, dtype=int) == BENIGN_LABEL]
    if len(benign_scores) == 0:
        raise ValueError("Validation split contains no benign rows for quantile calibration.")
    return float(np.quantile(benign_scores, float(quantile)))


def calculate_metrics(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores > threshold).astype(int)

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
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def calibrate_final_threshold(
    y_true,
    scores,
) -> tuple[float, str, pd.DataFrame]:
    """Select the final threshold using validation F1 only; save alternatives for diagnostics."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if set(np.unique(y_true)) != {BENIGN_LABEL, MALICIOUS_LABEL}:
        raise ValueError(
            "Final threshold calibration requires benign and malicious validation rows."
        )

    primary_threshold = _best_fbeta_threshold(y_true, scores, beta=1.0)
    candidate_specs: list[tuple[str, float]] = [
        ("validation_f1", primary_threshold),
    ]

    for quantile in FINAL_THRESHOLD_BENIGN_QUANTILES:
        candidate_specs.append(
            (
                f"benign_q{quantile:g}",
                _benign_quantile_threshold(y_true, scores, quantile),
            )
        )

    candidate_specs.append(
        ("validation_f2", _best_fbeta_threshold(y_true, scores, beta=2.0))
    )

    for fpr_cap in FINAL_THRESHOLD_FPR_CANDIDATES:
        candidate_specs.append(
            (
                f"recall_at_fpr_{fpr_cap:g}",
                _recall_at_fpr_cap_threshold(y_true, scores, fpr_cap),
            )
        )

    seen = set()
    rows = []
    for method, threshold in candidate_specs:
        key = round(float(threshold), 12)
        if key in seen:
            continue
        seen.add(key)
        metrics = calculate_metrics(y_true, scores, float(threshold))
        rows.append(
            {
                "threshold_method": method,
                "threshold": float(threshold),
                "selected_for_final": method == "validation_f1",
                **metrics,
            }
        )

    candidates = pd.DataFrame(rows).sort_values(
        by=["selected_for_final", "f1", "fpr"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return float(primary_threshold), "validation_f1", candidates


# ============================================================
# SHARED SPLIT / TRAINING HELPERS
# ============================================================


def _shared_split_signature(manifest: pd.DataFrame) -> str:
    columns = [
        column
        for column in [
            ORIGINAL_ROW_COL,
            SOURCE_FILE_COL,
            LABEL_COL,
            "Timestamp",
            "__Split",
        ]
        if column in manifest.columns
    ]
    payload = (
        manifest[columns]
        .sort_values(ORIGINAL_ROW_COL)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _drop_shared_split_helper_columns(
    splits: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        split_name: split_df.drop(
            columns=[ORIGINAL_ROW_COL],
            errors="ignore",
        ).copy()
        for split_name, split_df in splits.items()
    }


def train_final_model(
    dataset_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    expected_selected_features: list[str],
):
    params, score_mode = get_final_configuration()
    benign_train_df = benign_only(train_df)

    # Fit preprocessing only on benign training rows, exactly as in the baseline AE.
    preprocessor = DatasetPreprocessor(f"{dataset_name}_autoencoder_final_optimized")
    preprocessor.fit(benign_train_df)

    _, X_benign_train, y_benign_train, selected_features, _ = transform_and_select(
        preprocessor,
        benign_train_df,
        dataset_name,
        expected_features=expected_selected_features,
    )
    _, X_validation, y_validation, validation_features, _ = transform_and_select(
        preprocessor,
        validation_df,
        dataset_name,
        expected_features=selected_features,
    )

    if validation_features != selected_features:
        raise RuntimeError(f"{dataset_name}: validation feature order differs from training.")
    if set(np.unique(y_benign_train)) != {BENIGN_LABEL}:
        raise RuntimeError("Malicious rows reached final Autoencoder fitting.")

    keras.backend.clear_session()
    set_random_seed(FINAL_RANDOM_STATE)
    model = build_autoencoder(X_benign_train.shape[1], params)
    history, best_epoch, best_internal_val_loss = fit_autoencoder(
        model,
        X_benign_train,
        params,
        FINAL_RANDOM_STATE,
    )

    validation_reconstruction = reconstruct_matrix(
        model,
        X_validation,
        batch_size=int(params["batch_size"]),
    )
    validation_scores = score_from_reconstruction(
        X_validation,
        validation_reconstruction,
        score_mode,
    )
    threshold, threshold_method, threshold_candidates = calibrate_final_threshold(
        y_validation,
        validation_scores,
    )
    validation_metrics = calculate_metrics(
        y_validation,
        validation_scores,
        threshold,
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "threshold": threshold,
        "threshold_method": threshold_method,
        "threshold_candidates": threshold_candidates,
        "history": history,
        "validation_metrics": validation_metrics,
        "benign_training_rows": len(X_benign_train),
        "best_epoch": best_epoch,
        "best_internal_val_loss": best_internal_val_loss,
    }


def save_dataset_training_outputs(
    dataset_name: str,
    manifest: pd.DataFrame,
    split_metadata: dict,
    final: dict,
) -> None:
    params, score_mode = get_final_configuration()
    artifact_dir = ARTIFACT_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    signature = _shared_split_signature(manifest)

    final["model"].save(artifact_dir / "autoencoder.keras")
    final["preprocessor"].save(artifact_dir / "preprocessor.joblib")
    manifest.to_csv(artifact_dir / "split_manifest.csv", index=False)
    pd.DataFrame({"Feature": final["selected_features"]}).to_csv(
        artifact_dir / "selected_features.csv", index=False
    )

    # Legacy filename retained for evaluator compatibility; values are frozen, not searched here.
    with (artifact_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable_params(params), handle, indent=2)

    with (artifact_dir / "decision_rule.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split_signature": signature,
                "shared_split_manifest": str(
                    SHARED_SPLIT_ROOT / f"{dataset_name}_source_aware_split.csv"
                ),
                "split_strategy": split_metadata.get("mode"),
                "oversized_source": split_metadata.get("oversized_source"),
                "score_mode": score_mode,
                "threshold_method": final["threshold_method"],
                "threshold_calibration_version": THRESHOLD_CALIBRATION_VERSION,
                "threshold": float(final["threshold"]),
                "test_used_for_selection": False,
            },
            handle,
            indent=2,
        )

    with (artifact_dir / "training_strategy.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "Dense Autoencoder",
                "status": "final_optimized_O5",
                "benchmark_datasets": list(BENCHMARK_DATASETS),
                "hyperparameter_scope": "same fixed model hyperparameters for Dataset 2 and Dataset 3",
                "evaluation_strategy": (
                    "persisted source-aware 70/15/15 holdout shared across models"
                ),
                "split_mode": split_metadata.get("mode"),
                "oversized_source": split_metadata.get("oversized_source"),
                "imbalance_strategy": "benign-only reconstruction training",
                "preprocessing_fit_scope": "this dataset's benign training rows only",
                "detector_fit_scope": "this dataset's benign training rows only",
                "internal_early_stopping_scope": "10% split from benign training rows",
                "threshold_calibration_scope": "this dataset's full mixed validation split only",
                "final_threshold_selection": "maximize validation F1",
                "test_used_for_model_or_threshold_selection": False,
                "final_configuration": _jsonable_params(params),
                "score_mode": score_mode,
                "feature_count": len(final["selected_features"]),
                "dropped_feature_rule": (
                    "drop FWD/Bwd Init Win Bytes, Total Connection Flow Time, "
                    "Fwd/Bwd Header Length, and every *Flag* feature except exact "
                    "URG Flag Count if present"
                ),
                "benign_training_rows": int(final["benign_training_rows"]),
                "best_epoch": int(final["best_epoch"]),
                "best_internal_validation_loss": float(final["best_internal_val_loss"]),
            },
            handle,
            indent=2,
        )

    pd.DataFrame(final["history"].history).to_csv(
        result_dir / "final_training_history.csv", index=False
    )
    pd.DataFrame([final["validation_metrics"]]).to_csv(
        result_dir / "final_validation_metrics.csv", index=False
    )
    final["threshold_candidates"].to_csv(
        result_dir / "validation_threshold_candidates.csv", index=False
    )

    pd.DataFrame(
        [
            {
                "Dataset": dataset_name,
                "Split": split_name,
                "Rows": int(split_metadata["row_counts"][split_name]),
                "Fraction": float(split_metadata["row_ratios"][split_name]),
            }
            for split_name in ["train", "validation", "test"]
        ]
    ).to_csv(result_dir / "shared_split_sizes.csv", index=False)


def _artifacts_are_compatible(
    dataset_name: str,
    manifest: pd.DataFrame,
) -> bool:
    artifact_dir = ARTIFACT_ROOT / dataset_name
    required = [
        artifact_dir / "autoencoder.keras",
        artifact_dir / "preprocessor.joblib",
        artifact_dir / "selected_features.csv",
        artifact_dir / "best_hyperparameters.json",
        artifact_dir / "decision_rule.json",
    ]
    if not all(path.exists() for path in required):
        return False

    try:
        with (artifact_dir / "best_hyperparameters.json").open("r", encoding="utf-8") as handle:
            params = json.load(handle)
        if params != _jsonable_params(FINAL_PARAMS):
            return False

        with (artifact_dir / "decision_rule.json").open("r", encoding="utf-8") as handle:
            rule = json.load(handle)
        if rule.get("score_mode") != FINAL_SCORE_MODE:
            return False
        if rule.get("threshold_calibration_version") != THRESHOLD_CALIBRATION_VERSION:
            return False
        if rule.get("split_signature") != _shared_split_signature(manifest):
            return False

        features = pd.read_csv(artifact_dir / "selected_features.csv")["Feature"].tolist()
        if len(features) != EXPECTED_FINAL_FEATURE_COUNT:
            return False
    except Exception:
        return False

    return True


def train_dataset(
    dataset_name: str,
    force: bool = False,
    force_resplit: bool = False,
) -> None:
    print("\n" + "=" * 90)
    print(f"FINAL OPTIMIZED AUTOENCODER - {dataset_name.upper()}")
    print("=" * 90)

    input_path = DATASET_PATHS[dataset_name]
    if not input_path.exists():
        raise FileNotFoundError(f"Missing ingested dataset: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    splits, manifest, split_metadata = get_or_create_shared_split(
        dataset_name=dataset_name,
        df=df,
        split_root=SHARED_SPLIT_ROOT,
        force_rebuild=force_resplit,
    )
    splits = _drop_shared_split_helper_columns(splits)

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

    if not force and not force_resplit and _artifacts_are_compatible(dataset_name, manifest):
        print("[RESUME] Compatible final AE artifacts already exist. Skipping retraining.")
        return

    params, score_mode = get_final_configuration()
    print("\nFrozen FINAL shared configuration:")
    print("  encoder_hidden_dims: (64, 32)")
    print("  bottleneck_dim: 4")
    print("  learning_rate: 0.002")
    print("  loss: mse")
    print("  l2_regularization: 1e-05")
    print("  denoising_noise_std: 0.0")
    print("  batch_size: 256")
    print(f"  score_mode: {score_mode}")

    # Derive and verify the final feature schema from benign TRAIN only.
    benign_train_df = benign_only(train_df)
    schema_preprocessor = DatasetPreprocessor(f"{dataset_name}_final_ae_schema")
    schema_preprocessor.fit(benign_train_df)
    _, _, _, final_features, final_dropped = transform_and_select(
        schema_preprocessor,
        benign_train_df,
        dataset_name=f"{dataset_name}_final_ae_schema",
        expected_features=None,
    )

    if len(final_features) != EXPECTED_FINAL_FEATURE_COUNT:
        raise RuntimeError(
            f"{dataset_name}: expected {EXPECTED_FINAL_FEATURE_COUNT} final features, "
            f"got {len(final_features)}."
        )

    print(f"Final hardened feature count: {len(final_features)}")
    print("Dropped by final hardening rule:")
    for feature in final_dropped:
        print(f"  - {feature}")

    final = train_final_model(
        dataset_name=dataset_name,
        train_df=train_df,
        validation_df=validation_df,
        expected_selected_features=final_features,
    )
    save_dataset_training_outputs(
        dataset_name=dataset_name,
        manifest=manifest,
        split_metadata=split_metadata,
        final=final,
    )

    print(f"Validation threshold: {final['threshold']:.8f} ({final['threshold_method']})")
    print(f"Validation Recall / TPR: {final['validation_metrics']['recall_tpr']:.4f}")
    print(f"Validation FPR: {final['validation_metrics']['fpr']:.4f}")
    print(f"Validation F1: {final['validation_metrics']['f1']:.4f}")
    print(f"Validation PR-AUC: {final['validation_metrics']['pr_auc']:.4f}")
    print(f"Benign training rows: {final['benign_training_rows']:,}")
    print(f"Untouched test rows: {len(test_df):,}")


def train_all_datasets(
    force: bool = False,
    force_resplit: bool = False,
) -> None:
    save_final_configuration()
    print(
        "Using FINAL shared AE configuration for Dataset 2 and Dataset 3: "
        "43 hardened features, 64-32 encoder, bottleneck=4, LR=0.002, "
        "MSE loss, L2=1e-5, noise=0.0, batch=256, score=Top-5 MSE."
    )
    for dataset_name in BENCHMARK_DATASETS:
        train_dataset(
            dataset_name,
            force=force,
            force_resplit=force_resplit,
        )
