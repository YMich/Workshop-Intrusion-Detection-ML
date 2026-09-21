from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from random_forest_config import (
    ARTIFACT_ROOT as STATIC_ARTIFACT_ROOT,
    BASE_RF_PARAMS,
    BENCHMARK_DATASETS,
    DATASET_PATHS,
    LABEL_COL,
    MODEL_RANDOM_STATE,
    PREDICTION_THRESHOLD,
    PROJECT_ROOT,
    RESULT_ROOT as STATIC_RESULT_ROOT,
    ROW_INDEX_COL,
    SOURCE_FILE_COL,
    TIMESTAMP_COL,
)
from random_forest_preprocessing import FoldMedianImputer
from random_forest_utils import (
    build_random_forest,
    calculate_binary_metrics,
    compute_training_class_weights,
)

# =============================================================================
# O4 TEMPORAL RANDOM FOREST
# =============================================================================
# Controlled continuation of the static chain:
#
#   B0: 57 fixed
#   O1: 10 fixed
#   O2: 12 fixed
#   O3: 12 + nested shared tuning
#   O4: SAME O3 RF hyperparameters + causal endpoint-aware temporal context
#
# For unbiased OOF comparisons, each outer fold uses the exact O3 configuration
# selected for that fold in O3_outer_fold_selections.csv.
#
# IP addresses and Timestamp are NEVER model inputs. They are used only to
# construct causal endpoint sequences and inter-flow timing features.
# =============================================================================

TEMPORAL_RESULT_ROOT = (
    PROJECT_ROOT / "results" / "models_optimized" / "RF" / "temporal_chain"
)
TEMPORAL_ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts" / "models_optimized" / "RF" / "temporal_chain"
)

SRC_IP_ALIASES = ("Src IP", "Source IP", "SrcIP", "SourceIP")
DST_IP_ALIASES = ("Dst IP", "Destination IP", "DstIP", "DestinationIP")
TIMESTAMP_ALIASES = (TIMESTAMP_COL, "TimeStamp", "Flow Timestamp")

BASE_10_FEATURES = [
    "Fwd IAT Mean",
    "Flow IAT Std",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Down/Up Ratio",
    "Fwd Act Data Pkts",
]
FWD_PACKET_LENGTH_CV = "Fwd Packet Length CV"
TOTAL_FWD_LENGTH = "Total Length of Fwd Packet"
BASE_12_FEATURES = BASE_10_FEATURES + [FWD_PACKET_LENGTH_CV, TOTAL_FWD_LENGTH]
TOTAL_FWD_LENGTH_ALIASES = (
    "Total Length of Fwd Packet",
    "Total Length of Fwd Packets",
    "TotLen Fwd Pkts",
)

# Features whose recent history is summarized. These are deliberately behavioral
# and do not include direct identifiers or absolute timestamps.
ROLLING_BEHAVIOR_FEATURES = [
    "Fwd IAT Mean",
    "Flow IAT Std",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Down/Up Ratio",
    "Fwd Act Data Pkts",
]

INTERFLOW_DELTA = "InterFlow Delta"
HISTORY_COUNT = "Endpoint History Count"
ROLL_WINDOWS = (3, 5, 10)

VARIANT_ORDER = ["R0", "RDELTA", "RROLL3", "RROLL5", "RROLL10"]

REPORT_METRICS = [
    "precision",
    "recall_tpr",
    "fpr",
    "f1",
    "roc_auc",
    "pr_auc",
    "fp",
    "fn",
    "tp",
]


# =============================================================================
# BASIC HELPERS
# =============================================================================


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _resolve_alias(columns, aliases, role: str) -> str:
    columns = list(columns)
    exact = set(columns)
    for alias in aliases:
        if alias in exact:
            return alias

    normalized = {_normalize_name(c): c for c in columns}
    for alias in aliases:
        key = _normalize_name(alias)
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not resolve {role}. Tried aliases={list(aliases)}. "
        f"Available columns include {columns[:50]}"
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = _numeric(numerator)
    denominator = _numeric(denominator)
    out = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = numerator.notna() & denominator.notna()
    nonzero = valid & (denominator.abs() > 1e-12)
    out.loc[nonzero] = numerator.loc[nonzero] / denominator.loc[nonzero]
    out.loc[valid & ~nonzero] = 0.0
    return out.replace([np.inf, -np.inf], np.nan)


def _parse_timestamp(series: pd.Series, dataset_name: str) -> pd.Series:
    """Parse CICFlowMeter timestamps without exposing absolute time to the RF."""
    text = series.astype(str).str.strip()

    # pandas >=2 supports format='mixed'. Try both month-first and day-first and
    # keep whichever parses more rows.
    candidates = []
    for dayfirst in (False, True):
        try:
            parsed = pd.to_datetime(
                text,
                errors="coerce",
                format="mixed",
                dayfirst=dayfirst,
            )
        except (TypeError, ValueError):
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
        candidates.append(parsed)

    parsed = max(candidates, key=lambda s: int(s.notna().sum()))
    bad = int(parsed.isna().sum())
    if bad:
        fraction = bad / max(len(parsed), 1)
        if fraction > 0.001:
            examples = text[parsed.isna()].head(5).tolist()
            raise ValueError(
                f"{dataset_name}: failed to parse {bad}/{len(parsed)} timestamps "
                f"({fraction:.3%}). Examples={examples}"
            )
        print(
            f"[TEMPORAL] {dataset_name}: {bad} timestamps could not be parsed; "
            "those rows will have missing temporal deltas and will be fold-imputed."
        )
    return parsed


def _load_static_fold_assignment(dataset_name: str, n_rows: int) -> np.ndarray:
    path = STATIC_ARTIFACT_ROOT / "folds" / f"{dataset_name}_outer_fold_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing static-chain fold manifest: {path}\n"
            "Run run_rf_static_optimization.py --grid full first."
        )

    manifest = pd.read_csv(path, low_memory=False)
    required = {ROW_INDEX_COL, "Outer_Fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    manifest[ROW_INDEX_COL] = pd.to_numeric(
        manifest[ROW_INDEX_COL], errors="raise"
    ).astype(np.int64)
    manifest["Outer_Fold"] = pd.to_numeric(
        manifest["Outer_Fold"], errors="raise"
    ).astype(int)
    manifest = manifest.sort_values(ROW_INDEX_COL).reset_index(drop=True)

    expected = np.arange(n_rows, dtype=np.int64)
    actual = manifest[ROW_INDEX_COL].to_numpy(dtype=np.int64)
    if len(manifest) != n_rows or not np.array_equal(actual, expected):
        raise RuntimeError(
            f"{dataset_name}: static fold manifest does not map exactly to current rows."
        )
    return manifest["Outer_Fold"].to_numpy(dtype=int)


def _parse_depth(value):
    if pd.isna(value) or str(value).strip().lower() in {"", "none", "nan"}:
        return None
    return int(float(value))


def _parse_max_features(value):
    text = str(value).strip()
    if text.lower() == "sqrt":
        return "sqrt"
    return float(text)


def load_o3_params_by_fold() -> dict[int, dict]:
    path = STATIC_RESULT_ROOT / "O3_outer_fold_selections.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing O3 outer-fold selections: {path}\n"
            "Run the static chain first."
        )

    table = pd.read_csv(path, low_memory=False)
    required = {
        "Outer_Fold",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    result = {}
    for row in table.itertuples(index=False):
        fold = int(getattr(row, "Outer_Fold"))
        params = deepcopy(BASE_RF_PARAMS)
        params.update(
            {
                "n_estimators": int(getattr(row, "n_estimators")),
                "max_depth": _parse_depth(getattr(row, "max_depth")),
                "min_samples_leaf": int(getattr(row, "min_samples_leaf")),
                "max_features": _parse_max_features(getattr(row, "max_features")),
                "bootstrap": bool(getattr(row, "bootstrap")),
            }
        )
        result[fold] = params

    if not result:
        raise RuntimeError("No O3 outer-fold hyperparameters were loaded.")
    return result


def load_o3_final_params() -> dict:
    path = STATIC_ARTIFACT_ROOT / "O3_final_shared_hyperparameters.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing O3 final shared parameters: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "parameters" not in payload:
        raise ValueError(f"{path}: missing 'parameters'.")
    return payload["parameters"]


# =============================================================================
# ENDPOINT-AWARE CAUSAL TEMPORAL FEATURE ENGINEERING
# =============================================================================


def load_temporal_base(dataset_name: str) -> pd.DataFrame:
    path = DATASET_PATHS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name}: missing dataset: {path}")

    raw = pd.read_csv(path, low_memory=False).copy()
    raw[ROW_INDEX_COL] = np.arange(len(raw), dtype=np.int64)
    raw[LABEL_COL] = pd.to_numeric(raw[LABEL_COL], errors="raise").astype(int)

    src_col = _resolve_alias(raw.columns, SRC_IP_ALIASES, "source IP")
    dst_col = _resolve_alias(raw.columns, DST_IP_ALIASES, "destination IP")
    ts_col = _resolve_alias(raw.columns, TIMESTAMP_ALIASES, "timestamp")
    total_col = _resolve_alias(
        raw.columns, TOTAL_FWD_LENGTH_ALIASES, "total forward-packet length"
    )

    required = set(BASE_10_FEATURES) | {
        SOURCE_FILE_COL,
        LABEL_COL,
        src_col,
        dst_col,
        ts_col,
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        total_col,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{dataset_name}: missing temporal source columns: {missing}")

    # Refined static features from O2/O3.
    raw[FWD_PACKET_LENGTH_CV] = _safe_ratio(
        raw["Fwd Packet Length Std"],
        _numeric(raw["Fwd Packet Length Mean"]).abs(),
    )
    raw[TOTAL_FWD_LENGTH] = _numeric(raw[total_col])
    for feature in BASE_12_FEATURES:
        raw[feature] = _numeric(raw[feature])

    raw["_SrcEndpoint"] = raw[src_col].astype(str)
    raw["_DstEndpoint"] = raw[dst_col].astype(str)
    raw["_ParsedTimestamp"] = _parse_timestamp(raw[ts_col], dataset_name)

    group_cols = [SOURCE_FILE_COL, "_SrcEndpoint", "_DstEndpoint"]
    raw = raw.sort_values(
        group_cols + ["_ParsedTimestamp", ROW_INDEX_COL],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    grouped = raw.groupby(group_cols, sort=False, dropna=False)
    raw[HISTORY_COUNT] = grouped.cumcount().astype(np.int64)

    # Current inter-flow delta is causal: current timestamp minus immediately
    # preceding timestamp for the same endpoint sequence.
    raw[INTERFLOW_DELTA] = (
        grouped["_ParsedTimestamp"].diff().dt.total_seconds().clip(lower=0)
    )

    # Rolling summaries use PREVIOUS rows only. shift(1) is the explicit no-future
    # guard. Current values are already available separately as the 12 static inputs.
    for window in ROLL_WINDOWS:
        for feature in ROLLING_BEHAVIOR_FEATURES:
            mean_name = f"{feature} Prev{window} Mean"
            std_name = f"{feature} Prev{window} Std"
            raw[mean_name] = grouped[feature].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
            raw[std_name] = grouped[feature].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).std(ddof=0)
            )

        raw[f"{INTERFLOW_DELTA} Prev{window} Mean"] = grouped[
            INTERFLOW_DELTA
        ].transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        raw[f"{INTERFLOW_DELTA} Prev{window} Std"] = grouped[
            INTERFLOW_DELTA
        ].transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).std(ddof=0))
        raw[f"{INTERFLOW_DELTA} Prev{window} CV"] = _safe_ratio(
            raw[f"{INTERFLOW_DELTA} Prev{window} Std"],
            raw[f"{INTERFLOW_DELTA} Prev{window} Mean"].abs(),
        )

    # Restore original row order so static fold manifests line up exactly.
    raw = raw.sort_values(ROW_INDEX_COL, kind="mergesort").reset_index(drop=True)

    # Hard privacy/leakage assertions: raw identifiers/time cannot enter model schemas.
    if len(raw) != len(pd.read_csv(path, usecols=[LABEL_COL])):
        raise RuntimeError(f"{dataset_name}: temporal feature engineering changed row count.")

    print(
        f"Loaded temporal {dataset_name}: rows={len(raw):,}, "
        f"endpoint sequences={raw.groupby(group_cols, dropna=False).ngroups:,}"
    )
    return raw


def rolling_feature_names(window: int) -> list[str]:
    names = []
    for feature in ROLLING_BEHAVIOR_FEATURES:
        names.extend(
            [
                f"{feature} Prev{window} Mean",
                f"{feature} Prev{window} Std",
            ]
        )
    names.extend(
        [
            f"{INTERFLOW_DELTA} Prev{window} Mean",
            f"{INTERFLOW_DELTA} Prev{window} Std",
            f"{INTERFLOW_DELTA} Prev{window} CV",
        ]
    )
    return names


def variant_features(variant: str) -> list[str]:
    if variant == "R0":
        return list(BASE_12_FEATURES)
    if variant == "RDELTA":
        return list(BASE_12_FEATURES) + [INTERFLOW_DELTA]
    if variant.startswith("RROLL"):
        window = int(variant.replace("RROLL", ""))
        if window not in ROLL_WINDOWS:
            raise ValueError(f"Unsupported rolling window {window}")
        return list(BASE_12_FEATURES) + [INTERFLOW_DELTA] + rolling_feature_names(window)
    raise ValueError(f"Unknown temporal variant: {variant}")


def _prepare_model_frame(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    required = [ROW_INDEX_COL, SOURCE_FILE_COL, LABEL_COL, HISTORY_COUNT] + features
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing model-frame columns: {missing}")
    frame = raw[required].copy()
    for feature in features:
        frame[feature] = _numeric(frame[feature])
    return frame


# =============================================================================
# OOF EVALUATION WITH O3 PER-FOLD HYPERPARAMETERS
# =============================================================================


def _fold_metric_or_nan(y_true, prob) -> dict:
    labels = set(pd.Series(y_true).astype(int).unique().tolist())
    if labels == {0, 1}:
        return calculate_binary_metrics(y_true, prob, threshold=PREDICTION_THRESHOLD)
    # FULL5 filtering can occasionally leave a single-class test fold. Overall OOF
    # metrics remain valid; this row is retained for fold accounting.
    return {
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "precision": np.nan,
        "recall_tpr": np.nan,
        "specificity_tnr": np.nan,
        "fpr": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "mcc": np.nan,
        "tn": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tp": np.nan,
        "threshold": PREDICTION_THRESHOLD,
    }


def evaluate_variant_oof(
    dataset_name: str,
    raw_temporal: pd.DataFrame,
    outer_assignment: np.ndarray,
    params_by_fold: dict[int, dict],
    variant: str,
    min_history: int | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = variant_features(variant)
    model_df = _prepare_model_frame(raw_temporal, features)
    model_df["_OuterFold"] = outer_assignment

    if min_history is not None:
        model_df = model_df[model_df[HISTORY_COUNT] >= int(min_history)].copy()
        if model_df.empty:
            raise RuntimeError(f"{dataset_name}/{variant}: FULL{min_history} population is empty.")

    stage = variant if min_history is None else f"{variant}_FULL{min_history}"
    oof_parts = []
    fold_rows = []
    importance_parts = []

    folds = sorted(pd.to_numeric(model_df["_OuterFold"], errors="raise").unique().tolist())
    for fold in folds:
        fold = int(fold)
        if fold not in params_by_fold:
            raise RuntimeError(f"{stage}: missing O3 parameters for outer fold {fold}.")

        train_df = model_df[model_df["_OuterFold"] != fold].copy()
        test_df = model_df[model_df["_OuterFold"] == fold].copy()
        if train_df.empty or test_df.empty:
            continue

        train_sources = set(train_df[SOURCE_FILE_COL].astype(str).unique())
        test_sources = set(test_df[SOURCE_FILE_COL].astype(str).unique())
        if train_sources & test_sources:
            raise RuntimeError(f"{dataset_name}/{stage}/fold{fold}: SourceFile leakage.")

        y_train = train_df[LABEL_COL].astype(int)
        if set(y_train.unique().tolist()) != {0, 1}:
            raise RuntimeError(
                f"{dataset_name}/{stage}/fold{fold}: training side lacks one class."
            )
        y_test = test_df[LABEL_COL].astype(int)

        imputer = FoldMedianImputer(f"temporal_{dataset_name}_{stage}_fold{fold}")
        X_train = imputer.fit_transform(train_df[features])
        X_test = imputer.transform(test_df[features])
        weights = compute_training_class_weights(y_train)
        params = deepcopy(params_by_fold[fold])
        model = build_random_forest(
            params=params,
            class_weights=weights,
            random_state=MODEL_RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= PREDICTION_THRESHOLD).astype(int)
        fold_metrics = _fold_metric_or_nan(y_test, prob)
        fold_rows.append(
            {
                "Variant": stage,
                "Dataset": dataset_name,
                "Outer_Fold": fold,
                "Feature_Count": len(features),
                "Rows": len(test_df),
                **fold_metrics,
            }
        )

        oof_parts.append(
            pd.DataFrame(
                {
                    ROW_INDEX_COL: test_df[ROW_INDEX_COL].to_numpy(dtype=np.int64),
                    SOURCE_FILE_COL: test_df[SOURCE_FILE_COL].astype(str).to_numpy(),
                    HISTORY_COUNT: test_df[HISTORY_COUNT].to_numpy(dtype=np.int64),
                    "Outer_Fold": fold,
                    "Actual_Label": y_test.to_numpy(dtype=int),
                    "Malicious_Probability": prob,
                    "Predicted_Label": pred,
                }
            )
        )

        importance_parts.append(
            pd.DataFrame(
                {
                    "Variant": stage,
                    "Dataset": dataset_name,
                    "Outer_Fold": fold,
                    "Feature": features,
                    "Is_Temporal": [feature not in BASE_12_FEATURES for feature in features],
                    "MDI_Importance": model.feature_importances_,
                }
            )
        )

    if not oof_parts:
        raise RuntimeError(f"{dataset_name}/{stage}: no OOF predictions produced.")

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(ROW_INDEX_COL).reset_index(drop=True)
    if oof[ROW_INDEX_COL].duplicated().any():
        raise RuntimeError(f"{dataset_name}/{stage}: duplicate OOF row predictions.")

    labels = set(oof["Actual_Label"].astype(int).unique().tolist())
    if labels != {0, 1}:
        raise RuntimeError(f"{dataset_name}/{stage}: OOF population lacks one class: {labels}")

    overall = calculate_binary_metrics(
        oof["Actual_Label"].to_numpy(dtype=int),
        oof["Malicious_Probability"].to_numpy(dtype=float),
        threshold=PREDICTION_THRESHOLD,
    )
    summary = {
        "Variant": stage,
        "Dataset": dataset_name,
        "Feature_Count": len(features),
        "Min_History": 0 if min_history is None else int(min_history),
        "Rows_Evaluated": int(len(oof)),
        "Hyperparameter_Regime": "same_O3_per_outer_fold_shared_selection",
        **overall,
    }

    importance = (
        pd.concat(importance_parts, ignore_index=True)
        if importance_parts
        else pd.DataFrame()
    )
    return summary, oof, pd.DataFrame(fold_rows), importance


# =============================================================================
# CROSS-DATASET TRANSFER WITH O3 FINAL SHARED DEPLOYMENT CONFIGURATION
# =============================================================================


def _fit_full(
    frame: pd.DataFrame,
    features: list[str],
    params: dict,
    name: str,
):
    y = frame[LABEL_COL].astype(int)
    if set(y.unique().tolist()) != {0, 1}:
        raise RuntimeError(f"{name}: training population lacks one class.")
    imputer = FoldMedianImputer(name)
    X = imputer.fit_transform(frame[features])
    weights = compute_training_class_weights(y)
    model = build_random_forest(
        params=deepcopy(params),
        class_weights=weights,
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(X, y)
    return model, imputer


def evaluate_transfer(
    temporal_frames: dict[str, pd.DataFrame],
    variant: str,
    final_params: dict,
    min_history: int | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    features = variant_features(variant)
    stage = variant if min_history is None else f"{variant}_FULL{min_history}"
    rows = []
    pred_parts = []

    for source, target in (("dataset2", "dataset3"), ("dataset3", "dataset2")):
        source_df = _prepare_model_frame(temporal_frames[source], features)
        target_df = _prepare_model_frame(temporal_frames[target], features)
        if min_history is not None:
            source_df = source_df[source_df[HISTORY_COUNT] >= int(min_history)].copy()
            target_df = target_df[target_df[HISTORY_COUNT] >= int(min_history)].copy()

        model, imputer = _fit_full(
            source_df,
            features,
            final_params,
            f"transfer_{stage}_{source}",
        )
        X_target = imputer.transform(target_df[features])
        y_target = target_df[LABEL_COL].astype(int).to_numpy()
        prob = model.predict_proba(X_target)[:, 1]
        pred = (prob >= PREDICTION_THRESHOLD).astype(int)
        metrics = calculate_binary_metrics(
            y_target,
            prob,
            threshold=PREDICTION_THRESHOLD,
        )
        rows.append(
            {
                "Variant": stage,
                "Train_Dataset": source,
                "Test_Dataset": target,
                "Feature_Count": len(features),
                "Min_History": 0 if min_history is None else int(min_history),
                "Rows_Evaluated": len(target_df),
                "Hyperparameter_Regime": "O3_final_shared_deployment_config",
                **metrics,
            }
        )
        pred_parts.append(
            pd.DataFrame(
                {
                    "Variant": stage,
                    "Train_Dataset": source,
                    "Test_Dataset": target,
                    ROW_INDEX_COL: target_df[ROW_INDEX_COL].to_numpy(dtype=np.int64),
                    HISTORY_COUNT: target_df[HISTORY_COUNT].to_numpy(dtype=np.int64),
                    "Actual_Label": y_target,
                    "Malicious_Probability": prob,
                    "Predicted_Label": pred,
                }
            )
        )

    return rows, pd.concat(pred_parts, ignore_index=True)


# =============================================================================
# REPORT HELPERS
# =============================================================================


def build_history_summary(raw: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for label in (0, 1):
        subset = raw[raw[LABEL_COL] == label]
        history = subset[HISTORY_COUNT].astype(int)
        rows.append(
            {
                "Dataset": dataset_name,
                "Label": label,
                "Rows": len(subset),
                "Median_History_Count": float(history.median()),
                "Pct_History_GE_1": float((history >= 1).mean()),
                "Pct_History_GE_3": float((history >= 3).mean()),
                "Pct_History_GE_5": float((history >= 5).mean()),
                "Pct_History_GE_10": float((history >= 10).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_temporal_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    rows = []
    for (variant, dataset), group in importance.groupby(["Variant", "Dataset"], sort=False):
        by_fold = (
            group.groupby(["Outer_Fold", "Is_Temporal"], as_index=False)["MDI_Importance"]
            .sum()
        )
        temporal = by_fold[by_fold["Is_Temporal"]].groupby("Outer_Fold")["MDI_Importance"].sum()
        all_folds = sorted(group["Outer_Fold"].unique().tolist())
        shares = np.array([float(temporal.get(fold, 0.0)) for fold in all_folds], dtype=float)
        rows.append(
            {
                "Variant": variant,
                "Dataset": dataset,
                "Temporal_Importance_Mean": float(shares.mean()),
                "Temporal_Importance_Std": float(shares.std(ddof=0)),
                "Temporal_Importance_Min": float(shares.min()),
                "Temporal_Importance_Max": float(shares.max()),
            }
        )
    return pd.DataFrame(rows)


def build_variant_ranking(within: pd.DataFrame) -> pd.DataFrame:
    standard = within[within["Min_History"] == 0].copy()
    rows = []
    for variant, group in standard.groupby("Variant", sort=False):
        if set(group["Dataset"]) != set(BENCHMARK_DATASETS):
            continue
        rows.append(
            {
                "Variant": variant,
                "Feature_Count": int(group["Feature_Count"].iloc[0]),
                "PR_AUC_Mean_D2_D3": float(group["pr_auc"].mean()),
                "PR_AUC_Min_D2_D3": float(group["pr_auc"].min()),
                "Recall_Mean_D2_D3": float(group["recall_tpr"].mean()),
                "Recall_Min_D2_D3": float(group["recall_tpr"].min()),
                "F1_Mean_D2_D3": float(group["f1"].mean()),
                "FPR_Mean_D2_D3": float(group["fpr"].mean()),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ranking = ranking.sort_values(
        [
            "PR_AUC_Mean_D2_D3",
            "PR_AUC_Min_D2_D3",
            "Recall_Min_D2_D3",
            "F1_Mean_D2_D3",
        ],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking["Rank"] = np.arange(1, len(ranking) + 1)
    ranking["Selected_By_Ranking"] = ranking["Rank"] == 1
    return ranking


def build_full5_deltas(within: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_name in BENCHMARK_DATASETS:
        r0 = within[(within["Variant"] == "R0_FULL5") & (within["Dataset"] == dataset_name)]
        rr = within[(within["Variant"] == "RROLL5_FULL5") & (within["Dataset"] == dataset_name)]
        if len(r0) == 1 and len(rr) == 1:
            a, b = r0.iloc[0], rr.iloc[0]
            row = {
                "Evaluation": f"within_{dataset_name}",
                "Rows": int(b["Rows_Evaluated"]),
            }
            for metric in REPORT_METRICS:
                row[f"R0_{metric}"] = a[metric]
                row[f"RROLL5_{metric}"] = b[metric]
                row[f"Delta_{metric}"] = b[metric] - a[metric]
            rows.append(row)

    for source, target in (("dataset2", "dataset3"), ("dataset3", "dataset2")):
        r0 = transfer[
            (transfer["Variant"] == "R0_FULL5")
            & (transfer["Train_Dataset"] == source)
            & (transfer["Test_Dataset"] == target)
        ]
        rr = transfer[
            (transfer["Variant"] == "RROLL5_FULL5")
            & (transfer["Train_Dataset"] == source)
            & (transfer["Test_Dataset"] == target)
        ]
        if len(r0) == 1 and len(rr) == 1:
            a, b = r0.iloc[0], rr.iloc[0]
            row = {
                "Evaluation": f"transfer_{source}_to_{target}",
                "Rows": int(b["Rows_Evaluated"]),
            }
            for metric in REPORT_METRICS:
                row[f"R0_{metric}"] = a[metric]
                row[f"RROLL5_{metric}"] = b[metric]
                row[f"Delta_{metric}"] = b[metric] - a[metric]
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# TOP-LEVEL O4
# =============================================================================


def run_temporal_optimization() -> pd.DataFrame:
    TEMPORAL_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMPORAL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    params_by_fold = load_o3_params_by_fold()
    final_params = load_o3_final_params()

    print("=" * 96)
    print("RF O4 - ENDPOINT-AWARE CAUSAL TEMPORAL OPTIMIZATION")
    print("=" * 96)
    print("Static representation: frozen O2/O3 12 features")
    print("OOF hyperparameters: exact O3 shared selection for each outer fold")
    print("Sequence key: (SourceFile, Src IP, Dst IP)")
    print("Ordering: Timestamp, then Original_Row_Index")
    print("No raw IP or Timestamp is a model input")
    print("Temporal variants: R0, RDELTA, RROLL3, RROLL5, RROLL10")
    print(f"RROLL5 feature count: {len(variant_features('RROLL5'))} (expected exact legacy endpoint-aware schema: 32)")
    print("Strict control: R0_FULL5 vs RROLL5_FULL5 on identical >=5-history rows")

    raw = {name: load_temporal_base(name) for name in BENCHMARK_DATASETS}
    outer_assignment = {
        name: _load_static_fold_assignment(name, len(raw[name]))
        for name in BENCHMARK_DATASETS
    }

    # History-availability diagnostic.
    history_summary = pd.concat(
        [build_history_summary(raw[name], name) for name in BENCHMARK_DATASETS],
        ignore_index=True,
    )
    history_summary.to_csv(
        TEMPORAL_RESULT_ROOT / "endpoint_history_availability.csv", index=False
    )

    within_rows = []
    oof_parts = []
    fold_metric_parts = []
    importance_parts = []

    # Standard all-row endpoint-aware ablation.
    for variant in VARIANT_ORDER:
        print("\n" + "-" * 96)
        print(variant)
        print("-" * 96)
        for dataset_name in BENCHMARK_DATASETS:
            summary, oof, fold_metrics, importance = evaluate_variant_oof(
                dataset_name=dataset_name,
                raw_temporal=raw[dataset_name],
                outer_assignment=outer_assignment[dataset_name],
                params_by_fold=params_by_fold,
                variant=variant,
                min_history=None,
            )
            within_rows.append(summary)
            tagged = oof.copy()
            tagged["Variant"] = variant
            tagged["Dataset"] = dataset_name
            oof_parts.append(tagged)
            fold_metric_parts.append(fold_metrics)
            importance_parts.append(importance)
            print(
                f"{dataset_name}: recall={summary['recall_tpr']:.4f}, "
                f"fpr={summary['fpr']:.6f}, f1={summary['f1']:.4f}, "
                f"pr_auc={summary['pr_auc']:.4f}"
            )

    # Strict same-population FULL5 control. Both models train and test only on rows
    # that already have >=5 preceding flows for the same endpoint sequence.
    for variant in ("R0", "RROLL5"):
        for dataset_name in BENCHMARK_DATASETS:
            summary, oof, fold_metrics, importance = evaluate_variant_oof(
                dataset_name=dataset_name,
                raw_temporal=raw[dataset_name],
                outer_assignment=outer_assignment[dataset_name],
                params_by_fold=params_by_fold,
                variant=variant,
                min_history=5,
            )
            within_rows.append(summary)
            tagged = oof.copy()
            tagged["Variant"] = f"{variant}_FULL5"
            tagged["Dataset"] = dataset_name
            oof_parts.append(tagged)
            fold_metric_parts.append(fold_metrics)
            importance_parts.append(importance)

    within = pd.DataFrame(within_rows)
    within.to_csv(TEMPORAL_RESULT_ROOT / "temporal_within_oof_summary.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(
        TEMPORAL_RESULT_ROOT / "temporal_oof_predictions.csv", index=False
    )
    pd.concat(fold_metric_parts, ignore_index=True).to_csv(
        TEMPORAL_RESULT_ROOT / "temporal_outer_fold_metrics.csv", index=False
    )

    importance = pd.concat(importance_parts, ignore_index=True)
    importance.to_csv(
        TEMPORAL_RESULT_ROOT / "temporal_feature_importance_by_fold.csv", index=False
    )
    (
        importance.groupby(["Variant", "Dataset", "Feature", "Is_Temporal"])[
            "MDI_Importance"
        ]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "MDI_Importance_Mean", "std": "MDI_Importance_Std"})
        .to_csv(TEMPORAL_RESULT_ROOT / "temporal_feature_importance_mean.csv", index=False)
    )
    summarize_temporal_importance(importance).to_csv(
        TEMPORAL_RESULT_ROOT / "temporal_importance_share.csv", index=False
    )

    # Cross-dataset transfer, using the final O3 deployment configuration.
    transfer_rows = []
    transfer_pred_parts = []
    for variant in VARIANT_ORDER:
        rows, preds = evaluate_transfer(raw, variant, final_params, min_history=None)
        transfer_rows.extend(rows)
        transfer_pred_parts.append(preds)
    for variant in ("R0", "RROLL5"):
        rows, preds = evaluate_transfer(raw, variant, final_params, min_history=5)
        transfer_rows.extend(rows)
        transfer_pred_parts.append(preds)

    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv(TEMPORAL_RESULT_ROOT / "temporal_cross_dataset_summary.csv", index=False)
    pd.concat(transfer_pred_parts, ignore_index=True).to_csv(
        TEMPORAL_RESULT_ROOT / "temporal_cross_dataset_predictions.csv", index=False
    )

    ranking = build_variant_ranking(within)
    ranking.to_csv(TEMPORAL_RESULT_ROOT / "temporal_variant_ranking.csv", index=False)

    full5 = build_full5_deltas(within, transfer)
    full5.to_csv(TEMPORAL_RESULT_ROOT / "FULL5_same_population_control.csv", index=False)

    # Save exact feature schemas / protocol provenance.
    for variant in VARIANT_ORDER:
        pd.DataFrame({"Feature": variant_features(variant)}).to_csv(
            TEMPORAL_ARTIFACT_ROOT / f"{variant}_features.csv", index=False
        )

    with (TEMPORAL_ARTIFACT_ROOT / "temporal_experiment_definition.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "static_parent_stage": "O3_12_TUNED",
                "base_features": BASE_12_FEATURES,
                "sequence_grouping": [SOURCE_FILE_COL, "Src IP", "Dst IP"],
                "raw_grouping_identifiers_are_model_inputs": False,
                "raw_timestamp_is_model_input": False,
                "interflow_delta_definition": "current timestamp - previous timestamp in same endpoint sequence",
                "rolling_features_use_shift_1": True,
                "rolling_behavior_features": ROLLING_BEHAVIOR_FEATURES,
                "rolling_statistics": ["mean", "std", "interflow_delta_cv"],
                "rolling_std_min_periods": 1,
                "rolling_windows": list(ROLL_WINDOWS),
                "standard_variants": VARIANT_ORDER,
                "oof_hyperparameter_rule": "reuse exact O3 per-outer-fold shared selections",
                "transfer_hyperparameter_rule": "use O3 final shared deployment configuration",
                "prediction_threshold": PREDICTION_THRESHOLD,
                "strict_control": "R0_FULL5 vs RROLL5_FULL5; both train/test populations require Endpoint History Count >= 5",
            },
            handle,
            indent=2,
        )

    print("\n" + "=" * 96)
    print("TEMPORAL RF O4 COMPLETE")
    print("=" * 96)
    report_cols = [
        "Variant",
        "Dataset",
        "Feature_Count",
        "Rows_Evaluated",
        "precision",
        "recall_tpr",
        "fpr",
        "f1",
        "roc_auc",
        "pr_auc",
        "fp",
        "fn",
    ]
    print(within[report_cols].to_string(index=False))
    if not ranking.empty:
        print("\nVariant ranking (standard all-row OOF):")
        print(ranking.to_string(index=False))
    print("\nSaved key files:")
    for path in [
        TEMPORAL_RESULT_ROOT / "temporal_within_oof_summary.csv",
        TEMPORAL_RESULT_ROOT / "temporal_cross_dataset_summary.csv",
        TEMPORAL_RESULT_ROOT / "temporal_variant_ranking.csv",
        TEMPORAL_RESULT_ROOT / "FULL5_same_population_control.csv",
        TEMPORAL_RESULT_ROOT / "endpoint_history_availability.csv",
        TEMPORAL_RESULT_ROOT / "temporal_importance_share.csv",
    ]:
        print(path)

    return within
