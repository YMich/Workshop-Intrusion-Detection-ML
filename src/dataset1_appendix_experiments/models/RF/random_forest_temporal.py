from __future__ import annotations

import numpy as np
import pandas as pd

from random_forest_config import (
    BASE_10_FEATURES,
    BASE_12_FEATURES,
    DST_IP_ALIASES,
    FWD_PACKET_LENGTH_CV,
    HISTORY_COUNT,
    INTERFLOW_DELTA,
    LABEL_COL,
    ROLLING_BEHAVIOR_FEATURES,
    ROLL_WINDOW,
    ROW_INDEX_COL,
    SOURCE_FILE_COL,
    SRC_IP_ALIASES,
    TIMESTAMP_ALIASES,
    TOTAL_FWD_LENGTH,
    TOTAL_FWD_LENGTH_ALIASES,
)


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
        f"Could not resolve {role}. Tried aliases={list(aliases)}."
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
    text = series.astype(str).str.strip()
    candidates = []
    for dayfirst in (False, True):
        try:
            parsed = pd.to_datetime(
                text, errors="coerce", format="mixed", dayfirst=dayfirst
            )
        except (TypeError, ValueError):
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
        candidates.append(parsed)
    parsed = max(candidates, key=lambda s: int(s.notna().sum()))

    bad = int(parsed.isna().sum())
    if bad:
        frac = bad / max(len(parsed), 1)
        if frac > 0.001:
            examples = text[parsed.isna()].head(5).tolist()
            raise ValueError(
                f"{dataset_name}: failed to parse {bad}/{len(parsed)} timestamps "
                f"({frac:.3%}). Examples={examples}"
            )
        print(
            f"[TEMPORAL] {dataset_name}: {bad} timestamps could not be parsed; "
            "temporal missing values will be training-fold median imputed."
        )
    return parsed


def rolling_feature_names(window: int = ROLL_WINDOW) -> list[str]:
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


def final_feature_names() -> list[str]:
    features = list(BASE_12_FEATURES) + [INTERFLOW_DELTA] + rolling_feature_names()
    if len(features) != 32:
        raise RuntimeError(f"Final RROLL5 schema must contain 32 features, found {len(features)}")
    if len(features) != len(set(features)):
        raise RuntimeError("Final RROLL5 schema contains duplicate features.")
    return features


def build_temporal_features(
    raw: pd.DataFrame,
    dataset_name: str,
    require_label: bool = True,
) -> pd.DataFrame:
    """
    Build the exact final endpoint-aware causal RROLL5 representation.

    Sequence key:
        (SourceFile, Src IP, Dst IP)
    Order:
        Timestamp, then Original_Row_Index
    History:
        previous rows only via shift(1)
    Rolling standard deviation:
        min_periods=1, ddof=0
    Final schema:
        32 features, including InterFlow Delta Prev5 CV

    Raw SourceFile/IP/Timestamp columns are never predictive features.
    """
    raw = raw.copy()
    raw[ROW_INDEX_COL] = np.arange(len(raw), dtype=np.int64)

    if require_label:
        if LABEL_COL not in raw.columns:
            raise ValueError(f"{dataset_name}: missing {LABEL_COL}")
        raw[LABEL_COL] = pd.to_numeric(raw[LABEL_COL], errors="raise").astype(int)

    src_col = _resolve_alias(raw.columns, SRC_IP_ALIASES, "source IP")
    dst_col = _resolve_alias(raw.columns, DST_IP_ALIASES, "destination IP")
    ts_col = _resolve_alias(raw.columns, TIMESTAMP_ALIASES, "timestamp")
    total_col = _resolve_alias(
        raw.columns, TOTAL_FWD_LENGTH_ALIASES, "total forward-packet length"
    )

    required = set(BASE_10_FEATURES) | {
        SOURCE_FILE_COL,
        src_col,
        dst_col,
        ts_col,
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        total_col,
    }
    if require_label:
        required.add(LABEL_COL)
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{dataset_name}: missing temporal source columns: {missing}")

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
    raw[INTERFLOW_DELTA] = (
        grouped["_ParsedTimestamp"].diff().dt.total_seconds().clip(lower=0)
    )

    w = ROLL_WINDOW
    for feature in ROLLING_BEHAVIOR_FEATURES:
        raw[f"{feature} Prev{w} Mean"] = grouped[feature].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        raw[f"{feature} Prev{w} Std"] = grouped[feature].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std(ddof=0)
        )

    raw[f"{INTERFLOW_DELTA} Prev{w} Mean"] = grouped[INTERFLOW_DELTA].transform(
        lambda s: s.shift(1).rolling(w, min_periods=1).mean()
    )
    raw[f"{INTERFLOW_DELTA} Prev{w} Std"] = grouped[INTERFLOW_DELTA].transform(
        lambda s: s.shift(1).rolling(w, min_periods=1).std(ddof=0)
    )
    raw[f"{INTERFLOW_DELTA} Prev{w} CV"] = _safe_ratio(
        raw[f"{INTERFLOW_DELTA} Prev{w} Std"],
        raw[f"{INTERFLOW_DELTA} Prev{w} Mean"].abs(),
    )

    raw = raw.sort_values(ROW_INDEX_COL, kind="mergesort").reset_index(drop=True)

    # Hard assertions against accidental leakage into the final model schema.
    features = final_feature_names()
    forbidden = {
        SOURCE_FILE_COL,
        src_col,
        dst_col,
        ts_col,
        "_SrcEndpoint",
        "_DstEndpoint",
        "_ParsedTimestamp",
        HISTORY_COUNT,
        ROW_INDEX_COL,
    }
    if forbidden & set(features):
        raise RuntimeError(f"Forbidden fields entered feature schema: {forbidden & set(features)}")

    return raw


def load_temporal_dataset(path, dataset_name: str) -> pd.DataFrame:
    path = pd.io.common.stringify_path(path)
    raw = pd.read_csv(path, low_memory=False)
    frame = build_temporal_features(raw, dataset_name=dataset_name, require_label=True)
    print(
        f"Loaded {dataset_name}: rows={len(frame):,}, "
        f"final features={len(final_feature_names())}"
    )
    return frame
