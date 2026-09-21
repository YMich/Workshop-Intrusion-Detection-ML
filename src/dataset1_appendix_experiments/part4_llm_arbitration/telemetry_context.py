from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    BENIGN_DEVIATION_TOP_K,
    CONTEXT_VARIANTS,
    CURRENT_FLOW_FEATURES,
    ESCALATE_LOW_CONFIDENCE_AGREEMENTS,
    HARD_CASE_ROUTES,
    LOW_CONFIDENCE_PERCENTILE,
    SEQUENCE_LENGTH,
    STRONG_PERCENTILE,
    WEAK_PERCENTILE,
)
from pipeline_io import ROW_KEY, hybrid


TARGET = "COBALT_STRIKE_HTTPS_C2"
NON_TARGET = "NOT_COBALT_STRIKE_HTTPS_C2"
EPS = 1e-12


ROUTE_DESCRIPTIONS = {
    "RF_ARBITRATED_KEEP_MALICIOUS": (
        "The LSTM produced a borderline malicious decision. LITEMV did not "
        "meet its confirmation threshold, so RF arbitrated and confirmed "
        "malicious."
    ),
    "LITEMV_RF_FILTERED_LSTM_POSITIVE": (
        "The LSTM produced a borderline malicious decision. LITEMV did not "
        "confirm it, and RF also failed to confirm it, so the hybrid changed "
        "the final decision to non-C2."
    ),
    "JOINT_LITEMV_RF_RESCUE": (
        "The LSTM predicted non-C2, but both LITEMV and RF exceeded strict "
        "rescue thresholds, so the hybrid changed the final decision to C2."
    ),
}


def label_name(value: int) -> str:
    return TARGET if int(value) == 1 else NON_TARGET


def _support_margin(
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    margin = np.where(
        pred == 1,
        p - float(threshold),
        float(threshold) - p,
    )
    return pred, margin


@dataclass
class StrengthCalibration:
    lstm_by_class: dict[int, np.ndarray]
    litemv_by_class: dict[int, np.ndarray]
    rf_by_class: dict[int, np.ndarray]


def _build_class_reference(
    probabilities: np.ndarray,
    threshold: float,
) -> dict[int, np.ndarray]:
    pred, margin = _support_margin(probabilities, threshold)
    return {
        cls: np.sort(margin[pred == cls])
        for cls in (0, 1)
    }


def fit_strength_calibration(
    validation: pd.DataFrame,
    lstm_threshold: float,
    litemv_threshold: float,
    rf_threshold: float,
) -> StrengthCalibration:
    return StrengthCalibration(
        lstm_by_class=_build_class_reference(
            validation["LSTM_Probability"].to_numpy(dtype=float),
            lstm_threshold,
        ),
        litemv_by_class=_build_class_reference(
            validation["LITEMV_Probability"].to_numpy(dtype=float),
            litemv_threshold,
        ),
        rf_by_class=_build_class_reference(
            validation["RF_Probability"].to_numpy(dtype=float),
            rf_threshold,
        ),
    )


def _percentile(reference: np.ndarray, value: float) -> float:
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0:
        return 0.5
    return float(
        np.searchsorted(reference, float(value), side="right")
        / len(reference)
    )


def strength_name(percentile: float) -> str:
    if percentile < WEAK_PERCENTILE:
        return "WEAK"
    if percentile < STRONG_PERCENTILE:
        return "MODERATE"
    return "STRONG"


def _add_one_model_strength(
    out: pd.DataFrame,
    probability_col: str,
    prediction_col: str,
    percentile_col: str,
    strength_col: str,
    references: dict[int, np.ndarray],
    threshold: float,
) -> None:
    pred, margin = _support_margin(
        out[probability_col].to_numpy(dtype=float),
        threshold,
    )
    percentiles = [
        _percentile(references[int(p)], m)
        for p, m in zip(pred, margin)
    ]
    out[prediction_col] = pred
    out[percentile_col] = percentiles
    out[strength_col] = [strength_name(v) for v in percentiles]


def add_validation_strengths(
    frame: pd.DataFrame,
    calibration: StrengthCalibration,
    lstm_threshold: float,
    litemv_threshold: float,
    rf_threshold: float,
) -> pd.DataFrame:
    out = frame.copy()

    _add_one_model_strength(
        out,
        "LSTM_Probability",
        "LSTM_Pred",
        "LSTM_SupportPercentile",
        "LSTM_EvidenceStrength",
        calibration.lstm_by_class,
        lstm_threshold,
    )
    _add_one_model_strength(
        out,
        "LITEMV_Probability",
        "LITEMV_Pred",
        "LITEMV_SupportPercentile",
        "LITEMV_EvidenceStrength",
        calibration.litemv_by_class,
        litemv_threshold,
    )
    _add_one_model_strength(
        out,
        "RF_Probability",
        "RF_Pred",
        "RF_SupportPercentile",
        "RF_EvidenceStrength",
        calibration.rf_by_class,
        rf_threshold,
    )

    out["CombinedWeaknessPercentile"] = np.minimum.reduce(
        [
            out["LSTM_SupportPercentile"].to_numpy(dtype=float),
            out["LITEMV_SupportPercentile"].to_numpy(dtype=float),
            out["RF_SupportPercentile"].to_numpy(dtype=float),
        ]
    )
    return out


def _weak_confirmed_strong_rf_disagreement(row: pd.Series) -> bool:
    """Validation-discovered ambiguity worth contextual arbitration.

    The frozen hybrid accepted an LSTM-positive after LITEMV confirmation even
    though both positive models had only WEAK support and RF supplied STRONG
    benign evidence. This condition is label-free: it uses only quantities that
    are available at inference time.
    """
    return (
        str(row["DecisionRoute"]) == "LITEMV_CONFIRMED_LSTM_POSITIVE"
        and int(row["LSTM_Pred"]) == 1
        and int(row["LITEMV_Pred"]) == 1
        and int(row["RF_Pred"]) == 0
        and str(row["LSTM_EvidenceStrength"]) == "WEAK"
        and str(row["LITEMV_EvidenceStrength"]) == "WEAK"
        and str(row["RF_EvidenceStrength"]) == "STRONG"
    )


def escalation_reason(
    row: pd.Series,
    selection_mode: str = "all",
) -> str | None:
    targeted = _weak_confirmed_strong_rf_disagreement(row)

    if selection_mode == "weak-confirmed-only":
        return (
            "WEAK_LSTM_LITEMV_CONFIRMATION_WITH_STRONG_RF_BENIGN"
            if targeted
            else None
        )

    if selection_mode not in {"all", "final-union", "legacy"}:
        raise ValueError(f"Unknown hard-case selection mode: {selection_mode}")

    route = str(row["DecisionRoute"])
    if route in HARD_CASE_ROUTES:
        return f"HYBRID_HARD_ROUTE::{route}"

    if selection_mode in {"all", "final-union"} and targeted:
        return "WEAK_LSTM_LITEMV_CONFIRMATION_WITH_STRONG_RF_BENIGN"

    if (
        ESCALATE_LOW_CONFIDENCE_AGREEMENTS
        and float(row["CombinedWeaknessPercentile"])
        <= LOW_CONFIDENCE_PERCENTILE
    ):
        return "UNUSUALLY_LOW_VALIDATION_SUPPORT"

    return None


def select_hard_cases(
    frame: pd.DataFrame,
    selection_mode: str = "all",
) -> pd.DataFrame:
    out = frame.copy()
    out["EscalationReason"] = out.apply(
        lambda row: escalation_reason(row, selection_mode=selection_mode),
        axis=1,
    )
    return out[out["EscalationReason"].notna()].copy()


@dataclass
class BenignBaseline:
    medians: dict[str, float]
    iqrs: dict[str, float]


def _signed_log1p(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def fit_benign_baseline(train_raw: pd.DataFrame) -> BenignBaseline:
    benign = train_raw[train_raw[hybrid.LABEL_COL].astype(int) == 0]
    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}

    for feature in CURRENT_FLOW_FEATURES:
        if feature not in benign.columns:
            continue

        values = pd.to_numeric(
            benign[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue

        transformed = _signed_log1p(values.to_numpy(dtype=float))
        q25, median, q75 = np.quantile(
            transformed, [0.25, 0.50, 0.75]
        )
        iqr = float(q75 - q25)
        medians[feature] = float(median)
        iqrs[feature] = iqr if abs(iqr) > EPS else 1.0

    return BenignBaseline(medians=medians, iqrs=iqrs)


def current_flow_telemetry(row: pd.Series) -> dict:
    telemetry = {}
    for feature in CURRENT_FLOW_FEATURES:
        if feature not in row.index:
            continue
        value = pd.to_numeric(
            pd.Series([row[feature]]), errors="coerce"
        ).iloc[0]
        if pd.isna(value) or not np.isfinite(value):
            continue
        telemetry[feature] = float(value)
    return telemetry


def strongest_benign_deviations(
    row: pd.Series,
    baseline: BenignBaseline,
) -> list[dict]:
    deviations = []
    for feature, median in baseline.medians.items():
        if feature not in row.index:
            continue

        value = pd.to_numeric(
            pd.Series([row[feature]]), errors="coerce"
        ).iloc[0]
        if pd.isna(value) or not np.isfinite(value):
            continue

        transformed = float(_signed_log1p([float(value)])[0])
        signed_deviation = (
            transformed - median
        ) / baseline.iqrs[feature]

        deviations.append(
            {
                "feature": feature,
                "direction": (
                    "ABOVE_BENIGN_BASELINE"
                    if signed_deviation >= 0
                    else "BELOW_BENIGN_BASELINE"
                ),
                "absolute_robust_deviation": float(abs(signed_deviation)),
            }
        )

    deviations.sort(
        key=lambda item: item["absolute_robust_deviation"],
        reverse=True,
    )
    return deviations[:BENIGN_DEVIATION_TOP_K]


def _relative_dispersion(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    denominator = float(np.mean(np.abs(values)))
    if denominator <= EPS:
        return 0.0
    return float(np.std(values, ddof=0) / denominator)


def _numeric_summary(
    window: pd.DataFrame,
    feature: str,
) -> dict | None:
    if feature not in window.columns:
        return None
    values = pd.to_numeric(
        window[feature], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return None
    return {
        "median": float(np.median(values)),
        "coefficient_of_variation": _relative_dispersion(values),
    }


def sequence_context(
    full_frame: pd.DataFrame,
    current_row: pd.Series,
) -> dict:
    source = current_row[hybrid.SOURCE_FILE_COL]
    position = int(current_row[ROW_KEY])

    source_rows = full_frame[
        full_frame[hybrid.SOURCE_FILE_COL] == source
    ].sort_values(ROW_KEY, kind="mergesort")

    start = max(0, position - SEQUENCE_LENGTH + 1)
    window = source_rows[
        (source_rows[ROW_KEY].astype(int) >= start)
        & (source_rows[ROW_KEY].astype(int) <= position)
    ].copy()

    timestamps = pd.to_datetime(
        window[hybrid.TIMESTAMP_COL],
        errors="coerce",
        utc=True,
    ).sort_values()
    gaps = (
        timestamps.diff()
        .dt.total_seconds()
        .dropna()
        .to_numpy(dtype=float)
    )
    gaps = gaps[np.isfinite(gaps)]

    recent_behavior = {}
    for feature in CURRENT_FLOW_FEATURES:
        summary = _numeric_summary(window, feature)
        if summary is not None:
            recent_behavior[feature] = summary

    return {
        "flows_in_window": int(len(window)),
        "inter_flow_gap_seconds": {
            "median": float(np.median(gaps)) if len(gaps) else None,
            "coefficient_of_variation": (
                _relative_dispersion(gaps) if len(gaps) else 0.0
            ),
        },
        "recent_behavior_summary": recent_behavior,
        "recent_model_behavior": {
            "lstm_c2_fraction": float(window["LSTM_Pred"].astype(int).mean()),
            "litemv_c2_fraction": float(
                window["LITEMV_Pred"].astype(int).mean()
            ),
            "rf_c2_fraction": float(window["RF_Pred"].astype(int).mean()),
            "hybrid_c2_fraction": float(
                window["Hybrid_Pred"].astype(int).mean()
            ),
        },
    }


def _base_payload(
    case_id: str,
    row: pd.Series,
    full_frame: pd.DataFrame,
    baseline: BenignBaseline,
) -> dict:
    return {
        "case_id": case_id,
        "target_task": (
            "Detect Cobalt Strike Beacon command-and-control over HTTPS "
            "from network-flow metadata."
        ),
        "escalation_reason": str(row["EscalationReason"]),
        "existing_hybrid_decision": label_name(int(row["Hybrid_Pred"])),
        "hybrid_decision_route": str(row["DecisionRoute"]),
        "hybrid_route_interpretation": ROUTE_DESCRIPTIONS.get(
            str(row["DecisionRoute"]),
            "The frozen hybrid produced this route without LLM involvement.",
        ),
        "upstream_evidence": {
            "lstm": {
                "prediction": label_name(int(row["LSTM_Pred"])),
                "validation_calibrated_strength": str(
                    row["LSTM_EvidenceStrength"]
                ),
            },
            "litemv": {
                "prediction": label_name(int(row["LITEMV_Pred"])),
                "validation_calibrated_strength": str(
                    row["LITEMV_EvidenceStrength"]
                ),
            },
            "random_forest": {
                "prediction": label_name(int(row["RF_Pred"])),
                "validation_calibrated_strength": str(
                    row["RF_EvidenceStrength"]
                ),
            },
        },
        "current_flow_telemetry": current_flow_telemetry(row),
        "strongest_current_flow_benign_deviations": (
            strongest_benign_deviations(row, baseline)
        ),
        "sequence_context": sequence_context(full_frame, row),
        "interpretation_warning": (
            "Anomaly and repetition are contextual signals only; neither is "
            "automatically evidence of Cobalt Strike."
        ),
    }



# ---------------------------------------------------------------------------
# O1 — explicit behavioral summaries for LLM telemetry triage
# ---------------------------------------------------------------------------
# O1 is intentionally label-free and model-agnostic. It converts recent raw
# flow history into compact, interpretable summaries that the LLM does not have
# to derive mentally from many individual numeric fields.

_FEATURE_ALIASES = {
    "fwd_packets": (
        "Total Fwd Packet",
        "Total Fwd Packets",
        "Tot Fwd Pkts",
        "Total Fwd Packets",
    ),
    "bwd_packets": (
        "Total Bwd packets",
        "Total Bwd Packets",
        "Tot Bwd Pkts",
        "Total Backward Packets",
    ),
    "fwd_bytes": (
        "Total Length of Fwd Packets",
        "TotLen Fwd Pkts",
        "Subflow Fwd Bytes",
        "Subflow Fwd Byts",
    ),
    "bwd_bytes": (
        "Total Length of Bwd Packets",
        "TotLen Bwd Pkts",
        "Subflow Bwd Bytes",
        "Subflow Bwd Byts",
    ),
}


def _first_present_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        if name in frame.columns:
            return name
    return None


def _finite_numeric_series(frame: pd.DataFrame, column: str | None) -> np.ndarray:
    if column is None or column not in frame.columns:
        return np.asarray([], dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return values.to_numpy(dtype=float)


def _paired_ratio_values(
    frame: pd.DataFrame,
    numerator_col: str | None,
    denominator_col: str | None,
) -> np.ndarray:
    if (
        numerator_col is None
        or denominator_col is None
        or numerator_col not in frame.columns
        or denominator_col not in frame.columns
    ):
        return np.asarray([], dtype=float)

    pair = pd.DataFrame(
        {
            "numerator": pd.to_numeric(
                frame[numerator_col], errors="coerce"
            ),
            "denominator": pd.to_numeric(
                frame[denominator_col], errors="coerce"
            ),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        return np.asarray([], dtype=float)

    numerator = pair["numerator"].to_numpy(dtype=float)
    denominator = pair["denominator"].to_numpy(dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=np.abs(denominator) > EPS,
    )


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    if abs(float(denominator)) <= EPS:
        return None
    return float(numerator / denominator)


def _series_summary(values: np.ndarray, current_value: float | None = None) -> dict | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    median = float(np.median(values))
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    mad = float(np.median(np.abs(values - median)))
    cv = _relative_dispersion(values)

    result = {
        "count": int(len(values)),
        "median": median,
        "mean": mean,
        "std": std,
        "coefficient_of_variation": float(cv),
        "median_absolute_deviation": mad,
    }

    if current_value is not None and np.isfinite(current_value):
        result["current"] = float(current_value)
        result["current_to_recent_median_ratio"] = _safe_ratio(
            float(current_value), median
        )
        scale = max(abs(median), EPS)
        result["current_fractional_deviation_from_recent_median"] = float(
            abs(float(current_value) - median) / scale
        )

    return result


def _regularity_level(cv: float | None, count: int) -> str:
    if count < 3 or cv is None or not np.isfinite(cv):
        return "INSUFFICIENT_HISTORY"
    if cv <= 0.15:
        return "HIGH_REGULARITY"
    if cv <= 0.35:
        return "MODERATE_REGULARITY"
    return "LOW_REGULARITY"


def _stability_level(cv: float | None, count: int) -> str:
    if count < 3 or cv is None or not np.isfinite(cv):
        return "INSUFFICIENT_HISTORY"
    if cv <= 0.20:
        return "HIGH_STABILITY"
    if cv <= 0.50:
        return "MODERATE_STABILITY"
    return "LOW_STABILITY"


def _numeric_value(row: pd.Series, column: str | None) -> float | None:
    if column is None or column not in row.index:
        return None
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def behavioral_context(
    full_frame: pd.DataFrame,
    current_row: pd.Series,
) -> dict:
    """Compact label-free behavioral summaries for O1.

    The goal is not to create another detector. It exposes explicit temporal,
    directionality and size-consistency facts so Qwen can reason over them
    without reverse-engineering patterns from dozens of raw numbers.
    """
    source = current_row[hybrid.SOURCE_FILE_COL]
    position = int(current_row[ROW_KEY])

    source_rows = full_frame[
        full_frame[hybrid.SOURCE_FILE_COL] == source
    ].sort_values(ROW_KEY, kind="mergesort")

    start = max(0, position - SEQUENCE_LENGTH + 1)
    window = source_rows[
        (source_rows[ROW_KEY].astype(int) >= start)
        & (source_rows[ROW_KEY].astype(int) <= position)
    ].copy()

    # --- Timing / beaconing periodicity ------------------------------------
    timestamps = pd.to_datetime(
        window[hybrid.TIMESTAMP_COL], errors="coerce", utc=True
    ).sort_values()
    gaps = (
        timestamps.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
    )
    gaps = gaps[np.isfinite(gaps) & (gaps >= 0.0)]
    gap_summary = _series_summary(gaps)
    if gap_summary is None:
        timing = {
            "usable_inter_flow_gaps": 0,
            "regularity_level": "INSUFFICIENT_HISTORY",
        }
    else:
        timing = {
            "usable_inter_flow_gaps": int(len(gaps)),
            **gap_summary,
            "regularity_level": _regularity_level(
                gap_summary["coefficient_of_variation"], len(gaps)
            ),
        }
        if len(gaps) >= 1:
            timing["most_recent_gap_seconds"] = float(gaps[-1])
            timing["most_recent_gap_to_median_ratio"] = _safe_ratio(
                float(gaps[-1]), float(gap_summary["median"])
            )

    # --- Flow duration stability ------------------------------------------
    duration_summary = None
    if "Flow Duration" in window.columns:
        duration_values = _finite_numeric_series(window, "Flow Duration")
        duration_summary = _series_summary(
            duration_values,
            _numeric_value(current_row, "Flow Duration"),
        )
        if duration_summary is not None:
            duration_summary["stability_level"] = _stability_level(
                duration_summary["coefficient_of_variation"],
                duration_summary["count"],
            )

    # --- Directionality ----------------------------------------------------
    fwd_packet_col = _first_present_column(window, _FEATURE_ALIASES["fwd_packets"])
    bwd_packet_col = _first_present_column(window, _FEATURE_ALIASES["bwd_packets"])
    fwd_byte_col = _first_present_column(window, _FEATURE_ALIASES["fwd_bytes"])
    bwd_byte_col = _first_present_column(window, _FEATURE_ALIASES["bwd_bytes"])

    directionality: dict[str, object] = {}

    packet_ratios = _paired_ratio_values(
        window, fwd_packet_col, bwd_packet_col
    )
    if len(packet_ratios):
        ratios = packet_ratios
        current_ratio = _safe_ratio(
            _numeric_value(current_row, fwd_packet_col),
            _numeric_value(current_row, bwd_packet_col),
        )
        summary = _series_summary(ratios, current_ratio)
        if summary is not None:
            summary["stability_level"] = _stability_level(
                summary["coefficient_of_variation"], summary["count"]
            )
            directionality["forward_to_backward_packet_ratio"] = summary

    byte_ratios = _paired_ratio_values(
        window, fwd_byte_col, bwd_byte_col
    )
    if len(byte_ratios):
        ratios = byte_ratios
        current_ratio = _safe_ratio(
            _numeric_value(current_row, fwd_byte_col),
            _numeric_value(current_row, bwd_byte_col),
        )
        summary = _series_summary(ratios, current_ratio)
        if summary is not None:
            summary["stability_level"] = _stability_level(
                summary["coefficient_of_variation"], summary["count"]
            )
            directionality["forward_to_backward_byte_ratio"] = summary

    if "Down/Up Ratio" in window.columns:
        down_up = _finite_numeric_series(window, "Down/Up Ratio")
        summary = _series_summary(
            down_up, _numeric_value(current_row, "Down/Up Ratio")
        )
        if summary is not None:
            summary["stability_level"] = _stability_level(
                summary["coefficient_of_variation"], summary["count"]
            )
            directionality["down_up_ratio"] = summary

    # --- Repeated size / shape behavior -----------------------------------
    size_consistency: dict[str, object] = {}
    size_features = (
        "Total Length of Fwd Packets",
        "TotLen Fwd Pkts",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Bwd Packet Length Std",
    )
    seen = set()
    for feature in size_features:
        if feature in seen or feature not in window.columns:
            continue
        # Avoid reporting the same forward-byte quantity twice when both alias
        # columns exist and are numerically identical.
        if feature == "TotLen Fwd Pkts" and "Total Length of Fwd Packets" in window.columns:
            a = _finite_numeric_series(window, "Total Length of Fwd Packets")
            b = _finite_numeric_series(window, "TotLen Fwd Pkts")
            if len(a) == len(b) and len(a) and np.allclose(a, b, equal_nan=True):
                continue
        values = _finite_numeric_series(window, feature)
        summary = _series_summary(values, _numeric_value(current_row, feature))
        if summary is None:
            continue
        summary["stability_level"] = _stability_level(
            summary["coefficient_of_variation"], summary["count"]
        )
        size_consistency[feature] = summary
        seen.add(feature)

    # --- Compact facts for the LLM ----------------------------------------
    high_stability_features = [
        name
        for name, summary in size_consistency.items()
        if summary.get("stability_level") == "HIGH_STABILITY"
    ]
    stable_directionality = [
        name
        for name, summary in directionality.items()
        if isinstance(summary, dict)
        and summary.get("stability_level") == "HIGH_STABILITY"
    ]

    return {
        "flows_in_window": int(len(window)),
        "beaconing_timing": timing,
        "flow_duration_behavior": duration_summary,
        "directionality_behavior": directionality,
        "size_consistency_behavior": size_consistency,
        "compact_behavioral_facts": {
            "timing_is_highly_regular": (
                timing.get("regularity_level") == "HIGH_REGULARITY"
            ),
            "flow_duration_is_highly_stable": bool(
                duration_summary is not None
                and duration_summary.get("stability_level") == "HIGH_STABILITY"
            ),
            "highly_stable_directionality_metrics": stable_directionality,
            "highly_stable_size_metrics": high_stability_features,
        },
        "interpretation_note": (
            "These are descriptive sequence statistics only. Regularity, "
            "stability, or asymmetry are not by themselves proof of Cobalt "
            "Strike and must be interpreted together with the other evidence."
        ),
    }


def apply_context_variant(payload: dict, variant: str) -> dict:
    if variant not in CONTEXT_VARIANTS:
        raise ValueError(
            f"Unknown context variant {variant!r}; expected {CONTEXT_VARIANTS}."
        )

    out = dict(payload)
    if variant in {"full", "behavioral_o1"}:
        return out
    if variant == "no_sequence":
        out.pop("sequence_context", None)
        return out
    if variant == "no_benign_deviations":
        out.pop("strongest_current_flow_benign_deviations", None)
        return out
    if variant == "models_only":
        keep = {
            "case_id",
            "target_task",
            "escalation_reason",
            "existing_hybrid_decision",
            "hybrid_decision_route",
            "hybrid_route_interpretation",
            "upstream_evidence",
            "interpretation_warning",
        }
        return {key: value for key, value in out.items() if key in keep}
    if variant == "telemetry_only":
        out.pop("upstream_evidence", None)
        out.pop("hybrid_decision_route", None)
        out.pop("hybrid_route_interpretation", None)
        return out

    raise AssertionError("Unhandled context variant")


def build_visible_payload(
    case_id: str,
    row: pd.Series,
    full_frame: pd.DataFrame,
    baseline: BenignBaseline,
    context_variant: str = "full",
) -> dict:
    """
    Build exactly the information visible to Qwen.

    Ground truth, dataset name, SourceFile value, timestamps, filenames, IPs,
    ports, capture identities, and malware-family labels are deliberately absent.
    """
    payload = _base_payload(case_id, row, full_frame, baseline)
    if context_variant == "behavioral_o1":
        payload["behavioral_context_o1"] = behavioral_context(
            full_frame, row
        )
    return apply_context_variant(payload, context_variant)
