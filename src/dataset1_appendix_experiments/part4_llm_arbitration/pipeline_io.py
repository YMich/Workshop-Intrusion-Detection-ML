from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import HYBRID_DIR

# Import the authoritative final Part-3 hybrid without copying its model logic.
hybrid_text = str(HYBRID_DIR)
if hybrid_text not in sys.path:
    sys.path.insert(0, hybrid_text)

import hybrid_pipeline as hybrid  # noqa: E402


ROW_KEY = hybrid.ROW_KEY_COL


def _load_hybrid_rule(dataset_name: str) -> dict:
    path = hybrid.ARTIFACT_ROOT / dataset_name / "hybrid_rule.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen hybrid rule: {path}. Run the final hybrid first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _raw_with_sequence_keys(split_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the SourceFile-local chronological row key used by Part 3."""
    raw = split_df.copy()
    raw["__RawOrder"] = np.arange(len(raw), dtype=np.int64)
    raw["__ParsedTimestamp"] = pd.to_datetime(
        raw[hybrid.TIMESTAMP_COL],
        errors="coerce",
        utc=True,
    )
    if raw["__ParsedTimestamp"].isna().any():
        raise ValueError("At least one Timestamp could not be parsed.")

    blocks = []
    for _source, group in raw.groupby(
        hybrid.SOURCE_FILE_COL,
        sort=True,
        dropna=False,
    ):
        group = group.sort_values(
            ["__ParsedTimestamp", "__RawOrder"],
            kind="mergesort",
        ).copy()
        group[ROW_KEY] = np.arange(len(group), dtype=np.int64)
        blocks.append(group)

    return pd.concat(blocks, ignore_index=True).drop(
        columns=["__RawOrder", "__ParsedTimestamp"]
    )


def _component_predictions(
    frame: pd.DataFrame,
    rule: dict,
) -> pd.DataFrame:
    out = frame.copy()
    out["LSTM_Pred"] = (
        out["LSTM_Probability"].to_numpy(dtype=float)
        >= float(rule["lstm_threshold"])
    ).astype(int)
    out["LITEMV_Pred"] = (
        out["LITEMV_Probability"].to_numpy(dtype=float)
        >= float(rule["litemv_standalone_threshold"])
    ).astype(int)
    out["RF_Pred"] = (
        out["RF_Probability"].to_numpy(dtype=float)
        >= float(rule["rf_standalone_threshold"])
    ).astype(int)
    return out


def _apply_frozen_hybrid(
    frame: pd.DataFrame,
    rule: dict,
) -> pd.DataFrame:
    out = _component_predictions(frame, rule)

    applied = hybrid.apply_hybrid_cascade(
        lstm_probability=out["LSTM_Probability"].to_numpy(dtype=float),
        litemv_probability=out["LITEMV_Probability"].to_numpy(dtype=float),
        rf_probability=out["RF_Probability"].to_numpy(dtype=float),
        lstm_threshold=float(rule["lstm_threshold"]),
        positive_filter_cutoff=float(rule["positive_filter_cutoff"]),
        litemv_confirm_threshold=float(rule["litemv_confirm_threshold"]),
        rf_confirm_threshold=float(rule["rf_confirm_threshold"]),
        litemv_rescue_threshold=float(rule["litemv_rescue_threshold"]),
        rf_rescue_threshold=float(rule["rf_rescue_threshold"]),
        enabled=bool(rule.get("enabled", True)),
    )

    out["Hybrid_Pred"] = applied["prediction"].astype(int)
    out["DecisionRoute"] = hybrid.route_labels(applied)
    return out


def _load_validation_scored(
    dataset_name: str,
    rule: dict,
) -> pd.DataFrame:
    path = (
        hybrid.RESULT_ROOT
        / dataset_name
        / "validation_component_probabilities.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing hybrid validation probabilities: {path}. "
            "Run the final hybrid first."
        )
    frame = pd.read_csv(path)
    return _apply_frozen_hybrid(frame, rule)


def _load_test_scored(
    dataset_name: str,
    rule: dict,
) -> pd.DataFrame:
    path = hybrid.RESULT_ROOT / dataset_name / "test_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen hybrid test predictions: {path}. "
            "Run the final hybrid first."
        )

    saved = pd.read_csv(path)
    recomputed = _apply_frozen_hybrid(saved, rule)

    if "Predicted_Label" not in saved.columns:
        raise RuntimeError(
            f"{dataset_name}: test_predictions.csv lacks Predicted_Label."
        )

    if not np.array_equal(
        recomputed["Hybrid_Pred"].astype(int).to_numpy(),
        saved["Predicted_Label"].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            f"{dataset_name}: recomputed frozen hybrid predictions differ "
            "from saved Part-3 test predictions."
        )

    if "DecisionRoute" in saved.columns and not np.array_equal(
        recomputed["DecisionRoute"].astype(str).to_numpy(),
        saved["DecisionRoute"].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            f"{dataset_name}: recomputed hybrid routes differ from saved "
            "Part-3 test routes."
        )

    return recomputed


def _attach_raw_telemetry(
    scored: pd.DataFrame,
    raw_split: pd.DataFrame,
    dataset_name: str,
    split_name: str,
) -> pd.DataFrame:
    raw = _raw_with_sequence_keys(raw_split)

    # The scored table already carries authoritative Label/Timestamp columns.
    raw_features = raw.drop(
        columns=[hybrid.LABEL_COL, hybrid.TIMESTAMP_COL],
        errors="ignore",
    )

    keys = [hybrid.SOURCE_FILE_COL, ROW_KEY]
    merged = scored.merge(
        raw_features,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(scored):
        raise RuntimeError(
            f"{dataset_name}/{split_name}: raw telemetry alignment failed."
        )
    return merged


def build_dataset_state(
    dataset_name: str,
    target_split: str,
) -> dict:
    """
    Consume the frozen final hybrid outputs without retraining/re-scoring models.

    Validation is used only for evidence-strength calibration and development.
    Test is loaded only from the already-frozen Part-3 hybrid outputs.
    """
    if target_split not in {"validation", "test"}:
        raise ValueError("target_split must be validation or test.")

    splits, _manifest, split_metadata = hybrid.load_shared_splits(dataset_name)
    rule = _load_hybrid_rule(dataset_name)

    validation_scored = _attach_raw_telemetry(
        _load_validation_scored(dataset_name, rule),
        splits["validation"],
        dataset_name,
        "validation",
    )

    if target_split == "validation":
        target_scored = validation_scored.copy()
    else:
        target_scored = _attach_raw_telemetry(
            _load_test_scored(dataset_name, rule),
            splits["test"],
            dataset_name,
            "test",
        )

    return {
        "dataset": dataset_name,
        "split_mode": split_metadata.get("mode"),
        "train_raw": splits["train"].copy(),
        "validation_scored": validation_scored,
        "target_scored": target_scored,
        "hybrid_rule": rule,
        "lstm_threshold": float(rule["lstm_threshold"]),
        "litemv_threshold": float(rule["litemv_standalone_threshold"]),
        "rf_threshold": float(rule["rf_standalone_threshold"]),
    }
