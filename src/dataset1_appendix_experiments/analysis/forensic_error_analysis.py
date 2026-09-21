from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


def discover_project_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "data" / "ingested").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root containing src/ and data/ingested/.")


PROJECT_ROOT = discover_project_root()
RESULT_ROOT = PROJECT_ROOT / "results" / "dataset1_appendix"
OUT_ROOT = RESULT_ROOT / "analysis" / "forensic_errors"

MODEL_SPECS = {
    "RF": {
        "path": RESULT_ROOT / "models" / "RF" / "dataset1" / "oof_predictions.csv",
        "actual": "Actual_Label",
        "pred": "Predicted_Label",
    },
    "LSTM": {
        "path": RESULT_ROOT / "models" / "LSTM" / "dataset1" / "test_predictions.csv",
        "actual": "Label",
        "pred": "Predicted_Label",
    },
    "LITEMV": {
        "path": RESULT_ROOT / "models" / "LITEMV" / "dataset1" / "test_predictions.csv",
        "actual": "Label",
        "pred": "Predicted_Label",
    },
    "IF": {
        "path": RESULT_ROOT / "models" / "IF" / "dataset1" / "test_predictions.csv",
        "actual": "Label",
        "pred": "PredictedLabel",
    },
    "AE": {
        "path": RESULT_ROOT / "models" / "AE" / "dataset1" / "test_predictions.csv",
        "actual": "Label",
        "pred": "Predicted_Label",
    },
    "HYBRID": {
        "path": RESULT_ROOT / "part3_hybrid" / "dataset1" / "test_predictions.csv",
        "actual": "Label",
        "pred": "Predicted_Label",
    },
}


def _source_col(df: pd.DataFrame) -> str | None:
    for name in ("SourceFile", "Source File", "source_file"):
        if name in df.columns:
            return name
    return None


def _position_col(df: pd.DataFrame) -> str | None:
    for name in (
        "TargetRowInSplitSourceSegment",
        "SourcePosition",
        "SequencePosition",
        "Endpoint History Count",
    ):
        if name in df.columns:
            return name
    return None


def summarize_model(model_name: str, spec: dict) -> dict:
    path = Path(spec["path"])
    if not path.is_file():
        return {"Model": model_name, "Status": f"missing:{path}"}

    df = pd.read_csv(path, low_memory=False)
    actual_col = spec["actual"]
    pred_col = spec["pred"]
    if actual_col not in df.columns or pred_col not in df.columns:
        raise RuntimeError(
            f"{model_name}: expected {actual_col!r}/{pred_col!r} in {path}; "
            f"found {list(df.columns)}"
        )

    actual = pd.to_numeric(df[actual_col], errors="raise").astype(int)
    pred = pd.to_numeric(df[pred_col], errors="raise").astype(int)
    fp = df[(actual == 0) & (pred == 1)].copy()
    fn = df[(actual == 1) & (pred == 0)].copy()
    tp = int(((actual == 1) & (pred == 1)).sum())
    tn = int(((actual == 0) & (pred == 0)).sum())

    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fp.to_csv(out_dir / "false_positives.csv", index=False)
    fn.to_csv(out_dir / "false_negatives.csv", index=False)

    source_col = _source_col(df)
    position_col = _position_col(df)
    for name, errors in (("false_positives", fp), ("false_negatives", fn)):
        if source_col is None:
            pd.DataFrame(columns=["SourceFile", "ErrorCount"]).to_csv(
                out_dir / f"{name}_by_source.csv", index=False
            )
            continue
        grouped = errors.groupby(source_col, dropna=False).size().reset_index(name="ErrorCount")
        grouped = grouped.sort_values("ErrorCount", ascending=False, kind="mergesort")
        if position_col is not None and not errors.empty:
            position_stats = (
                errors.groupby(source_col, dropna=False)[position_col]
                .agg(["min", "median", "max"])
                .reset_index()
                .rename(
                    columns={
                        "min": "MinErrorPosition",
                        "median": "MedianErrorPosition",
                        "max": "MaxErrorPosition",
                    }
                )
            )
            grouped = grouped.merge(position_stats, on=source_col, how="left")
        grouped.to_csv(out_dir / f"{name}_by_source.csv", index=False)

    benign = int((actual == 0).sum())
    malicious = int((actual == 1).sum())
    fp_n = int(len(fp))
    fn_n = int(len(fn))
    return {
        "Model": model_name,
        "Status": "ok",
        "Rows": int(len(df)),
        "TN": tn,
        "FP": fp_n,
        "FN": fn_n,
        "TP": tp,
        "FPR": float(fp_n / benign) if benign else np.nan,
        "FNR": float(fn_n / malicious) if malicious else np.nan,
        "SourceColumn": source_col,
        "PositionColumn": position_col,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [summarize_model(name, spec) for name, spec in MODEL_SPECS.items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_ROOT / "dataset1_forensic_error_summary.csv", index=False)
    with (OUT_ROOT / "dataset1_forensic_error_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
