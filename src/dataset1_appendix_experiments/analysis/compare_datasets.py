from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def discover_project_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "results").exists():
            return candidate
    raise RuntimeError("Could not locate project root containing src/ and results/.")


PROJECT_ROOT = discover_project_root()
OUT_ROOT = PROJECT_ROOT / "results" / "dataset1_appendix" / "analysis"


def _json_record(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_path(model: str, dataset: str) -> Path:
    if dataset == "dataset1":
        base = PROJECT_ROOT / "results" / "dataset1_appendix" / "models" / model
    else:
        base = PROJECT_ROOT / "results" / "models_optimized_final" / model

    if model == "RF":
        return base / dataset / "oof_overall_metrics.json"
    return base / dataset / "test_metrics.json"


def main() -> None:
    rows = []
    for model in ("RF", "LSTM", "LITEMV", "IF", "AE"):
        for dataset in ("dataset1", "dataset2", "dataset3"):
            path = _metric_path(model, dataset)
            record = _json_record(path)
            if record is None:
                rows.append({"Model": model, "Dataset": dataset, "Status": f"missing:{path}"})
                continue
            row = {"Model": model, "Dataset": dataset, "Status": "ok"}
            for key in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall_tpr",
                "specificity_tnr",
                "fpr",
                "f1",
                "f2",
                "roc_auc",
                "pr_auc",
                "mcc",
                "tn",
                "fp",
                "fn",
                "tp",
                "threshold",
            ):
                if key in record:
                    row[key] = record[key]
            rows.append(row)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_ROOT / "d1_d2_d3_model_comparison.csv", index=False)
    print(frame.to_string(index=False))

    # Part-3 comparison when both appendix and original results exist.
    hybrid_parts = []
    d1 = PROJECT_ROOT / "results" / "dataset1_appendix" / "part3_hybrid" / "dataset1" / "test_ablation_metrics.csv"
    if d1.is_file():
        hybrid_parts.append(pd.read_csv(d1))
    original_root = PROJECT_ROOT / "results" / "part3_hybrid" / "final_lstm_litemv_rf"
    for dataset in ("dataset2", "dataset3"):
        path = original_root / dataset / "test_ablation_metrics.csv"
        if path.is_file():
            hybrid_parts.append(pd.read_csv(path))
    if hybrid_parts:
        pd.concat(hybrid_parts, ignore_index=True).to_csv(
            OUT_ROOT / "d1_d2_d3_part3_hybrid_comparison.csv", index=False
        )


if __name__ == "__main__":
    main()
