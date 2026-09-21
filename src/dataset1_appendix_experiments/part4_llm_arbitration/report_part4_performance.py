from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from config import DATASETS, RESULT_ROOT
from evaluation import metrics_from_predictions
from pipeline_io import ROW_KEY, hybrid


def _raw_llm_label(action: str, hybrid_pred: int) -> int:
    if action == "OVERRIDE_TO_C2":
        return 1
    if action == "OVERRIDE_TO_NON_C2":
        return 0
    return int(hybrid_pred)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).to_dict().items()
    }


def _safe_read_llm_outputs(path: Path, keys: list[str]) -> pd.DataFrame:
    """Read LLM outputs while treating a zero-byte file as zero escalations."""
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        # Older zero-escalation runs wrote a completely empty CSV (no header).
        # Keep the report reproducible without rerunning the LLM.
        return pd.DataFrame(
            columns=[
                *keys,
                "LLMRecommendedAction",
                "LLMConfidence",
                "LLMTargetSpecificity",
                "GateReason",
                "OverrideAccepted",
            ]
        )


def _subset_metrics(y_true, predictions) -> dict:
    """Metrics for the escalated subset; rates are undefined when it is empty."""
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if len(y_true) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "fpr": None,
            "fnr": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    return metrics_from_predictions(y_true, predictions)


def _fmt_metric(value) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def _dataset_report(dataset_name: str) -> dict:
    result_dir = RESULT_ROOT / "FINAL_FROZEN_TEST" / dataset_name / "test"
    predictions_path = result_dir / "part4_predictions.csv"
    escalated_path = result_dir / "escalated_cases_internal.csv"
    outputs_path = result_dir / "llm_outputs_unlabeled.csv"

    missing = [
        str(path)
        for path in (predictions_path, escalated_path, outputs_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing FINAL TEST result files. Run the frozen Part-4 TEST first. "
            f"Missing: {missing}"
        )

    predictions = pd.read_csv(predictions_path)
    escalated = pd.read_csv(escalated_path)

    keys = [hybrid.SOURCE_FILE_COL, ROW_KEY]
    outputs = _safe_read_llm_outputs(outputs_path, keys)
    label_col = hybrid.LABEL_COL

    selected = escalated[keys].drop_duplicates().merge(
        predictions,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if selected[label_col].isna().any():
        raise RuntimeError(
            f"Could not map every escalated {dataset_name} case to TEST labels."
        )

    selected = selected.merge(
        outputs,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_llm"),
    )
    if selected["LLMRecommendedAction"].isna().any():
        raise RuntimeError(
            f"Missing LLM output for at least one escalated {dataset_name} case."
        )

    y_true = selected[label_col].astype(int).to_numpy()
    hybrid_pred = selected["Hybrid_Pred"].astype(int).to_numpy()
    final_pred = selected["Part4_Final_Pred"].astype(int).to_numpy()

    raw_llm_pred = np.asarray(
        [
            _raw_llm_label(action, hp)
            for action, hp in zip(
                selected["LLMRecommendedAction"].astype(str),
                hybrid_pred,
            )
        ],
        dtype=int,
    )

    actual_changed = final_pred != hybrid_pred
    raw_changed = raw_llm_pred != hybrid_pred

    actual_corrected = int(
        np.sum(actual_changed & (hybrid_pred != y_true) & (final_pred == y_true))
    )
    actual_introduced = int(
        np.sum(actual_changed & (hybrid_pred == y_true) & (final_pred != y_true))
    )
    raw_corrected = int(
        np.sum(raw_changed & (hybrid_pred != y_true) & (raw_llm_pred == y_true))
    )
    raw_introduced = int(
        np.sum(raw_changed & (hybrid_pred == y_true) & (raw_llm_pred != y_true))
    )

    total_hybrid_errors = int(
        np.sum(
            predictions["Hybrid_Pred"].astype(int).to_numpy()
            != predictions[label_col].astype(int).to_numpy()
        )
    )
    escalated_hybrid_errors = int(np.sum(hybrid_pred != y_true))

    override_proposals = raw_changed
    proposal_count = int(np.sum(override_proposals))
    correct_proposals = int(
        np.sum(override_proposals & (raw_llm_pred == y_true))
    )

    report = {
        "dataset": dataset_name,
        "test_rows": int(len(predictions)),
        "escalated_cases": int(len(selected)),
        "escalation_rate": float(len(selected) / len(predictions)),
        "escalated_ground_truth": {
            "benign": int(np.sum(y_true == 0)),
            "malicious": int(np.sum(y_true == 1)),
        },
        "hybrid_errors_total_test": total_hybrid_errors,
        "hybrid_errors_inside_escalated_subset": escalated_hybrid_errors,
        "hybrid_error_coverage_by_selector": float(
            escalated_hybrid_errors / total_hybrid_errors
            if total_hybrid_errors
            else 0.0
        ),
        "escalated_subset_hybrid_metrics": _subset_metrics(
            y_true, hybrid_pred
        ),
        "escalated_subset_raw_llm_recommendation_metrics": _subset_metrics(
            y_true, raw_llm_pred
        ),
        "escalated_subset_gated_part4_metrics": _subset_metrics(
            y_true, final_pred
        ),
        "llm_recommended_actions": _count_dict(
            selected["LLMRecommendedAction"]
        ),
        "llm_confidence": _count_dict(selected["LLMConfidence"]),
        "llm_target_specificity": _count_dict(
            selected["LLMTargetSpecificity"]
        ),
        "gate_reasons": _count_dict(selected["GateReason"]),
        "raw_override_proposals": proposal_count,
        "raw_override_proposal_accuracy": (
            float(correct_proposals / proposal_count)
            if proposal_count
            else None
        ),
        "raw_corrected_hybrid_errors": raw_corrected,
        "raw_introduced_new_errors": raw_introduced,
        "raw_net_error_benefit": raw_corrected - raw_introduced,
        "accepted_overrides": int(selected["OverrideAccepted"].astype(bool).sum()),
        "accepted_corrected_hybrid_errors": actual_corrected,
        "accepted_introduced_new_errors": actual_introduced,
        "accepted_net_error_benefit": actual_corrected - actual_introduced,
    }
    return report


def main() -> None:
    reports = [_dataset_report(dataset) for dataset in DATASETS]

    out_dir = RESULT_ROOT / "FINAL_FROZEN_TEST"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "part4_3_arbitration_performance.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)

    rows = []
    for report in reports:
        hybrid_m = report["escalated_subset_hybrid_metrics"]
        raw_m = report["escalated_subset_raw_llm_recommendation_metrics"]
        gated_m = report["escalated_subset_gated_part4_metrics"]
        rows.append(
            {
                "Dataset": report["dataset"],
                "TestRows": report["test_rows"],
                "Escalated": report["escalated_cases"],
                "EscalationRate": report["escalation_rate"],
                "HybridErrorsTotal": report["hybrid_errors_total_test"],
                "EscalatedHybridErrors": report[
                    "hybrid_errors_inside_escalated_subset"
                ],
                "ErrorCoverage": report[
                    "hybrid_error_coverage_by_selector"
                ],
                "RawOverrideProposals": report["raw_override_proposals"],
                "RawCorrections": report["raw_corrected_hybrid_errors"],
                "RawIntroduced": report["raw_introduced_new_errors"],
                "AcceptedOverrides": report["accepted_overrides"],
                "AcceptedCorrections": report[
                    "accepted_corrected_hybrid_errors"
                ],
                "AcceptedIntroduced": report[
                    "accepted_introduced_new_errors"
                ],
                "HybridEscalatedF1": hybrid_m["f1"],
                "RawLLMEscalatedF1": raw_m["f1"],
                "GatedPart4EscalatedF1": gated_m["f1"],
            }
        )

    csv_path = out_dir / "part4_3_arbitration_performance.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\nPART 4.3 — FINAL FROZEN TEST ARBITRATION PERFORMANCE")
    print("=" * 72)
    for report in reports:
        print(f"\n{report['dataset'].upper()}")
        print(
            f"  escalated: {report['escalated_cases']}/"
            f"{report['test_rows']} "
            f"({100.0 * report['escalation_rate']:.4f}%)"
        )
        print(
            "  hybrid errors covered by selector: "
            f"{report['hybrid_errors_inside_escalated_subset']}/"
            f"{report['hybrid_errors_total_test']} "
            f"({100.0 * report['hybrid_error_coverage_by_selector']:.2f}%)"
        )
        print(
            f"  raw LLM override proposals: {report['raw_override_proposals']} | "
            f"corrected={report['raw_corrected_hybrid_errors']} | "
            f"introduced={report['raw_introduced_new_errors']}"
        )
        print(
            f"  accepted overrides: {report['accepted_overrides']} | "
            f"corrected={report['accepted_corrected_hybrid_errors']} | "
            f"introduced={report['accepted_introduced_new_errors']}"
        )
        print(
            "  escalated-subset F1: "
            f"hybrid={_fmt_metric(report['escalated_subset_hybrid_metrics']['f1'])}, "
            f"raw_llm={_fmt_metric(report['escalated_subset_raw_llm_recommendation_metrics']['f1'])}, "
            f"gated_part4={_fmt_metric(report['escalated_subset_gated_part4_metrics']['f1'])}"
        )
        print(f"  actions: {report['llm_recommended_actions']}")
        print(f"  confidence: {report['llm_confidence']}")
        print(f"  gate reasons: {report['gate_reasons']}")

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
