from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from config import (
    CONTEXT_VARIANTS,
    DATASETS,
    FINAL_TEST_CONTEXT_VARIANT,
    FINAL_TEST_DEVICE_MODE,
    FINAL_TEST_PROFILE_NAME,
    FINAL_TEST_PROMPT_VARIANT,
    FINAL_TEST_SELECTION_MODE,
    HALF_GPU_OFFLOAD_LAYERS,
    KEEP_MODEL_RESIDENT,
    MODEL_NAME,
    OLLAMA_OPTIONS,
    PROMPT_VARIANTS,
    QWEN_TRANSFORMER_LAYERS,
    RESULT_ROOT,
)
from evaluation import metrics_from_predictions
from llm_arbitrator import apply_safety_gate, infer_one
from pipeline_io import ROW_KEY, build_dataset_state, hybrid
from prompts import get_prompt_pair
from telemetry_context import (
    add_validation_strengths,
    build_visible_payload,
    fit_benign_baseline,
    fit_strength_calibration,
    select_hard_cases,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_prompt_artifacts(output_dir, prompt_variant: str) -> None:
    system_prompt, user_prompt_template = get_prompt_pair(prompt_variant)
    (output_dir / "system_prompt.txt").write_text(
        system_prompt + "\n", encoding="utf-8"
    )
    (output_dir / "user_prompt_template.txt").write_text(
        user_prompt_template + "\n", encoding="utf-8"
    )


def _write_payloads(payloads: list[dict], path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=False)
                + "\n"
            )


def run_dataset(
    dataset_name: str,
    stage: str,
    dry_run: bool,
    max_cases: int | None,
    context_variant: str,
    prompt_variant: str,
    device_mode: str,
    selection_mode: str,
) -> dict:
    print("\n" + "=" * 110)
    print(
        f"PART 4 LOCAL LLM ARBITRATION | {dataset_name.upper()} | "
        f"{stage.upper()} | context={context_variant} | prompt={prompt_variant}"
    )
    print("=" * 110)

    if stage == "test":
        frozen = {
            "context_variant": FINAL_TEST_CONTEXT_VARIANT,
            "prompt_variant": FINAL_TEST_PROMPT_VARIANT,
            "selection_mode": FINAL_TEST_SELECTION_MODE,
            "device_mode": FINAL_TEST_DEVICE_MODE,
        }
        actual = {
            "context_variant": context_variant,
            "prompt_variant": prompt_variant,
            "selection_mode": selection_mode,
            "device_mode": device_mode,
        }
        mismatches = {
            key: (actual[key], expected)
            for key, expected in frozen.items()
            if actual[key] != expected
        }
        if mismatches:
            raise ValueError(
                "FINAL TEST configuration is frozen from validation. "
                f"Refusing modified TEST settings: {mismatches}"
            )
        if max_cases is not None:
            raise ValueError(
                "--max-cases is forbidden on the FINAL TEST run."
            )
        if dry_run:
            raise ValueError(
                "--dry-run is disabled for FINAL TEST. Run the frozen TEST once."
            )

    state = build_dataset_state(dataset_name, target_split=stage)

    calibration = fit_strength_calibration(
        state["validation_scored"],
        lstm_threshold=state["lstm_threshold"],
        litemv_threshold=state["litemv_threshold"],
        rf_threshold=state["rf_threshold"],
    )

    target = add_validation_strengths(
        state["target_scored"],
        calibration,
        lstm_threshold=state["lstm_threshold"],
        litemv_threshold=state["litemv_threshold"],
        rf_threshold=state["rf_threshold"],
    )

    baseline = fit_benign_baseline(state["train_raw"])
    hard_cases = select_hard_cases(
        target, selection_mode=selection_mode
    ).sort_values(
        [hybrid.SOURCE_FILE_COL, ROW_KEY],
        kind="mergesort",
    )

    if max_cases is not None:
        hard_cases = hard_cases.head(int(max_cases))

    print(
        f"Rows={len(target):,} | escalated={len(hard_cases):,} | "
        f"model={MODEL_NAME} | device_mode={device_mode} | "
        f"selection_mode={selection_mode}"
        + (
            f" | GPU layers={HALF_GPU_OFFLOAD_LAYERS}/"
            f"{QWEN_TRANSFORMER_LAYERS}"
            if device_mode == "split50"
            else ""
        )
    )

    if stage == "test":
        output_dir = (
            RESULT_ROOT
            / "FINAL_FROZEN_TEST"
            / dataset_name
            / "test"
        )
    else:
        output_dir = RESULT_ROOT / context_variant
        if prompt_variant != "baseline":
            output_dir = output_dir / f"prompt_{prompt_variant}"
        if selection_mode != "all":
            output_dir = output_dir / f"selector_{selection_mode}"
        output_dir = output_dir / dataset_name / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    complete_marker = output_dir / "FINAL_TEST_COMPLETE.json"
    if stage == "test" and complete_marker.is_file():
        completed = json.loads(complete_marker.read_text(encoding="utf-8"))
        print(
            "FINAL TEST already completed for this dataset; "
            "reusing the frozen saved result without rerunning Qwen."
        )
        return completed["summary"]

    _write_prompt_artifacts(output_dir, prompt_variant)

    selection_columns = [
        hybrid.SOURCE_FILE_COL,
        ROW_KEY,
        "DecisionRoute",
        "EscalationReason",
        "Hybrid_Pred",
        "LSTM_Pred",
        "LITEMV_Pred",
        "RF_Pred",
        "LSTM_EvidenceStrength",
        "LITEMV_EvidenceStrength",
        "RF_EvidenceStrength",
    ]
    hard_cases[selection_columns].to_csv(
        output_dir / "escalated_cases_internal.csv",
        index=False,
    )

    payload_records: list[tuple[pd.Series, dict]] = []
    payloads: list[dict] = []
    for ordinal, (_index, row) in enumerate(
        hard_cases.iterrows(), start=1
    ):
        # Anonymous within-run identifier: no dataset/source information encoded.
        case_id = f"case_{ordinal:05d}"
        payload = build_visible_payload(
            case_id=case_id,
            row=row,
            full_frame=target,
            baseline=baseline,
            context_variant=context_variant,
        )
        payload_records.append((row, payload))
        payloads.append(payload)

    # Exact model-visible contexts for Section 4.2. These contain no ground truth
    # or source/capture identifiers.
    _write_payloads(payloads, output_dir / "llm_payloads.jsonl")

    system_prompt, user_prompt_template = get_prompt_pair(prompt_variant)
    prompt_manifest = {
        "model": MODEL_NAME,
        "ollama_options": OLLAMA_OPTIONS,
        "final_test_profile": (
            FINAL_TEST_PROFILE_NAME if stage == "test" else None
        ),
        "configuration_frozen_before_test": bool(stage == "test"),
        "context_variant": context_variant,
        "prompt_variant": prompt_variant,
        "prompt_only_optimization": bool(prompt_variant != "baseline"),
        "telemetry_context_schema": (
            "O1_EXPLICIT_BEHAVIORAL_SUMMARIES"
            if context_variant == "behavioral_o1"
            else "BASELINE"
        ),
        "context_only_optimization": bool(
            context_variant == "behavioral_o1"
        ),
        "selection_mode": selection_mode,
        "device_mode": device_mode,
        "device_assignment": (
            "18_of_36_transformer_layers_offloaded_to_gpu_per_request"
            if device_mode == "split50"
            else (
                "cpu_only_num_gpu_0"
                if device_mode == "cpu"
                else "ollama_gpu_auto_offload"
            )
        ),
        "qwen_transformer_layers": int(QWEN_TRANSFORMER_LAYERS),
        "requested_gpu_layers": (
            int(HALF_GPU_OFFLOAD_LAYERS)
            if device_mode == "split50"
            else (0 if device_mode == "cpu" else None)
        ),
        "system_prompt_sha256": _sha256_text(system_prompt),
        "user_prompt_template_sha256": _sha256_text(user_prompt_template),
        "ground_truth_sent_to_llm": False,
        "dataset_identity_sent_to_llm": False,
        "sourcefile_sent_to_llm": False,
        "ip_or_port_sent_to_llm": False,
    }
    with (output_dir / "prompt_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(prompt_manifest, handle, indent=2)

    if dry_run:
        summary = {
            "dataset": dataset_name,
            "stage": stage,
            "rows": int(len(target)),
            "escalated_cases": int(len(hard_cases)),
            "context_variant": context_variant,
            "prompt_variant": prompt_variant,
            "selection_mode": selection_mode,
            "device_mode": device_mode,
            "keep_model_resident": bool(KEEP_MODEL_RESIDENT),
            "dry_run": True,
        }
        with (output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2)
        print(
            "Dry run complete: hard cases and exact anonymized LLM payloads "
            "were written; Ollama was not called."
        )
        return summary

    checkpoint_path = output_dir / "llm_outputs_checkpoint.csv"
    outputs = []
    completed_keys = set()
    if stage == "test" and checkpoint_path.is_file():
        checkpoint = pd.read_csv(checkpoint_path)
        required_resume = {hybrid.SOURCE_FILE_COL, ROW_KEY}
        if not required_resume.issubset(checkpoint.columns):
            raise RuntimeError(
                "Existing FINAL TEST checkpoint is malformed; refusing to "
                "restart or overwrite it."
            )
        outputs = checkpoint.to_dict(orient="records")
        completed_keys = {
            (str(row[hybrid.SOURCE_FILE_COL]), int(row[ROW_KEY]))
            for row in outputs
        }
        print(
            f"Resuming FINAL TEST checkpoint: {len(outputs):,}/"
            f"{len(payload_records):,} arbitration cases already completed."
        )

    for ordinal, (row, payload) in enumerate(payload_records, start=1):
        resume_key = (
            str(row[hybrid.SOURCE_FILE_COL]),
            int(row[ROW_KEY]),
        )
        if resume_key in completed_keys:
            continue
        case_id = payload["case_id"]
        mode_text = (
            f"GPU+CPU({HALF_GPU_OFFLOAD_LAYERS}/"
            f"{QWEN_TRANSFORMER_LAYERS} GPU layers)"
            if device_mode == "split50"
            else device_mode.upper()
        )
        print(
            f"\n[{ordinal:03d}/{len(payload_records):03d}] {case_id} | "
            f"{row['DecisionRoute']} | hybrid={int(row['Hybrid_Pred'])} | "
            f"device={mode_text}"
        )

        decision, runtime = infer_one(
            payload,
            execution_mode=device_mode,
            prompt_variant=prompt_variant,
        )
        gated = apply_safety_gate(
            hybrid_label=int(row["Hybrid_Pred"]),
            decision=decision,
            lstm_strength=str(row["LSTM_EvidenceStrength"]),
            litemv_strength=str(row["LITEMV_EvidenceStrength"]),
            rf_strength=str(row["RF_EvidenceStrength"]),
        )

        outputs.append(
            {
                "CaseID": case_id,
                "ExecutionMode": runtime.execution_mode,
                "RequestedGPULayers": runtime.requested_gpu_layers,
                "TotalTransformerLayers": runtime.total_transformer_layers,
                hybrid.SOURCE_FILE_COL: row[hybrid.SOURCE_FILE_COL],
                ROW_KEY: int(row[ROW_KEY]),
                "DecisionRoute": str(row["DecisionRoute"]),
                "EscalationReason": str(row["EscalationReason"]),
                "HybridPred": int(row["Hybrid_Pred"]),
                "LLMRecommendedAction": decision.recommended_action,
                "LLMConfidence": decision.confidence,
                "LLMTargetSpecificity": decision.target_specificity,
                "LLMEvidence": json.dumps(
                    decision.evidence, ensure_ascii=False
                ),
                "LLMConflictSummary": decision.conflict_summary,
                "OverrideAccepted": bool(gated.override_accepted),
                "GateReason": gated.gate_reason,
                "FinalPred": int(gated.final_label),
                "WallSeconds": float(runtime.wall_seconds),
                "GPUPreC": runtime.pre_gpu_c,
                "GPUMaxC": runtime.max_gpu_c,
                "GPUPostC": runtime.post_gpu_c,
                "GPUPreMemoryMB": runtime.pre_gpu_memory_mb,
                "GPUMaxMemoryMB": runtime.max_gpu_memory_mb,
                "GPUPostMemoryMB": runtime.post_gpu_memory_mb,
                "GPUPeakIncrementMB": runtime.peak_gpu_memory_increment_mb,
                "PromptTokens": int(runtime.prompt_tokens),
                "CompletionTokens": int(runtime.completion_tokens),
            }
        )

        # Persist progress after every completed arbitration so a hardware abort
        # does not discard earlier decisions from a long validation run.
        pd.DataFrame(outputs).to_csv(checkpoint_path, index=False)

    outputs_df = pd.DataFrame(outputs)
    if outputs_df.empty:
        # Preserve a valid, headered artifact when the frozen selector escalates
        # zero cases.  This is a legitimate experimental outcome, not an error.
        outputs_df = pd.DataFrame(
            columns=[
                "CaseID",
                "ExecutionMode",
                "RequestedGPULayers",
                "TotalTransformerLayers",
                hybrid.SOURCE_FILE_COL,
                ROW_KEY,
                "DecisionRoute",
                "EscalationReason",
                "HybridPred",
                "LLMRecommendedAction",
                "LLMConfidence",
                "LLMTargetSpecificity",
                "LLMEvidence",
                "LLMConflictSummary",
                "OverrideAccepted",
                "GateReason",
                "FinalPred",
                "WallSeconds",
                "GPUPreC",
                "GPUMaxC",
                "GPUPostC",
                "GPUPreMemoryMB",
                "GPUMaxMemoryMB",
                "GPUPostMemoryMB",
                "GPUPeakIncrementMB",
                "PromptTokens",
                "CompletionTokens",
            ]
        )

    # Saved before labels are joined for evaluation.
    outputs_df.to_csv(
        output_dir / "llm_outputs_unlabeled.csv",
        index=False,
    )

    final_pred = target["Hybrid_Pred"].astype(int).to_numpy().copy()
    if not outputs_df.empty:
        position_lookup = {
            (row[hybrid.SOURCE_FILE_COL], int(row[ROW_KEY])): pos
            for pos, (_idx, row) in enumerate(target.iterrows())
        }
        for row in outputs_df.itertuples(index=False):
            key = (
                getattr(row, hybrid.SOURCE_FILE_COL),
                int(getattr(row, ROW_KEY)),
            )
            final_pred[position_lookup[key]] = int(row.FinalPred)

    y_true = target[hybrid.LABEL_COL].astype(int).to_numpy()
    hybrid_pred = target["Hybrid_Pred"].astype(int).to_numpy()

    hybrid_metrics = metrics_from_predictions(y_true, hybrid_pred)
    part4_metrics = metrics_from_predictions(y_true, final_pred)

    changed = final_pred != hybrid_pred
    corrected = int(
        np.sum(changed & (hybrid_pred != y_true) & (final_pred == y_true))
    )
    introduced = int(
        np.sum(changed & (hybrid_pred == y_true) & (final_pred != y_true))
    )

    predictions = target[
        [
            hybrid.SOURCE_FILE_COL,
            ROW_KEY,
            hybrid.LABEL_COL,
            "DecisionRoute",
            "LSTM_Pred",
            "LITEMV_Pred",
            "RF_Pred",
            "Hybrid_Pred",
            "LSTM_EvidenceStrength",
            "LITEMV_EvidenceStrength",
            "RF_EvidenceStrength",
        ]
    ].copy()
    predictions["Part4_Final_Pred"] = final_pred
    predictions.to_csv(
        output_dir / "part4_predictions.csv",
        index=False,
    )

    metrics_table = pd.DataFrame(
        [
            {
                "Dataset": dataset_name,
                "Stage": stage,
                "System": "Part3_LSTM_LITEMV_RF_Hybrid",
                **hybrid_metrics,
            },
            {
                "Dataset": dataset_name,
                "Stage": stage,
                "System": "Part4_Qwen_Gated",
                **part4_metrics,
            },
        ]
    )
    metrics_table.to_csv(output_dir / "metrics.csv", index=False)

    if not outputs_df.empty:
        runtime_columns = [
            "WallSeconds",
            "GPUPreC",
            "GPUMaxC",
            "GPUPostC",
            "GPUPreMemoryMB",
            "GPUMaxMemoryMB",
            "GPUPostMemoryMB",
            "GPUPeakIncrementMB",
            "PromptTokens",
            "CompletionTokens",
        ]
        outputs_df[runtime_columns].describe(include="all").to_csv(
            output_dir / "runtime_summary.csv"
        )

    summary = {
        "dataset": dataset_name,
        "stage": stage,
        "final_test_profile": (
            FINAL_TEST_PROFILE_NAME if stage == "test" else None
        ),
        "configuration_frozen_before_test": bool(stage == "test"),
        "model": MODEL_NAME,
        "ollama_options": OLLAMA_OPTIONS,
        "context_variant": context_variant,
        "prompt_variant": prompt_variant,
        "prompt_only_optimization": bool(prompt_variant != "baseline"),
        "telemetry_context_schema": (
            "O1_EXPLICIT_BEHAVIORAL_SUMMARIES"
            if context_variant == "behavioral_o1"
            else "BASELINE"
        ),
        "context_only_optimization": bool(
            context_variant == "behavioral_o1"
        ),
        "selection_mode": selection_mode,
        "rows": int(len(target)),
        "escalated_cases": int(len(hard_cases)),
        "device_mode": device_mode,
        "keep_model_resident": bool(KEEP_MODEL_RESIDENT),
        "qwen_transformer_layers": int(QWEN_TRANSFORMER_LAYERS),
        "requested_gpu_layers": (
            int(HALF_GPU_OFFLOAD_LAYERS)
            if device_mode == "split50"
            else (0 if device_mode == "cpu" else None)
        ),
        "requested_cpu_transformer_layers": (
            int(QWEN_TRANSFORMER_LAYERS - HALF_GPU_OFFLOAD_LAYERS)
            if device_mode == "split50"
            else (int(QWEN_TRANSFORMER_LAYERS) if device_mode == "cpu" else None)
        ),
        "accepted_overrides": int(
            outputs_df["OverrideAccepted"].sum()
            if not outputs_df.empty
            else 0
        ),
        "corrected_hybrid_errors": corrected,
        "introduced_new_errors": introduced,
        "net_error_benefit": corrected - introduced,
        "hybrid_metrics": hybrid_metrics,
        "part4_metrics": part4_metrics,
        "peak_observed_gpu_memory_mb": (
            int(outputs_df["GPUMaxMemoryMB"].dropna().max())
            if (
                not outputs_df.empty
                and outputs_df["GPUMaxMemoryMB"].notna().any()
            )
            else None
        ),
        "peak_observed_llm_gpu_increment_mb": (
            int(outputs_df["GPUPeakIncrementMB"].dropna().max())
            if (
                not outputs_df.empty
                and outputs_df["GPUPeakIncrementMB"].notna().any()
            )
            else None
        ),
        "ground_truth_sent_to_llm": False,
        "identifiers_sent_to_llm": False,
    }

    with (output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    if stage == "test":
        final_marker = {
            "profile": FINAL_TEST_PROFILE_NAME,
            "configuration_frozen_before_test": True,
            "context_variant": FINAL_TEST_CONTEXT_VARIANT,
            "prompt_variant": FINAL_TEST_PROMPT_VARIANT,
            "selection_mode": FINAL_TEST_SELECTION_MODE,
            "device_mode": FINAL_TEST_DEVICE_MODE,
            "summary": summary,
        }
        with (output_dir / "FINAL_TEST_COMPLETE.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(final_marker, handle, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Part 4: conservative local-Qwen arbitration on top of the frozen "
            "LSTM -> LITEMV -> RF hybrid."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
    )
    parser.add_argument(
        "--stage",
        choices=["validation", "test"],
        default="validation",
    )
    parser.add_argument(
        "--context-variant",
        choices=CONTEXT_VARIANTS,
        default=FINAL_TEST_CONTEXT_VARIANT,
        help=(
            "Frozen TEST uses behavioral_o1. Validation may override this for "
            "documented sensitivity experiments."
        ),
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default=FINAL_TEST_PROMPT_VARIANT,
        help=(
            "Prompt formulation. FINAL TEST is frozen to semantic_o3; other "
            "variants remain available only for validation sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=["all", "final-union", "legacy", "weak-confirmed-only"],
        default=FINAL_TEST_SELECTION_MODE,
        help=(
            "Hard-case selector. final-union = explicit final-validation selector "
            "(legacy hard routes plus the validation-justified weak LSTM/LITEMV "
            "confirmation vs strong RF benign disagreement); all uses the same union "
            "and is reserved for the eventual frozen TEST configuration; legacy = "
            "original routes only; "
            "weak-confirmed-only = run only the 9 newly discovered validation "
            "cases for the targeted experiment."
        ),
    )
    parser.add_argument(
        "--device-mode",
        choices=["split50", "gpu", "cpu"],
        default=FINAL_TEST_DEVICE_MODE,
        help=(
            "LLM execution policy. split50 offloads 18 of Qwen3-8B's 36 "
            "transformer layers to GPU on every request; the remaining 18 "
            "run on CPU."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build/select cases and exact payloads but do not call Ollama.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Development cap; forbidden on TEST.",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    summaries = [
        run_dataset(
            dataset_name,
            stage=args.stage,
            dry_run=args.dry_run,
            max_cases=args.max_cases,
            context_variant=args.context_variant,
            prompt_variant=args.prompt_variant,
            device_mode=args.device_mode,
            selection_mode=args.selection_mode,
        )
        for dataset_name in datasets
    ]

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    run_summary_path = RESULT_ROOT / (
        f"{args.stage}_{args.context_variant}_{args.prompt_variant}_{args.selection_mode}_{args.device_mode}_run_summary.json"
    )
    with run_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)


if __name__ == "__main__":
    main()
