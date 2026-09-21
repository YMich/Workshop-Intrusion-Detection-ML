"""Run validation-only Part-4 ablation experiments.

This script exists only to reproduce the validation experiments used to compare
prompt/context/selector variants. It cannot run TEST. The final Part-4 system is
``run_final_part4.py``.
"""

from __future__ import annotations

import argparse
import json

from arbitration_pipeline import run_dataset
from config import (
    CONTEXT_VARIANTS,
    DATASETS,
    FINAL_TEST_CONTEXT_VARIANT,
    FINAL_TEST_DEVICE_MODE,
    FINAL_TEST_PROMPT_VARIANT,
    FINAL_TEST_SELECTION_MODE,
    PROMPT_VARIANTS,
    RESULT_ROOT,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only Part-4 ablation runner. This script cannot access "
            "the final TEST stage."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
    )
    parser.add_argument(
        "--context-variant",
        choices=CONTEXT_VARIANTS,
        default=FINAL_TEST_CONTEXT_VARIANT,
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default=FINAL_TEST_PROMPT_VARIANT,
    )
    parser.add_argument(
        "--selection-mode",
        choices=["all", "final-union", "legacy", "weak-confirmed-only"],
        default=FINAL_TEST_SELECTION_MODE,
    )
    parser.add_argument(
        "--device-mode",
        choices=["split50", "gpu", "cpu"],
        default=FINAL_TEST_DEVICE_MODE,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build selected cases/payloads without calling Ollama.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional validation-only development cap.",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    summaries = [
        run_dataset(
            dataset_name=dataset_name,
            stage="validation",
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
    summary_path = RESULT_ROOT / (
        "validation_ablation_"
        f"{args.context_variant}_{args.prompt_variant}_"
        f"{args.selection_mode}_{args.device_mode}_run_summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    print(f"\nValidation ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
