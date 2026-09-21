"""Run the frozen final Part-4 LLM arbitration system.

This is the authoritative Part-4 entry point for the final submission.
It exposes only dataset selection. The model, prompt, context representation,
hard-case selector, device policy, and safety gate are frozen in ``config.py``
from validation-only development.

The script never performs prompt selection, context selection, model
hyperparameter search, or TEST-time configuration changes.
"""

from __future__ import annotations

import argparse
import json

from arbitration_pipeline import run_dataset
from config import (
    DATASETS,
    FINAL_TEST_CONTEXT_VARIANT,
    FINAL_TEST_DEVICE_MODE,
    FINAL_TEST_PROFILE_NAME,
    FINAL_TEST_PROMPT_VARIANT,
    FINAL_TEST_SELECTION_MODE,
    MODEL_NAME,
    RESULT_ROOT,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen final Part-4 Qwen arbitration system on the "
            "authoritative hybrid TEST outputs."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
        help="Dataset to evaluate. Default: both Dataset 2 and Dataset 3.",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("FINAL PART-4 LLM ARBITRATION")
    print("=" * 100)
    print(f"Profile:   {FINAL_TEST_PROFILE_NAME}")
    print(f"Model:     {MODEL_NAME}")
    print(f"Prompt:    {FINAL_TEST_PROMPT_VARIANT}")
    print(f"Context:   {FINAL_TEST_CONTEXT_VARIANT}")
    print(f"Selector:  {FINAL_TEST_SELECTION_MODE}")
    print(f"Device:    {FINAL_TEST_DEVICE_MODE}")
    print("Stage:     TEST (frozen)")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    summaries = [
        run_dataset(
            dataset_name=dataset_name,
            stage="test",
            dry_run=False,
            max_cases=None,
            context_variant=FINAL_TEST_CONTEXT_VARIANT,
            prompt_variant=FINAL_TEST_PROMPT_VARIANT,
            device_mode=FINAL_TEST_DEVICE_MODE,
            selection_mode=FINAL_TEST_SELECTION_MODE,
        )
        for dataset_name in datasets
    ]

    output_dir = RESULT_ROOT / "FINAL_FROZEN_TEST"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "final_part4_run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "profile": FINAL_TEST_PROFILE_NAME,
                "model": MODEL_NAME,
                "prompt_variant": FINAL_TEST_PROMPT_VARIANT,
                "context_variant": FINAL_TEST_CONTEXT_VARIANT,
                "selection_mode": FINAL_TEST_SELECTION_MODE,
                "device_mode": FINAL_TEST_DEVICE_MODE,
                "stage": "test",
                "summaries": summaries,
            },
            handle,
            indent=2,
        )

    print(f"\nFinal run summary: {summary_path}")


if __name__ == "__main__":
    main()
