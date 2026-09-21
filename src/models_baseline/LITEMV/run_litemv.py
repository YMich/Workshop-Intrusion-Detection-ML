from __future__ import annotations

import argparse

from litemv import (
    DATASET_PATHS,
    SHARED_PARAMS,
    describe_compute_device,
    train_all_datasets,
)
from litemv_evaluation import evaluate_all_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "LITEMV experiment using the persisted shared source-aware split, "
            "a fixed shared D2/D3 model configuration, and validation-only threshold "
            "calibration."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PATHS.keys()),
        default=None,
        help="Run only one dataset. Default: dataset2 and dataset3.",
    )

    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Train/calibrate but skip final test evaluation.",
    )

    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Evaluate already-trained artifacts without retraining.",
    )

    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help=(
            "Rebuild the shared source-aware split manifest. Normally leave "
            "this OFF so LITEMV uses exactly the same rows as the other models."
        ),
    )

    args = parser.parse_args()

    if args.training_only and args.evaluation_only:
        parser.error(
            "--training-only and --evaluation-only cannot be used together."
        )

    if args.evaluation_only and args.force_resplit:
        parser.error(
            "--force-resplit cannot be combined with --evaluation-only."
        )

    return args


def main():
    args = parse_args()

    print("=" * 100)
    print("LITEMV - SOURCE-AWARE MULTIVARIATE TIME-SERIES CLASSIFIER")
    print("=" * 100)
    print(describe_compute_device())
    print(
        "Split rule: reuse persisted source-aware train/validation/test rows."
    )
    print(
        "Representation: causal SourceFile-local flow windows; "
        "zero-left-padded prefixes."
    )
    print(
        "Feature rule: fixed model-agnostic 57-feature baseline; "
        "no LSTM-specific O1/O2 optimization."
    )
    print(
        "Configuration rule: one frozen shared LITEMV configuration for Dataset 2 "
        "and Dataset 3; no hyperparameter search is performed during execution."
    )
    print(
        "Threshold rule: calibrated separately per dataset on validation only; "
        "test remains untouched."
    )
    print(f"Shared parameters: {SHARED_PARAMS}")

    if not args.evaluation_only:
        print("\nPHASE 1 - TRAINING / VALIDATION / THRESHOLD CALIBRATION")
        train_all_datasets(
            only_dataset=args.dataset,
            force_resplit=args.force_resplit,
        )

    if args.training_only:
        print("\nTraining complete; final test evaluation skipped by request.")
        return

    print("\nPHASE 2 - FINAL TEST EVALUATION")
    summary = evaluate_all_datasets(
        only_dataset=args.dataset,
    )

    print("\n" + "=" * 100)
    print("LITEMV COMPLETE")
    print("=" * 100)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
