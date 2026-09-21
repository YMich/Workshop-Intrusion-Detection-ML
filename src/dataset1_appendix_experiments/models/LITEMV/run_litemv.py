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
            "Final optimized LITEMV O1 using the persisted source-aware split, "
            "frozen shared D2/D3 model hyperparameters applied to D1, and validation-only threshold "
            "calibration."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PATHS.keys()),
        default=None,
        help="Run only one dataset. Default: dataset1.",
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
    print("LITEMV FINAL O1 - 43-FEATURE HARDENED MULTIVARIATE CLASSIFIER")
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
        "Feature rule: common 57-feature schema -> accepted O1 hardening -> "
        "43 final features."
    )
    print(
        "Hyperparameter rule: the frozen shared D2/D3 LITEMV configuration applied unchanged to Dataset 1."
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
