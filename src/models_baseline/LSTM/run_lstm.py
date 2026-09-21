import argparse

from lstm import (
    DATASET_PATHS,
    describe_compute_device,
    train_all_datasets,
)
from lstm_evaluation import evaluate_all_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Baseline LSTM experiment on Dataset 2 and Dataset 3 using one fixed "
            "shared configuration, source-aware 70/15/15 splits, and "
            "validation-only threshold calibration."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PATHS.keys()),
        default=None,
        help=(
            "Evaluate only one benchmark dataset. Fixed baseline training reproduces "
            "both Dataset 2 and Dataset 3."
        ),
    )

    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Train/calibrate but skip test evaluation.",
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
            "this off so all models can reuse exactly the same rows."
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

    if args.dataset is not None and not args.evaluation_only:
        parser.error(
            "--dataset can only be used with --evaluation-only. Baseline training "
            "reproduces Dataset 2 and Dataset 3 together."
        )

    return args


def main():
    args = parse_args()

    print("=" * 90)
    print("LSTM - D2/D3 SHARED-HYPERPARAMETER HOLDOUT EXPERIMENT")
    print("=" * 90)
    print(describe_compute_device())
    print(
        "Split rule: whole SourceFiles first; if one source dominates, only "
        "that source is split chronologically 70/15/15."
    )
    print(
        "Configuration rule: fixed shared baseline; batch_size=128, with no "
        "hyperparameter search during execution."
    )
    print(
        "Threshold rule: calibrated separately per dataset on validation only; "
        "test remains untouched."
    )

    if not args.evaluation_only:
        print("\nPHASE 1 - TRAINING / VALIDATION / THRESHOLD CALIBRATION")
        train_all_datasets(
            force_resplit=args.force_resplit,
        )

    if args.training_only:
        print("\nTraining complete; test evaluation skipped by request.")
        return

    print("\nPHASE 2 - FINAL TEST EVALUATION")
    summary = evaluate_all_datasets(only_dataset=args.dataset)

    print("\n" + "=" * 90)
    print("LSTM COMPLETE")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
