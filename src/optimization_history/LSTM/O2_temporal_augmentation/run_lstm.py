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
            "LSTM Optimization 2 on Dataset 2 and Dataset 3: retain the accepted "
            "43-feature O1 representation and add malicious-only temporal/rate "
            "time-warp augmentation while reusing the exact shared hyperparameters, "
            "source-aware split, and validation-only threshold calibration."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PATHS.keys()),
        default=None,
        help=(
            "Evaluate only one benchmark dataset. O2 training still runs both Dataset 2 "
            "and Dataset 3 to preserve the shared experiment design."
        ),
    )

    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Train/tune/calibrate but skip test evaluation.",
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
            "--dataset can only be used with --evaluation-only. O2 training is executed "
            "for Dataset 2 and Dataset 3 together."
        )

    return args


def main():
    args = parse_args()

    print("=" * 90)
    print("LSTM O2 - TEMPORAL/RATE AUGMENTATION ON ACCEPTED O1")
    print("=" * 90)
    print(describe_compute_device())
    print(
        "Split rule: whole SourceFiles first; if one source dominates, only "
        "that source is split chronologically 70/15/15."
    )
    print(
        "Hyperparameter rule: reuse the exact baseline shared D2/D3 hyperparameters; "
        "no hyperparameter is re-tuned in O2."
    )
    print(
        "Threshold rule: calibrated separately per dataset on validation only; "
        "test remains untouched."
    )

    if not args.evaluation_only:
        print("\nPHASE 1 - O2 TRAINING / VALIDATION / THRESHOLD CALIBRATION")
        train_all_datasets(
            force_resplit=args.force_resplit,
        )

    if args.training_only:
        print("\nTraining complete; test evaluation skipped by request.")
        return

    print("\nPHASE 2 - O2 FINAL TEST EVALUATION")
    summary = evaluate_all_datasets(only_dataset=args.dataset)

    print("\n" + "=" * 90)
    print("LSTM COMPLETE")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
