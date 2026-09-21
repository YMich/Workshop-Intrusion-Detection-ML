import argparse

from autoencoder import DATASET_PATHS, train_all_datasets
from autoencoder_evaluation import evaluate_all_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Dataset-2/Dataset-3 Autoencoder experiment using one shared "
            "hyperparameter configuration and the persisted source-aware 70/15/15 split."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PATHS.keys()),
        default=None,
        help=(
            "Evaluate one benchmark dataset only. Fixed baseline training runs both "
            "Dataset 2 and Dataset 3."
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
        "--force",
        action="store_true",
        help="Retrain even if compatible Autoencoder artifacts already exist.",
    )

    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help=(
            "Rebuild the shared split manifest. Normally leave this off so "
            "LSTM, Autoencoder and Isolation Forest reuse the same rows."
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
    print("AUTOENCODER - D2/D3 SHARED-HYPERPARAMETER HOLDOUT")
    print("=" * 90)
    print(
        "Split: persisted 70/15/15 source-aware manifest; whole SourceFiles "
        "whenever possible, chronological fallback only for an oversized source."
    )
    print(
        "Configuration rule: fixed shared baseline; learning_rate=0.002, "
        "loss=Huber, with no hyperparameter search during execution."
    )
    print(
        "One-class rule: preprocessing and Autoencoder fitting use benign training "
        "rows only; validation remains mixed."
    )
    print(
        "Score rule: Top-5 MSE is shared across D2/D3. Final threshold is calibrated "
        "separately per dataset on validation only; test remains untouched."
    )

    if not args.evaluation_only:
        print("\nPHASE 1 - TRAINING / VALIDATION / THRESHOLD CALIBRATION")
        train_all_datasets(
            force=args.force,
            force_resplit=args.force_resplit,
        )

    if args.training_only:
        print("\nTraining complete; test evaluation skipped by request.")
        return

    print("\nPHASE 2 - FINAL SHARED TEST EVALUATION")

    summary = evaluate_all_datasets(only_dataset=args.dataset)

    print("\n" + "=" * 90)
    print("AUTOENCODER COMPLETE")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
