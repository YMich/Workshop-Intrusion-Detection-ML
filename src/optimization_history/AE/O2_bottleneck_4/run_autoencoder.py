import argparse

from autoencoder import DATASET_PATHS, train_all_datasets, train_dataset
from autoencoder_evaluation import evaluate_all_datasets, evaluate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Autoencoder Optimization 2 (bottleneck constriction) using "
            "the persisted shared source-aware 70/15/15 split."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PATHS.keys()),
        default=None,
        help="Run one benchmark dataset only. Default: Dataset 2 and Dataset 3.",
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

    return args


def main():
    args = parse_args()

    print("=" * 90)
    print("AUTOENCODER O2 - BOTTLENECK CONSTRICTION")
    print("=" * 90)
    print(
        "Split: persisted 70/15/15 source-aware manifest; whole SourceFiles "
        "whenever possible, chronological fallback only for an oversized source."
    )
    print(
        "Retained O1 rule: keep the same 43-feature brittle-feature-pruned schema."
    )
    print(
        "O2 rule: change bottleneck_dim from 8 to 4 only; keep LR=0.002, Huber, "
        "L2=1e-5, noise=0.02, batch=256 and Top-5 MSE fixed; recalibrate "
        "threshold on O2 validation only."
    )

    if not args.evaluation_only:
        print("\nPHASE 1 - O2 TRAINING / VALIDATION / THRESHOLD CALIBRATION")
        if args.dataset is None:
            train_all_datasets(
                force=args.force,
                force_resplit=args.force_resplit,
            )
        else:
            train_dataset(
                args.dataset,
                force=args.force,
                force_resplit=args.force_resplit,
            )

    if args.training_only:
        print("\nTraining complete; test evaluation skipped by request.")
        return

    print("\nPHASE 2 - O2 FINAL TEST EVALUATION")

    if args.dataset is None:
        summary = evaluate_all_datasets()
    else:
        summary = evaluate_all_datasets(only_dataset=args.dataset)

    print("\n" + "=" * 90)
    print("AUTOENCODER O2 COMPLETE")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
