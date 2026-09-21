import argparse

from autoencoder import DATASET_PATHS, train_all_datasets, train_dataset
from autoencoder_evaluation import evaluate_all_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Dataset-1 appendix Autoencoder using the frozen final D2/D3 configuration using the "
            "fixed O5 configuration and the persisted source-aware 70/15/15 split."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PATHS.keys()),
        default=None,
        help="Run one benchmark dataset only. Default: Dataset 1.",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Train/calibrate but skip final test evaluation.",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Evaluate already-trained final artifacts without retraining.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if compatible final artifacts already exist.",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help=(
            "Rebuild the shared split manifest. Normally leave this OFF so AE, "
            "LSTM, IF, and other models reuse the same rows."
        ),
    )

    args = parser.parse_args()
    if args.training_only and args.evaluation_only:
        parser.error("--training-only and --evaluation-only cannot be used together.")
    if args.evaluation_only and args.force_resplit:
        parser.error("--force-resplit cannot be combined with --evaluation-only.")
    return args


def main():
    args = parse_args()

    print("=" * 90)
    print("DATASET 1 APPENDIX - FINAL OPTIMIZED AUTOENCODER")
    print("=" * 90)
    print(
        "Split: persisted source-aware 70/15/15 manifest; whole SourceFiles whenever "
        "possible, chronological fallback only for an oversized source."
    )
    print(
        "Features: final hardened 43-feature schema; environment-dependent Init Win, "
        "connection-time/header, and TCP flag features removed."
    )
    print(
        "Fixed shared model: encoder 64-32, bottleneck=4, LR=0.002, MSE loss, "
        "L2=1e-5, denoising=0.0, batch=256, Top-5 MSE score."
    )
    print(
        "Threshold: calibrated independently per dataset on validation F1 only; "
        "test remains untouched until final evaluation."
    )

    if not args.evaluation_only:
        print("\nPHASE 1 - FINAL TRAINING / VALIDATION / THRESHOLD CALIBRATION")
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

    print("\nPHASE 2 - FINAL TEST EVALUATION")
    summary = evaluate_all_datasets(only_dataset=args.dataset)

    print("\n" + "=" * 90)
    print("FINAL OPTIMIZED AUTOENCODER COMPLETE")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
