import argparse

from isolation_forest import (
    BENCHMARK_DATASETS,
    FINAL_IF_PARAMS,
    FINAL_RANDOM_STATE,
    evaluate_all,
    evaluate_dataset,
    train_all,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the final retained Isolation Forest: O1 hardened 43-feature schema "
            "with the frozen shared D2/D3 hyperparameter configuration applied unchanged to Dataset 1."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=list(BENCHMARK_DATASETS),
        default=None,
        help="Evaluation-only: evaluate one dataset's already-trained final artifacts.",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Evaluate existing final artifacts without retraining.",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help=(
            "Explicitly rebuild the shared source-aware split. Normally DO NOT use this; "
            "the final controlled benchmark should reuse the persisted split."
        ),
    )
    args = parser.parse_args()

    if args.dataset is not None and not args.evaluation_only:
        parser.error("--dataset is supported only together with --evaluation-only.")
    if args.evaluation_only and args.force_resplit:
        parser.error("--force-resplit cannot be used with --evaluation-only.")
    return args


def main():
    args = parse_args()

    print("=" * 92)
    print("FINAL OPTIMIZED ISOLATION FOREST - RETAINED O1")
    print("=" * 92)
    print("Feature optimization: 57 -> 43 hardened features.")
    print("Rejected optimizations O2/O3/O4 are NOT part of this final pipeline.")
    print("Shared hyperparameters:")
    for key, value in FINAL_IF_PARAMS.items():
        print(f"  {key}: {value}")
    print(f"  random_state: {FINAL_RANDOM_STATE}")
    print("Training: benign TRAIN only.")
    print("Threshold: per-dataset max-F1 on mixed VALIDATION only.")
    print("TEST is untouched until the threshold is frozen.")

    if args.evaluation_only:
        if args.dataset is not None:
            print(evaluate_dataset(args.dataset))
        else:
            print(evaluate_all().to_string(index=False))
        return

    print(train_all(force_resplit=args.force_resplit).to_string(index=False))


if __name__ == "__main__":
    main()
