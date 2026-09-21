import argparse
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from random_forest_training import train_all_datasets  # noqa: E402
from random_forest_evaluation import evaluate_all_datasets  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Dataset-2/Dataset-3 Random Forest baseline using one fixed "
            "shared configuration and source-aware group OOF evaluation."
        )
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Run fixed-configuration training and skip OOF result evaluation.",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Evaluate existing Random Forest OOF results/artifacts without retraining.",
    )
    args = parser.parse_args()
    if args.training_only and args.evaluation_only:
        parser.error("--training-only and --evaluation-only cannot be used together.")
    return args


def main():
    args = parse_args()
    print("=" * 90)
    print("RANDOM FOREST - FIXED D2/D3 BASELINE GROUP OOF")
    print("=" * 90)
    print(
        "Evaluation rule: source-aware grouped outer folds provide OOF predictions; "
        "there is no inner hyperparameter search."
    )
    print(
        "Fixed configuration: n_estimators=128, max_depth=10, "
        "min_samples_leaf=1, max_features=0.5, bootstrap=True."
    )
    print(
        "Class imbalance rule: balanced class weights are recomputed from each "
        "current training fold only."
    )
    print("Prediction threshold: fixed at 0.50.")

    if not args.evaluation_only:
        print("\nPHASE 1: FIXED-CONFIGURATION GROUP-OOF TRAINING")
        train_all_datasets()
    if args.training_only:
        print("\nTraining complete; OOF evaluation skipped by request.")
        return
    print("\nPHASE 2: OUT-OF-FOLD EVALUATION")
    summary = evaluate_all_datasets()
    print("\n" + "=" * 90)
    print("RANDOM FOREST BASELINE COMPLETE")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
