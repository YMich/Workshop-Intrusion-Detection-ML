import argparse
from isolation_forest_o1 import DATASET_PATHS, evaluate_dataset, train_all, train_dataset, locate_shared_if_hyperparameters


def parse_args():
    parser = argparse.ArgumentParser(description="Isolation Forest O1: controlled 57->43 brittle-feature pruning ablation")
    parser.add_argument("--dataset", choices=list(DATASET_PATHS), default=None)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--force-resplit", action="store_true")
    args = parser.parse_args()
    if args.evaluation_only and args.force_resplit:
        parser.error("--force-resplit cannot be combined with --evaluation-only")
    return args


def main():
    args = parse_args()
    print("=" * 90)
    print("ISOLATION FOREST O1 - BRITTLE FEATURE PRUNING")
    print("=" * 90)
    print("Only optimization change: baseline 57 features -> hardened 43-feature schema.")
    print("Preprocessing remains signed-log1p + RobustScaler.")
    print("One-class benign training and validation max-F1 threshold remain unchanged.")
    print("Shared D2/D3 IF hyperparameters are loaded from the existing baseline artifacts and verified identical.")
    print("Do not use --force-resplit for the controlled comparison.")

    if args.evaluation_only:
        if args.dataset:
            result = evaluate_dataset(args.dataset)
            print(result)
        else:
            from isolation_forest_o1 import evaluate_all
            print(evaluate_all().to_string(index=False))
        return

    if args.dataset:
        params, source = locate_shared_if_hyperparameters()
        result = train_dataset(args.dataset, params, source, force_resplit=args.force_resplit)
        print(result)
    else:
        print(train_all(force_resplit=args.force_resplit).to_string(index=False))


if __name__ == "__main__":
    main()
