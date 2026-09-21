from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: Path, *args: str) -> None:
    command = [sys.executable, str(script), *args]
    print("\n" + "=" * 100)
    print("RUN:", " ".join(command))
    print("=" * 100)
    subprocess.run(command, cwd=script.parent, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-1 appendix experiments without modifying final D2/D3 source code."
    )
    parser.add_argument(
        "--phase",
        choices=["models", "cross", "hybrid", "analysis", "part4", "all-no-llm"],
        default="models",
    )
    parser.add_argument(
        "--model",
        choices=["RF", "LSTM", "LITEMV", "IF", "AE"],
        default=None,
        help="Optional: run only one model during --phase models or --phase cross.",
    )
    args = parser.parse_args()

    def wanted(name: str) -> bool:
        return args.model is None or args.model == name

    if args.phase in {"models", "all-no-llm"}:
        if wanted("RF"):
            run(ROOT / "models" / "RF" / "run_random_forest.py")
        if wanted("LSTM"):
            run(ROOT / "models" / "LSTM" / "run_lstm.py")
        if wanted("LITEMV"):
            run(ROOT / "models" / "LITEMV" / "run_litemv.py", "--dataset", "dataset1")
        if wanted("IF"):
            run(ROOT / "models" / "IF" / "run_isolation_forest.py")
        if wanted("AE"):
            run(ROOT / "models" / "AE" / "run_autoencoder.py", "--dataset", "dataset1")

    if args.phase in {"cross", "all-no-llm"}:
        if wanted("RF"):
            run(ROOT / "models" / "RF" / "cross_dataset_evaluation.py")
        if wanted("LSTM"):
            run(ROOT / "models" / "LSTM" / "cross_dataset_evaluation.py")
        if wanted("LITEMV"):
            run(ROOT / "models" / "LITEMV" / "run_litemv_full_target_transfer.py")
        if wanted("IF"):
            run(ROOT / "models" / "IF" / "cross_dataset_evaluation.py")
        if wanted("AE"):
            run(ROOT / "models" / "AE" / "cross_dataset_evaluation.py")

    if args.phase in {"hybrid", "all-no-llm"}:
        run(ROOT / "part3_hybrid" / "hybrid_pipeline.py", "--dataset", "dataset1")

    if args.phase in {"analysis", "all-no-llm"}:
        run(ROOT / "analysis" / "forensic_error_analysis.py")
        run(ROOT / "analysis" / "compare_datasets.py")

    if args.phase == "part4":
        run(ROOT / "part4_llm_arbitration" / "run_part4.py", "--stage", "test", "--dataset", "dataset1")
        run(ROOT / "part4_llm_arbitration" / "report_part4_performance.py")


if __name__ == "__main__":
    main()
