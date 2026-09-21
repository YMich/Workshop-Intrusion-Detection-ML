from __future__ import annotations

"""
One-command launcher for the final Part-3 hybrid intrusion-detection pipeline.

Final architecture:
    LSTM -> LITEMV -> RF

Important artifact rule
-----------------------
Part 3 uses the frozen fitted LSTM/LITEMV snapshot that generated the report.
Those fitted weights are stored under:

    artifacts/hybrid_pipeline/base_models/<dataset>/{LSTM,LITEMV}/

They are intentionally separate from:

    artifacts/models_optimized_final/

which belongs to the independently reproducible Part-2 model runs.  This
separation prevents a fresh stochastic Part-2 neural-model retraining from
silently replacing the fitted model instance used by the report-facing Part-3
cascade.

The launcher never retrains the Part-3 neural base models.  It validates the
frozen snapshot and then runs hybrid_pipeline.py.  The RF arbiter is fit inside
hybrid_pipeline.py on TRAIN only; routing thresholds are calibrated on
VALIDATION only; TEST is reporting-only.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


DATASETS = ("dataset2", "dataset3")


def discover_project_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (
            (candidate / "src").is_dir()
            and (candidate / "data" / "ingested").is_dir()
        ):
            return candidate
    raise RuntimeError(
        "Could not locate project root. Expected a parent containing "
        "src/ and data/ingested/."
    )


PROJECT_ROOT = discover_project_root()
HYBRID_DIR = PROJECT_ROOT / "src" / "hybrid_pipeline"
HYBRID_SCRIPT = HYBRID_DIR / "hybrid_pipeline.py"

LSTM_DIR = PROJECT_ROOT / "src" / "models_optimized_final" / "LSTM"
LITEMV_DIR = PROJECT_ROOT / "src" / "models_optimized_final" / "LITEMV"
RF_DIR = PROJECT_ROOT / "src" / "models_optimized_final" / "RF"

HYBRID_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "hybrid_pipeline"
PART3_BASE_MODEL_ROOT = HYBRID_ARTIFACT_ROOT / "base_models"
HYBRID_RESULT_ROOT = PROJECT_ROOT / "results" / "hybrid_pipeline"
SNAPSHOT_MANIFEST = PART3_BASE_MODEL_ROOT / "PART3_BASE_MODEL_MANIFEST.json"


REQUIRED_SOURCE_FILES = (
    HYBRID_SCRIPT,
    LSTM_DIR / "lstm.py",
    LITEMV_DIR / "litemv.py",
    RF_DIR / "random_forest_config.py",
    RF_DIR / "random_forest_preprocessing.py",
    RF_DIR / "random_forest_temporal.py",
    RF_DIR / "random_forest_training.py",
    RF_DIR / "random_forest_utils.py",
)


def validate_sources() -> None:
    missing = [path for path in REQUIRED_SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The final hybrid package cannot run because required source "
            "files are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )


def lstm_required_artifacts(dataset_name: str) -> tuple[Path, ...]:
    root = PART3_BASE_MODEL_ROOT / dataset_name / "LSTM"
    return (
        root / "lstm_model.keras",
        root / "selected_features.csv",
        root / "best_hyperparameters.json",
        root / "decision_rule.json",
        root / "split_manifest.csv",
        root / "training_strategy.json",
    )


def litemv_required_artifacts(dataset_name: str) -> tuple[Path, ...]:
    root = PART3_BASE_MODEL_ROOT / dataset_name / "LITEMV"
    return (
        root / "litemv.keras",
        root / "selected_features.json",
        root / "shared_hyperparameters.json",
        root / "decision_rule.json",
    )


def _missing(paths: tuple[Path, ...]) -> list[Path]:
    return [path for path in paths if not path.is_file()]


def _threshold(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    value = data.get("threshold")
    return None if value is None else float(value)


def artifact_status(dataset_names: list[str]) -> dict:
    result = {}
    for dataset_name in dataset_names:
        lstm_paths = lstm_required_artifacts(dataset_name)
        litemv_paths = litemv_required_artifacts(dataset_name)
        lstm_missing = _missing(lstm_paths)
        litemv_missing = _missing(litemv_paths)
        result[dataset_name] = {
            "lstm_ready": not lstm_missing,
            "litemv_ready": not litemv_missing,
            "lstm_missing": [str(path) for path in lstm_missing],
            "litemv_missing": [str(path) for path in litemv_missing],
            "lstm_threshold": _threshold(
                PART3_BASE_MODEL_ROOT
                / dataset_name
                / "LSTM"
                / "decision_rule.json"
            ),
            "litemv_threshold": _threshold(
                PART3_BASE_MODEL_ROOT
                / dataset_name
                / "LITEMV"
                / "decision_rule.json"
            ),
        }
    return result


def print_status(status: dict) -> None:
    print("\nFROZEN PART-3 BASE-MODEL READINESS")
    print("=" * 100)
    for dataset_name, info in status.items():
        print(
            f"{dataset_name}: "
            f"LSTM={'READY' if info['lstm_ready'] else 'MISSING'} | "
            f"LITEMV={'READY' if info['litemv_ready'] else 'MISSING'}"
        )
        if info["lstm_threshold"] is not None:
            print(f"  frozen LSTM threshold:   {info['lstm_threshold']:.12f}")
        if info["litemv_threshold"] is not None:
            print(f"  frozen LITEMV threshold: {info['litemv_threshold']:.12f}")
        for path in info["lstm_missing"]:
            print(f"  missing LSTM:   {path}")
        for path in info["litemv_missing"]:
            print(f"  missing LITEMV: {path}")


def validate_frozen_snapshot(target_datasets: list[str]) -> None:
    status = artifact_status(target_datasets)
    print_status(status)
    if any(
        not info["lstm_ready"] or not info["litemv_ready"]
        for info in status.values()
    ):
        raise RuntimeError(
            "Frozen Part-3 base-model snapshot is incomplete. Restore the "
            "submission artifacts under artifacts/hybrid_pipeline/base_models/. "
            "Do not retrain or overwrite Part-2 models to repair this."
        )

    # The manifest is provenance/audit metadata. Required files above are the
    # execution boundary, but a missing manifest should still be visible.
    if not SNAPSHOT_MANIFEST.is_file():
        print(
            "\nWARNING: snapshot provenance manifest is missing: "
            f"{SNAPSHOT_MANIFEST}"
        )


def run_command(command: list[str], cwd: Path, label: str) -> None:
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)
    print("Working directory:", cwd)
    print("Command:", " ".join(command))
    print()
    subprocess.run(command, cwd=str(cwd), check=True)


def run_final_hybrid(dataset: str | None) -> None:
    command = [sys.executable, str(HYBRID_SCRIPT)]
    if dataset is not None:
        command.extend(["--dataset", dataset])
    run_command(
        command,
        cwd=HYBRID_DIR,
        label="RUNNING FINAL FROZEN PART-3 LSTM -> LITEMV -> RF HYBRID",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the final fixed Part-3 LSTM -> LITEMV -> RF hybrid using "
            "the frozen report-facing base-model snapshot."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Run one benchmark dataset. Default: dataset2 and dataset3.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source files and frozen Part-3 base-model readiness.",
    )
    parser.add_argument(
        "--retrain-base-models",
        action="store_true",
        help=(
            "Disabled for the final submission. Part-3 uses a frozen fitted "
            "snapshot; fresh retraining belongs to Part 2 and must not replace it."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_sources()

    if args.retrain_base_models:
        raise RuntimeError(
            "--retrain-base-models is intentionally disabled in the final "
            "Part-3 runner. The report-facing cascade uses a frozen fitted "
            "LSTM/LITEMV snapshot stored separately from Part-2 artifacts."
        )

    target_datasets = [args.dataset] if args.dataset else list(DATASETS)

    print("=" * 100)
    print("FINAL HYBRID PIPELINE — FROZEN PART-3 SNAPSHOT")
    print("=" * 100)
    print(f"Python:       {sys.executable}")
    print(f"Project root: {PROJECT_ROOT}")
    print("Architecture: LSTM -> LITEMV -> RF")
    print("Base models:  artifacts/hybrid_pipeline/base_models/")
    print("Datasets:     " + ", ".join(target_datasets))

    validate_frozen_snapshot(target_datasets)
    if args.check_only:
        return

    run_final_hybrid(args.dataset)

    print("\n" + "=" * 100)
    print("FINAL HYBRID PIPELINE COMPLETE")
    print("=" * 100)
    print(f"Results:   {HYBRID_RESULT_ROOT}")
    print(f"Artifacts: {HYBRID_ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()
