from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from random_forest_config import BENCHMARK_DATASETS, PROJECT_ROOT
from rf_temporal_optimization import (
    evaluate_transfer,
    load_o3_final_params,
    load_temporal_base,
    variant_features,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "models_optimized"
    / "RF"
    / "temporal_reproduction_check"
)

EXPECTED_FEATURE_COUNT = 32
EXPECTED_CORE_PARAMS = {
    "n_estimators": 256,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
}


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    features = variant_features("RROLL5")
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"RROLL5 schema has {len(features)} features; expected {EXPECTED_FEATURE_COUNT}."
        )

    required_temporal = {
        "InterFlow Delta Prev5 Mean",
        "InterFlow Delta Prev5 Std",
        "InterFlow Delta Prev5 CV",
    }
    missing = sorted(required_temporal - set(features))
    if missing:
        raise RuntimeError(f"Exact RROLL5 temporal schema is missing: {missing}")

    params = load_o3_final_params()

    print("=" * 100)
    print("RROLL5 EXACT-REPRODUCTION CROSS-DATASET CHECK")
    print("=" * 100)
    print(f"Feature count: {len(features)}")
    print("Restored behavior: rolling std uses min_periods=1")
    print("Restored feature: InterFlow Delta Prev5 CV")
    print("Cross-dataset RF hyperparameters loaded from O3 final shared configuration:")
    for key in [
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
    ]:
        print(f"  {key}={params.get(key)!r}")

    mismatches = {
        key: (params.get(key), value)
        for key, value in EXPECTED_CORE_PARAMS.items()
        if params.get(key) != value
    }
    if mismatches:
        print("\nWARNING: O3 final parameters differ from the configuration used in the earlier endpoint-aware result:")
        for key, (actual, expected) in mismatches.items():
            print(f"  {key}: loaded={actual!r}, earlier={expected!r}")
        print("The run will still use the actual O3 final shared parameters, so interpret numerical differences accordingly.")

    frames = {name: load_temporal_base(name) for name in BENCHMARK_DATASETS}
    rows, preds = evaluate_transfer(
        temporal_frames=frames,
        variant="RROLL5",
        final_params=params,
        min_history=None,
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_ROOT / "RROLL5_exact_cross_dataset_summary.csv", index=False)
    preds.to_csv(RESULT_ROOT / "RROLL5_exact_cross_dataset_predictions.csv", index=False)
    pd.DataFrame({"Feature": features}).to_csv(
        RESULT_ROOT / "RROLL5_exact_32_features.csv", index=False
    )
    with (RESULT_ROOT / "RROLL5_exact_reproduction_definition.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "variant": "RROLL5",
                "feature_count": len(features),
                "rolling_std_min_periods": 1,
                "interflow_delta_cv_restored": True,
                "sequence_key": ["SourceFile", "Src IP", "Dst IP"],
                "raw_ip_model_inputs": False,
                "raw_timestamp_model_input": False,
                "hyperparameters": params,
            },
            handle,
            indent=2,
        )

    print("\nCross-dataset summary:")
    cols = [
        "Train_Dataset",
        "Test_Dataset",
        "Feature_Count",
        "precision",
        "recall_tpr",
        "fpr",
        "f1",
        "roc_auc",
        "pr_auc",
        "fp",
        "fn",
        "tp",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\nSaved to: {RESULT_ROOT}")
    print("\nHistorical values to compare against (approximate):")
    print("  D2 -> D3: recall=0.4557, F1=0.615, PR-AUC=0.727, FPR=0.00151")
    print("  D3 -> D2: recall=0.4588, F1=0.617, PR-AUC=0.586, FPR=0.000545")


if __name__ == "__main__":
    main()
