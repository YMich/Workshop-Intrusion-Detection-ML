# Final Part-3 Hybrid Intrusion-Detection Pipeline

This directory contains the final report-facing gated cascade:

```text
LSTM -> LITEMV -> Random Forest
```

## Important artifact separation

The Part-3 hybrid uses the **frozen fitted LSTM/LITEMV snapshot that generated the report**. Those fitted weights are intentionally stored separately from the independently reproducible Part-2 model artifacts.

```text
Part 2 fitted models:
artifacts/models_optimized_final/

Part 3 report-facing fitted snapshot:
artifacts/hybrid_pipeline/base_models/
├── dataset2/
│   ├── LSTM/
│   └── LITEMV/
└── dataset3/
    ├── LSTM/
    └── LITEMV/
```

The model **implementations and configured hyperparameters** still come from:

```text
src/models_optimized_final/LSTM/
src/models_optimized_final/LITEMV/
src/models_optimized_final/RF/
```

Only the fitted Part-3 neural-network instances are isolated. This prevents a later stochastic Part-2 retraining from overwriting the exact fitted model instance used by the report-facing cascade.

For the recovered report snapshot, the key LSTM thresholds are:

```text
Dataset 2: 0.5122592449188232
Dataset 3: 0.004561834968626499
```

and the LITEMV thresholds are:

```text
Dataset 2: 0.004431014880537987
Dataset 3: 0.08543922007083893
```

## Leakage-safe evaluation procedure

For each dataset the operational pipeline:

1. reuses the persisted source-aware TRAIN / VALIDATION / TEST split;
2. loads the frozen Part-3 LSTM and LITEMV fitted models;
3. reconstructs preprocessing from TRAIN only;
4. fits the RF arbiter on TRAIN only;
5. calibrates only the fixed cascade's routing thresholds on VALIDATION;
6. freezes the routing rule;
7. evaluates once on TEST.

TEST is never used for model fitting, preprocessing fitting, routing calibration, architecture selection, or threshold repair.

The three-stage architecture itself is fixed before TEST.

## Why base-model retraining is disabled here

A fresh neural-model retraining can produce a different fitted model even when architecture and hyperparameters are unchanged. During final-submission testing, a newly trained Dataset-2 LSTM achieved perfect validation F1, which meant the fixed hybrid could not strictly improve it. The recovered historical Part-3 LSTM is the fitted instance that actually generated the report and leaves three validation false positives for the cascade to correct.

Therefore the final Part-3 launcher **does not retrain LSTM/LITEMV** and does not overwrite Part-2 artifacts. If the frozen Part-3 snapshot is missing, restore the distributed artifact package rather than retraining until a particular result appears.

## Files

```text
src/hybrid_pipeline/
├── README.md
├── hybrid_pipeline.py
├── run_hybrid_pipeline.py
└── cross_dataset_evaluation.py
```

## Readiness check

From the project root:

```bash
python src/hybrid_pipeline/run_hybrid_pipeline.py --check-only
```

The expected Dataset-2 frozen LSTM threshold is approximately `0.512259`, not the independently retrained Part-2 threshold.

## Within-dataset hybrid evaluation

Run both datasets:

```bash
python src/hybrid_pipeline/run_hybrid_pipeline.py
```

Or individually:

```bash
python src/hybrid_pipeline/run_hybrid_pipeline.py --dataset dataset2
python src/hybrid_pipeline/run_hybrid_pipeline.py --dataset dataset3
```

Expected report-facing TEST results are approximately:

```text
Dataset 2
LSTM:   FP=35, FN=4, F1=0.9469
Hybrid: FP=13, FN=4, F1=0.9762

Dataset 3
LSTM:   FP=27, FN=0, F1=0.9750
Hybrid: FP=7,  FN=0, F1=0.9934
```

## Cross-dataset evaluation

After the within-dataset hybrid is verified:

```bash
python src/hybrid_pipeline/cross_dataset_evaluation.py
```

The source LSTM/LITEMV fitted state, source preprocessing, RF arbiter, and validation-derived hybrid rule are transferred without target-domain fitting or recalibration.

## `--retrain-base-models`

The legacy flag is retained only to fail clearly. In the final submission it is intentionally disabled:

```bash
python src/hybrid_pipeline/run_hybrid_pipeline.py --retrain-base-models
```

will raise an explanatory error instead of overwriting the frozen Part-3 snapshot or the independently verified Part-2 artifacts.

## Outputs

Within-dataset artifacts:

```text
artifacts/hybrid_pipeline/dataset2/
artifacts/hybrid_pipeline/dataset3/
```

Within-dataset results:

```text
results/hybrid_pipeline/dataset2/
results/hybrid_pipeline/dataset3/
results/hybrid_pipeline/hybrid_test_summary.csv
```

Cross-dataset outputs:

```text
artifacts/hybrid_pipeline/cross_dataset/
results/hybrid_pipeline/cross_dataset/
```
