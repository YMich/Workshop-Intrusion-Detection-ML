# Final Optimized Models

This directory contains the final accepted Dataset-2 / Dataset-3 model implementations used by the project.

## Submission rule

All model hyperparameters in this directory are **frozen**. The final source code does not perform grid search, random search, sensitivity sweeps, or model-hyperparameter selection. Dataset-specific quantities that must be learned from data—training-fitted preprocessing, class weights, early-stopping epoch, and validation-only decision thresholds—remain dataset-specific.

## Models

- `RF/` — final endpoint-aware Random Forest with the fixed RROLL5 representation.
- `LSTM/` — final 43-feature sequence model with malicious-only temporal/rate augmentation.
- `LITEMV/` — final 43-feature LITEMV sequence model.
- `AE/` — final 43-feature benign-only autoencoder.
- `IF/` — final 43-feature benign-only Isolation Forest retained for supplementary evaluation.

The final hybrid uses LSTM, LITEMV, and Random Forest. AE remains a separately evaluated final model; IF is retained as supplementary one-class evidence.

## Required project structure

These models expect the repository to contain:

```text
data/ingested/
src/preprocessing/
src/feature_engineering/
src/models_optimized_final/
```

The ingested datasets are the execution starting point for the submitted ML pipeline. Raw-PCAP extraction and ingestion do not need to be rerun when `data/ingested/*.csv` is present.

## Typical commands

Run individual models from the project root:

```bash
python src/models_optimized_final/RF/run_random_forest.py
python src/models_optimized_final/LSTM/run_lstm.py
python src/models_optimized_final/LITEMV/run_litemv.py
python src/models_optimized_final/AE/run_autoencoder.py
python src/models_optimized_final/IF/run_isolation_forest.py
```

Each model README documents its own artifacts, evaluation entry points, and cross-dataset support.
