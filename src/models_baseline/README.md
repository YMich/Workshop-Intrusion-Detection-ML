# Baseline Models

This directory contains the fixed baseline implementations used for comparison with the optimized/final models.

## Important submission rule

The submitted baseline code contains **no model hyperparameter search or optimization stage**. The model configurations were frozen before this submission package was prepared and are used directly for Dataset 2 and Dataset 3.

Validation-only decision-threshold calibration, early stopping, class-weight calculation, and source-aware split construction remain part of normal model training and evaluation. These operations are not model-hyperparameter searches.

## Frozen shared configurations

| Model | Fixed shared baseline configuration |
| --- | --- |
| Random Forest | `n_estimators=128`, `max_depth=10`, `min_samples_leaf=1`, `max_features=0.5`, `bootstrap=True` |
| LITEMV | `sequence_length=40`, `n_filters=32`, `kernel_size=40`, `conv_stride=1`, `learning_rate=0.001`, `batch_size=64`, train/eval stride `5/1` |
| Autoencoder | encoder `64-32`, bottleneck `8`, `learning_rate=0.002`, `loss=Huber`, `L2=1e-5`, noise `0.02`, `batch_size=256`, score `Top-5 MSE` |
| LSTM | `sequence_length=20`, LSTM units `32`, dense units `32`, `dropout=0.2`, `learning_rate=0.001`, `batch_size=128`, `L2=1e-5` |
| Isolation Forest | `n_estimators=200`, `max_samples=512`, `max_features=1.0`, `bootstrap=False`, `contamination=auto` |

## Directory layout

```text
models_baseline/
├── README.md
├── AE/
├── IF/
├── LITEMV/
├── LSTM/
└── RF/
```

Each model directory contains its training entry point, evaluation code, and required local helpers. `__pycache__`, `.pyc`, sensitivity scripts, optimization scripts, and tuning-only support files are intentionally excluded.

## Data and shared dependencies

The baseline implementations consume the finalized datasets under:

```text
data/ingested/
```

They also rely on the repository's shared preprocessing and feature-engineering modules where imported by the individual model implementations.

## Reproducibility boundary

The baseline code is intended to reproduce the reported baseline models using the frozen configurations above. It does not attempt to rediscover, rank, compare, or optimize model hyperparameters during execution.

## Running the baselines

From the project root, run a model with:

```bash
python src/models_baseline/RF/run_random_forest.py
python src/models_baseline/LSTM/run_lstm.py
python src/models_baseline/LITEMV/run_litemv.py
python src/models_baseline/AE/run_autoencoder.py
python src/models_baseline/IF/run_isolation_forest.py
```

Each command trains with the frozen configuration and then evaluates the corresponding held-out/OOF outputs. Use the model-specific `--help` option for training-only or evaluation-only modes.

The same frozen values are also recorded in `baseline_configurations.json`.

### Random Forest evaluation

The submitted Random Forest code does not run an inner hyperparameter-selection loop. It applies the frozen shared RF configuration directly inside each source-aware outer fold, producing out-of-fold predictions while keeping each fold's imputation and class weights training-only.
