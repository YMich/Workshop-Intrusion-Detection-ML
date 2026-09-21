# Preprocessing and Normalization

This directory contains the shared numerical preprocessing implementation used
throughout the project.

## Directory contents

```text
src/preprocessing/
├── README.md
├── __init__.py
├── preprocessing.py
├── preprocessing_config.py
└── run_preprocessing.py
```

## Core preprocessing

`DatasetPreprocessor` implements the numerical transformation used by the
models:

```text
non-finite values
      ↓
NaN
      ↓
median imputation
      ↓
signed log1p
      ↓
RobustScaler (median / IQR)
```

The signed-log transformation is:

```text
sign(x) * ln(1 + |x|)
```

Metadata and bookkeeping fields such as `Flow ID`, endpoint information,
`Timestamp`, `SourceFile`, and `Label` are retained by this module so later
stages can use them for grouping, source-aware splitting, temporal ordering,
sequence construction, or evaluation. Downstream feature/model code controls
which fields are actually predictive inputs.

## Two valid usage modes

This module is used in two different ways.

### 1. Final split-aware model preprocessing

This is the preprocessing path used for the final reported model evaluation.

The final model pipeline first establishes its source-aware split and then fits
the preprocessor only on the permitted training reference data:

```text
data/ingested/
      ↓
source-aware split
      ↓
TRAIN
      ↓
DatasetPreprocessor.fit(TRAIN)
      ↓
frozen medians + signed log1p + RobustScaler
      ↓
transform TRAIN / VALIDATION / TEST
```

For the supervised LSTM and LITEMV pipelines, preprocessing is fitted on their
TRAIN split.

For the one-class Autoencoder and Isolation Forest pipelines, preprocessing is
fitted on **benign TRAIN only**.

Validation, TEST, and cross-dataset target data are transformed using the
already-fitted source/training preprocessing parameters. They are not used to
fit those parameters.

### 2. Standalone Milestone-2 preprocessing snapshots

`run_preprocessing.py` is retained to reproduce the earlier standalone
Milestone-2 preprocessing stage.

It independently fits one `DatasetPreprocessor` to each complete Dataset 1,
Dataset 2, and Dataset 3 CSV and writes:

```text
data/preprocessed/
├── dataset1_preprocessed.csv
├── dataset2_preprocessed.csv
└── dataset3_preprocessed.csv
```

with corresponding preprocessing artifacts under:

```text
artifacts/preprocessing/
```

This standalone full-dataset operation is useful for reproducing the
Milestone-2 preprocessing snapshots. It is **not** the preprocessing procedure
used for final held-out model evaluation.

## Do I need to run this folder before the final models?

**No.**

If the submission already contains the finalized files under:

```text
data/ingested/
```

the final model runners import `DatasetPreprocessor` directly and fit it at the
correct training scope themselves.

You should not run `run_preprocessing.py` as a prerequisite to the final model
training/evaluation pipeline.

## Running the standalone Milestone-2 snapshot stage

From the project root:

```bash
python src/preprocessing/run_preprocessing.py
```

The runner prints a warning explaining that it is reproducing the standalone
full-dataset Milestone-2 snapshots rather than the final split-aware evaluation
pipeline.

## Files

### `preprocessing.py`

Contains:

- `signed_log1p`
- schema/header validation helpers
- timestamp parsing
- non-finite-value cleanup
- numeric conversion
- `DatasetPreprocessor`
- `preprocess_all_datasets`

`DatasetPreprocessor.fit(df)` learns the median-imputation values and
`RobustScaler` parameters from the dataframe supplied by the caller.

`DatasetPreprocessor.transform(df)` applies those frozen parameters without
refitting.

### `preprocessing_config.py`

Defines portable project-relative paths, metadata fields, and the common
RobustScaler settings.

No machine-specific project path is required.

### `run_preprocessing.py`

Reproduces the standalone Milestone-2 full-dataset preprocessing snapshots.
It is retained for stage-level reproducibility and is not used by the final
model evaluation runners.

## Reproducibility and leakage boundary

The numerical preprocessing implementation is shared, but the **fit scope is
controlled by the caller**.

For final evaluation:

- LSTM/LITEMV: fit on TRAIN only.
- Autoencoder/Isolation Forest: fit on benign TRAIN only.
- Validation and TEST: transform only.
- Cross-dataset target data: transform only using source-fitted parameters.

This keeps the final evaluation preprocessing isolated from held-out data.
