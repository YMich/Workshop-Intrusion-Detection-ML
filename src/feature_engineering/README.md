# Feature Engineering and Selection

This directory contains the shared feature-selection and feature-engineering logic used by the final intrusion-detection models.

## Directory Contents

```text
src/feature_engineering/
├── README.md
├── __init__.py
├── feature_config.py
├── feature_selection.py
└── run_feature_selection.py
```

## Purpose

The final submission begins from the finalized CICFlowMeter datasets stored under:

```text
data/ingested/
```

The feature-engineering stage converts these ingested datasets into the common predictive feature representation used by the model pipelines.

The shared stage is intentionally model-agnostic. It applies the common feature-selection and engineering rules before any model-specific pruning, temporal representation, sequence construction, or anomaly-scoring logic is applied.

The common stage produces the ordered 57-feature schema used as the starting point for the final model implementations.

## Files

### `feature_config.py`

Defines the shared configuration for the feature-engineering stage, including project paths, dataset locations, metadata fields, excluded fields, and expected feature-selection settings.

### `feature_selection.py`

Implements the shared feature-selection and feature-engineering logic.

Its responsibilities include:

- preserving the canonical schema expected by the downstream models;
- excluding non-predictive metadata from the model feature set;
- applying the feature-selection decisions established during the project;
- maintaining a consistent feature order across datasets;
- returning the selected predictive features together with any metadata required by downstream processing.

Fields such as IP addresses, ports, timestamps, `Flow ID`, and `SourceFile` may still be retained in the working dataframe when required for grouping, sequencing, temporal ordering, provenance, or evaluation, but they are not used directly as predictive features by this shared feature-selection stage.

### `run_feature_selection.py`

Provides a standalone entry point for reproducing the shared feature-selection stage and saving its outputs for inspection.

Run it from the project root with:

```bash
python src/feature_engineering/run_feature_selection.py
```

Generated feature-selection artifacts are written under the project's feature-selection artifact directory and include the selected feature list and feature-drop information produced by the implementation.

## Use by the Final Models

The final model pipelines import and call the functions in `feature_selection.py` directly.

Therefore:

> **You do not need to run `run_feature_selection.py` before training or evaluating the final models.**

The standalone runner is included for reproducibility, inspection, and independent verification of the common feature-selection stage.

The normal model pipeline is conceptually:

```text
data/ingested/
      |
      v
training-fitted preprocessing
      |
      v
shared feature selection / engineering
      |
      v
common 57-feature representation
      |
      v
model-specific processing
      |
      +--> Random Forest
      +--> LSTM
      +--> LITEMV
      +--> Autoencoder
```

Some final models apply additional model-specific hardening or temporal engineering after this shared stage. Those operations are implemented inside the corresponding model directories and are not part of this shared module.

## Reproducibility and Leakage Controls

The feature-engineering code is designed to be used only after the train/validation/test boundaries required by the relevant model pipeline have been established.

Data-dependent preprocessing parameters are fitted on training data only and then applied unchanged to validation and test data.

Metadata required for source-aware splitting, temporal ordering, or evaluation is kept separate from the predictive feature schema.

The shared feature list and ordering are checked so that downstream models do not silently train or evaluate with an inconsistent schema.

## Submission Scope

The final submission starts from the finalized ingested datasets under:

```text
data/ingested/
```

Raw-PCAP extraction and CICFlowMeter execution are outside the execution scope of the final submission package.

This directory is required by the final model implementations and must remain at:

```text
src/feature_engineering/
```

within the submitted project structure.
