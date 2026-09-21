# Part 2 — Sample-Level Forensic Error Analysis

This directory reproduces the Dataset-2 / Dataset-3 sample-level forensic analysis used for Section 2.1 of the report.

It is an **analysis-only** stage. It does not train models, tune hyperparameters, change decision thresholds, or modify model artifacts.

## Directory contents

```text
src/part2_analysis/
├── README.md
├── build_forensic_error_dataset.py
└── plot_forensic_report_figures.py
```

## Evaluation contract

The analysis preserves each baseline model's leakage-safe evaluation protocol:

- **Random Forest** — full source-aware outer-fold out-of-fold (OOF) predictions generated with the frozen shared baseline RF configuration.
- **LITEMV / Autoencoder / LSTM** — predictions from the persisted shared source-aware held-out TEST split.

For cross-model row-level overlap only, the already-generated RF OOF ledger is restricted to the same held-out row indices used by LITEMV, Autoencoder, and LSTM. This restricted RF view is not used for the official RF confusion matrix, metrics, source-level error analysis, or feature-error comparisons.

## Prerequisites

This folder is **not** the first stage of the pipeline. It consumes baseline prediction files and artifacts created by the fixed baseline model implementations.

Expected prediction files include:

```text
results/models_baseline/RandomForest/<dataset>/oof_predictions.csv
results/models_baseline/LITEMV/<dataset>/test_predictions.csv
results/models_baseline/AE/<dataset>/test_predictions.csv
results/models_baseline/LSTM/<dataset>/test_predictions.csv
```

It also uses the finalized ingested datasets and supporting split/feature artifacts under:

```text
data/ingested/
artifacts/splits/
artifacts/feature_selection/
artifacts/models_baseline/
```

If these baseline artifacts/results are already included in the submission, no model training is required before running this analysis.

## Step 1 — Build the frozen forensic snapshot

From the project root:

```bash
python src/part2_analysis/build_forensic_error_dataset.py
```

By default the snapshot is written to:

```text
results/forensic_analysis/final_baseline_v1/
```

The script validates the prediction provenance, expected feature schema, row coverage, labels, thresholds, and evaluation protocol before exporting the forensic tables.

To analyze only one dataset:

```bash
python src/part2_analysis/build_forensic_error_dataset.py --dataset dataset2
```

or:

```bash
python src/part2_analysis/build_forensic_error_dataset.py --dataset dataset3
```

If a snapshot with the same name already exists, use a new `--snapshot-name`. `--force` should be used only when intentional replacement is required.

## Step 2 — Generate report figures

After the snapshot exists, run:

```bash
python src/part2_analysis/plot_forensic_report_figures.py
```

The figures are written by default to:

```text
results/forensic_analysis/report_plots/final_baseline_v1/
```

The plotting script produces PNG and PDF report figures together with CSV exports of the plotted data.

The generated outputs include:

- combined D2/D3 confusion matrices;
- SourceFile-level FP/FN concentration figures;
- feature characteristics associated with FP/FN errors;
- cross-model error-overlap analysis.

## Feature-analysis scope

The forensic feature comparisons use the fixed baseline Step-3 57-feature schema in interpretable/raw units. The analysis does not introduce optimized-model feature pruning, temporal feature engineering, augmentation, or model-specific search logic.

## Reproducibility boundary

This folder reads existing model outputs only. It deliberately does not:

- retrain any model;
- perform model hyperparameter search or sensitivity analysis;
- recalibrate model decision thresholds;
- alter train/validation/test assignments;
- use optimized-model predictions in the baseline forensic snapshot.

The submitted baseline model implementations use frozen shared configurations. The Random Forest OOF predictions therefore correspond to source-aware outer-fold evaluation with that fixed configuration rather than inner hyperparameter selection.
