# Workshop in Intrusion Detection using ML

## AI-Driven Detection of Cobalt Strike HTTPS C2 from Network-Flow Metadata

This repository contains the **source code** for the final implementation, optimization, evaluation, hybrid ensemble, and local-LLM arbitration stages of the Workshop in Intrusion Detection using Machine Learning project.

The project detects **Cobalt Strike Beacon command-and-control traffic over HTTPS** using **flow-level metadata only**. Payload contents are not used as predictive input.

> **Important:** this repository intentionally contains baseline code, retained optimization stages, final models, and appendix experiments.  
> The authoritative final Dataset-2 / Dataset-3 model implementations are in:
>
> `src/models_optimized_final/`

Large datasets, trained model artifacts, frozen results, and the final report are distributed separately through the accompanying **Google Drive package**.

---

# 1. Repository / Google Drive Split

The project is distributed in two parts.

## GitHub — source code

GitHub contains:

```text
WorkshopID/
├── README.md
├── .gitignore
├── src/
└── data/
    ├── README.md
    ├── raw/
    │   └── .gitkeep
    └── ingested/
        └── .gitkeep
```

The GitHub repository contains the code and the expected data-directory structure, but **does not contain the large datasets, trained artifacts, or frozen result archives**.

## Google Drive — large/frozen project files

Google Drive contains the large project material:

```text
WorkshopID_GoogleDrive/
├── data/
│   ├── raw/                     # Optional unless reproducing PCAP extraction
│   └── ingested/
│       ├── dataset1_ingested.csv
│       ├── dataset2_ingested.csv
│       └── dataset3_ingested.csv
├── model_artifacts/
├── results/
└── report/
    └── Milestone 3 Final Report ... .docx
```

For a complete local reproduction, copy the Google Drive directories into the root of the cloned repository so the local project becomes:

```text
WorkshopID/
├── README.md
├── .gitignore
├── src/
├── data/
│   ├── raw/
│   └── ingested/
│       ├── dataset1_ingested.csv
│       ├── dataset2_ingested.csv
│       └── dataset3_ingested.csv
├── model_artifacts/
└── results/
```

The final report may remain outside the repository; it does not need to be copied into the code tree.

---

# 2. Start Here

For the final Dataset-2 / Dataset-3 implementation, the authoritative model code is:

```text
src/models_optimized_final/
```

The principal source directories are:

```text
src/
├── preprocessing/                  Shared preprocessing/data utilities
├── models_baseline/                Baseline models used for comparison
├── models_optimized_final/         AUTHORITATIVE FINAL MODELS
│   ├── LSTM/
│   ├── LITEMV/
│   ├── AE/
│   ├── IF/
│   └── RF/
├── optimization_history/           Retained optimization-stage implementations
├── hybrid_pipeline/                Part 3 LSTM → LITEMV → RF gated cascade
├── part4_llm/                      Part 4 local-Qwen arbitration
└── dataset1_appendix_experiments/  Dataset-1 appendix-only experiments
```

If the goal is simply to inspect the final submitted models, begin with:

```text
src/models_optimized_final/
```

Do **not** interpret `models_baseline/`, `optimization_history/`, or `dataset1_appendix_experiments/` as alternative final implementations.

---

# 3. Project Scope

The detection target is:

```text
Cobalt Strike Beacon C2 over HTTPS
```

MITRE ATT&CK mapping:

```text
T1071.001 — Application Layer Protocol: Web Protocols
```

The project intentionally restricts detection to information derived from network-flow metadata.

Examples include:

- flow timing;
- packet counts;
- packet lengths;
- byte counts;
- traffic rates;
- forward/backward behavior;
- TCP flags;
- TCP-window-related features;
- causal temporal context from preceding flows.

Direct identifiers are not used as ordinary predictive input. Metadata such as `SourceFile`, IP addresses, and timestamps may be retained internally where required for:

- source-aware splitting;
- chronological ordering;
- sequence construction;
- endpoint-history construction;
- forensic attribution.

They are not supplied to the final classifier as ordinary identifier features.

---

# 4. Datasets

The main final evaluation uses:

```text
Dataset 2
Dataset 3
```

Dataset 1 is retained as an **appendix/generalization dataset**.

Dataset 1 is not used to choose the final Dataset-2 / Dataset-3:

- architecture;
- feature set;
- model hyperparameters;
- optimization-acceptance decisions.

Expected local ingested paths are:

```text
data/ingested/dataset1_ingested.csv
data/ingested/dataset2_ingested.csv
data/ingested/dataset3_ingested.csv
```

Dataset 1 is required only for reproducing the appendix evaluation.

The ingested files are intentionally excluded from GitHub and must be copied from the Google Drive package.

---

# 5. Data Extraction and Ingestion

Network traffic originates from PCAP captures and is converted to flow-level records using **CICFlowMeter**.

The ingestion stage standardizes extracted flows into a common schema.

`SourceFile` is retained as metadata so that flows from the same capture can later be grouped correctly during splitting, temporal sequencing, and forensic evaluation.

---

# 6. Leakage Prevention

The project avoids generic random row-level splitting.

## LSTM / LITEMV / Autoencoder / Isolation Forest

The shared source-aware split is approximately:

```text
70% training
15% validation
15% test
```

Whole `SourceFile` captures are kept in one partition whenever possible.

If one source is too large for a reasonable grouped split, only that oversized source may be divided into non-overlapping chronological segments.

The partition is established **before fitting any data-dependent preprocessing**.

## Random Forest

Random Forest uses grouped cross-validation with `SourceFile` as the grouping variable.

The main within-dataset evaluation uses:

```text
5 outer folds
3 inner folds
```

The inner folds are used for model-selection decisions. The outer folds produce the report-facing **out-of-fold (OOF)** predictions.

This distinction matters:

> The final fitted RF deployment artifact is not the source of the report's within-dataset OOF metrics.

The frozen OOF prediction ledgers are the authoritative evidence for those values.

---

# 7. Removed Leakage / Metadata Fields

Direct or environment-identifying fields were removed from ordinary predictive input.

Examples include:

```text
Flow ID
Src IP
Dst IP
Src Port
Dst Port
Protocol
Timestamp
ICMP Code
ICMP Type
SourceType
SourceDataset
```

Constant and redundant columns identified during earlier analysis were also removed.

The exact final ordered model feature sets are stored with the fitted artifacts, typically through files such as:

```text
selected_features.csv
selected_features.json
```

Those saved schemas should be treated as authoritative.

---

# 8. Preprocessing

Data-dependent preprocessing is fitted only on the permitted training population.

## LSTM / LITEMV / Autoencoder

The numerical preprocessing pipeline uses:

1. invalid/non-finite value handling;
2. median imputation;
3. signed `log1p`;
4. robust scaling using training-derived median/IQR statistics.

Signed log transformation:

```text
sign(x) × log(1 + |x|)
```

This reduces the effect of extreme heavy-tailed flow values while preserving sign.

For the Autoencoder, benign-only training constraints are respected when fitting the one-class anomaly-detection pipeline.

## Random Forest

Random Forest uses training-fold median imputation and does not require neural-network-style scaling.

## Isolation Forest

Isolation Forest uses its saved fitted preprocessing state and selected-feature schema.

Do not refit preprocessing on test or transfer-target data.

---

# 9. Final Feature Representations

The final D2/D3 model logic is shared between datasets. There is no dataset-specific feature-selection branch.

Final representations:

```text
LSTM         43 hardened features + causal sequence context
LITEMV       43 hardened features + causal sequence context
Autoencoder  43 hardened features
IF           43 hardened features
RF           32-feature endpoint-aware causal RROLL5 representation
```

---

# 10. Final Models

## 10.1 LSTM

Authoritative code:

```text
src/models_optimized_final/LSTM/
```

Final representation:

```text
43 environment-hardened features
sequence length = 20
training stride = 5
validation/test stride = 1
```

Shared configured architecture:

```text
LSTM units          = 32
Dense units         = 32
Dropout             = 0.20
Recurrent dropout   = 0
L2                  = 1e-5
Learning rate       = 0.001
Batch size          = 128
Loss                = Binary Cross-Entropy
Maximum epochs      = 25
Early-stop patience = 4
```

The retained final optimization additionally performs **training-only malicious temporal/rate augmentation**.

Two augmented variants are generated per malicious `SourceFile`.

Validation and test data are never augmented.

The final operating threshold is calibrated separately from each dataset's validation predictions.

---

## 10.2 LITEMV

Authoritative code:

```text
src/models_optimized_final/LITEMV/
```

Final representation:

```text
43 environment-hardened features
sequence length = 40
training stride = 5
evaluation stride = 1
```

Shared configured architecture:

```text
Filters             = 32
Kernel size         = 40
Convolution stride  = 1
Activation          = ReLU
Learning rate       = 0.001
Batch size          = 64
Loss                = Binary Cross-Entropy
Maximum epochs      = 500
Early-stop patience = 40
```

The retained model uses the environment-hardening reduction from the earlier 57-feature representation to 43 features.

The decision threshold is calibrated on validation data separately for each dataset.

---

## 10.3 Autoencoder

Authoritative code:

```text
src/models_optimized_final/AE/
```

The Autoencoder is a benign-only one-class anomaly detector.

Final representation:

```text
43 environment-hardened features
```

Final configuration:

```text
Encoder             = 64 → 32
Bottleneck          = 4
Hidden activation   = ReLU
Output activation   = Linear
L2                  = 1e-5
Learning rate       = 0.002
Loss                = MSE
Batch size          = 256
Maximum epochs      = 20
Early-stop patience = 3
Denoising noise     = 0.0
```

Final anomaly scoring:

```text
Top-5 MSE
```

The score is the mean of the five largest per-feature squared reconstruction errors.

The decision threshold is derived from validation evidence, never from the test set.

---

## 10.4 Isolation Forest

Authoritative code:

```text
src/models_optimized_final/IF/
```

The final IF uses the hardened 43-feature representation.

The authoritative fitted model, preprocessor, selected features, and decision rule are stored in the separate Google Drive artifact package under:

```text
model_artifacts/models_optimized_final/IF/
```

---

## 10.5 Random Forest

Authoritative code:

```text
src/models_optimized_final/RF/
```

The final RF uses a **32-feature endpoint-aware causal RROLL5 representation**.

Flows are grouped by:

```text
(SourceFile, Src IP, Dst IP)
```

and ordered chronologically.

Temporal features use only the previous five flows from the same endpoint sequence.

The current flow is excluded from its own temporal history.

`SourceFile`, `Src IP`, and `Dst IP` are therefore used to build temporal groups, not as ordinary predictive features.

Final shared configuration:

```text
n_estimators       = 256
criterion          = gini
max_depth          = None
min_samples_split  = 2
min_samples_leaf   = 2
max_features       = sqrt
bootstrap          = True
threshold          = 0.50
```

Balanced class weights are computed from the applicable training partition/fold.

---

# 11. Shared Hyperparameters Across D2 and D3

The final implementation uses the same **configured model hyperparameters** for Dataset 2 and Dataset 3.

Shared examples include:

```text
architecture
sequence length
number of units / filters
dropout
learning rate
batch size
tree configuration
feature logic
```

Some quantities are intentionally fitted separately because they are learned properties of the training data rather than model-design hyperparameters.

These may differ by dataset:

```text
preprocessing medians
robust-scaling IQR values
class weights
actual early-stopping epoch
fitted model weights
validation-calibrated decision threshold
```

Random Forest keeps a fixed threshold of:

```text
0.50
```

Dataset-specific fitted quantities do not represent different model architectures.

---

# 12. Baseline Models

Directory:

```text
src/models_baseline/
```

These models are retained for:

- forensic error analysis;
- optimization comparisons;
- sensitivity analysis;
- before/after measurements;
- report traceability.

They are **not** the final optimized D2/D3 implementations.

---

# 13. Optimization History

Directory:

```text
src/optimization_history/
```

Optimization evidence is distributed across the GitHub code and Google Drive artifacts/results:

```text
src/optimization_history/
model_artifacts/optimization_history/
results/optimization_history/
results/optimization_runs/
```

The separation is intentional:

- `src/` — experiment implementation;
- `model_artifacts/` — fitted experimental state;
- `results/` — measured outputs and comparisons.

## Retained LSTM path

```text
Baseline
  ↓
O1 — 57 → 43 environment-hardening feature reduction
  ↓
O2 — malicious temporal/rate training augmentation
  ↓
Final LSTM
```

Prefix-aware/attention experiments are not part of the final model.

## Retained LITEMV path

```text
Baseline
  ↓
O1 — 57 → 43 environment-hardening feature reduction
  ↓
Final LITEMV
```

Later balancing/augmentation alternatives are not retained.

## Retained RF path

```text
B0 — 57-feature baseline
 ↓
O1 — 10-feature lean representation
 ↓
O2 — 12-feature refined representation
 ↓
O3 — tuned static RF
 ↓
O4 — endpoint-aware RROLL5 temporal representation
 ↓
Final RF
```

RROLL5-specific O5 retuning was evaluated but rejected; final RF remains O4.

## Retained Autoencoder path

```text
Baseline
 ↓
O1 — feature hardening
 ↓
O2 — bottleneck = 4
 ↓
O3 — MSE training loss
 ↓
O4 — reconstruction-score experiment
 ↓
O5 — no denoising + Top-5 MSE
 ↓
Final Autoencoder
```

O6/MAE alternatives are not final.

## Retained Isolation Forest path

```text
Baseline
 ↓
O1 — 57 → 43 feature hardening
 ↓
Final IF
```

---

# 14. RF FULL5 Verification

Frozen evidence:

```text
results/optimization_history/RF/O4_RROLL5/temporal_chain/
FULL5_same_population_control.csv
```

This experiment evaluates static RF and RROLL5 on the same restricted population of flows with at least five previous same-endpoint observations.

Dataset-3 control values:

```text
Static RF:
Recall = 38.70%
F1     = 0.541
FP     = 221

RROLL5:
Recall = 49.44%
F1     = 0.660
FP     = 17
```

This supports the conclusion that previous-flow behavior contributes useful information rather than the effect being explained only by history length.

---

# 15. RF OOF vs Deployment Artifact

## Within-dataset RF results

Report-facing RF within-dataset results are based on frozen **outer-fold OOF prediction ledgers**.

## Final fitted RF artifact

The fitted final RF artifact is intended for:

- deployment-style inference;
- cross-dataset transfer;
- downstream integration requiring one fitted model.

Therefore:

> Do not run the full-fit RF on its own training data and compare that result with the report's OOF metrics.

They are different evaluation protocols.

---

# 16. Cross-Dataset Evaluation

Strict directional transfer is evaluated as:

```text
Dataset 2 → Dataset 3
Dataset 3 → Dataset 2
```

For transfer:

- source data provide the model;
- source data provide fitted preprocessing;
- source data provide the threshold;
- target data are not used for training;
- target data are not used to fit preprocessing;
- target data are not used to calibrate thresholds;
- target labels are used only for final evaluation.

This is intentionally a no-target-adaptation protocol.

---

# 17. Part 3 — Hybrid Pipeline

Authoritative source:

```text
src/hybrid_pipeline/
```

Architecture:

```text
LSTM
  ↓
LITEMV
  ↓
Random Forest
```

This is a **gated cascade**, not majority voting.

Roles:

1. LSTM — primary detector.
2. LITEMV — additional confirmation on selected cases.
3. RF — conservative arbiter when required by the learned routing logic.

Held-out TEST results:

```text
Dataset 2:
LSTM F1   ≈ 0.9469
Hybrid F1 ≈ 0.9762
FP: 35 → 13
FN: 4 → 4

Dataset 3:
LSTM F1   ≈ 0.9750
Hybrid F1 ≈ 0.9934
FP: 27 → 7
FN: 0 → 0
```

The RF component used in Part 3 is a separate TRAIN-only fit of the same final 32-feature RROLL5 configuration.

Its standalone Part-3 metrics therefore do not need to equal the Part-2 OOF or full-transfer RF metrics.

---

# 18. Part 4 — Local LLM Arbitration

Authoritative source:

```text
src/part4_llm/
```

The LLM is a selective arbitrator, not a standalone IDS.

Final frozen configuration:

```text
Model             = qwen3:8b-q4_K_M
Prompt variant    = semantic_o3
Context variant   = behavioral_o1
Selection mode    = final-union
temperature       = 0.0
top_p             = 0.9
top_k             = 20
seed              = 42
context window    = 4096
max output tokens = 512
```

Execution used a 36-layer Qwen model with 18 requested transformer layers offloaded to GPU.

The LLM is not given:

```text
ground-truth label
dataset identity
SourceFile
capture filename
IP address
port
malware-family label
```

Final TEST escalation counts:

```text
Dataset 2: 25 / 14,609
Dataset 3: 32 / 15,076
Combined : 57 / 29,685
```

No LLM override passed the final safety gate.

Final metrics therefore remain:

```text
Dataset 2:
Hybrid F1 = 0.9762
Part-4 F1 = 0.9762

Dataset 3:
Hybrid F1 = 0.9934
Part-4 F1 = 0.9934
```

The report uses **frozen LLM outputs**. Rerunning Qwen is not required to reproduce the submitted report.

---

# 19. Part-4 Evidence Note

Authoritative frozen evidence is stored in the Google Drive results package under:

```text
results/part4_llm/lstm_litemv_rf_arbitration/FINAL_FROZEN_TEST/
```

Use the per-dataset directories:

```text
FINAL_FROZEN_TEST/dataset2/test/
FINAL_FROZEN_TEST/dataset3/test/
```

and:

```text
FINAL_FROZEN_TEST/part4_3_arbitration_performance.csv
```

The file:

```text
FINAL_FROZEN_TEST/final_part4_run_summary.json
```

contains only Dataset 3 in its internal `summaries` array and should not be used as the sole combined D2/D3 evidence source.

---

# 20. Dataset-1 Appendix

Source:

```text
src/dataset1_appendix_experiments/
```

Dataset 1 is used only for appendix/generalization evaluation.

It was not used to select final D2/D3:

- features;
- hyperparameters;
- architecture;
- optimization acceptance.

---

# 21. Frozen Evaluation Policy

Report-facing results are frozen.

Evaluation/reproduction should load the saved:

```text
model
preprocessor
selected-feature schema
persisted split assignment
decision threshold
```

where applicable.

Do **not**:

```text
retrain models
refit preprocessing
recalibrate thresholds
rebuild splits
adapt to transfer-target data
replace report-facing metrics
rerun the LLM and substitute new outputs for the frozen TEST evidence
```

Evaluation-only execution exists to reproduce the frozen inference path, not to define a new experiment.

---

# 22. Model Artifacts

Downloaded from Google Drive and placed locally under:

```text
model_artifacts/
```

Final fitted model artifacts are under:

```text
model_artifacts/models_optimized_final/
```

Typical contents include:

```text
.keras or .joblib model files
preprocessor.joblib
selected_features.csv
selected_features.json
decision_rule.json
best_hyperparameters.json
training metadata
split metadata
```

The exact files differ by architecture.

---

# 23. Frozen Results

Downloaded from Google Drive and placed locally under:

```text
results/
```

Important result families:

```text
results/models_optimized_final/
results/optimization_history/
results/optimization_runs/
results/forensic_analysis/
results/hybrid_pipeline/
results/part4_llm/
results/dataset1_appendix/
```

Prediction ledgers are intentionally retained to allow reported metrics and FP/FN analyses to be reconstructed.

---

# 24. Reproducibility Environment

Persisted scikit-learn artifacts were created with:

```text
scikit-learn 1.7.2
```

Using the same version is recommended for strict artifact compatibility.

Independent final verification was also performed under:

```text
scikit-learn 1.9.0
```

which emitted the expected `InconsistentVersionWarning` when loading 1.7.2 estimators.

Fresh verification nevertheless reproduced the final binary classifications for the persisted scikit-learn models.

The final verification environment used Python 3.12.

Core dependency families include:

```text
numpy
pandas
scikit-learn
joblib
tensorflow / keras
```

Part 4 requires Ollama plus the specified local Qwen model only if one intentionally reruns the LLM stage.

Frozen Part-4 report reproduction does not require an LLM rerun.

---

# 25. Random Seeds and Determinism

Primary reproducibility seed:

```text
42
```

Where applicable:

- Python RNG is seeded;
- NumPy RNG is seeded;
- TensorFlow RNG is seeded;
- deterministic TensorFlow operations are enabled;
- RF uses deterministic random-state behavior;
- source-aware splitting is deterministic.

Sensitivity/stability experiments additionally used:

```text
17
42
73
```

These do not represent separate final model configurations.

---

# 26. Safe RF Deployment-Style Inference

Inference-only script:

```text
src/models_optimized_final/RF/predict_random_forest.py
```

Example:

```bash
python src/models_optimized_final/RF/predict_random_forest.py \
    --model-dataset dataset2 \
    --input data/ingested/dataset3_ingested.csv \
    --output rf_d2_to_d3_predictions.csv
```

This loads the Dataset-2 final RF artifact and applies it to Dataset 3.

Do not use deployment-style inference as a replacement for the report's within-dataset RF OOF evaluation.

---

# 27. Reviewer Navigation Guide

If you are reviewing the project, use this order.

## Final standalone models

```text
src/models_optimized_final/
```

## Final fitted artifacts

```text
model_artifacts/models_optimized_final/
```

## Final standalone results

```text
results/models_optimized_final/
```

## Part 3 hybrid

```text
src/hybrid_pipeline/
model_artifacts/hybrid_pipeline/
results/hybrid_pipeline/
```

## Part 4 LLM

```text
src/part4_llm/
results/part4_llm/
```

## Optimization traceability

```text
src/optimization_history/
model_artifacts/optimization_history/
results/optimization_history/
results/optimization_runs/
```

## Dataset-1 appendix

```text
src/dataset1_appendix_experiments/
model_artifacts/dataset1_appendix/
results/dataset1_appendix/
```

---

# 28. Important Review Warnings

### Final code

Use:

```text
src/models_optimized_final/
```

not `models_baseline/`.

### Frozen evaluation

Do not retrain when attempting to reproduce report metrics.

### Thresholds

Do not recalibrate on TEST or transfer targets.

### RF

Do not compare full-fit training predictions with OOF report metrics.

### Temporal grouping

Do not interpret `SourceFile` or IP addresses used for grouping as predictive RF features.

### Dataset 1

Do not interpret Dataset 1 as a model-selection dataset.

### Part 4

Do not rerun Qwen and replace the frozen TEST outputs when reproducing the submitted report.

---

# 29. Final Verification Status

Before packaging, the project underwent a final read-only verification pass covering:

- code integrity;
- persisted splits;
- source-aware separation;
- preprocessing state;
- selected-feature schemas;
- model artifacts;
- prediction ledgers;
- frozen metrics;
- cross-dataset evaluation;
- RF OOF evidence;
- RF FULL5 evidence;
- Part-3 hybrid results;
- Part-4 LLM evidence;
- optimization-history traceability;
- report/result consistency.

Fresh inference was performed for the final:

```text
LSTM
LITEMV
Autoencoder
Isolation Forest
Random Forest
```

and reproduced the report-facing classifications.

For Random Forest transfer:

```text
D2 → D3: 0 classification mismatches
D3 → D2: 0 classification mismatches
```

Small numerical-score differences observed under different library/runtime environments did not change final classifications.

---

# 30. Final Source of Truth

If historical files describe different stages, use this precedence:

```text
1. src/models_optimized_final/
2. model_artifacts/models_optimized_final/
3. results/models_optimized_final/
4. frozen Part-3 / Part-4 directories
5. optimization history
6. baseline implementations
```

The final report should be interpreted against the final/frozen paths, not abandoned intermediate experiments.

---

# 31. GitHub Upload Checklist

Before pushing the repository, confirm that GitHub contains:

```text
README.md
.gitignore
src/
data/README.md
data/raw/.gitkeep
data/ingested/.gitkeep
```

and **does not contain**:

```text
dataset CSVs
PCAP files
trained .keras models
trained .joblib models
large frozen result ledgers
Google Drive archives
local virtual environments
```

The complete large/frozen project state should remain in Google Drive.

---

# 32. Summary

The repository intentionally contains baseline, optimization, final-model, hybrid, LLM, and appendix code because the project requires traceability across the full experimental process.

The key distinction is:

```text
FINAL MODELS
    src/models_optimized_final/

BASELINES
    src/models_baseline/

OPTIMIZATION HISTORY
    src/optimization_history/

FINAL HYBRID
    src/hybrid_pipeline/

FINAL LLM STAGE
    src/part4_llm/

APPENDIX-ONLY EXPERIMENTS
    src/dataset1_appendix_experiments/
```

For normal review, start with:

```text
src/models_optimized_final/
```

Then combine the GitHub code with the separately distributed Google Drive `data/`, `model_artifacts/`, and `results/` directories when full local reproduction is required.
