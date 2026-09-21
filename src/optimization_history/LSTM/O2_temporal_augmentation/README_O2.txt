LSTM Optimization 2 — Temporal/Rate Time-Warp Augmentation
============================================================

Purpose
-------
Continue from the accepted 43-feature O1 model. O2 changes only the TRAINING
representation by adding synthetic time-warped variants of malicious SourceFiles.
Validation and test data are never augmented.

Accepted O1 remains active
--------------------------
- Drops FWD Init Win Bytes, Bwd Init Win Bytes, Total Connection Flow Time,
  Fwd Header Length, and Bwd Header Length.
- Drops every flag-related feature except exact aggregate "URG Flag Count".
- Expected model input remains the same 43 features as accepted O1.

O2 augmentation
---------------
For every malicious SourceFile in the TRAIN split, create two synthetic copies:
- one faster/compressed variant with s sampled in [0.50, 0.90]
- one slower/expanded variant with s sampled in [1.10, 2.00]

Raw-space transformation:
- time-valued retained features: x' = s * x
  (Flow Duration, IAT families, Active/Idle timing families when retained)
- rate-valued retained features: x' = x / s
  (features ending /s and retained *Rate Avg fields)

Reproducibility
---------------
AUGMENTATION_RANDOM_STATE = 42.
The preprocessor is fitted on ORIGINAL training rows only. Synthetic rows are
then transformed using that frozen train-fitted signed-log1p + RobustScaler.
Thus augmentation does not alter imputation medians or scaler parameters.

Class weighting
---------------
O1 class weights are calculated from ORIGINAL training sequences only. To avoid
turning augmentation into an implicit class-weight change, the malicious class
weight is divided equally across the original malicious sequence plus its two
synthetic variants. Benign weight is unchanged.

Unchanged from O1
-----------------
- shared source-aware split manifest
- 43-feature O1 schema
- signed-log1p + RobustScaler
- shared D2/D3 model hyperparameters
- architecture
- sequence length and stride
- early stopping
- validation-only per-dataset threshold calibration
- untouched test set

Folder placement
----------------
This folder can stay nested, e.g.
  Workshop_final/src/models_same_hyper_improved/LSTM/o2_temporal_augmentation/
Path discovery no longer depends on a fixed number of parent directories.

Run
---
From this folder:
  python run_lstm.py

Do NOT use --force-resplit for the controlled O1 -> O2 comparison.

Outputs
-------
results/models_same_hyper/LSTM/optimization_2_temporal_augmentation/
  lstm_test_summary.csv
  optimization_2_vs_o1.csv
  optimization_2_vs_baseline.csv
  dataset2/augmentation_manifest.csv
  dataset2/false_positives_by_source.csv
  dataset2/false_negatives_by_source.csv
  dataset3/augmentation_manifest.csv
  dataset3/false_positives_by_source.csv
  dataset3/false_negatives_by_source.csv

Primary decision
----------------
Compare O2 against accepted O1, not against the rejected 29-feature experiment.
O1 reference performance:
- D2: FP=30, FN=3, Recall=0.9915, Precision=0.9208, F1=0.9549
- D3: FP=24, FN=0, Recall=1.0000, Precision=0.9564, F1=0.9777

O2 is useful if it preserves D3's zero-FN behavior and D2's near-perfect recall
while improving robustness/precision or at least not materially increasing FPs.
