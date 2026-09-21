ISOLATION FOREST O1 — BRITTLE FEATURE PRUNING
=============================================

Purpose
-------
Controlled feature-hardening experiment for Section 2.2.

Only optimization change
------------------------
Baseline selected features: 57
O1 selected features:      43

Mandatory drops:
- FWD Init Win Bytes
- Bwd Init Win Bytes
- Total Connection Flow Time
- Fwd Header Length
- Bwd Header Length
- every feature containing "Flag" except exact aggregate "URG Flag Count"

Held fixed
----------
- persisted source-aware split
- signed-log1p + RobustScaler preprocessing
- benign-only Isolation Forest fitting
- SAME shared D2/D3 Isolation Forest hyperparameters from existing baseline artifacts
- random_state=42 final fit
- anomaly score = -IsolationForest.score_samples(X)
- validation-only max-F1 threshold calibration
- untouched test set

The script deliberately refuses to run if it cannot locate and verify the existing
shared D2/D3 Isolation Forest hyperparameters. It does not perform a new sensitivity
search, because that would confound the O1 feature-pruning experiment.

Run
---
python run_isolation_forest.py

Do NOT add --force-resplit for the controlled comparison.

Expected outputs
----------------
artifacts/models_same_hyper/IsolationForest/optimization_1_feature_pruning/
results/models_same_hyper/IsolationForest/optimization_1_feature_pruning/

Main summary:
results/models_same_hyper/IsolationForest/optimization_1_feature_pruning/
    isolation_forest_o1_test_summary.csv
