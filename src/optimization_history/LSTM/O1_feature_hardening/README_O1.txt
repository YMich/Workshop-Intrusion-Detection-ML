Optimization 1 — Brittle / Environment-Dependent Feature Removal

Drop entirely from the LSTM input:
- FWD Init Win Bytes
- Bwd Init Win Bytes
- Total Connection Flow Time
- Fwd Header Length
- Bwd Header Length
- Every flag-related feature EXCEPT the exact aggregate feature: URG Flag Count

Flag rule:
- If a selected feature name contains "Flag", it is removed unless its exact
  normalized name is "URG Flag Count".
- This removes aggregate SYN Flag Count, FIN Flag Count, ACK Flag Count,
  PSH Flag Count, RST Flag Count, ECE Flag Count, and CWR Flag Count.
- It also removes every directional Fwd/Bwd flag field, including Fwd/Bwd RST,
  PSH, URG, and any other directional flag field that survives feature selection.
- URG Flag Count is retained only if that exact aggregate feature exists in the
  selected schema. No other flag-related feature is retained.

Unchanged controls:
- signed-log1p + RobustScaler preprocessing
- shared source-aware split manifests
- baseline shared D2/D3 LSTM hyperparameters
- class weighting procedure
- early stopping procedure
- validation-only threshold calibration
- sequence construction and stride
- random seed

Important:
- Run the completed baseline experiment first. O1 loads:
  artifacts/models_same_hyper/LSTM/shared_best_hyperparameters.json
- O1 DOES NOT overwrite baseline outputs.
- O1 writes under:
  artifacts/models_same_hyper/LSTM/optimization_1_brittle_feature_removal/
  results/models_same_hyper/LSTM/optimization_1_brittle_feature_removal/

Run from the LSTM model directory as usual:
  python run_lstm.py

Key outputs after evaluation:
- optimization_1_vs_baseline.csv
- dataset2/test_metrics.csv
- dataset2/false_negatives.csv
- dataset2/false_negatives_by_source.csv
- dataset2/false_positives_by_source.csv
- dataset3/test_metrics.csv
- dataset3/false_negatives_by_source.csv
- dataset3/false_positives_by_source.csv

For Section 2.2, inspect especially:
- Dataset 2 total FN
- Dataset 2 IcedID FN concentration by SourceFile
- Dataset 2 recall/F1/PR-AUC
- Dataset 3 FP and FN
- Dataset 3 recall/F1/PR-AUC
