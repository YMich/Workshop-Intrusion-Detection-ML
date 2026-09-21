AUTOENCODER O4 — FULL-FEATURE MSE ANOMALY SCORE

O4 is a score-only controlled experiment on top of O3.

NO MODEL RETRAINING
-------------------
The exact O3 model weights, O3 fitted preprocessor, O3 feature schema, and O3
source-aware split are reused. This prevents stochastic retraining differences
from contaminating the scoring comparison.

FROZEN FROM O3
--------------
Features: 43 (O1 brittle-feature-pruned schema)
Encoder: 64 -> 32
Bottleneck: 4
Learning rate used to train O3: 0.002
Training loss: MSE
L2: 1e-5
Denoising noise: 0.02
Batch size: 256
Preprocessing: signed-log1p + RobustScaler fitted on O3 benign training data

ONLY O4 CHANGE
--------------
Anomaly score: Top-5 MSE -> full-feature MSE across all 43 retained features.

The threshold is recalibrated on each dataset's validation split because the
numeric score scale changes when the scoring rule changes. Test data are never
used for threshold selection.

HYPOTHESIS
----------
If Dataset-3 malicious flows create small reconstruction deviations distributed
across many features, full MSE may retain that aggregate evidence better than a
Top-5-only score. This is a hypothesis to be tested, not an assumed fact.

OUTPUTS
-------
artifacts/models_same_hyper/Autoencoder/optimization_4_full_mse_score/
results/models_same_hyper/Autoencoder/optimization_4_full_mse_score/

Key files:
- autoencoder_o4_test_summary.csv
- optimization_4_vs_o3.csv
- dataset*/test_metrics.csv
- dataset*/test_predictions.csv
- dataset*/confusion_matrix.csv
- validation_threshold_candidates.csv
