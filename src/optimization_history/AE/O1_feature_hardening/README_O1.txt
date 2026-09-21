AUTOENCODER O1 — BRITTLE FEATURE REMOVAL (SHARED FIXED CONFIGURATION)
====================================================================

Controlled experiment
---------------------
O1 changes ONLY the input feature set. Dataset 2 and Dataset 3 both use the
previously agreed shared Autoencoder configuration:

  encoder_hidden_dims = (64, 32)
  bottleneck_dim       = 8
  learning_rate        = 0.002
  loss                 = Huber (delta=1.0)
  l2_regularization    = 1e-5
  denoising_noise_std  = 0.02
  batch_size           = 256
  anomaly score        = Top-5 MSE

Thresholds are still calibrated separately per dataset using validation data
only. Test data is untouched until final evaluation.

O1 feature removal
------------------
Explicitly remove:
  - FWD Init Win Bytes
  - Bwd Init Win Bytes
  - Total Connection Flow Time
  - Fwd Header Length
  - Bwd Header Length

Also remove every selected feature containing 'Flag' except exact aggregate
'URG Flag Count', if present. Expected feature count: 57 -> 43.

Run
---
python run_autoencoder.py

Do not use --force-resplit unless you intentionally want to rebuild the shared
source-aware split.

Outputs
-------
artifacts/models_same_hyper/Autoencoder/optimization_1_brittle_feature_removal/
results/models_same_hyper/Autoencoder/optimization_1_brittle_feature_removal/

The evaluator attempts to compare against:
results/models_same_hyper/Autoencoder/baseline/<dataset>/test_metrics.csv
If that shared-baseline result folder does not exist yet, O1 evaluation still
completes and prints a warning; the O1 metrics remain valid, but the automatic
delta table cannot be produced until the shared baseline metrics are present.
