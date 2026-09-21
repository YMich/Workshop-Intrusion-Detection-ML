AUTOENCODER O3 — MSE TRAINING LOSS

Purpose
-------
O3 builds directly on O2. It retains the same 43-feature O1-pruned schema and the
4-dimensional bottleneck introduced in O2. The ONLY model change is:

    training loss: Huber -> MSE

Frozen settings
---------------
Encoder: 64 -> 32
Bottleneck: 4
Learning rate: 0.002
Loss: MSE
L2: 1e-5
Denoising noise std: 0.02
Batch size: 256
Anomaly score: Top-5 MSE
Preprocessing: existing signed-log1p + RobustScaler
Feature schema: same 43 features as O1/O2
Threshold: recalibrated per dataset on validation only

Why
---
This tests whether MSE training, which penalizes large reconstruction residuals
quadratically, improves benign/malicious score separation after the O2 bottleneck
constriction. Note that Huber vs MSE affects model training; the detector anomaly
score itself remains Top-5 MSE in both O2 and O3.

Outputs
-------
artifacts/models_same_hyper/Autoencoder/optimization_3_mse_loss/
results/models_same_hyper/Autoencoder/optimization_3_mse_loss/

If O2 results exist, O3 writes optimization_3_vs_o2.csv automatically.
