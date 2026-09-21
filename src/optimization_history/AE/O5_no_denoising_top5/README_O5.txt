AUTOENCODER O5 — REMOVE DENOISING

Purpose
-------
O5 builds directly on O3 (not O4). It keeps the same 43-feature O1-pruned schema,
4-dimensional bottleneck, MSE training loss, and Top-5 MSE anomaly score.
The ONLY model change is:

    denoising_noise_std: 0.02 -> 0.0

Frozen settings
---------------
Encoder: 64 -> 32
Bottleneck: 4
Learning rate: 0.002
Loss: MSE
L2: 1e-5
Denoising noise std: 0.0
Batch size: 256
Anomaly score: Top-5 MSE
Preprocessing: existing signed-log1p + RobustScaler
Feature schema: same 43 features as O1/O2/O3
Threshold: recalibrated per dataset on validation only

Why
---
This tests whether Gaussian denoising makes the benign manifold too tolerant of
small reconstruction deviations and therefore hides stealthier malicious flows.
The hypothesis is experimental; O5 is designed to test it without changing the
architecture, loss, score mode, feature schema, split, or preprocessing.

Outputs
-------
artifacts/models_same_hyper/Autoencoder/optimization_5_no_denoising/
results/models_same_hyper/Autoencoder/optimization_5_no_denoising/

If O3 results exist, O5 writes optimization_5_vs_o3.csv automatically.
