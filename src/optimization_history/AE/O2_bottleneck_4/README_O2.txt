AUTOENCODER O2 — BOTTLENECK CONSTRICTION

Purpose
-------
O2 builds directly on O1. It RETAINS the same 43-feature brittle-feature-pruned
representation and changes exactly one model hyperparameter:

    bottleneck_dim: 8 -> 4

Everything else remains fixed for Dataset 2 and Dataset 3:

    encoder_hidden_dims = (64, 32)
    learning_rate       = 0.002
    loss                = Huber (delta=1.0)
    l2_regularization   = 1e-5
    denoising_noise_std = 0.02
    batch_size          = 256
    anomaly score       = Top-5 MSE
    preprocessing       = signed-log1p + RobustScaler

The source-aware split is reused unchanged. The preprocessor is fitted only on
benign training rows, and the operating threshold is recalibrated separately on
each dataset's validation partition. Test data is untouched until final evaluation.

Outputs
-------
artifacts/models_same_hyper/Autoencoder/optimization_2_bottleneck_4/
results/models_same_hyper/Autoencoder/optimization_2_bottleneck_4/

If O1 results exist at:
results/models_same_hyper/Autoencoder/optimization_1_brittle_feature_removal/
then O2 also writes optimization_2_vs_o1.csv automatically.

Run
---
python run_autoencoder.py

Do not use --force-resplit for the controlled comparison.
