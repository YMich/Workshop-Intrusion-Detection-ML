# Final Autoencoder

Final benign-only autoencoder used for Dataset 2 and Dataset 3.

## Frozen model configuration

- selected representation: 43 hardened features
- encoder: `64 -> 32`
- bottleneck: `4`
- loss: MSE
- learning rate: `0.002`
- L2: `1e-5`
- denoising noise: `0.0`
- batch size: `256`
- anomaly score: Top-5 feature reconstruction MSE

No model-hyperparameter search is performed by the final code. The model is fitted on benign TRAIN rows only. The decision threshold is calibrated separately for each dataset from VALIDATION scores; TEST is untouched until final evaluation.

## Run

```bash
python src/models_optimized_final/AE/run_autoencoder.py
```

Strict cross-dataset transfer:

```bash
python src/models_optimized_final/AE/cross_dataset_evaluation.py
```

Artifacts and results are written under:

```text
artifacts/models_optimized_final/AE/
results/models_optimized_final/AE/
```

`best_hyperparameters.json` is retained as a legacy artifact filename for evaluator compatibility; it contains the frozen final configuration and is not produced by a runtime search.
