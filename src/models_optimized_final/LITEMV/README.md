# Final LITEMV

Final accepted LITEMV sequence model for Dataset 2 and Dataset 3.

## Frozen model configuration

- final representation: 43 hardened features
- sequence length: `40`
- `n_filters=32`
- `kernel_size=40`
- convolution stride: `1`
- learning rate: `0.001`
- batch size: `64`
- training/evaluation sequence stride: `5 / 1`

The model configuration is identical for Dataset 2 and Dataset 3. No model-hyperparameter search is performed. Preprocessing state, class weights, early-stopping epoch, and validation-calibrated decision threshold are learned separately per dataset.

## Run

```bash
python src/models_optimized_final/LITEMV/run_litemv.py
```

Full-target strict transfer evaluation:

```bash
python src/models_optimized_final/LITEMV/run_litemv_full_target_transfer.py
```

Artifacts and results are written under:

```text
artifacts/models_optimized_final/LITEMV/
results/models_optimized_final/LITEMV/
```
