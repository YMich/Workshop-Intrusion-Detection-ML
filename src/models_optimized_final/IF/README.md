# Final Isolation Forest

Final 43-feature Isolation Forest implementation retained for supplementary evaluation.

## Frozen model configuration

- `n_estimators=200`
- `max_samples=512`
- `max_features=1.0`
- `bootstrap=False`
- `contamination="auto"`
- random state `42`

The model is fitted on benign TRAIN rows only. No model-hyperparameter search is performed. The anomaly threshold is calibrated on VALIDATION only and TEST is reserved for final evaluation.

## Run

```bash
python src/models_optimized_final/IF/run_isolation_forest.py
```

Strict cross-dataset transfer:

```bash
python src/models_optimized_final/IF/cross_dataset_evaluation.py
```

Artifacts and results are written under:

```text
artifacts/models_optimized_final/IF/
results/models_optimized_final/IF/
```

`best_hyperparameters.json` is a legacy artifact filename containing the frozen configuration; it does not imply a search in this submitted code.
