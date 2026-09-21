# Final Random Forest

Final endpoint-aware Random Forest used by the project.

## Frozen model configuration

- `n_estimators=256`
- `criterion="gini"`
- `max_depth=None`
- `min_samples_split=2`
- `min_samples_leaf=2`
- `max_features="sqrt"`
- `bootstrap=True`

The model uses the fixed final endpoint-aware causal RROLL5 temporal representation. No feature-window search or model-hyperparameter search is performed by the submitted final code. Source-aware out-of-fold evaluation is used for within-dataset assessment.

## Run

```bash
python src/models_optimized_final/RF/run_random_forest.py
```

Prediction with persisted deployment artifacts is provided by:

```bash
python src/models_optimized_final/RF/predict_random_forest.py --help
```

Artifacts and results are written under:

```text
artifacts/models_optimized_final/RF/
results/models_optimized_final/RF/
```
