# Part 3 Hybrid Analysis / Ablation

This folder preserves the validation-only Part-3 ablation analysis separately from the final operational hybrid in `src/hybrid_pipeline/`.

The analysis uses the same final model implementations under `src/models_optimized_final/`, but it consumes the **recovered report-facing Part-3 fitted LSTM/LITEMV evidence**, not the independently retrained Part-2 neural-model results.

The fitted Part-3 base snapshot is stored under:

```text
artifacts/hybrid_pipeline/base_models/
```

and the frozen component probabilities used by the report-facing cascade are read from:

```text
results/hybrid_pipeline/<dataset>/validation_component_probabilities.csv
results/hybrid_pipeline/<dataset>/test_predictions.csv
```

The RF analysis stage is fit on TRAIN only. Hybrid routing candidates are evaluated on VALIDATION only. TEST is reporting-only.

Run both datasets:

```bash
python src/part3_hybrid_analysis/run_part3_hybrid_ablation.py
```

Or one dataset:

```bash
python src/part3_hybrid_analysis/run_part3_hybrid_ablation.py --dataset dataset2
python src/part3_hybrid_analysis/run_part3_hybrid_ablation.py --dataset dataset3
```

Outputs are written under:

```text
results/part3_hybrid_analysis/final_lstm_litemv_rf/
artifacts/part3_hybrid_analysis/final_lstm_litemv_rf/
```

This folder is supplementary analysis. The final deployed/report-facing cascade remains the fixed three-stage `LSTM -> LITEMV -> RF` implementation in `src/hybrid_pipeline/`.
