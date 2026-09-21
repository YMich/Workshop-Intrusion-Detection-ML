# Exact endpoint-aware RROLL5 restoration

This patch restores the two temporal-engineering details from the earlier endpoint-aware RROLL5 experiment:

1. Rolling standard deviations use `min_periods=1` with `ddof=0`.
2. Every rolling window includes the coefficient of variation of inter-flow delta:

   `InterFlow Delta PrevW CV = PrevW Std / abs(PrevW Mean)`

For RROLL5 the schema is therefore exactly **32 model features**:

- 12 current-flow features
- 1 current inter-flow delta
- 16 previous-5 mean/std summaries for 8 behavioral variables
- 3 previous-5 inter-flow-delta summaries: mean, std, CV

The sequence key remains `(SourceFile, Src IP, Dst IP)`, and raw IPs/Timestamp are never model inputs.

## First run: cheap reproduction check

Do **not** rerun O5 yet. First run only:

```powershell
python run_rf_temporal_reproduction_check.py
```

This trains only the two cross-dataset RROLL5 transfer models using the frozen O3 final shared hyperparameters. It should be much faster than O4/O5.

Compare the output with the earlier endpoint-aware result:

- D2 -> D3: recall ~0.4557, F1 ~0.615, PR-AUC ~0.727, FPR ~0.00151
- D3 -> D2: recall ~0.4588, F1 ~0.617, PR-AUC ~0.586, FPR ~0.000545

If those reproduce approximately, then rerun O4 with:

```powershell
python run_rf_temporal_optimization.py
```

Only after O4 is correct should O5 be rerun. Existing O5 results were produced with the unintended 31-feature RROLL5 representation and should not be used as final results.

Because `rf_temporal_hyperparameter_verification.py` imports `variant_features()` and `load_temporal_base()` from `rf_temporal_optimization.py`, replacing that file with this patch automatically makes the next O5 run use the corrected 32-feature RROLL5 representation.
