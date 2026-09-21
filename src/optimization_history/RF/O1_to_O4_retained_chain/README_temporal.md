# RF O4 — Endpoint-aware temporal optimization

Add these files to:

```text
D:\Documents\Workshop_final\src\models_optimized\RF\
```

Do **not** replace the static-chain files. This stage consumes the outputs already produced by:

```powershell
python run_rf_static_optimization.py --grid full
```

It expects:

```text
artifacts/models_optimized/RF/static_chain/folds/dataset2_outer_fold_manifest.csv
artifacts/models_optimized/RF/static_chain/folds/dataset3_outer_fold_manifest.csv
artifacts/models_optimized/RF/static_chain/O3_final_shared_hyperparameters.json
results/models_optimized/RF/static_chain/O3_outer_fold_selections.csv
```

Run:

```powershell
D:\Documents\Workshop_final\src\.venv\Scripts\python.exe run_rf_temporal_optimization.py
```

## Controlled O4 experiment

The 12 static O2/O3 features remain fixed. For OOF evaluation, every outer fold reuses the exact shared O3 RF configuration that was selected without seeing that outer test fold.

Sequence unit:

```text
(SourceFile, Src IP, Dst IP)
```

Rows are ordered causally by Timestamp. IPs and raw Timestamp are never model features.

Variants:

```text
R0      = current 12 features only
RDELTA  = R0 + current InterFlow Delta
RROLL3  = R0 + InterFlow Delta + previous-3-flow mean/std summaries
RROLL5  = R0 + InterFlow Delta + previous-5-flow mean/std summaries
RROLL10 = R0 + InterFlow Delta + previous-10-flow mean/std summaries
```

Rolling behavior summaries use only previous rows (`shift(1)`), for:

- Fwd IAT Mean
- Flow IAT Std
- Fwd Packet Length Mean
- Fwd Packet Length Std
- Bwd Packet Length Mean
- Bwd Packet Length Std
- Down/Up Ratio
- Fwd Act Data Pkts
- InterFlow Delta

The stage also runs the strict same-population control:

```text
R0_FULL5 vs RROLL5_FULL5
```

Both models are trained and evaluated only on rows with at least 5 previous flows for the same endpoint sequence. This separates temporal-content benefit from simple history-availability bias.

Key outputs:

```text
results/models_optimized/RF/temporal_chain/
    temporal_within_oof_summary.csv
    temporal_cross_dataset_summary.csv
    temporal_variant_ranking.csv
    FULL5_same_population_control.csv
    endpoint_history_availability.csv
    temporal_importance_share.csv
    temporal_feature_importance_mean.csv
```
