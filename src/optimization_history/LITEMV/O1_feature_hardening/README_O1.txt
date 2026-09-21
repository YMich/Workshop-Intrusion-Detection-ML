LITEMV O1 — 43 hardened features

Place run_litemv_o1_hardened_features.py in:
D:\Documents\Workshop_final\src\models_same_hyper\LITEMV

Run:
    python run_litemv_o1_hardened_features.py

Normally DO NOT use --force-resplit.

O1 retrains D2 and D3 because the LITEMV input changes from 57 to 43 features.
All other B0 architecture/training/threshold rules remain unchanged.

Outputs:
artifacts/models_same_hyper/LITEMV/optimization_o1_hardened_features/
results/models_same_hyper/LITEMV/optimization_o1_hardened_features/

The same run also prints and saves:
- B0 vs O1 within-dataset source-test comparison
- B0 vs O1 D2->D3 validation/test transfer
- B0 vs O1 D3->D2 validation/test transfer

The previous source-robust threshold experiment is diagnostic only; it is not O1.
