# Dataset 1 Appendix Experiments

This folder is intentionally separate from `src/models_optimized_final/` because Dataset 1 is an additional evaluation extension reported in Appendix J. It is included as supplementary reproducibility code and does not modify the final D2/D3 pipeline.
It extends the already-selected D2/D3 pipelines to Dataset 1 without changing the final D2/D3 source implementations or overwriting their artifacts/results.

> **Submission role:** This folder is supplementary Appendix-J reproducibility code. The main final pipeline remains the Dataset-2/Dataset-3 implementation under `src/models_optimized_final/` and `src/hybrid_pipeline/`.

## Experimental contract

- Dataset 1 input: `data/ingested/dataset1_ingested.csv`.
- Model architecture/hyperparameters are frozen to the final D2/D3 values.
- No Dataset-1 grid search or model-hyperparameter re-selection is performed.
- Dataset-specific learned quantities remain dataset-specific: fitted preprocessing, class weights, early-stopping epoch, and validation-calibrated decision threshold where applicable.
- LSTM/LITEMV/IF/AE share the same persisted Dataset-1 source-aware 70/15/15 split under `artifacts/dataset1_appendix/splits/`.
- RF preserves its final source-aware OOF evaluation protocol rather than being forced into the neural-model split protocol.
- Dataset-1 TEST is untouched until model/threshold selection is complete.
- All appendix artifacts/results are isolated under `artifacts/dataset1_appendix/` and `results/dataset1_appendix/`.

## Folder placement

Copy this complete folder to:

```text
<PROJECT_ROOT>/src/dataset1_appendix_experiments/
```

The existing project must still contain:

```text
src/preprocessing/
src/feature_engineering/
src/models_optimized_final/
data/ingested/dataset1_ingested.csv
data/ingested/dataset2_ingested.csv
data/ingested/dataset3_ingested.csv
```

## Quick start

This appendix is orchestrated through `run_dataset1_appendix.py`. You do not need to manually run the individual Dataset-1 model scripts first.

For all Dataset-1 appendix stages except the local LLM:

```bash
python run_dataset1_appendix.py --phase all-no-llm
```

Part 4 is kept separate because it requires a running local Ollama model.

## 1. Dataset-1 local model evaluation

From `src/dataset1_appendix_experiments`:

```bash
python run_dataset1_appendix.py --phase models
```

This runs:

1. Random Forest — final 32-feature endpoint-aware RROLL5, source-aware OOF, fixed RF hyperparameters.
2. LSTM — final O1+O2 model, frozen D2/D3 shared hyperparameters, D1-specific preprocessing/class weights/early stopping/validation threshold.
3. LITEMV — final O1 43-feature model, frozen shared hyperparameters, D1-specific learned quantities.
4. Isolation Forest — final O1 43-feature model, benign D1 TRAIN only, D1 validation threshold.
5. Autoencoder — final 43-feature/O5 model, benign D1 TRAIN only, D1 validation threshold.

## 2. Six-way cross-dataset generalization

After D1 artifacts exist, run:

```bash
python run_dataset1_appendix.py --phase cross
```

Each model evaluates all ordered directions:

```text
D1 -> D2
D1 -> D3
D2 -> D1
D2 -> D3
D3 -> D1
D3 -> D2
```

For D1 as source, the appendix D1 artifact is used. For D2/D3 as source, the existing final artifact under `artifacts/models_optimized_final/` is used. Target data are never used for fitting preprocessing, threshold calibration, or model training.

## 3. Part 3 hybrid on Dataset 1

Run after the D1 LSTM and LITEMV outputs exist:

```bash
python run_dataset1_appendix.py --phase hybrid
```

The Dataset-1 copy retains the Part-3 validation-only rule search. RF is fitted on D1 TRAIN only; LSTM/LITEMV consume their frozen D1 validation/test predictions.

For Part-4 compatibility the appendix hybrid additionally saves:

```text
results/dataset1_appendix/part3_hybrid/dataset1/validation_component_probabilities.csv
artifacts/dataset1_appendix/part3_hybrid/dataset1/hybrid_rule.json
```

and stable `Predicted_Label` / `DecisionRoute` aliases in the D1 hybrid test predictions.

## 4. Forensic FP/FN analysis and D1/D2/D3 comparison

```bash
python run_dataset1_appendix.py --phase analysis
```

Outputs include:

```text
results/dataset1_appendix/analysis/forensic_errors/
results/dataset1_appendix/analysis/d1_d2_d3_model_comparison.csv
```

The forensic utility exports FP/FN rows and source-level concentration tables for RF, LSTM, LITEMV, IF, AE, and the hybrid.

## 5. Part 4 on Dataset 1

Part 4 is intentionally tested with the exact same final LLM configuration used for D2/D3:

```text
Model: qwen3:8b-q4_K_M
Prompt: semantic_o3
Context: behavioral_o1
Selector: final-union
Safety gate: unchanged
Inference parameters: unchanged
```

Only dataset routing and output paths are changed. There is no D1 prompt tuning.

Part 4 is deliberately evaluated on the same three-model LSTM -> LITEMV -> RF cascade assumed by the original Part-4 design. The Dataset-1 Part-3 ablation still reports its validation-selected architecture separately; if that selected architecture is simpler than the triple cascade, report the Part-4 result as an additional robustness/transfer experiment rather than silently redefining the Dataset-1 Part-3 winner.

Prerequisites:

```bash
pip install -r part4_llm_arbitration/requirements_part4.txt
ollama pull qwen3:8b-q4_K_M
```

Then run:

```bash
python run_dataset1_appendix.py --phase part4
```

This performs the frozen TEST arbitration on Dataset 1 and then produces the Part-4.3 report.

## 6. Everything except the local LLM

```bash
python run_dataset1_appendix.py --phase all-no-llm
```

The LLM phase is deliberately separate because it requires a running local Ollama model and may be substantially slower.

## Important Part-3 / Part-4 compatibility note

The main final D2/D3 hybrid implementation lives at `src/hybrid_pipeline/hybrid_pipeline.py`. The Dataset-1 appendix keeps its own local Part-3 implementation so that Appendix-J experiments remain isolated from the main final pipeline.

Dataset-1 Part 4 requires the same logical interface and output artifacts used by the final hybrid workflow, including the hybrid rule and validation component probabilities. The local appendix Part-3 implementation does not expose that interface identically, so this appendix includes a small compatibility API/output layer.

The compatibility layer is local to `src/dataset1_appendix_experiments/`. It does **not** modify `src/hybrid_pipeline/`, `src/models_optimized_final/`, or the final D2/D3 artifacts/results.

## Cross-dataset compatibility fallback

Older D2/D3 `preprocessor.joblib` artifacts may fail to unpickle under a different pandas version
(e.g. `StringDtype` state incompatibility). Cross-dataset scripts now first try the frozen artifact.
If and only if loading fails for this compatibility reason, they deterministically rebuild the
preprocessor from the persisted SOURCE training split using the original fitting rule:

- LSTM / LITEMV: all SOURCE training rows.
- Isolation Forest / Autoencoder: benign SOURCE training rows only.

The TARGET dataset is never used for preprocessing fit, threshold calibration, or model training.
The stored source model, source selected-feature schema, and source decision threshold remain frozen.

To resume one cross-dataset model without rerunning earlier ones:

```bash
python run_dataset1_appendix.py --phase cross --model LSTM
python run_dataset1_appendix.py --phase cross --model LITEMV
python run_dataset1_appendix.py --phase cross --model IF
python run_dataset1_appendix.py --phase cross --model AE
```
