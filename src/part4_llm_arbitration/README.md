# Part 4 — Final Local LLM Arbitration

This directory contains the submitted Part-4 system: a frozen local Qwen
arbitration layer applied after the authoritative final hybrid IDS.

```text
LSTM -> LITEMV -> RF -> frozen hybrid decision
                         |
                         v
                 hard-case selector
                         |
                         v
                  Qwen3-8B Q4_K_M
                         |
                         v
                 deterministic safety gate
                         |
                         v
                 Part-4 final prediction
```

## Where is the final Part-4 model?

Part 4 does **not** train or fine-tune a new neural network. The pretrained LLM
weights are supplied by Ollama:

```text
qwen3:8b-q4_K_M
```

The submitted Part-4 system is the frozen combination of:

```text
Base LLM:       qwen3:8b-q4_K_M
Prompt:         semantic_o3
Context:        behavioral_o1
Selector:       final-union
Safety gate:    deterministic conservative override gate
Temperature:    0.0
Top-p:          0.9
Top-k:          20
Seed:           42
Context window: 4096
Max output:     512
Thinking:       disabled
Execution:      split50 (18/36 transformer layers requested on GPU)
```

The model weights are therefore external, while the complete project-specific
arbitration method is contained in this directory.

## Authoritative final entry point

The final submitted Part-4 implementation is:

```text
run_final_part4.py
```

Run from the project root:

```bash
python src/part4_llm_arbitration/run_final_part4.py
```

or for one dataset:

```bash
python src/part4_llm_arbitration/run_final_part4.py --dataset dataset2
python src/part4_llm_arbitration/run_final_part4.py --dataset dataset3
```

`run_final_part4.py` exposes only dataset selection. It does **not** expose
prompt, context, selector, device-policy, case-cap, or dry-run switches for TEST.
Those settings are frozen in `config.py` from validation-only development.

The underlying pipeline also independently verifies the frozen TEST profile and
refuses changed TEST settings.

## Validation-only ablation code

The experiments used to compare prompt/context/selector variants are separated
from the final system:

```text
run_part4_ablation.py
```

Example:

```bash
python src/part4_llm_arbitration/run_part4_ablation.py \
    --dataset dataset2 \
    --prompt-variant semantic_o3 \
    --context-variant behavioral_o1 \
    --selection-mode final-union
```

This runner is **validation-only**. It has no TEST-stage option.

These ablations reproduce the Part-4 development analysis; they do not perform
model hyperparameter search or modify the frozen final Qwen weights.

## Runtime prerequisites

Install the Part-4 Python dependencies:

```bash
pip install -r src/part4_llm_arbitration/requirements_part4.txt
```

Install/pull the exact local LLM:

```bash
ollama pull qwen3:8b-q4_K_M
```

The authoritative final hybrid implementation must be present at:

```text
src/hybrid_pipeline/hybrid_pipeline.py
```

and its required persisted artifacts/results must already exist. Part 4 consumes
the frozen hybrid outputs; it does not retrain the Part-3 models.

## Source files

```text
src/part4_llm_arbitration/
├── README.md
├── __init__.py
├── config.py
├── arbitration_pipeline.py
├── run_final_part4.py
├── run_part4_ablation.py
├── llm_arbitrator.py
├── pipeline_io.py
├── prompts.py
├── telemetry_context.py
├── evaluation.py
├── report_part4_performance.py
├── FINAL_SUBMISSION_MANIFEST.json
└── requirements_part4.txt
```

### `config.py`

Contains the frozen final model/runtime/gating configuration and the documented
validation variants.

### `arbitration_pipeline.py`

Contains the shared end-to-end arbitration implementation: loading hybrid state,
validation-relative evidence calibration, hard-case selection, privacy-safe
payload construction, Ollama inference, deterministic safety gating,
checkpointing, prediction assembly, and metrics.

### `run_final_part4.py`

The **authoritative final Part-4 runner**. It always executes the frozen TEST
profile and permits only dataset selection.

### `run_part4_ablation.py`

Validation-only ablation runner used to reproduce prompt/context/selector
comparisons. It cannot run TEST.

### `llm_arbitrator.py`

Calls `qwen3:8b-q4_K_M` through Ollama, validates structured model output,
collects runtime telemetry, implements the thermal guard, and applies the
conservative deterministic override gate.

### `prompts.py`

Contains the exact prompt variants, including the final `semantic_o3` prompt.

### `telemetry_context.py`

Builds the model-visible behavioral context, validation-relative evidence
strengths, hard-case selector, and privacy-safe payload.

### `pipeline_io.py`

Loads/reconstructs the authoritative frozen Part-3 hybrid state without copying
or retraining the hybrid models.

### `evaluation.py`

Computes classification metrics from the saved predictions.

### `report_part4_performance.py`

Report-only utility for the completed frozen TEST. It does not call Qwen or
change any prediction.

## Privacy and leakage controls

The model-visible payload deliberately excludes:

- ground-truth labels;
- dataset identity;
- `SourceFile` / capture names;
- IP addresses and ports;
- filenames;
- malware-family labels.

Exact model-visible payloads are written to `llm_payloads.jsonl` so the LLM
input can be audited.

## Frozen TEST safeguards

For TEST, the implementation enforces:

```text
context  = behavioral_o1
prompt   = semantic_o3
selector = final-union
device   = split50
```

TEST forbids:

- alternative prompt/context/selector settings;
- dry-run mode;
- case caps;
- TEST-time configuration selection.

The runner checkpoints each completed arbitration. A completed dataset receives
`FINAL_TEST_COMPLETE.json`; rerunning the final command reuses the completed
result instead of sending the same cases to Qwen again.

## Outputs

Frozen final TEST results are written under:

```text
results/part4_llm/lstm_litemv_rf_arbitration/FINAL_FROZEN_TEST/
```

Per-dataset outputs include the exact prompts/payloads, selected internal cases,
unlabeled Qwen outputs, final predictions, metrics, runtime statistics, summary,
and completion marker.

After TEST completes, generate the Part-4.3 report-only analysis with:

```bash
python src/part4_llm_arbitration/report_part4_performance.py
```

## Submission boundary

There is **no model hyperparameter search** in the final Part-4 execution path.
The final Qwen configuration and project-specific arbitration settings are
frozen before TEST. Validation-only prompt/context/selector ablations are kept
in a separate runner solely to reproduce the documented development analysis.
