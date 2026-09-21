from __future__ import annotations

from pathlib import Path


def discover_project_root() -> Path:
    """Locate the project root from the authoritative final hybrid pipeline."""
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        hybrid = candidate / "src" / "hybrid_pipeline" / "hybrid_pipeline.py"
        if hybrid.is_file():
            return candidate
    raise RuntimeError(
        "Could not locate project root. Expected "
        "src/hybrid_pipeline/hybrid_pipeline.py."
    )


PROJECT_ROOT = discover_project_root()
PART4_DIR = Path(__file__).resolve().parent
HYBRID_DIR = PROJECT_ROOT / "src" / "hybrid_pipeline"

DATASETS = ("dataset2", "dataset3")

# ---------------------------------------------------------------------------
# Local LLM — frozen inference configuration
# ---------------------------------------------------------------------------
# Pin the exact quantized Ollama tag for reproducibility.
MODEL_NAME = "qwen3:8b-q4_K_M"
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 20,
    "seed": 42,
    "num_ctx": 4096,
    "num_predict": 512,
}
THINK = False
# Keep Qwen resident between requests to avoid repeated model reloads.
# The thermal guard may still unload it when cooling is required.
KEEP_ALIVE = "30m"
KEEP_MODEL_RESIDENT = True

# Qwen3-8B has 36 transformer layers. For thermal-safe local execution, the
# default Part-4 runtime offloads exactly half (18) to GPU and leaves the other
# half on CPU. This is a layer split; auxiliary tensors/KV/graph memory mean the
# measured CPU/GPU utilization will not necessarily be exactly 50.0/50.0.
QWEN_TRANSFORMER_LAYERS = 36
HALF_GPU_OFFLOAD_LAYERS = 18

# ---------------------------------------------------------------------------
# Hard-case escalation policy
# ---------------------------------------------------------------------------
# These routes contain genuine secondary-model intervention/conflict in the
# frozen LSTM -> LITEMV -> RF cascade. Straightforward LSTM_DIRECT and
# LITEMV_CONFIRMED_LSTM_POSITIVE cases remain untouched.
HARD_CASE_ROUTES = {
    "RF_ARBITRATED_KEEP_MALICIOUS",
    "LITEMV_RF_FILTERED_LSTM_POSITIVE",
    "JOINT_LITEMV_RF_RESCUE",
}

# Optional validation-ablation expansion. Keep False for the frozen final
# configuration; alternative behavior is restricted to validation-only experiments.
ESCALATE_LOW_CONFIDENCE_AGREEMENTS = False
LOW_CONFIDENCE_PERCENTILE = 0.05

# Validation-relative evidence-strength bins.
WEAK_PERCENTILE = 0.50
STRONG_PERCENTILE = 0.90

# ---------------------------------------------------------------------------
# Deterministic safety gate
# ---------------------------------------------------------------------------
ALLOW_AUTOMATIC_OVERRIDES = True
REQUIRE_HIGH_CONFIDENCE_FOR_OVERRIDE = True
REQUIRE_DIRECTIONAL_SPECIFICITY = True
REQUIRE_AT_LEAST_ONE_WEAK_UPSTREAM_MODEL = True

# ---------------------------------------------------------------------------
# Telemetry exposed to the LLM
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 20
CURRENT_FLOW_FEATURES = (
    "Flow Duration",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Min",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Down/Up Ratio",
    "Total Length of Fwd Packets",
    "TotLen Fwd Pkts",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Std",
)
BENIGN_DEVIATION_TOP_K = 5

PROMPT_VARIANTS = (
    "baseline",
    "independent_o2",
    "semantic_o3",
)

CONTEXT_VARIANTS = (
    "full",
    "behavioral_o1",
    "no_sequence",
    "no_benign_deviations",
    "models_only",
    "telemetry_only",
)


# ---------------------------------------------------------------------------
# FINAL FROZEN TEST PROFILE — selected on validation only
# ---------------------------------------------------------------------------
FINAL_TEST_CONTEXT_VARIANT = "behavioral_o1"
FINAL_TEST_PROMPT_VARIANT = "semantic_o3"
FINAL_TEST_SELECTION_MODE = "final-union"
FINAL_TEST_DEVICE_MODE = "split50"
FINAL_TEST_PROFILE_NAME = "frozen_validation_selected_v1"

# ---------------------------------------------------------------------------
# GPU monitoring / thermal safety
# ---------------------------------------------------------------------------
GPU_MONITOR_ENABLED = True
THERMAL_GUARD_ENABLED = True
RESUME_TEMP_C = 50
PRECALL_LIMIT_C = 60
HARD_ABORT_TEMP_C = 74
WATCHDOG_POLL_SECONDS = 0.5
COOLDOWN_POLL_SECONDS = 3.0

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "part4_llm"
    / "lstm_litemv_rf_arbitration"
)
