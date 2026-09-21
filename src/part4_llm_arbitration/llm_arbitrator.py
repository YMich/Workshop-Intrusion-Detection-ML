from __future__ import annotations

import atexit
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import from_json

from config import (
    ALLOW_AUTOMATIC_OVERRIDES,
    COOLDOWN_POLL_SECONDS,
    GPU_MONITOR_ENABLED,
    HARD_ABORT_TEMP_C,
    HALF_GPU_OFFLOAD_LAYERS,
    KEEP_ALIVE,
    KEEP_MODEL_RESIDENT,
    MODEL_NAME,
    OLLAMA_OPTIONS,
    PRECALL_LIMIT_C,
    QWEN_TRANSFORMER_LAYERS,
    REQUIRE_AT_LEAST_ONE_WEAK_UPSTREAM_MODEL,
    REQUIRE_DIRECTIONAL_SPECIFICITY,
    REQUIRE_HIGH_CONFIDENCE_FOR_OVERRIDE,
    RESUME_TEMP_C,
    THERMAL_GUARD_ENABLED,
    THINK,
    WATCHDOG_POLL_SECONDS,
)
from prompts import build_user_prompt, get_prompt_pair


class ArbitrationDecision(BaseModel):
    recommended_action: Literal[
        "KEEP_HYBRID",
        "OVERRIDE_TO_C2",
        "OVERRIDE_TO_NON_C2",
    ]
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    target_specificity: Literal[
        "NON_SPECIFIC",
        "SUPPORTS_C2",
        "CONTRADICTS_C2",
    ]
    evidence: list[str] = Field(min_length=1, max_length=4)
    conflict_summary: str


def _fail_closed_decision(reason: str) -> ArbitrationDecision:
    """Return a regression-safe decision when structured output cannot be recovered.

    This never changes the hybrid prediction. It is deliberately conservative: a
    malformed/truncated LLM response must not become an automatic override.
    """
    return ArbitrationDecision(
        recommended_action="KEEP_HYBRID",
        confidence="LOW",
        target_specificity="NON_SPECIFIC",
        evidence=[
            "Structured LLM output was incomplete or invalid; the hybrid decision was retained safely."
        ],
        conflict_summary=f"Fail-closed structured-output fallback: {reason}",
    )


def _parse_arbitration_decision(content: str) -> ArbitrationDecision:
    """Parse Qwen's structured response, recovering safely from EOF truncation.

    Ollama is already given ArbitrationDecision.model_json_schema(), but a response
    can still hit num_predict=512 while writing a final explanatory string. In that
    case the classification fields may be complete even though the trailing JSON is
    missing a closing quote/brace. We first use normal strict Pydantic validation.
    If that fails, pydantic-core's allow_partial parser recovers only fully completed
    JSON values. We preserve a recovered action only when it is itself schema-valid;
    otherwise we fail closed to KEEP_HYBRID.

    This does NOT change the model, prompt, selector, inference parameters, safety
    gate, or 512-token output limit used in the report.
    """
    try:
        return ArbitrationDecision.model_validate_json(content)
    except ValidationError as strict_error:
        try:
            partial = from_json(content, allow_partial=True)
        except (ValueError, TypeError) as partial_error:
            print(
                "[LLM OUTPUT WARNING] Invalid structured response could not be "
                "recovered; retaining the hybrid decision safely."
            )
            return _fail_closed_decision(
                f"strict={strict_error.errors()[0].get('type', 'validation_error')}; "
                f"partial={type(partial_error).__name__}"
            )

        if not isinstance(partial, dict):
            print(
                "[LLM OUTPUT WARNING] Partial structured response was not a JSON "
                "object; retaining the hybrid decision safely."
            )
            return _fail_closed_decision("partial JSON was not an object")

        recovered = dict(partial)
        # Narrative fields are not allowed to block a completed classification.
        # If truncation occurred while writing them, replace them with an audit note.
        evidence = recovered.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            recovered["evidence"] = [
                "The model response was truncated after emitting its classification fields."
            ]
        recovered.setdefault(
            "conflict_summary",
            "The structured response was truncated after the completed decision fields; "
            "the decision was recovered without changing the safety gate.",
        )

        try:
            decision = ArbitrationDecision.model_validate(recovered)
        except ValidationError:
            print(
                "[LLM OUTPUT WARNING] Partial JSON did not contain a complete valid "
                "decision; retaining the hybrid decision safely."
            )
            return _fail_closed_decision("partial JSON lacked a complete valid decision")

        print(
            "[LLM OUTPUT WARNING] Truncated structured JSON detected; recovered the "
            "completed decision fields with partial-JSON parsing."
        )
        return decision


@dataclass
class RuntimeInfo:
    execution_mode: str
    requested_gpu_layers: int | None
    total_transformer_layers: int
    wall_seconds: float
    pre_gpu_c: int | None
    max_gpu_c: int | None
    post_gpu_c: int | None
    pre_gpu_memory_mb: int | None
    max_gpu_memory_mb: int | None
    post_gpu_memory_mb: int | None
    peak_gpu_memory_increment_mb: int | None
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GatedDecision:
    final_label: int
    override_accepted: bool
    gate_reason: str


def _gpu_stats() -> tuple[int, int]:
    """Return (max temperature C, max used-memory MB) across visible GPUs."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        rows.append((int(float(parts[0])), int(float(parts[1]))))

    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU telemetry.")

    return max(temp for temp, _ in rows), max(mem for _, mem in rows)


def _safe_gpu_stats() -> tuple[int | None, int | None]:
    if not (GPU_MONITOR_ENABLED or THERMAL_GUARD_ENABLED):
        return None, None
    try:
        return _gpu_stats()
    except Exception:
        if THERMAL_GUARD_ENABLED:
            raise
        return None, None


def _unload_model() -> None:
    try:
        subprocess.run(
            ["ollama", "stop", MODEL_NAME],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        pass


def shutdown_model() -> None:
    """Explicitly release the resident Ollama model."""
    _unload_model()


# Always release the model when the Python process exits normally.
atexit.register(shutdown_model)


def wait_until_cool() -> None:
    if not THERMAL_GUARD_ENABLED:
        return

    _unload_model()
    while True:
        temp, _memory = _gpu_stats()
        if temp <= RESUME_TEMP_C:
            print(f"[THERMAL] GPU={temp}C <= {RESUME_TEMP_C}C. Ready.")
            return
        print(f"[THERMAL] GPU={temp}C; waiting for <= {RESUME_TEMP_C}C.")
        time.sleep(COOLDOWN_POLL_SECONDS)


class _GpuWatchdog:
    def __init__(self):
        self.triggered = False
        self.max_temp: int | None = None
        self.max_memory_mb: int | None = None
        self._stop = threading.Event()
        self._thread = None

    def _record(self) -> None:
        temp, memory = _gpu_stats()
        self.max_temp = (
            temp if self.max_temp is None else max(self.max_temp, temp)
        )
        self.max_memory_mb = (
            memory
            if self.max_memory_mb is None
            else max(self.max_memory_mb, memory)
        )
        if THERMAL_GUARD_ENABLED and temp >= HARD_ABORT_TEMP_C:
            print(
                f"\n[THERMAL ABORT] GPU reached {temp}C "
                f"(limit={HARD_ABORT_TEMP_C}C)."
            )
            self.triggered = True
            _unload_model()

    def _run(self):
        while not self._stop.wait(WATCHDOG_POLL_SECONDS):
            try:
                self._record()
            except Exception:
                if THERMAL_GUARD_ENABLED:
                    self.triggered = True
                    _unload_model()
                return
            if self.triggered:
                return

    def __enter__(self):
        if GPU_MONITOR_ENABLED or THERMAL_GUARD_ENABLED:
            try:
                self._record()
            except Exception:
                if THERMAL_GUARD_ENABLED:
                    raise
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return False


def infer_one(
    payload: dict,
    execution_mode: Literal["split50", "gpu", "cpu"] = "split50",
    prompt_variant: str = "baseline",
) -> tuple[ArbitrationDecision, RuntimeInfo]:
    """Run one arbitration request using the requested CPU/GPU offload mode.

    split50 is the default thermal-safe mode: Qwen3-8B has 36 transformer
    layers, and Ollama is asked to offload 18 of them to the GPU while the
    remaining transformer layers execute on CPU. Generation parameters remain
    identical across execution modes.
    """
    if execution_mode not in {"split50", "gpu", "cpu"}:
        raise ValueError(
            "execution_mode must be 'split50', 'gpu', or 'cpu'; "
            f"got {execution_mode!r}"
        )

    uses_gpu = execution_mode in {"split50", "gpu"}

    if uses_gpu and THERMAL_GUARD_ENABLED:
        temp, _memory = _gpu_stats()
        if temp > RESUME_TEMP_C:
            wait_until_cool()
        temp, _memory = _gpu_stats()
        if temp >= PRECALL_LIMIT_C:
            wait_until_cool()

    if uses_gpu:
        pre_temp, pre_memory = _safe_gpu_stats()
    else:
        pre_temp, pre_memory = None, None

    request_options = dict(OLLAMA_OPTIONS)
    requested_gpu_layers: int | None
    if execution_mode == "split50":
        requested_gpu_layers = int(HALF_GPU_OFFLOAD_LAYERS)
        request_options["num_gpu"] = requested_gpu_layers
    elif execution_mode == "cpu":
        requested_gpu_layers = 0
        request_options["num_gpu"] = 0
    else:
        # Preserve Ollama's normal automatic/full GPU offload behavior.
        requested_gpu_layers = None

    system_prompt, _user_template = get_prompt_pair(prompt_variant)

    started = time.perf_counter()

    try:
        if uses_gpu:
            watchdog = _GpuWatchdog()
            with watchdog:
                response = chat(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_user_prompt(payload, prompt_variant=prompt_variant)},
                    ],
                    format=ArbitrationDecision.model_json_schema(),
                    think=THINK,
                    options=request_options,
                    keep_alive=KEEP_ALIVE,
                )

            if watchdog.triggered:
                raise RuntimeError(
                    "GPU thermal watchdog triggered; Qwen was unloaded."
                )

            post_temp, post_memory = _safe_gpu_stats()
            max_temp = watchdog.max_temp
            max_memory = watchdog.max_memory_mb
        else:
            response = chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(payload, prompt_variant=prompt_variant)},
                ],
                format=ArbitrationDecision.model_json_schema(),
                think=THINK,
                options=request_options,
                keep_alive=KEEP_ALIVE,
            )
            post_temp, post_memory = None, None
            max_temp, max_memory = None, None

        decision = _parse_arbitration_decision(response.message.content)
        elapsed = time.perf_counter() - started

        peak_increment = None
        if max_memory is not None and pre_memory is not None:
            peak_increment = max(0, int(max_memory) - int(pre_memory))

        return decision, RuntimeInfo(
            execution_mode=execution_mode,
            requested_gpu_layers=requested_gpu_layers,
            total_transformer_layers=int(QWEN_TRANSFORMER_LAYERS),
            wall_seconds=float(elapsed),
            pre_gpu_c=pre_temp,
            max_gpu_c=max_temp,
            post_gpu_c=post_temp,
            pre_gpu_memory_mb=pre_memory,
            max_gpu_memory_mb=max_memory,
            post_gpu_memory_mb=post_memory,
            peak_gpu_memory_increment_mb=peak_increment,
            prompt_tokens=int(response.prompt_eval_count or 0),
            completion_tokens=int(response.eval_count or 0),
        )
    finally:
        # Fast resident mode avoids reloading ~5.2 GB of model weights for every
        # case. The thermal guard still unloads Qwen immediately if cooling is
        # required, and an atexit handler releases it when the run finishes.
        if not KEEP_MODEL_RESIDENT:
            _unload_model()


def apply_safety_gate(
    hybrid_label: int,
    decision: ArbitrationDecision,
    lstm_strength: str,
    litemv_strength: str,
    rf_strength: str,
) -> GatedDecision:
    hybrid_label = int(hybrid_label)

    if decision.recommended_action == "KEEP_HYBRID":
        return GatedDecision(
            final_label=hybrid_label,
            override_accepted=False,
            gate_reason="LLM_KEEP_HYBRID",
        )

    requested_label = (
        1 if decision.recommended_action == "OVERRIDE_TO_C2" else 0
    )

    if requested_label == hybrid_label:
        return GatedDecision(
            final_label=hybrid_label,
            override_accepted=False,
            gate_reason="LLM_ACTION_MATCHES_HYBRID",
        )

    if not ALLOW_AUTOMATIC_OVERRIDES:
        return GatedDecision(
            final_label=hybrid_label,
            override_accepted=False,
            gate_reason="AUTOMATIC_OVERRIDES_DISABLED",
        )

    if (
        REQUIRE_HIGH_CONFIDENCE_FOR_OVERRIDE
        and decision.confidence != "HIGH"
    ):
        return GatedDecision(
            final_label=hybrid_label,
            override_accepted=False,
            gate_reason="REJECTED_NOT_HIGH_CONFIDENCE",
        )

    if REQUIRE_DIRECTIONAL_SPECIFICITY:
        expected = (
            "SUPPORTS_C2" if requested_label == 1 else "CONTRADICTS_C2"
        )
        if decision.target_specificity != expected:
            return GatedDecision(
                final_label=hybrid_label,
                override_accepted=False,
                gate_reason="REJECTED_SPECIFICITY_MISMATCH",
            )

    if (
        REQUIRE_AT_LEAST_ONE_WEAK_UPSTREAM_MODEL
        and "WEAK"
        not in {str(lstm_strength), str(litemv_strength), str(rf_strength)}
    ):
        return GatedDecision(
            final_label=hybrid_label,
            override_accepted=False,
            gate_reason="REJECTED_NO_WEAK_UPSTREAM_MODEL",
        )

    return GatedDecision(
        final_label=requested_label,
        override_accepted=True,
        gate_reason="STRICT_GATE_ACCEPTED_OVERRIDE",
    )
