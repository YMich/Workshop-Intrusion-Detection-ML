import subprocess
import sys
import time
from pathlib import Path

from pipeline_config import PROJECT_ROOT


# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Order matters:
# Dataset 1 benign MUST run first because it creates the
# reference schema used by the other PCAP extraction scripts.
# CICIDS2017 Friday belongs to Dataset 1 benign. It is prepared
# after the PCAP jobs so the aligned Friday CSV is available to the
# downstream ingestion stage together with the other extracted CSVs.
SCRIPTS = [
    "run_benign_cicflowmeter_aligned.py",
    "run_malicious_cicflowmeter_aligned.py",
    "run_dataset2_hikari_benign_https_aligned_parallel.py",
    "run_dataset2_malicious_config_aligned_parallel.py",
    "run_dataset3_benign_ctu_https_aligned_parallel.py",
    "run_dataset3_malicious_config_aligned_parallel.py",
    "prepare_cicids2017_friday_benign_https.py",
]

# These are imported by the runnable scripts and therefore must
# be present in the same directory, but they are NOT run directly.
REQUIRED_SHARED_MODULES = [
    "pipeline_config.py",
    "cicflowmeter_common.py",
    "config_based_malicious_common.py",
]


# ============================================================
# HELPERS
# ============================================================

def verify_files_exist() -> None:
    required = SCRIPTS + REQUIRED_SHARED_MODULES
    missing = [name for name in required if not (SCRIPT_DIR / name).is_file()]

    if missing:
        print("\n[ERROR] Missing required files:")
        for name in missing:
            print(f"  - {name}")
        print(f"\nExpected directory:\n{SCRIPT_DIR}")
        sys.exit(1)


def run_script(script_name: str, index: int, total: int) -> None:
    script_path = SCRIPT_DIR / script_name

    print("\n" + "=" * 90)
    print(f"[{index}/{total}] RUNNING: {script_name}")
    print("=" * 90)

    start = time.perf_counter()

    # sys.executable guarantees that every script uses the same
    # Python interpreter/environment as this runner.
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR),
    )

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print("\n" + "!" * 90)
        print(f"[FAILED] {script_name}")
        print(f"Exit code: {result.returncode}")
        print(f"Elapsed:   {elapsed:.1f} seconds")
        print("Stopping the pipeline so later datasets are not generated from an incomplete run.")
        print("!" * 90)
        sys.exit(result.returncode)

    print("\n" + "-" * 90)
    print(f"[OK] {script_name}")
    print(f"Elapsed: {elapsed:.1f} seconds")
    print("-" * 90)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    verify_files_exist()

    print("=" * 90)
    print("CICFLOWMETER FULL EXTRACTION PIPELINE")
    print("=" * 90)
    print(f"Python:       {sys.executable}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Script dir:   {SCRIPT_DIR}")
    print(f"Scripts:      {len(SCRIPTS)}")
    print()
    print("Execution order:")
    for i, name in enumerate(SCRIPTS, start=1):
        print(f"  {i}. {name}")

    pipeline_start = time.perf_counter()

    for i, script_name in enumerate(SCRIPTS, start=1):
        run_script(script_name, i, len(SCRIPTS))

    total_elapsed = time.perf_counter() - pipeline_start

    print("\n" + "=" * 90)
    print("ALL CICFLOWMETER EXTRACTION SCRIPTS COMPLETED SUCCESSFULLY")
    print("=" * 90)
    print(f"Total elapsed: {total_elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
