import os
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# Resolve the repository root from this file location:
#   <PROJECT_ROOT>/src/extraction/pipeline_config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# TOP-LEVEL PROJECT DIRECTORIES
# ============================================================

DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
METADATA_ROOT = DATA_ROOT / "metadata"
EXTRACTED_ROOT = DATA_ROOT / "extracted"
WORK_ROOT = PROJECT_ROOT / "work"


# ============================================================
# CICFLOWMETER DOCKER IMAGE
# ============================================================

# The original extraction environment used this local image ID.
# For another machine, set CICFLOWMETER_DOCKER_IMAGE to any installed
# equivalent image name/tag or image ID before running extraction.
CICFLOWMETER_DOCKER_IMAGE = os.environ.get(
    "CICFLOWMETER_DOCKER_IMAGE",
    "1453ede6f6e9",
)


# ============================================================
# RAW INPUT DIRECTORIES
# These folders contain the authoritative source PCAP/PCAPNG files.
# ============================================================

DATASET1_BENIGN_INPUT_DIR = RAW_ROOT / "dataset1" / "benign"
DATASET1_MALICIOUS_INPUT_DIR = RAW_ROOT / "dataset1" / "malicious"

DATASET2_BENIGN_INPUT_DIR = RAW_ROOT / "dataset2" / "benign"
DATASET2_MALICIOUS_INPUT_DIR = RAW_ROOT / "dataset2" / "malicious"

DATASET3_BENIGN_INPUT_DIR = RAW_ROOT / "dataset3" / "benign"
DATASET3_MALICIOUS_INPUT_DIR = RAW_ROOT / "dataset3" / "malicious"

# CICIDS2017 Friday belongs to Dataset 1 benign.
# It lives in the same raw benign folder as the rest of Dataset 1.
CICIDS2017_FRIDAY_INPUT_CSV = DATASET1_BENIGN_INPUT_DIR / "friday.csv"


# ============================================================
# METADATA / AUXILIARY INPUTS
# ============================================================

MTA_CONFIG_JSON = METADATA_ROOT / "mta_config.json"
HIKARI_ALLFLOWMETER_CSV = METADATA_ROOT / "ALLFLOWMETER_HIKARI2021.csv"


# ============================================================
# CICFLOWMETER TEMPORARY WORK DIRECTORIES
# These are generated/disposable. They are NOT final datasets.
# ============================================================

RUN_ROOT = WORK_ROOT / "cicflowmeter_run"
CIC_INPUT_DIR = RUN_ROOT / "input"
CIC_OUTPUT_DIR = RUN_ROOT / "output"

PARALLEL_ROOT = WORK_ROOT / "cicflowmeter_parallel_jobs"
DATASET2_BENIGN_PARALLEL_CIC_ROOT = PARALLEL_ROOT / "dataset2_benign"
DATASET2_MALICIOUS_PARALLEL_CIC_ROOT = PARALLEL_ROOT / "dataset2_malicious"
DATASET3_BENIGN_PARALLEL_CIC_ROOT = PARALLEL_ROOT / "dataset3_benign"
DATASET3_MALICIOUS_PARALLEL_CIC_ROOT = PARALLEL_ROOT / "dataset3_malicious"


# ============================================================
# FINAL CICFLOWMETER CSV OUTPUT DIRECTORIES
# ============================================================

DATASET1_BENIGN_OUTPUT_DIR = (
    EXTRACTED_ROOT / "dataset1" / "benign_cicflowmeter_aligned"
)

DATASET1_MALICIOUS_OUTPUT_DIR = (
    EXTRACTED_ROOT / "dataset1" / "malicious_cicflowmeter_https_aligned"
)

DATASET2_BENIGN_OUTPUT_DIR = (
    EXTRACTED_ROOT / "dataset2" / "benign_cicflowmeter_https_aligned"
)

DATASET2_MALICIOUS_OUTPUT_DIR = (
    EXTRACTED_ROOT / "dataset2" / "malicious_cicflowmeter_config_aligned"
)

DATASET3_BENIGN_OUTPUT_DIR = (
    EXTRACTED_ROOT / "dataset3" / "benign_cicflowmeter_https_aligned"
)

DATASET3_MALICIOUS_OUTPUT_DIR = (
    EXTRACTED_ROOT / "dataset3" / "malicious_cicflowmeter_config_aligned"
)



# CICIDS2017 Friday is filtered/normalized into Dataset 1 benign output.
# This makes it a normal Dataset 1 benign CSV for later stages.
CICIDS2017_FRIDAY_OUTPUT_CSV = (
    DATASET1_BENIGN_OUTPUT_DIR / "friday_benign_https_aligned.csv"
)

# Dataset 1 benign defines the canonical final CSV schema used by the
# other five extraction scripts. run_all_extractions.py runs it first.
REFERENCE_SCHEMA_DIR = DATASET1_BENIGN_OUTPUT_DIR



