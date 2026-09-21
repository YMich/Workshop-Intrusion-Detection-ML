"""Configuration for the canonical dataset-ingestion stage.

This module maps the aligned CICFlowMeter CSV directories produced by
``src/extraction`` to the finalized CSV files consumed by the ML pipeline.

No machine-specific absolute paths are used. ``PROJECT_ROOT`` is derived from
this file's location:

    <PROJECT_ROOT>/src/ingestion/ingestion_config.py
"""

from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
EXTRACTED_ROOT = DATA_ROOT / "extracted"
INGESTED_ROOT = DATA_ROOT / "ingested"


# =============================================================================
# EXTRACTED INPUT DIRECTORIES
# =============================================================================

# Dataset 1
DATASET1_BENIGN_DIR = (
    EXTRACTED_ROOT / "dataset1" / "benign_cicflowmeter_aligned"
)
DATASET1_MALICIOUS_DIR = (
    EXTRACTED_ROOT / "dataset1" / "malicious_cicflowmeter_https_aligned"
)

# Dataset 2
DATASET2_BENIGN_DIR = (
    EXTRACTED_ROOT / "dataset2" / "benign_cicflowmeter_https_aligned"
)
DATASET2_MALICIOUS_DIR = (
    EXTRACTED_ROOT / "dataset2" / "malicious_cicflowmeter_config_aligned"
)

# Dataset 3
DATASET3_BENIGN_DIR = (
    EXTRACTED_ROOT / "dataset3" / "benign_cicflowmeter_https_aligned"
)
DATASET3_MALICIOUS_DIR = (
    EXTRACTED_ROOT / "dataset3" / "malicious_cicflowmeter_config_aligned"
)


# =============================================================================
# INGESTED OUTPUT FILES
# =============================================================================

DATASET1_INGESTED_CSV = INGESTED_ROOT / "dataset1_ingested.csv"
DATASET2_INGESTED_CSV = INGESTED_ROOT / "dataset2_ingested.csv"
DATASET3_INGESTED_CSV = INGESTED_ROOT / "dataset3_ingested.csv"


# =============================================================================
# CANONICAL INGESTION COLUMNS
# =============================================================================

LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"
SOURCE_FILE_COL = "SourceFile"

# These fields are intentionally retained by ingestion because later stages may
# need them for source-aware splitting, temporal ordering, endpoint grouping, or
# provenance. They are not automatically predictive model features.
REQUIRED_IDENTIFIER_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    TIMESTAMP_COL,
]

# Extraction writes a Label column. Ingestion validates that the column exists
# but deliberately overwrites its contents from the authoritative benign /
# malicious directory assignment.
REQUIRED_COLUMNS = REQUIRED_IDENTIFIER_COLUMNS + [LABEL_COL]
