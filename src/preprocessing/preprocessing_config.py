from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# File location:
#   <PROJECT_ROOT>\src\preprocessing\preprocessing_config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"

INGESTED_ROOT = DATA_ROOT / "ingested"
PREPROCESSED_ROOT = DATA_ROOT / "preprocessed"

PREPROCESSING_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
)


# ============================================================
# INPUT DATASETS
# ============================================================

DATASET1_INGESTED_CSV = (
    INGESTED_ROOT
    / "dataset1_ingested.csv"
)

DATASET2_INGESTED_CSV = (
    INGESTED_ROOT
    / "dataset2_ingested.csv"
)

DATASET3_INGESTED_CSV = (
    INGESTED_ROOT
    / "dataset3_ingested.csv"
)


# ============================================================
# STANDALONE MILESTONE-2 OUTPUT DATASETS
# ============================================================

DATASET1_PREPROCESSED_CSV = (
    PREPROCESSED_ROOT
    / "dataset1_preprocessed.csv"
)

DATASET2_PREPROCESSED_CSV = (
    PREPROCESSED_ROOT
    / "dataset2_preprocessed.csv"
)

DATASET3_PREPROCESSED_CSV = (
    PREPROCESSED_ROOT
    / "dataset3_preprocessed.csv"
)


# ============================================================
# STANDALONE MILESTONE-2 PREPROCESSING ARTIFACTS
# ============================================================

DATASET1_PREPROCESSOR_ARTIFACT = (
    PREPROCESSING_ARTIFACT_ROOT
    / "dataset1_preprocessor.joblib"
)

DATASET2_PREPROCESSOR_ARTIFACT = (
    PREPROCESSING_ARTIFACT_ROOT
    / "dataset2_preprocessor.joblib"
)

DATASET3_PREPROCESSOR_ARTIFACT = (
    PREPROCESSING_ARTIFACT_ROOT
    / "dataset3_preprocessor.joblib"
)


# ============================================================
# PIPELINE COLUMNS
# ============================================================

LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"
SOURCE_FILE_COL = "SourceFile"

# These are retained unchanged for the later
# Feature Selection & Engineering stage.
METADATA_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "SourceFile",
]

PASSTHROUGH_COLUMNS = (
    METADATA_COLUMNS
    + [LABEL_COL]
)


# ============================================================
# NORMALIZATION SETTINGS
# ============================================================

# Shared numerical preprocessing used by both the standalone Milestone-2
# snapshots and the split-aware final model pipelines:
#
#   1. signed log1p:
#          sign(x) * ln(1 + |x|)
#
#   2. RobustScaler:
#          center = median
#          scale  = IQR
#
ROBUST_SCALER_WITH_CENTERING = True
ROBUST_SCALER_WITH_SCALING = True
ROBUST_SCALER_QUANTILE_RANGE = (
    25.0,
    75.0,
)
ROBUST_SCALER_UNIT_VARIANCE = False
