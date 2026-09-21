from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# File location:
#   <PROJECT_ROOT>\src\feature_engineering\feature_config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"

PREPROCESSED_ROOT = (
    DATA_ROOT / "preprocessed"
)

FEATURE_ROOT = (
    DATA_ROOT / "features"
)

FEATURE_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "feature_selection"
)


# ============================================================
# INPUT DATASETS
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
# OUTPUT DATASETS
# ============================================================

DATASET1_FEATURES_CSV = (
    FEATURE_ROOT
    / "dataset1_features.csv"
)

DATASET2_FEATURES_CSV = (
    FEATURE_ROOT
    / "dataset2_features.csv"
)

DATASET3_FEATURES_CSV = (
    FEATURE_ROOT
    / "dataset3_features.csv"
)


# ============================================================
# PIPELINE COLUMNS
# ============================================================

LABEL_COL = "Label"

TIMESTAMP_COL = "Timestamp"

SOURCE_FILE_COL = "SourceFile"

# Metadata/identifier columns are retained through Step 3 for ALL models.
# They are not predictive features and are not part of the 57-feature count.
#
# Model-specific handling happens later in Model Training.
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


# ============================================================
# FEATURE REMOVAL - REPORT SECTION 3.2
# ============================================================

# Raw non-predictive fields.
RAW_NON_PREDICTIVE_COLUMNS = [
    "ICMP Code",
    "ICMP Type",
]


# Reported constant columns.
#
# SourceType and SourceDataset existed in earlier intermediate data but are
# not expected in the new canonical extraction schema. They are dropped only
# if present.
REPORT_CONSTANT_COLUMNS = [
    "URG Flag Count",
    "Bwd URG Flags",
    "Fwd URG Flags",
    "SourceType",
    "SourceDataset",
]


# Final universal correlation/RMI drop list reported in Section 3.2.
CORRELATION_RMI_DROP_COLUMNS = [
    "Average Packet Size",
    "Bwd PSH Flags",
    "Bwd Packet/Bulk Avg",
    "Bwd Segment Size Avg",
    "Flow Duration",
    "Flow IAT Max",
    "Flow IAT Mean",
    "Flow Packets/s",
    "Fwd IAT Max",
    "Fwd IAT Std",
    "Fwd IAT Total",
    "Fwd RST Flags",
    "Fwd Segment Size Avg",
    "Idle Max",
    "Idle Mean",
    "Packet Length Mean",
    "Subflow Bwd Packets",
    "Total Bwd packets",
    "Total Fwd Packet",
]


# Final Milestone 2 predictive schema:
# 57 numeric predictive features + Label.
EXPECTED_FINAL_FEATURE_COUNT = 57
