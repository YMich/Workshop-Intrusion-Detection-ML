from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Locate the Workshop_final root without relying on a fixed directory depth."""
    start = Path(start).resolve()
    for candidate in (start.parent, *start.parents):
        if (
            (candidate / "data" / "ingested").exists()
            and (candidate / "src").exists()
        ):
            return candidate
    raise RuntimeError(
        "Could not locate project root containing data/ingested and src/. "
        f"Started from: {start}"
    )


PROJECT_ROOT = _find_project_root(Path(__file__))
DATA_ROOT = PROJECT_ROOT / "data"
INGESTED_ROOT = DATA_ROOT / "ingested"

DATASET_PATHS = {
    "dataset1": INGESTED_ROOT / "dataset1_ingested.csv",
}
BENCHMARK_DATASETS = tuple(DATASET_PATHS.keys())

RESULT_ROOT = PROJECT_ROOT / "results" / "dataset1_appendix" / "models" / "RF"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset1_appendix" / "models" / "RF"

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
TIMESTAMP_COL = "Timestamp"
ROW_INDEX_COL = "Original_Row_Index"

# Final evaluation protocol.
OUTER_CV_FOLDS = 5
CV_SHUFFLE = True
CV_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.50
CLASS_WEIGHT_METHOD = "balanced"

# -----------------------------------------------------------------------------
# FINAL SHARED RF HYPERPARAMETERS
# -----------------------------------------------------------------------------
# Frozen final values from Dataset 2/3; applied unchanged to Dataset 1.
FINAL_RF_PARAMS = {
    "n_estimators": 256,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "n_jobs": -1,
    "verbose": 0,
    "warm_start": False,
    "ccp_alpha": 0.0,
    "max_samples": None,
}

# Exact final endpoint-aware RROLL5 representation.
BASE_10_FEATURES = [
    "Fwd IAT Mean",
    "Flow IAT Std",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Down/Up Ratio",
    "Fwd Act Data Pkts",
]

FWD_PACKET_LENGTH_CV = "Fwd Packet Length CV"
TOTAL_FWD_LENGTH = "Total Length of Fwd Packet"
BASE_12_FEATURES = BASE_10_FEATURES + [
    FWD_PACKET_LENGTH_CV,
    TOTAL_FWD_LENGTH,
]

ROLLING_BEHAVIOR_FEATURES = [
    "Fwd IAT Mean",
    "Flow IAT Std",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Down/Up Ratio",
    "Fwd Act Data Pkts",
]

INTERFLOW_DELTA = "InterFlow Delta"
HISTORY_COUNT = "Endpoint History Count"
ROLL_WINDOW = 5

SRC_IP_ALIASES = ("Src IP", "Source IP", "SrcIP", "SourceIP")
DST_IP_ALIASES = ("Dst IP", "Destination IP", "DstIP", "DestinationIP")
TIMESTAMP_ALIASES = (TIMESTAMP_COL, "TimeStamp", "Flow Timestamp")
TOTAL_FWD_LENGTH_ALIASES = (
    "Total Length of Fwd Packet",
    "Total Length of Fwd Packets",
    "TotLen Fwd Pkts",
)
