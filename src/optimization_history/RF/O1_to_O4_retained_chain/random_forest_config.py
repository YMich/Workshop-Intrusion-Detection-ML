from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Expected location:
#   <PROJECT_ROOT>/src/models_optimized/RF/random_forest_config.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
INGESTED_ROOT = DATA_ROOT / "ingested"

RESULT_ROOT = PROJECT_ROOT / "results" / "models_optimized" / "RF" / "static_chain"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models_optimized" / "RF" / "static_chain"

# ============================================================================
# INPUT DATASETS
# ============================================================================

DATASET_PATHS = {
    "dataset2": INGESTED_ROOT / "dataset2_ingested.csv",
    "dataset3": INGESTED_ROOT / "dataset3_ingested.csv",
}
BENCHMARK_DATASETS = tuple(DATASET_PATHS.keys())

# ============================================================================
# COLUMNS / CV
# ============================================================================

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
TIMESTAMP_COL = "Timestamp"
ROW_INDEX_COL = "Original_Row_Index"

OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 3
CV_SHUFFLE = True
CV_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.50

# ============================================================================
# B0 / O1 / O2 FIXED BASELINE RF
# ============================================================================

# This is the final shared RF configuration reported for the original
# 57-feature baseline. It is deliberately frozen for B0, O1 and O2 so that
# 57 -> 10 -> 12 changes only the feature representation.
BASE_RF_PARAMS = {
    "n_estimators": 128,
    "criterion": "gini",
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.5,
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

# Compatibility values used by helper functions in random_forest_utils.py.
FOCUSED_N_ESTIMATORS = [128, 256]
FOCUSED_MAX_FEATURES = ["sqrt", 0.5]
CLASS_WEIGHT_METHOD = "balanced"
PRIMARY_SELECTION_METRIC = "pr_auc_mean_across_datasets"
