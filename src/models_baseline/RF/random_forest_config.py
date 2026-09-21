from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
INGESTED_ROOT = DATA_ROOT / "ingested"

MODEL_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models_baseline" / "RandomForest"
RESULT_ROOT = PROJECT_ROOT / "results" / "models_baseline" / "RandomForest"
SHARED_HYPERPARAMETER_PATH = MODEL_ARTIFACT_ROOT / "shared_configuration.json"

DATASET_PATHS = {
    "dataset2": INGESTED_ROOT / "dataset2_ingested.csv",
    "dataset3": INGESTED_ROOT / "dataset3_ingested.csv",
}
BENCHMARK_DATASETS = tuple(DATASET_PATHS.keys())

LABEL_COL = "Label"
SOURCE_FILE_COL = "SourceFile"
TIMESTAMP_COL = "Timestamp"
ROW_INDEX_COL = "Original_Row_Index"

# Group-aware OOF evaluation. SourceFile is the grouping variable.
OUTER_CV_FOLDS = 5
CV_SHUFFLE = True
CV_RANDOM_STATE = 42
CLASS_WEIGHT_METHOD = "balanced"

# Frozen shared baseline configuration for Dataset 2 and Dataset 3.
# No inner-CV hyperparameter search is performed.
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

MODEL_RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.50
