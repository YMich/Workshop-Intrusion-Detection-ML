from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from random_forest_config import ARTIFACT_ROOT, PREDICTION_THRESHOLD, ROW_INDEX_COL
from random_forest_preprocessing import FoldMedianImputer
from random_forest_temporal import build_temporal_features, final_feature_names


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply one of the fitted final RF models to an ingested CICFlowMeter CSV. "
            "Temporal features are rebuilt causally from the supplied CSV."
        )
    )
    parser.add_argument(
        "--model-dataset",
        choices=["dataset1"],
        required=True,
        help="Which fitted final model artifact to load.",
    )
    parser.add_argument("--input", required=True, help="Input ingested CSV.")
    parser.add_argument("--output", required=True, help="Output predictions CSV.")
    args = parser.parse_args()

    artifact_dir = ARTIFACT_ROOT / args.model_dataset
    model = joblib.load(artifact_dir / "random_forest_final.joblib")
    imputer = FoldMedianImputer.load(artifact_dir / "median_imputer_final.joblib")

    raw = pd.read_csv(args.input, low_memory=False)
    temporal = build_temporal_features(
        raw, dataset_name=Path(args.input).stem, require_label=False
    )
    features = final_feature_names()
    X = temporal[features].copy()
    for feature in features:
        X[feature] = pd.to_numeric(X[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    X = imputer.transform(X)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= PREDICTION_THRESHOLD).astype(int)

    out = pd.DataFrame(
        {
            ROW_INDEX_COL: temporal[ROW_INDEX_COL].to_numpy(dtype=np.int64),
            "Malicious_Probability": prob,
            "Predicted_Label": pred,
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
