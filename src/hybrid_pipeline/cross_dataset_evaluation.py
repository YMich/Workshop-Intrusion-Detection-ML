from __future__ import annotations

"""
Strict cross-dataset evaluation of the FINAL fixed hybrid cascade.

Architecture:
    LSTM -> LITEMV -> RF

Directions:
    dataset2 -> dataset3
    dataset3 -> dataset2

Protocol for source A -> target B
---------------------------------
SOURCE A ONLY:
    - source-aware TRAIN split fits/reconstructs LSTM/LITEMV preprocessing
    - trained source LSTM/LITEMV model artifacts are loaded
    - source component thresholds are used
    - RF arbiter is trained on source TRAIN only
    - hybrid routing thresholds are calibrated on source VALIDATION only

TARGET B:
    - the ENTIRE target dataset is evaluation-only
    - no target preprocessing fit
    - no target model fitting
    - no target threshold/routing calibration
    - no target adaptation of any kind
    - target labels are used only to compute final metrics

This matches the existing individual-model cross-dataset convention where
a source-trained model is applied directly to the full target dataset.

Outputs
-------
results/hybrid_pipeline/cross_dataset/
    hybrid_cross_dataset_summary.csv
    cross_dataset_component_comparison.csv
    dataset2_to_dataset3/
        source_validation_rule_search.csv
        source_hybrid_rule.json
        cross_dataset_metrics.csv
        cross_dataset_component_metrics.csv
        cross_dataset_predictions.csv
        cross_dataset_by_source.csv
    dataset3_to_dataset2/
        ...

Run
---
    python cross_dataset_evaluation.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import hybrid_pipeline as hp


DIRECTIONS = (
    ("dataset2", "dataset3"),
    ("dataset3", "dataset2"),
)

RESULT_ROOT = (
    hp.PROJECT_ROOT
    / "results"
    / "hybrid_pipeline"
    / "cross_dataset"
)

ARTIFACT_ROOT = (
    hp.PROJECT_ROOT
    / "artifacts"
    / "hybrid_pipeline"
    / "cross_dataset"
)


# =============================================================================
# HELPERS
# =============================================================================


def _component_record(
    model_name: str,
    source_dataset: str,
    target_dataset: str,
    y_true: np.ndarray,
    prediction: np.ndarray,
    evaluation_rows: int,
    evaluation_sources: int,
) -> dict:
    metrics = hp.metrics_from_predictions(
        y_true,
        prediction,
    )

    return {
        "Model": model_name,
        "SourceDataset": source_dataset,
        "TargetDataset": target_dataset,
        "EvaluationRows": int(evaluation_rows),
        "EvaluationSourceFiles": int(evaluation_sources),
        "TargetUsedForTraining": False,
        "TargetUsedForPreprocessingFit": False,
        "TargetUsedForThresholdCalibration": False,
        "TargetUsedForHybridRoutingCalibration": False,
        "TargetAdaptation": False,
        **metrics,
    }


def _source_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for source_file, group in predictions.groupby(
        hp.SOURCE_FILE_COL,
        dropna=False,
        sort=True,
    ):
        y = (
            group[hp.LABEL_COL]
            .astype(int)
            .to_numpy()
        )

        hybrid = (
            group["Hybrid_Predicted_Label"]
            .astype(int)
            .to_numpy()
        )

        rows.append(
            {
                hp.SOURCE_FILE_COL: source_file,
                "Rows": int(len(group)),
                "ActualLabel": (
                    int(y[0])
                    if len(set(y.tolist())) == 1
                    else -1
                ),
                "PredictedMalicious": int(
                    (hybrid == 1).sum()
                ),
                "Errors": int(
                    (hybrid != y).sum()
                ),
                "MeanLSTMProbability": float(
                    group[
                        "LSTM_Probability"
                    ].mean()
                ),
                "MeanLITEMVProbability": float(
                    group[
                        "LITEMV_Probability"
                    ].mean()
                ),
                "MeanRFProbability": float(
                    group[
                        "RF_Probability"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def _save_confusion(
    y_true: np.ndarray,
    prediction: np.ndarray,
    path: Path,
) -> None:
    hp.save_confusion_matrix_csv(
        y_true=y_true,
        predictions=prediction,
        path=path,
    )


# =============================================================================
# STRICT TRANSFER
# =============================================================================


def evaluate_transfer(
    source_dataset: str,
    target_dataset: str,
) -> tuple[dict, list[dict]]:
    print("\n" + "=" * 110)
    print(
        f"STRICT HYBRID CROSS-DATASET TRANSFER | "
        f"{source_dataset.upper()} -> {target_dataset.upper()}"
    )
    print("=" * 110)

    output_dir = (
        RESULT_ROOT
        / f"{source_dataset}_to_{target_dataset}"
    )
    artifact_dir = (
        ARTIFACT_ROOT
        / f"{source_dataset}_to_{target_dataset}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. SOURCE-ONLY train / validation data
    # -------------------------------------------------------------------------
    source_splits, _manifest, source_split_metadata = (
        hp.load_shared_splits(source_dataset)
    )

    source_train = source_splits["train"]
    source_validation = source_splits["validation"]

    print(
        f"Source rows | train={len(source_train):,} | "
        f"validation={len(source_validation):,} | "
        f"split_mode={source_split_metadata.get('mode')}"
    )

    # -------------------------------------------------------------------------
    # 2. Build/load every component using SOURCE ONLY
    # -------------------------------------------------------------------------
    print("Loading source LSTM...")
    lstm_bundle = hp.load_lstm_bundle(
        source_dataset,
        source_train,
    )

    print("Loading source LITEMV...")
    litemv_bundle = hp.load_litemv_bundle(
        source_dataset,
        source_train,
    )

    print("Training RF arbiter on source TRAIN only...")
    rf_bundle = hp.train_rf_arbiter(
        source_dataset,
        source_train,
    )

    # -------------------------------------------------------------------------
    # 3. SOURCE VALIDATION ONLY: calibrate fixed hybrid routing rule
    # -------------------------------------------------------------------------
    print("Calibrating hybrid routing rule on source VALIDATION only...")

    source_validation_scores = hp.align_outputs(
        hp.score_lstm_split(
            source_dataset,
            "cross_source_validation",
            source_validation,
            lstm_bundle,
        ),
        hp.score_litemv_split(
            source_dataset,
            "cross_source_validation",
            source_validation,
            litemv_bundle,
        ),
        hp.score_rf_split(
            source_dataset,
            "cross_source_validation",
            source_validation,
            rf_bundle,
        ),
    )

    (
        rule_search,
        selected_rule,
        source_lstm_validation_metrics,
    ) = hp.search_hybrid_rule(
        validation=source_validation_scores,
        lstm_threshold=lstm_bundle["threshold"],
        litemv_threshold=litemv_bundle["threshold"],
        rf_threshold=rf_bundle["threshold"],
    )

    rule_search.to_csv(
        output_dir
        / "source_validation_rule_search.csv",
        index=False,
    )

    selected_rule.update(
        {
            "SourceDataset": source_dataset,
            "TargetDataset": target_dataset,
            "SourceSplitMode": (
                source_split_metadata.get("mode")
            ),
            "LSTMThreshold": float(
                lstm_bundle["threshold"]
            ),
            "LITEMVStandaloneThreshold": float(
                litemv_bundle["threshold"]
            ),
            "RFStandaloneThreshold": float(
                rf_bundle["threshold"]
            ),
            "SourceLSTMValidationMetrics": (
                source_lstm_validation_metrics
            ),
            "TargetUsedForTraining": False,
            "TargetUsedForPreprocessingFit": False,
            "TargetUsedForThresholdCalibration": False,
            "TargetUsedForHybridRoutingCalibration": False,
            "TargetAdaptation": False,
            "TransferProtocol": (
                "strict_source_only_training_validation_calibration_"
                "full_target_evaluation"
            ),
        }
    )

    with (
        artifact_dir
        / "source_hybrid_rule.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            hp.json_safe(selected_rule),
            handle,
            indent=2,
        )

    print(
        "Frozen source rule | "
        f"LSTM threshold={selected_rule['LSTMThreshold']:.6f} | "
        f"positive cutoff="
        f"{selected_rule['positive_filter_cutoff']:.6f} | "
        f"LITEMV confirm="
        f"{selected_rule['litemv_confirm_threshold']:.6f} | "
        f"RF confirm="
        f"{selected_rule['rf_confirm_threshold']:.6f}"
    )

    # -------------------------------------------------------------------------
    # 4. Load ENTIRE target dataset. From here onward it is evaluation only.
    # -------------------------------------------------------------------------
    target_path = Path(
        hp.lstm.DATASET_PATHS[target_dataset]
    )

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Missing target dataset: {target_path}"
        )

    target_df = pd.read_csv(
        target_path,
        low_memory=False,
    )

    print(
        f"Target evaluation population | "
        f"rows={len(target_df):,} | "
        f"SourceFiles="
        f"{target_df[hp.SOURCE_FILE_COL].nunique():,}"
    )
    print(
        "NO target fitting / preprocessing fit / threshold calibration / "
        "hybrid calibration."
    )

    # -------------------------------------------------------------------------
    # 5. Apply source-trained components directly to FULL target
    # -------------------------------------------------------------------------
    target_scores = hp.align_outputs(
        hp.score_lstm_split(
            target_dataset,
            f"cross_{source_dataset}_to_{target_dataset}",
            target_df,
            lstm_bundle,
        ),
        hp.score_litemv_split(
            target_dataset,
            f"cross_{source_dataset}_to_{target_dataset}",
            target_df,
            litemv_bundle,
        ),
        hp.score_rf_split(
            target_dataset,
            f"cross_{source_dataset}_to_{target_dataset}",
            target_df,
            rf_bundle,
        ),
    )

    if len(target_scores) != len(target_df):
        raise RuntimeError(
            f"{source_dataset}->{target_dataset}: expected one "
            f"decision per target flow ({len(target_df):,}); "
            f"got {len(target_scores):,}."
        )

    y_target = (
        target_scores[hp.LABEL_COL]
        .astype(int)
        .to_numpy()
    )

    lp = target_scores[
        "LSTM_Probability"
    ].to_numpy(dtype=float)

    vp = target_scores[
        "LITEMV_Probability"
    ].to_numpy(dtype=float)

    rp = target_scores[
        "RF_Probability"
    ].to_numpy(dtype=float)

    # Component decisions use SOURCE thresholds only.
    lstm_prediction = (
        lp >= float(lstm_bundle["threshold"])
    ).astype(int)

    litemv_prediction = (
        vp >= float(litemv_bundle["threshold"])
    ).astype(int)

    rf_prediction = (
        rp >= float(rf_bundle["threshold"])
    ).astype(int)

    # Hybrid routing thresholds are frozen SOURCE-validation values.
    applied = hp.apply_hybrid_cascade(
        lstm_probability=lp,
        litemv_probability=vp,
        rf_probability=rp,
        lstm_threshold=float(
            selected_rule["LSTMThreshold"]
        ),
        positive_filter_cutoff=float(
            selected_rule[
                "positive_filter_cutoff"
            ]
        ),
        litemv_confirm_threshold=float(
            selected_rule[
                "litemv_confirm_threshold"
            ]
        ),
        rf_confirm_threshold=float(
            selected_rule[
                "rf_confirm_threshold"
            ]
        ),
        litemv_rescue_threshold=float(
            selected_rule[
                "litemv_rescue_threshold"
            ]
        ),
        rf_rescue_threshold=float(
            selected_rule[
                "rf_rescue_threshold"
            ]
        ),
    )

    hybrid_prediction = applied[
        "prediction"
    ].astype(int)

    # -------------------------------------------------------------------------
    # 6. Metrics — target labels used ONLY here
    # -------------------------------------------------------------------------
    component_rows = [
        _component_record(
            "LSTM",
            source_dataset,
            target_dataset,
            y_target,
            lstm_prediction,
            len(target_scores),
            target_df[
                hp.SOURCE_FILE_COL
            ].nunique(),
        ),
        _component_record(
            "LITEMV",
            source_dataset,
            target_dataset,
            y_target,
            litemv_prediction,
            len(target_scores),
            target_df[
                hp.SOURCE_FILE_COL
            ].nunique(),
        ),
        _component_record(
            "RF",
            source_dataset,
            target_dataset,
            y_target,
            rf_prediction,
            len(target_scores),
            target_df[
                hp.SOURCE_FILE_COL
            ].nunique(),
        ),
        _component_record(
            "LSTM_LITEMV_RF",
            source_dataset,
            target_dataset,
            y_target,
            hybrid_prediction,
            len(target_scores),
            target_df[
                hp.SOURCE_FILE_COL
            ].nunique(),
        ),
    ]

    component_metrics = pd.DataFrame(
        component_rows
    )

    component_metrics.to_csv(
        output_dir
        / "cross_dataset_component_metrics.csv",
        index=False,
    )

    hybrid_result = dict(
        component_metrics.loc[
            component_metrics["Model"]
            == "LSTM_LITEMV_RF"
        ]
        .iloc[0]
        .to_dict()
    )

    pd.DataFrame(
        [hybrid_result]
    ).to_csv(
        output_dir
        / "cross_dataset_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 7. Flow-level evidence
    # -------------------------------------------------------------------------
    prediction_table = target_scores.copy()

    prediction_table[
        "LSTM_Source_Threshold"
    ] = float(lstm_bundle["threshold"])

    prediction_table[
        "LITEMV_Source_Threshold"
    ] = float(litemv_bundle["threshold"])

    prediction_table[
        "RF_Source_Threshold"
    ] = float(rf_bundle["threshold"])

    prediction_table[
        "LSTM_Predicted_Label"
    ] = lstm_prediction

    prediction_table[
        "LITEMV_Predicted_Label"
    ] = litemv_prediction

    prediction_table[
        "RF_Predicted_Label"
    ] = rf_prediction

    prediction_table[
        "Hybrid_Predicted_Label"
    ] = hybrid_prediction

    prediction_table[
        "Hybrid_DecisionRoute"
    ] = hp.route_labels(applied)

    prediction_table[
        "TransferSourceDataset"
    ] = source_dataset

    prediction_table[
        "TransferTargetDataset"
    ] = target_dataset

    prediction_table.to_csv(
        output_dir
        / "cross_dataset_predictions.csv",
        index=False,
    )

    _source_summary(
        prediction_table
    ).to_csv(
        output_dir
        / "cross_dataset_by_source.csv",
        index=False,
    )

    _save_confusion(
        y_true=y_target,
        prediction=hybrid_prediction,
        path=(
            output_dir
            / "cross_dataset_confusion_matrix.csv"
        ),
    )

    # -------------------------------------------------------------------------
    # 8. Transfer-specific comparison against source-trained LSTM
    # -------------------------------------------------------------------------
    lstm_metrics = hp.metrics_from_predictions(
        y_target,
        lstm_prediction,
    )
    hybrid_metrics = hp.metrics_from_predictions(
        y_target,
        hybrid_prediction,
    )

    delta = {
        "SourceDataset": source_dataset,
        "TargetDataset": target_dataset,
        "LSTM_F1": float(
            lstm_metrics["f1"]
        ),
        "Hybrid_F1": float(
            hybrid_metrics["f1"]
        ),
        "Delta_F1": float(
            hybrid_metrics["f1"]
            - lstm_metrics["f1"]
        ),
        "LSTM_FPR": float(
            lstm_metrics["fpr"]
        ),
        "Hybrid_FPR": float(
            hybrid_metrics["fpr"]
        ),
        "Delta_FPR": float(
            hybrid_metrics["fpr"]
            - lstm_metrics["fpr"]
        ),
        "LSTM_FNR": float(
            lstm_metrics["fnr"]
        ),
        "Hybrid_FNR": float(
            hybrid_metrics["fnr"]
        ),
        "Delta_FNR": float(
            hybrid_metrics["fnr"]
            - lstm_metrics["fnr"]
        ),
        "Corrected_LSTM_False_Positives": int(
            np.sum(
                (y_target == 0)
                & (lstm_prediction == 1)
                & (hybrid_prediction == 0)
            )
        ),
        "Introduced_New_False_Positives": int(
            np.sum(
                (y_target == 0)
                & (lstm_prediction == 0)
                & (hybrid_prediction == 1)
            )
        ),
        "Rescued_LSTM_False_Negatives": int(
            np.sum(
                (y_target == 1)
                & (lstm_prediction == 0)
                & (hybrid_prediction == 1)
            )
        ),
        "Introduced_New_False_Negatives": int(
            np.sum(
                (y_target == 1)
                & (lstm_prediction == 1)
                & (hybrid_prediction == 0)
            )
        ),
        "TotalChangedDecisions": int(
            np.sum(
                lstm_prediction
                != hybrid_prediction
            )
        ),
    }

    pd.DataFrame(
        [delta]
    ).to_csv(
        output_dir
        / "hybrid_vs_lstm_transfer.csv",
        index=False,
    )

    print("\nCross-dataset component results:")
    print(
        component_metrics[
            [
                "Model",
                "accuracy",
                "precision",
                "recall_tpr",
                "fpr",
                "fnr",
                "f1",
                "fp",
                "fn",
            ]
        ].to_string(index=False)
    )

    print(
        "\nHybrid vs source-trained LSTM transfer | "
        f"DeltaF1={delta['Delta_F1']:+.6f} | "
        f"DeltaFPR={delta['Delta_FPR']:+.6f} | "
        f"DeltaFNR={delta['Delta_FNR']:+.6f}"
    )

    return hybrid_result, component_rows


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    hybrid_rows = []
    component_rows = []

    for source_dataset, target_dataset in DIRECTIONS:
        hybrid_result, rows = evaluate_transfer(
            source_dataset,
            target_dataset,
        )
        hybrid_rows.append(hybrid_result)
        component_rows.extend(rows)

    hybrid_summary = pd.DataFrame(
        hybrid_rows
    )

    component_summary = pd.DataFrame(
        component_rows
    )

    hybrid_summary.to_csv(
        RESULT_ROOT
        / "hybrid_cross_dataset_summary.csv",
        index=False,
    )

    component_summary.to_csv(
        RESULT_ROOT
        / "cross_dataset_component_comparison.csv",
        index=False,
    )

    methodology = {
        "Directions": [
            "dataset2_to_dataset3",
            "dataset3_to_dataset2",
        ],
        "TargetPopulation": "entire_target_dataset",
        "Architecture": "LSTM -> LITEMV -> RF fixed gated cascade",
        "SourceTrainingOnly": True,
        "SourcePreprocessingOnly": True,
        "SourceComponentThresholdsOnly": True,
        "SourceHybridRoutingCalibrationOnly": True,
        "TargetUsedForTraining": False,
        "TargetUsedForPreprocessingFit": False,
        "TargetUsedForThresholdCalibration": False,
        "TargetUsedForHybridRoutingCalibration": False,
        "TargetAdaptation": False,
        "TargetLabelsUsedOnlyForMetrics": True,
        "ComparisonModelsProducedInSameRun": [
            "LSTM",
            "LITEMV",
            "RF",
            "LSTM_LITEMV_RF",
        ],
    }

    with (
        RESULT_ROOT
        / "cross_dataset_methodology.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            methodology,
            handle,
            indent=2,
        )

    print("\n" + "=" * 110)
    print("HYBRID CROSS-DATASET SUMMARY")
    print("=" * 110)
    print(
        hybrid_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 110)
    print("ALL COMPONENTS — SAME CROSS-DATASET POPULATIONS")
    print("=" * 110)
    print(
        component_summary[
            [
                "Model",
                "SourceDataset",
                "TargetDataset",
                "f1",
                "fpr",
                "fnr",
                "precision",
                "recall_tpr",
                "fp",
                "fn",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nResults:   {RESULT_ROOT}"
    )
    print(
        f"Artifacts: {ARTIFACT_ROOT}"
    )


if __name__ == "__main__":
    main()
