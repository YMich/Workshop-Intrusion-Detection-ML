from __future__ import annotations

"""
LITEMV O1 — environment-dependent feature hardening (57 -> 43).

This is the FIRST optimization after the 57-feature LITEMV baseline.

ONLY intended change from B0:
    predictive feature schema: 57 -> 43.

Everything else stays unchanged:
    - persisted source-aware split
    - train-fitted preprocessing
    - shared LITEMV hyperparameters for D2/D3
    - sequence length and strides
    - class weighting
    - early stopping
    - validation-only threshold calibration
    - untouched source test
    - strict cross-dataset transfer protocol

Accepted O1 hardening rule:
    Mandatory drops:
      FWD Init Win Bytes
      Bwd Init Win Bytes
      Total Connection Flow Time
      Fwd Header Length
      Bwd Header Length

    Also drop every selected feature whose normalized name contains "flag",
    EXCEPT "URG Flag Count".

The code asserts exactly 14 drops and exactly 43 retained features.

Cross-dataset analysis:
    B0 D2 -> D3 validation/test
    O1 D2 -> D3 validation/test
    B0 D3 -> D2 validation/test
    O1 D3 -> D2 validation/test

For cross transfer, source preprocessor/model/features/threshold are frozen.
No target fitting, recalibration, or adaptation is allowed.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from litemv import (
    ARTIFACT_ROOT as B0_ARTIFACT_ROOT,
    RESULT_ROOT as B0_RESULT_ROOT,
    BENCHMARK_DATASETS,
    EVAL_SEQUENCE_STRIDE,
    EXPECTED_FEATURE_COUNT,
    RANDOM_STATE,
    SHARED_PARAMS,
    SOURCE_FILE_COL,
    TRAIN_SEQUENCE_STRIDE,
    VALIDATION_FPR_CAP,
    _jsonable_params,
    _save_history,
    _save_model_summary,
    build_litemv_model,
    build_sequences,
    calculate_class_weights,
    calculate_metrics,
    calibrate_validation_threshold,
    fit_litemv,
    load_dataset_and_split,
    predict_probabilities,
    predictions_from_probabilities,
    set_random_seed,
    split_summary,
    transform_and_select,
)
from preprocessing import DatasetPreprocessor


# =============================================================================
# O1 CONTRACT
# =============================================================================

STAGE_B0 = "B0_57_BASELINE"
STAGE_O1 = "O1_43_HARDENED"

B0_FEATURE_COUNT = 57
O1_FEATURE_COUNT = 43

O1_MANDATORY_DROPS = {
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Total Connection Flow Time",
    "Fwd Header Length",
    "Bwd Header Length",
}
O1_ALLOWED_FLAG_FEATURE = "URG Flag Count"

O1_ARTIFACT_ROOT = (
    B0_ARTIFACT_ROOT / "optimization_o1_hardened_features"
)
O1_RESULT_ROOT = (
    B0_RESULT_ROOT / "optimization_o1_hardened_features"
)


# =============================================================================
# HELPERS
# =============================================================================

def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_keras_model(path: Path) -> keras.Model:
    try:
        return keras.models.load_model(
            path, compile=False, safe_mode=False
        )
    except TypeError:
        return keras.models.load_model(path, compile=False)


def normalize_feature_name(name: str) -> str:
    return " ".join(str(name).strip().split()).casefold()


def is_o1_flag_drop(name: str) -> bool:
    normalized = normalize_feature_name(name)
    return (
        "flag" in normalized
        and normalized
        != normalize_feature_name(O1_ALLOWED_FLAG_FEATURE)
    )


def resolve_o1_features(
    baseline_features: list[str],
) -> tuple[list[str], list[str]]:
    """Apply the finalized 57->43 hardening rule exactly."""
    baseline_features = list(baseline_features)

    if len(baseline_features) != B0_FEATURE_COUNT:
        raise ValueError(
            f"O1 requires the finalized {B0_FEATURE_COUNT}-feature "
            f"B0 schema; got {len(baseline_features)}."
        )

    missing = sorted(
        O1_MANDATORY_DROPS - set(baseline_features)
    )
    if missing:
        raise ValueError(
            "O1 mandatory hardening features are absent from B0: "
            f"{missing}"
        )

    dropped = [
        feature
        for feature in baseline_features
        if (
            feature in O1_MANDATORY_DROPS
            or is_o1_flag_drop(feature)
        )
    ]
    dropped_set = set(dropped)
    kept = [
        feature
        for feature in baseline_features
        if feature not in dropped_set
    ]

    if len(dropped) != B0_FEATURE_COUNT - O1_FEATURE_COUNT:
        raise RuntimeError(
            "O1 must implement exactly 57 -> 43. "
            f"Resolved {len(dropped)} drops: {dropped}"
        )

    if len(kept) != O1_FEATURE_COUNT:
        raise RuntimeError(
            f"O1 produced {len(kept)} features instead of 43."
        )

    # URG Flag Count is an allowed exception only IF it exists in the
    # finalized B0 schema. In the current project schema it may already have
    # been removed upstream as constant/uninformative, so O1 must not require
    # its presence.
    bad_flags = [
        feature
        for feature in kept
        if (
            "flag" in normalize_feature_name(feature)
            and normalize_feature_name(feature)
            != normalize_feature_name(O1_ALLOWED_FLAG_FEATURE)
        )
    ]
    if bad_flags:
        raise RuntimeError(
            "Non-approved flag features survived O1: "
            f"{bad_flags}"
        )

    return kept, dropped


def transform_to_b0(
    preprocessor: DatasetPreprocessor,
    raw_df: pd.DataFrame,
    dataset_name: str,
    expected_baseline_features: list[str] | None = None,
):
    selected, features, dropped, manifest = (
        transform_and_select(
            preprocessor,
            raw_df,
            dataset_name=dataset_name,
            expected_features=expected_baseline_features,
        )
    )
    if len(features) != B0_FEATURE_COUNT:
        raise RuntimeError(
            f"{dataset_name}: expected B0=57, got {len(features)}."
        )
    return selected, list(features), dropped, manifest


# =============================================================================
# O1 TRAINING
# =============================================================================

def train_o1_dataset(
    dataset_name: str,
    force_resplit: bool = False,
) -> dict:
    print("\n" + "=" * 100)
    print(f"LITEMV O1 57 -> 43 TRAINING - {dataset_name.upper()}")
    print("=" * 100)

    set_random_seed(RANDOM_STATE)
    tf.keras.backend.clear_session()

    splits, _manifest, split_metadata = load_dataset_and_split(
        dataset_name,
        force_resplit=force_resplit,
    )
    print(split_summary(splits).to_string(index=False))

    train_raw = splits["train"]
    val_raw = splits["validation"]

    # Same B0 preprocessing: fit TRAIN only.
    preprocessor = DatasetPreprocessor(dataset_name)
    preprocessor.fit(train_raw)

    train_b0, b0_features, step3_dropped, step3_manifest = (
        transform_to_b0(
            preprocessor,
            train_raw,
            dataset_name=f"{dataset_name}_litemv_o1_train",
        )
    )
    val_b0, val_b0_features, _, _ = transform_to_b0(
        preprocessor,
        val_raw,
        dataset_name=f"{dataset_name}_litemv_o1_validation",
        expected_baseline_features=b0_features,
    )
    if val_b0_features != b0_features:
        raise RuntimeError(
            "Validation B0 feature schema differs from training."
        )

    o1_features, o1_dropped = resolve_o1_features(
        b0_features
    )

    sequence_length = int(SHARED_PARAMS["sequence_length"])

    X_train, y_train, _train_meta = build_sequences(
        train_b0,
        o1_features,
        sequence_length,
        TRAIN_SEQUENCE_STRIDE,
    )
    X_val, y_val, val_meta = build_sequences(
        val_b0,
        o1_features,
        sequence_length,
        EVAL_SEQUENCE_STRIDE,
    )

    if len(y_val) != len(val_raw):
        raise RuntimeError(
            f"{dataset_name}: expected one validation prediction "
            f"per flow ({len(val_raw):,}); got {len(y_val):,}."
        )

    class_weights = calculate_class_weights(y_train)

    print(f"\nB0 features: {len(b0_features)}")
    print(f"O1 features: {len(o1_features)}")
    print(f"O1 drops ({len(o1_dropped)}):")
    for feature in o1_dropped:
        print(f"  - {feature}")
    print(
        f"Train sequences: {len(y_train):,} | {X_train.shape}"
    )
    print(
        f"Validation sequences: {len(y_val):,} | {X_val.shape}"
    )
    print(f"Class weights: {class_weights}")
    print(f"Frozen shared LITEMV params: {SHARED_PARAMS}")

    model = build_litemv_model(
        feature_count=len(o1_features),
        params=SHARED_PARAMS,
    )

    history, best_epoch = fit_litemv(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_val,
        y_validation=y_val,
        params=SHARED_PARAMS,
        class_weights=class_weights,
    )

    val_prob = predict_probabilities(
        model,
        X_val,
        batch_size=int(SHARED_PARAMS["batch_size"]),
    )

    # Important: threshold procedure stays exactly B0.
    calibration = calibrate_validation_threshold(
        y_val,
        val_prob,
    )
    threshold = float(calibration["threshold"])
    val_metrics = calculate_metrics(
        y_val,
        val_prob,
        threshold,
    )
    val_pred = predictions_from_probabilities(
        val_prob,
        threshold,
    )

    artifact_dir = O1_ARTIFACT_ROOT / dataset_name
    result_dir = O1_RESULT_ROOT / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    model.save(artifact_dir / "litemv_o1.keras")
    preprocessor.save(artifact_dir / "preprocessor.joblib")

    save_json(
        artifact_dir / "baseline_features.json",
        b0_features,
    )
    save_json(
        artifact_dir / "selected_features.json",
        o1_features,
    )
    save_json(
        artifact_dir / "o1_dropped_features.json",
        o1_dropped,
    )
    save_json(
        artifact_dir / "shared_hyperparameters.json",
        _jsonable_params(SHARED_PARAMS),
    )
    save_json(
        artifact_dir / "decision_rule.json",
        {
            "stage": STAGE_O1,
            "calibration_scope": "validation_only_per_dataset",
            "method": calibration["method"],
            "threshold": threshold,
            "validation_fpr_cap": VALIDATION_FPR_CAP,
            "test_used_for_threshold_selection": False,
            "cross_dataset_used_for_threshold_selection": False,
            "all_validation_candidates": calibration["all_candidates"],
        },
    )
    save_json(
        artifact_dir / "training_metadata.json",
        {
            "stage": STAGE_O1,
            "dataset": dataset_name,
            "split_mode": split_metadata.get("mode"),
            "baseline_feature_count": len(b0_features),
            "o1_feature_count": len(o1_features),
            "o1_drop_count": len(o1_dropped),
            "sequence_length": sequence_length,
            "train_sequence_stride": TRAIN_SEQUENCE_STRIDE,
            "eval_sequence_stride": EVAL_SEQUENCE_STRIDE,
            "best_epoch": int(best_epoch),
            "architecture": "LITEMV",
            "hyperparameter_scope": "shared_dataset2_dataset3",
            "only_intended_change_from_b0": (
                "57_to_43_environment_dependent_feature_hardening"
            ),
            "class_weights": {
                str(k): float(v)
                for k, v in class_weights.items()
            },
            "test_used_for_training_or_calibration": False,
        },
    )

    pd.DataFrame({"Feature": b0_features}).to_csv(
        artifact_dir / "baseline_features.csv",
        index=False,
    )
    pd.DataFrame({"Feature": o1_features}).to_csv(
        artifact_dir / "selected_features.csv",
        index=False,
    )
    pd.DataFrame(
        {"DroppedFeature": o1_dropped}
    ).to_csv(
        artifact_dir / "o1_dropped_features.csv",
        index=False,
    )
    step3_manifest.to_csv(
        artifact_dir / "step3_feature_manifest.csv",
        index=False,
    )
    pd.DataFrame(
        {"DroppedFeature": step3_dropped}
    ).to_csv(
        artifact_dir / "step3_dropped_features.csv",
        index=False,
    )
    split_summary(splits).to_csv(
        artifact_dir / "split_summary.csv",
        index=False,
    )

    _save_history(
        history,
        result_dir / "training_history.csv",
    )
    _save_model_summary(
        model,
        artifact_dir / "model_summary.txt",
    )

    val_table = val_meta.copy()
    val_table["MaliciousProbability"] = val_prob
    val_table["Predicted_Label"] = val_pred
    val_table["ValidationCalibratedThreshold"] = threshold
    val_table.to_csv(
        result_dir / "validation_predictions.csv",
        index=False,
    )
    pd.DataFrame([val_metrics]).to_csv(
        result_dir / "validation_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        calibration["all_candidates"]
    ).to_csv(
        result_dir / "threshold_candidates.csv",
        index=False,
    )

    print("\nO1 validation-only calibration:")
    print(
        f"  method={calibration['method']} | "
        f"threshold={threshold:.6f} | "
        f"Recall={val_metrics['recall_tpr']:.4f} | "
        f"FPR={val_metrics['fpr']:.4f} | "
        f"F2={val_metrics['f2']:.4f} | "
        f"PR-AUC={val_metrics['pr_auc']:.4f}"
    )
    print("  TEST has not been used.")

    del X_train, X_val
    tf.keras.backend.clear_session()

    return {
        "Stage": STAGE_O1,
        "Dataset": dataset_name,
        "FeatureCount": len(o1_features),
        "BestEpoch": int(best_epoch),
        "ThresholdMethod": calibration["method"],
        "Threshold": threshold,
        **val_metrics,
    }


# =============================================================================
# SOURCE BUNDLES
# =============================================================================

def load_b0_bundle(dataset_name: str) -> dict:
    artifact_dir = B0_ARTIFACT_ROOT / dataset_name
    model_path = artifact_dir / "litemv.keras"
    prep_path = artifact_dir / "preprocessor.joblib"
    features_path = artifact_dir / "selected_features.json"
    decision_path = artifact_dir / "decision_rule.json"

    for path in (
        model_path,
        prep_path,
        features_path,
        decision_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing B0 artifact: {path}"
            )

    features = list(load_json(features_path))
    if len(features) != B0_FEATURE_COUNT:
        raise RuntimeError(
            f"{dataset_name} B0 has {len(features)} features."
        )

    decision = load_json(decision_path)
    return {
        "stage": STAGE_B0,
        "model": load_keras_model(model_path),
        "preprocessor": DatasetPreprocessor.load(prep_path),
        "baseline_features": features,
        "model_features": features,
        "threshold": float(decision["threshold"]),
        "threshold_method": str(decision["method"]),
    }


def load_o1_bundle(dataset_name: str) -> dict:
    artifact_dir = O1_ARTIFACT_ROOT / dataset_name
    model_path = artifact_dir / "litemv_o1.keras"
    prep_path = artifact_dir / "preprocessor.joblib"
    b0_features_path = artifact_dir / "baseline_features.json"
    features_path = artifact_dir / "selected_features.json"
    decision_path = artifact_dir / "decision_rule.json"

    for path in (
        model_path,
        prep_path,
        b0_features_path,
        features_path,
        decision_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing O1 artifact: {path}"
            )

    baseline_features = list(
        load_json(b0_features_path)
    )
    model_features = list(load_json(features_path))
    expected, _ = resolve_o1_features(
        baseline_features
    )
    if model_features != expected:
        raise RuntimeError(
            f"{dataset_name}: O1 43-feature schema changed."
        )

    decision = load_json(decision_path)
    return {
        "stage": STAGE_O1,
        "model": load_keras_model(model_path),
        "preprocessor": DatasetPreprocessor.load(prep_path),
        "baseline_features": baseline_features,
        "model_features": model_features,
        "threshold": float(decision["threshold"]),
        "threshold_method": str(decision["method"]),
    }


# =============================================================================
# STRICT EVALUATION / TRANSFER
# =============================================================================

def evaluate_source_bundle(
    source_dataset: str,
    target_dataset: str,
    target_split: str,
    bundle: dict,
) -> tuple[dict, pd.DataFrame]:
    target_splits, _manifest, target_split_metadata = (
        load_dataset_and_split(
            target_dataset,
            force_resplit=False,
        )
    )
    target_raw = target_splits[target_split]

    target_b0, target_b0_features, _, _ = (
        transform_to_b0(
            bundle["preprocessor"],
            target_raw,
            dataset_name=(
                f"{bundle['stage']}_{source_dataset}_to_"
                f"{target_dataset}_{target_split}"
            ),
            expected_baseline_features=(
                bundle["baseline_features"]
            ),
        )
    )
    if target_b0_features != bundle["baseline_features"]:
        raise RuntimeError(
            f"{source_dataset}->{target_dataset}/{target_split}: "
            "source B0 feature schema/order changed."
        )

    X, y, meta = build_sequences(
        target_b0,
        bundle["model_features"],
        int(SHARED_PARAMS["sequence_length"]),
        EVAL_SEQUENCE_STRIDE,
    )

    if len(y) != len(target_raw):
        raise RuntimeError(
            f"{bundle['stage']} {source_dataset}->{target_dataset}/"
            f"{target_split}: expected {len(target_raw):,} predictions, "
            f"got {len(y):,}."
        )

    prob = predict_probabilities(
        bundle["model"],
        X,
        batch_size=int(SHARED_PARAMS["batch_size"]),
    )
    pred = predictions_from_probabilities(
        prob,
        bundle["threshold"],
    )
    metrics = calculate_metrics(
        y,
        prob,
        bundle["threshold"],
    )

    row = {
        "Stage": bundle["stage"],
        "SourceDataset": source_dataset,
        "TargetDataset": target_dataset,
        "Direction": f"{source_dataset}_to_{target_dataset}",
        "TargetSplit": target_split,
        "Transfer": source_dataset != target_dataset,
        "TargetSplitMode": target_split_metadata.get("mode"),
        "FeatureCount": len(bundle["model_features"]),
        "SequenceLength": int(
            SHARED_PARAMS["sequence_length"]
        ),
        "ThresholdMethod": bundle["threshold_method"],
        "SourceValidationThreshold": bundle["threshold"],
        "Rows": int(len(y)),
        "SourceFiles": int(
            meta[SOURCE_FILE_COL].nunique()
        ),
        **metrics,
    }

    pred_table = meta.copy()
    pred_table["MaliciousProbability"] = prob
    pred_table["Predicted_Label"] = pred
    pred_table["SourceValidationThreshold"] = (
        bundle["threshold"]
    )
    pred_table["Stage"] = bundle["stage"]
    pred_table["SourceDataset"] = source_dataset
    pred_table["TargetDataset"] = target_dataset
    pred_table["TargetSplit"] = target_split

    del X
    return row, pred_table


def run_b0_vs_o1_evaluation(
    only_dataset: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = (
        [only_dataset]
        if only_dataset is not None
        else list(BENCHMARK_DATASETS)
    )

    within_rows = []
    cross_rows = []

    for source in sources:
        print("\n" + "#" * 100)
        print(f"B0 VS O1 EVALUATION - SOURCE {source.upper()}")
        print("#" * 100)

        for loader in (load_b0_bundle, load_o1_bundle):
            bundle = loader(source)

            # Within-dataset untouched source test.
            row, pred = evaluate_source_bundle(
                source_dataset=source,
                target_dataset=source,
                target_split="test",
                bundle=bundle,
            )
            within_rows.append(row)

            within_dir = (
                O1_RESULT_ROOT
                / "comparison"
                / bundle["stage"]
                / source
            )
            within_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            pd.DataFrame([row]).to_csv(
                within_dir / "source_test_metrics.csv",
                index=False,
            )
            pred.to_csv(
                within_dir / "source_test_predictions.csv",
                index=False,
            )

            print(
                f"{bundle['stage']} source test: "
                f"Recall={row['recall_tpr']:.4f} | "
                f"FPR={row['fpr']:.4f} | "
                f"F2={row['f2']:.4f} | "
                f"PR-AUC={row['pr_auc']:.4f}"
            )

            # True transfer.
            for target in BENCHMARK_DATASETS:
                if target == source:
                    continue

                direction = f"{source}_to_{target}"
                cross_dir = (
                    O1_RESULT_ROOT
                    / "cross_dataset"
                    / bundle["stage"]
                    / direction
                )
                cross_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                save_json(
                    cross_dir / "protocol.json",
                    {
                        "stage": bundle["stage"],
                        "source_dataset": source,
                        "target_dataset": target,
                        "source_model_frozen": True,
                        "source_preprocessor_frozen": True,
                        "source_feature_schema_frozen": True,
                        "source_validation_threshold_frozen": True,
                        "target_preprocessor_fit": False,
                        "target_threshold_calibration": False,
                        "target_model_adaptation": False,
                        "target_validation_role": (
                            "diagnostic_robustness_only"
                        ),
                        "target_test_role": (
                            "reporting_only_not_stage_selection"
                        ),
                    },
                )

                for split in ("validation", "test"):
                    cross_row, cross_pred = (
                        evaluate_source_bundle(
                            source_dataset=source,
                            target_dataset=target,
                            target_split=split,
                            bundle=bundle,
                        )
                    )
                    cross_rows.append(cross_row)

                    pd.DataFrame([cross_row]).to_csv(
                        cross_dir / f"{split}_metrics.csv",
                        index=False,
                    )
                    cross_pred.to_csv(
                        cross_dir / f"{split}_predictions.csv",
                        index=False,
                    )

                    print(
                        f"  {bundle['stage']} {direction} {split}: "
                        f"Recall={cross_row['recall_tpr']:.4f} | "
                        f"FPR={cross_row['fpr']:.4f} | "
                        f"F2={cross_row['f2']:.4f} | "
                        f"PR-AUC={cross_row['pr_auc']:.4f}"
                    )

            tf.keras.backend.clear_session()

    within_df = pd.DataFrame(within_rows)
    cross_df = pd.DataFrame(cross_rows)

    O1_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    within_df.to_csv(
        O1_RESULT_ROOT
        / "b0_vs_o1_within_dataset_test_summary.csv",
        index=False,
    )
    cross_df.to_csv(
        O1_RESULT_ROOT
        / "b0_vs_o1_cross_dataset_summary.csv",
        index=False,
    )

    return within_df, cross_df


# =============================================================================
# MAIN
# =============================================================================

def run(
    only_dataset: str | None = None,
    training_only: bool = False,
    evaluation_only: bool = False,
    force_resplit: bool = False,
) -> None:
    print("=" * 100)
    print("LITEMV O1 - ENVIRONMENT HARDENING 57 -> 43")
    print("=" * 100)
    print("Only intended change from B0: 57 -> 43 features.")
    print("Same LITEMV architecture and shared D2/D3 hyperparameters.")
    print("Same B0 validation-only threshold calibration.")
    print("Strict cross-dataset transfer: no target adaptation.")
    print("Baseline overwrite: NO.")

    datasets = (
        [only_dataset]
        if only_dataset is not None
        else list(BENCHMARK_DATASETS)
    )

    if not evaluation_only:
        rows = [
            train_o1_dataset(
                dataset_name=name,
                force_resplit=force_resplit,
            )
            for name in datasets
        ]
        O1_RESULT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )
        pd.DataFrame(rows).to_csv(
            O1_RESULT_ROOT / "o1_validation_summary.csv",
            index=False,
        )

    if not training_only:
        within_df, cross_df = (
            run_b0_vs_o1_evaluation(
                only_dataset=only_dataset,
            )
        )

        print("\n" + "=" * 100)
        print("B0 VS O1 - WITHIN-DATASET TEST")
        print("=" * 100)
        print(
            within_df[
                [
                    "Stage",
                    "SourceDataset",
                    "FeatureCount",
                    "SourceValidationThreshold",
                    "precision",
                    "recall_tpr",
                    "fpr",
                    "f1",
                    "f2",
                    "roc_auc",
                    "pr_auc",
                    "mcc",
                    "fp",
                    "fn",
                    "tp",
                ]
            ].to_string(index=False)
        )

        print("\n" + "=" * 100)
        print("B0 VS O1 - TRUE CROSS-DATASET TRANSFER")
        print("=" * 100)
        print(
            cross_df[
                [
                    "Stage",
                    "Direction",
                    "TargetSplit",
                    "FeatureCount",
                    "SourceValidationThreshold",
                    "precision",
                    "recall_tpr",
                    "fpr",
                    "f1",
                    "f2",
                    "roc_auc",
                    "pr_auc",
                    "mcc",
                    "fp",
                    "fn",
                    "tp",
                ]
            ].to_string(index=False)
        )

    save_json(
        O1_ARTIFACT_ROOT / "o1_protocol.json",
        {
            "stage": STAGE_O1,
            "baseline_feature_count": B0_FEATURE_COUNT,
            "o1_feature_count": O1_FEATURE_COUNT,
            "mandatory_drops": sorted(
                O1_MANDATORY_DROPS
            ),
            "allowed_flag_feature": (
                O1_ALLOWED_FLAG_FEATURE
            ),
            "rule": (
                'drop every selected feature whose normalized '
                'name contains "flag", except URG Flag Count'
            ),
            "expected_drop_count": 14,
            "architecture_changed": False,
            "shared_hyperparameters_changed": False,
            "split_changed": False,
            "preprocessing_procedure_changed": False,
            "threshold_calibration_procedure_changed": False,
            "target_adaptation_in_cross_dataset": False,
        },
    )

    print("\nResults:")
    print(f"  {O1_RESULT_ROOT}")
    print("Artifacts:")
    print(f"  {O1_ARTIFACT_ROOT}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(BENCHMARK_DATASETS),
        default=None,
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Normally DO NOT use this.",
    )
    args = parser.parse_args()

    if args.training_only and args.evaluation_only:
        parser.error(
            "--training-only and --evaluation-only "
            "cannot be used together."
        )
    return args


def main():
    args = parse_args()
    run(
        only_dataset=args.dataset,
        training_only=args.training_only,
        evaluation_only=args.evaluation_only,
        force_resplit=args.force_resplit,
    )


if __name__ == "__main__":
    main()
