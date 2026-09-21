from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from random_forest_config import (  # noqa: E402
    BENCHMARK_DATASETS,
    DATASET_PATHS,
    FOCUSED_MAX_FEATURES,
    FOCUSED_N_ESTIMATORS,
    MODEL_ARTIFACT_ROOT,
    PREDICTION_THRESHOLD,
    RESULT_ROOT,
    SHARED_HYPERPARAMETER_PATH,
    SHARED_SELECTION_RESULT_DIR,
)
from random_forest_utils import (  # noqa: E402
    METRIC_COLUMNS,
    calculate_binary_metrics,
)


# ============================================================
# OOF PLOTS -- SAME EVALUATION LOGIC
# ============================================================


def save_oof_plots(
    y_true,
    probabilities,
    predictions,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malicious"],
    ).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Random Forest - Nested-CV OOF Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "oof_confusion_matrix.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
        name="Random Forest OOF",
    )
    ax.set_title("Random Forest - Nested-CV OOF ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "oof_roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
        name="Random Forest OOF",
    )
    ax.set_title("Random Forest - Nested-CV OOF Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "oof_precision_recall_curve.png", dpi=300)
    plt.close(fig)


def build_fold_stability_summary(outer_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(outer_metrics[metric], errors="raise")
        rows.append(
            {
                "Metric": metric,
                "Fold_Mean": float(values.mean()),
                "Fold_Std": float(values.std(ddof=0)),
                "Fold_Min": float(values.min()),
                "Fold_Max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# SHARED HYPERPARAMETER SENSITIVITY / STABILITY PLOTS
# ============================================================


def _parameter_order(parameter: str):
    if parameter == "n_estimators":
        return FOCUSED_N_ESTIMATORS
    if parameter == "max_features":
        return FOCUSED_MAX_FEATURES
    raise ValueError(parameter)


def save_shared_sensitivity_plots() -> None:
    raw_path = SHARED_SELECTION_RESULT_DIR / "shared_inner_runs.csv"
    summary_path = (
        SHARED_SELECTION_RESULT_DIR / "shared_final_sensitivity_summary.csv"
    )

    if not raw_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Shared RF sensitivity results are missing. Run training first."
        )

    raw = pd.read_csv(raw_path, low_memory=False)
    summary = pd.read_csv(summary_path, low_memory=False)
    plot_dir = SHARED_SELECTION_RESULT_DIR / "sensitivity_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for parameter in ["n_estimators", "max_features"]:
        column = f"param_{parameter}"
        order = _parameter_order(parameter)
        labels = [str(value) for value in order]
        x = np.arange(len(order))

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        joint_values = []

        for dataset_name in BENCHMARK_DATASETS:
            dataset_rows = raw[raw["Dataset"] == dataset_name].copy()
            means = []
            stds = []

            for value in order:
                if parameter == "n_estimators":
                    mask = pd.to_numeric(
                        dataset_rows[column], errors="coerce"
                    ) == int(value)
                else:
                    mask = dataset_rows[column].astype(str) == str(value)

                values = pd.to_numeric(
                    dataset_rows.loc[mask, "pr_auc"], errors="raise"
                )
                means.append(float(values.mean()))
                stds.append(float(values.std(ddof=0)))

            ax.errorbar(
                x,
                means,
                yerr=stds,
                marker="o",
                capsize=4,
                label=dataset_name,
            )
            joint_values.append(np.asarray(means, dtype=float))

        joint_mean = np.mean(np.vstack(joint_values), axis=0)
        ax.plot(
            x,
            joint_mean,
            marker="o",
            linestyle="--",
            label="Equal-weight D2/D3 mean",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Inner-validation PR-AUC")
        ax.set_title(f"Shared RF Sensitivity - {parameter}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"sensitivity_{parameter}.png", dpi=300)
        plt.close(fig)

    # Direct comparison of all four focused configurations.
    ordered = summary.sort_values(
        ["param_n_estimators", "param_max_features"],
        key=lambda col: col.astype(str),
        kind="mergesort",
    ).reset_index(drop=True)

    labels = [
        f"n={int(row.param_n_estimators)}, mf={row.param_max_features}"
        for row in ordered.itertuples()
    ]
    x = np.arange(len(ordered))
    y = pd.to_numeric(
        ordered["pr_auc_mean_across_datasets"], errors="raise"
    ).to_numpy(dtype=float)
    yerr = pd.to_numeric(
        ordered["pr_auc_std_across_all_inner_runs"], errors="raise"
    ).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Equal-weight D2/D3 inner-validation PR-AUC")
    ax.set_title("Shared RF Focused Configuration Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "shared_configuration_comparison.png", dpi=300)
    plt.close(fig)


# ============================================================
# SHARED-HYPERPARAMETER VERIFICATION
# ============================================================


def _load_shared_hyperparameters() -> dict:
    if not SHARED_HYPERPARAMETER_PATH.exists():
        raise FileNotFoundError(
            f"Missing shared RF hyperparameter artifact: {SHARED_HYPERPARAMETER_PATH}"
        )
    with SHARED_HYPERPARAMETER_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_shared_dataset_artifact(dataset_name: str, shared: dict) -> None:
    path = MODEL_ARTIFACT_ROOT / dataset_name / "final_hyperparameters.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing RF final hyperparameters: {path}")

    with path.open("r", encoding="utf-8") as handle:
        local = json.load(handle)

    if local.get("parameters") != shared.get("parameters"):
        raise RuntimeError(
            f"{dataset_name}: final RF hyperparameters do not match the shared artifact."
        )


def verify_outer_fold_shared_selections(dataset_name: str, oof: pd.DataFrame) -> None:
    """Verify that this dataset used the joint D2/D3 selection in every outer fold."""
    selection_path = (
        SHARED_SELECTION_RESULT_DIR / "shared_outer_fold_selections.csv"
    )
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Missing shared outer-fold selection artifact: {selection_path}"
        )

    expected = pd.read_csv(selection_path, low_memory=False)
    required = {"Outer_Fold", "Selected_Shared_Config_ID"}
    missing = required - set(oof.columns)
    if missing:
        raise RuntimeError(
            f"{dataset_name}: OOF predictions are missing shared-selection columns: "
            f"{sorted(missing)}"
        )

    for row in expected.itertuples(index=False):
        fold = int(row.outer_fold)
        expected_id = str(row.config_id)
        actual_ids = set(
            oof.loc[
                pd.to_numeric(oof["Outer_Fold"], errors="raise").astype(int) == fold,
                "Selected_Shared_Config_ID",
            ].astype(str)
        )
        if actual_ids != {expected_id}:
            raise RuntimeError(
                f"{dataset_name}: outer fold {fold} used {sorted(actual_ids)}, "
                f"expected shared config {expected_id}."
            )


# ============================================================
# DATASET OOF EVALUATION
# ============================================================


def evaluate_dataset(dataset_name: str) -> dict:
    print("\n" + "=" * 90)
    print(f"RANDOM FOREST NESTED-CV OOF EVALUATION - {dataset_name.upper()}")
    print("=" * 90)

    result_dir = RESULT_ROOT / dataset_name
    oof_path = result_dir / "oof_predictions.csv"
    outer_metrics_path = result_dir / "outer_fold_metrics.csv"

    for path in [oof_path, outer_metrics_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required nested-CV result not found: {path}"
            )

    shared = _load_shared_hyperparameters()
    verify_shared_dataset_artifact(dataset_name, shared)

    oof = pd.read_csv(oof_path, low_memory=False)
    outer_metrics = pd.read_csv(outer_metrics_path, low_memory=False)
    verify_outer_fold_shared_selections(dataset_name, oof)

    y_true = pd.to_numeric(
        oof["Actual_Label"], errors="raise"
    ).astype(int).to_numpy()
    probabilities = pd.to_numeric(
        oof["Malicious_Probability"], errors="raise"
    ).to_numpy(dtype=float)
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)

    overall_metrics = calculate_binary_metrics(
        y_true,
        probabilities,
        threshold=PREDICTION_THRESHOLD,
    )
    stability = build_fold_stability_summary(outer_metrics)
    stability.to_csv(
        result_dir / "outer_fold_stability_summary.csv",
        index=False,
    )

    metric_summary_rows = []
    for metric in METRIC_COLUMNS:
        stability_row = stability[stability["Metric"] == metric].iloc[0]
        metric_summary_rows.append(
            {
                "Metric": metric,
                "OOF_Overall": overall_metrics[metric],
                "Outer_Fold_Mean": stability_row["Fold_Mean"],
                "Outer_Fold_Std": stability_row["Fold_Std"],
                "Outer_Fold_Min": stability_row["Fold_Min"],
                "Outer_Fold_Max": stability_row["Fold_Max"],
            }
        )

    pd.DataFrame(metric_summary_rows).to_csv(
        result_dir / "nested_cv_metric_summary.csv",
        index=False,
    )

    with (result_dir / "oof_overall_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(overall_metrics, handle, indent=2)

    pd.DataFrame([overall_metrics]).to_csv(
        result_dir / "oof_overall_metrics.csv",
        index=False,
    )

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual_Benign", "Actual_Malicious"],
        columns=["Predicted_Benign", "Predicted_Malicious"],
    ).to_csv(result_dir / "oof_confusion_matrix.csv")

    save_oof_plots(
        y_true=y_true,
        probabilities=probabilities,
        predictions=predictions,
        output_dir=result_dir,
    )

    print(f"OOF rows: {len(oof):,}")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {overall_metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall / TPR: {overall_metrics['recall_tpr']:.4f}")
    print(f"FPR: {overall_metrics['fpr']:.4f}")
    print(f"F1: {overall_metrics['f1']:.4f}")
    print(f"ROC-AUC: {overall_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {overall_metrics['pr_auc']:.4f}")

    print("\nOuter-fold stability:")
    for metric in ["recall_tpr", "fpr", "f1", "roc_auc", "pr_auc"]:
        row = stability[stability["Metric"] == metric].iloc[0]
        print(
            f"  {metric}: mean={row['Fold_Mean']:.4f}, "
            f"std={row['Fold_Std']:.4f}, "
            f"min={row['Fold_Min']:.4f}, max={row['Fold_Max']:.4f}"
        )

    return {
        "Dataset": dataset_name,
        "HyperparameterScope": "shared_dataset2_dataset3",
        "SharedHyperparametersVerified": True,
        **overall_metrics,
    }


def evaluate_all_datasets() -> pd.DataFrame:
    if tuple(DATASET_PATHS.keys()) != BENCHMARK_DATASETS:
        raise RuntimeError("RF benchmark dataset configuration is inconsistent.")

    rows = [evaluate_dataset(name) for name in BENCHMARK_DATASETS]
    summary = pd.DataFrame(rows)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        RESULT_ROOT / "random_forest_nested_cv_summary.csv",
        index=False,
    )

    save_shared_sensitivity_plots()
    return summary
