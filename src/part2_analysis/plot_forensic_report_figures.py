#!/usr/bin/env python3
"""
Create report-ready figures for Milestone 3, Section 2.1 from the final D2/D3
forensic snapshot produced by build_forensic_error_dataset.py.

Figures created
---------------
1. Combined confusion matrices: 4 models x 2 datasets using each model's native baseline evaluation protocol.
2. Dataset 2 SourceFile-level FP/FN concentration heatmap.
3. Dataset 3 SourceFile-level FP/FN concentration heatmap.
4. Dataset 2 feature characteristics of FP/FN errors (top rank-biserial effects).
5. Dataset 3 feature characteristics of FP/FN errors.
6. Cross-model error overlap for Dataset 2 and Dataset 3.

The script reads the frozen forensic snapshot only. It does not retrain models,
change thresholds, or modify the snapshot.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "random_forest",
    "litemv",
    "autoencoder",
    "lstm",
]

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "litemv": "LITEMV",
    "autoencoder": "Autoencoder",
    "lstm": "LSTM",
}

DATASET_ORDER = ["dataset2", "dataset3"]


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def require_columns(df: pd.DataFrame, required: Iterable[str], description: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{description} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def ensure_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}:\n{path}")


def short_source_label(source_file: str, width: int = 46) -> str:
    name = Path(str(source_file)).name
    if name.lower().endswith(".csv"):
        name = name[:-4]
    return "\n".join(
        textwrap.wrap(
            name,
            width=width,
            break_long_words=False,
            break_on_hyphens=True,
        )
    )


def short_feature_label(feature: str, width: int = 28) -> str:
    return "\n".join(
        textwrap.wrap(
            str(feature),
            width=width,
            break_long_words=False,
            break_on_hyphens=True,
        )
    )


def save_figure(fig: plt.Figure, output_stem: Path, dpi: int) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIGURE 2.1A -- COMBINED CONFUSION MATRICES
# =============================================================================


def plot_combined_confusion_matrices(
    snapshot_dir: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    rows = []
    for dataset_name in DATASET_ORDER:
        path = snapshot_dir / dataset_name / "error_counts.csv"
        ensure_file(path, f"{dataset_name} error_counts.csv")
        table = pd.read_csv(path)
        require_columns(
            table,
            ["Model", "TN", "FP", "FN", "TP"],
            f"{dataset_name} error_counts.csv",
        )
        rows.append(table)

    all_counts = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.3))
    image = None

    for i, dataset_name in enumerate(DATASET_ORDER):
        dataset_rows = all_counts[all_counts["Dataset"] == dataset_name]
        for j, model_name in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            match = dataset_rows[dataset_rows["Model"] == model_name]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one confusion row for {dataset_name}/{model_name}; found {len(match)}"
                )

            row = match.iloc[0]
            cm = np.array(
                [
                    [int(row["TN"]), int(row["FP"])],
                    [int(row["FN"]), int(row["TP"])],
                ],
                dtype=float,
            )

            # Row-normalized percentages let models remain interpretable even when
            # benign and malicious class counts differ strongly.
            row_totals = cm.sum(axis=1, keepdims=True)
            normalized = np.divide(
                cm,
                row_totals,
                out=np.zeros_like(cm),
                where=row_totals > 0,
            )

            image = ax.imshow(normalized, vmin=0.0, vmax=1.0, aspect="equal")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Benign", "Malicious"], fontsize=8.5)
            ax.set_yticklabels(["Benign", "Malicious"], fontsize=8.5)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)
            ds = dataset_name.replace("dataset", "D")
            ax.set_title(f"{ds} — {MODEL_DISPLAY[model_name]}", fontsize=10.5, fontweight="bold")

            for r in range(2):
                for c in range(2):
                    count = int(cm[r, c])
                    pct = 100.0 * normalized[r, c]
                    ax.text(
                        c,
                        r,
                        f"{count:,}\n{pct:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

    if image is not None:
        # Put the shared colorbar in its own axis outside the 2x4 matrix grid.
        # Using fig.colorbar(..., ax=axes) together with tight_layout can pull the
        # colorbar into the right-most panels and cover the LSTM matrices.
        cbar_ax = fig.add_axes([0.935, 0.16, 0.014, 0.68])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("Row-normalized proportion")

    fig.suptitle(
        "Section 2.1 — Final-baseline confusion matrices by native evaluation protocol",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "RF uses full source-aware OOF predictions with the frozen shared RF configuration; "
        "LITEMV/AE/LSTM use the persisted shared held-out test split.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    # Manual spacing is intentional here because the shared colorbar has its own
    # axis. This reserves the entire right margin for the colorbar and prevents
    # it from overlapping the D2/D3 LSTM panels.
    fig.subplots_adjust(
        left=0.06,
        right=0.91,
        bottom=0.09,
        top=0.89,
        wspace=0.30,
        hspace=0.32,
    )

    stem = output_dir / "figure_2_1a_combined_confusion_matrices"
    save_figure(fig, stem, dpi)
    all_counts.to_csv(stem.with_name(stem.name + "_data.csv"), index=False)


# =============================================================================
# FIGURES 2.1B / 2.1C -- SOURCE-LEVEL ERROR CONCENTRATION
# =============================================================================


def build_source_error_matrices(
    error_by_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    require_columns(
        error_by_source,
        ["Model", "SourceFile", "FP", "FN", "SourceErrorRate"],
        "error_by_source.csv",
    )

    work = error_by_source.copy()
    work["FP"] = pd.to_numeric(work["FP"], errors="coerce").fillna(0).astype(int)
    work["FN"] = pd.to_numeric(work["FN"], errors="coerce").fillna(0).astype(int)
    work["Errors"] = work["FP"] + work["FN"]
    work["SourceErrorRate"] = pd.to_numeric(work["SourceErrorRate"], errors="coerce").fillna(0.0)

    # Keep every source with at least one error from at least one model.
    error_sources = (
        work.groupby("SourceFile", dropna=False)["Errors"].sum()
        .loc[lambda series: series > 0]
        .index
    )
    work = work[work["SourceFile"].isin(error_sources)].copy()
    if work.empty:
        raise ValueError("No SourceFile-level FP/FN errors found to plot.")

    count_matrix = work.pivot_table(
        index="SourceFile",
        columns="Model",
        values="Errors",
        aggfunc="sum",
        fill_value=0,
    )
    rate_matrix = work.pivot_table(
        index="SourceFile",
        columns="Model",
        values="SourceErrorRate",
        aggfunc="max",
        fill_value=0.0,
    )

    for model in MODEL_ORDER:
        if model not in count_matrix.columns:
            count_matrix[model] = 0
        if model not in rate_matrix.columns:
            rate_matrix[model] = 0.0
    count_matrix = count_matrix[MODEL_ORDER]
    rate_matrix = rate_matrix[MODEL_ORDER]

    total_errors = count_matrix.sum(axis=1)
    count_matrix = count_matrix.loc[total_errors.sort_values(ascending=False).index]
    rate_matrix = rate_matrix.reindex(count_matrix.index)

    source_type = {}
    for source, group in work.groupby("SourceFile", sort=False):
        fp = int(group["FP"].sum())
        fn = int(group["FN"].sum())
        if fp > 0 and fn == 0:
            source_type[source] = "FP"
        elif fn > 0 and fp == 0:
            source_type[source] = "FN"
        else:
            source_type[source] = "FP+FN"

    source_type_series = pd.Series(source_type).reindex(count_matrix.index)
    return count_matrix, rate_matrix, source_type_series


def plot_error_concentration_heatmap(
    snapshot_dir: Path,
    dataset_name: str,
    output_dir: Path,
    figure_label: str,
    dpi: int,
) -> None:
    path = snapshot_dir / dataset_name / "error_by_source.csv"
    ensure_file(path, f"{dataset_name} error_by_source.csv")
    df = pd.read_csv(path)
    counts, rates, source_types = build_source_error_matrices(df)

    values = rates.to_numpy(dtype=float)
    n_sources = len(rates)
    fig_height = max(5.2, 0.58 * n_sources + 1.9)
    fig, ax = plt.subplots(figsize=(11.0, fig_height))

    image = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_DISPLAY[model] for model in MODEL_ORDER])

    labels = []
    for source in rates.index:
        kind = str(source_types.loc[source])
        labels.append(f"[{kind}] {short_source_label(source)}")
    ax.set_yticks(np.arange(n_sources))
    ax.set_yticklabels(labels, fontsize=8.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("SourceFile")
    ds_number = dataset_name.replace("dataset", "")
    ax.set_title(
        f"Dataset {ds_number} — FP/FN concentration by SourceFile and model",
        fontweight="bold",
        pad=12,
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            count = int(counts.iloc[i, j])
            rate = float(values[i, j])
            text = "–" if count == 0 else f"{count:,}\n({100.0 * rate:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.2)

    ax.set_xticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_sources, 1), minor=True)
    ax.grid(which="minor", linewidth=0.6, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Source-specific error rate")

    fig.text(
        0.5,
        0.012,
        "Cell = error count and source-specific error rate. RF uses full OOF evaluation; LITEMV/AE/LSTM use the shared held-out test.",
        ha="center",
        va="bottom",
        fontsize=8.4,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    stem = output_dir / f"{figure_label}_{dataset_name}_error_concentration"
    save_figure(fig, stem, dpi)

    export = counts.copy()
    export.insert(0, "ErrorType", source_types)
    for model in MODEL_ORDER:
        export[f"{MODEL_DISPLAY[model]} ErrorRate"] = rates[model]
    export = export.rename(columns=MODEL_DISPLAY)
    export.index.name = "SourceFile"
    export.to_csv(stem.with_name(stem.name + "_data.csv"))


# =============================================================================
# FIGURES 2.1D / 2.1E -- FEATURE CHARACTERISTICS OF FP/FN ERRORS
# =============================================================================


def select_top_feature_effects(
    contrasts: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    require_columns(
        contrasts,
        [
            "Model",
            "Comparison",
            "Feature",
            "RankBiserialEffect",
            "AbsRankBiserialEffect",
            "FDR_Q",
            "ErrorN",
            "ReferenceN",
        ],
        "feature_error_contrasts.csv",
    )

    work = contrasts.copy()
    work["RankBiserialEffect"] = pd.to_numeric(work["RankBiserialEffect"], errors="coerce")
    work["AbsRankBiserialEffect"] = pd.to_numeric(work["AbsRankBiserialEffect"], errors="coerce")
    work["FDR_Q"] = pd.to_numeric(work["FDR_Q"], errors="coerce")
    work = work.dropna(subset=["RankBiserialEffect", "AbsRankBiserialEffect"])

    selected = (
        work.sort_values(
            ["Model", "Comparison", "AbsRankBiserialEffect"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .groupby(["Model", "Comparison"], group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return selected


def plot_feature_error_effects(
    snapshot_dir: Path,
    dataset_name: str,
    output_dir: Path,
    figure_label: str,
    dpi: int,
    top_n: int,
) -> None:
    path = snapshot_dir / dataset_name / "feature_error_contrasts.csv"
    ensure_file(path, f"{dataset_name} feature_error_contrasts.csv")
    contrasts = pd.read_csv(path)
    selected = select_top_feature_effects(contrasts, top_n=top_n)

    comparisons = ["FP_vs_TN", "FN_vs_TP"]
    fig, axes = plt.subplots(4, 2, figsize=(14.4, 15.5), sharex=True)

    for row_idx, model_name in enumerate(MODEL_ORDER):
        for col_idx, comparison in enumerate(comparisons):
            ax = axes[row_idx, col_idx]
            sub = selected[
                (selected["Model"] == model_name)
                & (selected["Comparison"] == comparison)
            ].copy()

            if sub.empty:
                ax.text(0.5, 0.5, "No analyzable errors", ha="center", va="center", transform=ax.transAxes)
                ax.set_yticks([])
            else:
                sub = sub.sort_values("AbsRankBiserialEffect", ascending=True)
                y = np.arange(len(sub))
                effects = sub["RankBiserialEffect"].to_numpy(dtype=float)
                ax.barh(y, effects)
                labels = [short_feature_label(feature) for feature in sub["Feature"]]
                ax.set_yticks(y)
                ax.set_yticklabels(labels, fontsize=8.0)

                for y_pos, (_, item) in enumerate(sub.iterrows()):
                    q = item["FDR_Q"]
                    marker = "*" if pd.notna(q) and float(q) < 0.05 else ""
                    x = float(item["RankBiserialEffect"])
                    offset = 0.025 if x >= 0 else -0.025
                    ax.text(
                        x + offset,
                        y_pos,
                        f"{x:+.2f}{marker}",
                        va="center",
                        ha="left" if x >= 0 else "right",
                        fontsize=7.8,
                    )

            ax.axvline(0.0, linewidth=1.0)
            ax.set_xlim(-1.05, 1.05)
            ax.grid(axis="x", alpha=0.2)

            comparison_title = "FP vs TN" if comparison == "FP_vs_TN" else "FN vs TP"
            ax.set_title(
                f"{MODEL_DISPLAY[model_name]} — {comparison_title}",
                fontsize=10.5,
                fontweight="bold",
            )
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Rank-biserial effect (error group − correct reference)")

    ds_number = dataset_name.replace("dataset", "")
    fig.suptitle(
        f"Dataset {ds_number} — strongest feature characteristics of false decisions",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "Positive effect: error samples tend to have higher values; negative effect: lower values. "
        "* FDR q < 0.05. Features are each model's actual final input schema, shown in raw/interpretable units.",
        ha="center",
        va="bottom",
        fontsize=8.4,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))

    stem = output_dir / f"{figure_label}_{dataset_name}_feature_error_effects"
    save_figure(fig, stem, dpi)
    selected.to_csv(stem.with_name(stem.name + "_data.csv"), index=False)


# =============================================================================
# FIGURE 2.1F -- CROSS-MODEL ERROR OVERLAP
# =============================================================================


def plot_cross_model_overlap(
    snapshot_dir: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=False)
    export_rows = []

    for ax, dataset_name in zip(axes, DATASET_ORDER):
        path = snapshot_dir / dataset_name / "cross_model_error_overlap_summary.csv"
        ensure_file(path, f"{dataset_name} cross-model overlap summary")
        summary = pd.read_csv(path)
        require_columns(summary, ["Label", "ModelsWrong", "Rows"], str(path))

        summary["Label"] = pd.to_numeric(summary["Label"], errors="raise").astype(int)
        summary["ModelsWrong"] = pd.to_numeric(summary["ModelsWrong"], errors="raise").astype(int)
        summary["Rows"] = pd.to_numeric(summary["Rows"], errors="raise").astype(int)
        export_rows.append(summary.assign(Dataset=dataset_name))

        # Focus on actual errors; zero-wrong rows would dominate the scale.
        work = summary[summary["ModelsWrong"] >= 1].copy()
        grouped = (
            work.groupby(["ModelsWrong", "Label"])["Rows"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=[1, 2, 3, 4], fill_value=0)
        )
        for label in [0, 1]:
            if label not in grouped.columns:
                grouped[label] = 0
        grouped = grouped[[0, 1]]

        x = np.arange(1, 5)
        width = 0.36
        ax.bar(x - width / 2, grouped[0].to_numpy(), width=width, label="Benign errors (FP)")
        ax.bar(x + width / 2, grouped[1].to_numpy(), width=width, label="Malicious errors (FN)")
        ax.set_xticks(x)
        ax.set_xlabel("Number of models wrong on the same sample")
        ax.set_ylabel("Held-out sample count")
        ds_number = dataset_name.replace("dataset", "")
        ax.set_title(f"Dataset {ds_number}", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8.5)

    fig.suptitle(
        "Cross-model overlap of false decisions",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Counts exclude rows correctly classified by all four models. Shared errors indicate samples difficult across model families.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    stem = output_dir / "figure_2_1f_cross_model_error_overlap"
    save_figure(fig, stem, dpi)
    pd.concat(export_rows, ignore_index=True).to_csv(
        stem.with_name(stem.name + "_data.csv"),
        index=False,
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    try:
        project_root = script_path.parents[2]
    except IndexError:
        project_root = Path.cwd()

    default_snapshot = (
        project_root / "results" / "forensic_analysis" / "final_baseline_v1"
    )
    default_output = (
        project_root
        / "results"
        / "forensic_analysis"
        / "report_plots"
        / "final_baseline_v1"
    )

    parser = argparse.ArgumentParser(
        description="Create report-ready D2/D3 FINAL BASELINE forensic figures for Section 2.1."
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=default_snapshot,
        help=f"Frozen forensic snapshot directory (default: {default_snapshot})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output directory for PNG/PDF/CSV files (default: {default_output})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution; default 300 DPI.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=5,
        help="Top absolute rank-biserial feature effects per model/comparison; default 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = args.snapshot_dir.resolve()
    output_dir = args.output_dir.resolve()

    marker = snapshot_dir / "FORENSIC_SNAPSHOT.json"
    ensure_file(marker, "FORENSIC_SNAPSHOT.json marker")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("SECTION 2.1 - FINAL BASELINE FORENSIC REPORT FIGURES")
    print("=" * 92)
    print(f"Snapshot: {snapshot_dir}")
    print(f"Output:   {output_dir}")
    print()

    plot_combined_confusion_matrices(snapshot_dir, output_dir, args.dpi)
    print("[OK] Figure 2.1a - combined D2/D3 confusion matrices")

    plot_error_concentration_heatmap(
        snapshot_dir,
        "dataset2",
        output_dir,
        "figure_2_1b",
        args.dpi,
    )
    print("[OK] Figure 2.1b - Dataset 2 SourceFile error concentration")

    plot_error_concentration_heatmap(
        snapshot_dir,
        "dataset3",
        output_dir,
        "figure_2_1c",
        args.dpi,
    )
    print("[OK] Figure 2.1c - Dataset 3 SourceFile error concentration")

    plot_feature_error_effects(
        snapshot_dir,
        "dataset2",
        output_dir,
        "figure_2_1d",
        args.dpi,
        args.top_features,
    )
    print("[OK] Figure 2.1d - Dataset 2 FP/FN feature characteristics")

    plot_feature_error_effects(
        snapshot_dir,
        "dataset3",
        output_dir,
        "figure_2_1e",
        args.dpi,
        args.top_features,
    )
    print("[OK] Figure 2.1e - Dataset 3 FP/FN feature characteristics")

    plot_cross_model_overlap(snapshot_dir, output_dir, args.dpi)
    print("[OK] Figure 2.1f - cross-model error overlap")

    print()
    print("Created:")
    for path in sorted(output_dir.glob("figure_2_1*")):
        print(f"  {path.name}")
    print("=" * 92)


if __name__ == "__main__":
    main()
