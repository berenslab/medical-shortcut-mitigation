"""
Prevalence ablation plot.

Example usage:
---------------
python src/shortcut/visualization/plot_prevalence_ablation.py \
    --csvs out/prevalence_ablation/morpho_mnist.csv \
        out/prevalence_ablation/chexpert.csv \
        out/prevalence_ablation/oct.csv \
    --labels MNIST CheXpert OCT

Minimal valid example CSV:
---------------
Method,Prev1_mean,Prev5_mean,Prev10_mean,Prev25_mean,Prev50_mean
Baseline,72.1,73.0,74.2,75.6,76.1
Rebalancing,72.5,73.4,74.8,76.1,76.9
AdvCl,73.0,73.8,75.1,76.4,77.2
AdvCl+Rebal,73.4,74.2,75.6,76.9,77.8
MMD,72.8,73.5,74.9,76.0,76.8
MMD+Rebal,73.2,74.0,75.3,76.5,77.4
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from shortcut.utils import scale_rcparams, setup_fonts_and_style


def extract_prevs(df):
    return sorted(
        int(c.replace("Prev", "").replace("_mean", ""))
        for c in df.columns
        if c.endswith("_mean")
    )


def save_figure(outdir, name, formats=("png", "pdf"), dpi=300):
    os.makedirs(outdir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(outdir, f"{name}.{fmt}")
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")


def load_datasets(csvs, labels):
    datasets = {}
    for i, csv in enumerate(csvs):
        label = labels[i] if labels else os.path.splitext(os.path.basename(csv))[0]
        datasets[label] = pd.read_csv(csv)
    return datasets


def plot_relative_drop(datasets, outdir, formats, use_symlog=False, zero_line=True):
    """
    Plot relative AUROC improvement over Baseline for multiple datasets (side-by-side),
    with clean axes, light grid, large markers, and well-ordered legend.
    """

    n_datasets = len(datasets)
    fig_width, fig_height = 5 * n_datasets, 5
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_datasets,
        figsize=(fig_width, fig_height),
        sharey=True,
    )

    scale_rcparams(fig_width, fig_height, scale_factor=1.8)

    if n_datasets == 1:
        axes = [axes]

    # Colorblind-friendly palette
    colors = [
        "#000000",
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#D55E00",
        "#56B4E9",
        "#F0E442",
    ]
    markers = ["o", "^", "s", "D", "v", "P", "X", "*"]

    # Collect base methods globally
    method_families = []
    for df in datasets.values():
        for m in df["Method"]:
            base = m.replace("+Rebal", "")
            if base not in method_families:
                method_families.append(base)

    color_dict = {m: colors[i % len(colors)] for i, m in enumerate(method_families)}
    marker_dict = {m: markers[i % len(markers)] for i, m in enumerate(method_families)}

    # Track global y-limits
    global_min, global_max = float("inf"), float("-inf")

    # Plot each dataset
    for ax, (dataset, df) in zip(axes, datasets.items()):
        prevs = extract_prevs(df)
        baseline_row = df[df["Method"] == "Baseline"].iloc[0]

        for _, row in df.iterrows():
            method = row["Method"]
            if method == "Baseline":
                continue

            if "+Rebal" in method:
                base_method = method.replace("+Rebal", "")
                linestyle = "--"
            else:
                base_method = method
                linestyle = "-"

            drops = [
                row[f"Prev{p}_mean"] - baseline_row[f"Prev{p}_mean"] for p in prevs
            ]
            global_min = min(global_min, min(drops))
            global_max = max(global_max, max(drops))

            ax.plot(
                prevs,
                drops,
                color=color_dict[base_method],
                marker=marker_dict[base_method],
                linestyle=linestyle,
                linewidth=2,
                markersize=9,
                alpha=0.8,
            )

        ax.set_title(dataset)
        ax.set_xlabel("Conditional prevalence (%)")

        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        # Light grid lines
        ax.grid(True, which="major", color="gray", alpha=0.3)
        ax.minorticks_off()

        # Ticks
        ax.tick_params(axis="both", which="major", direction="out", length=6)

        # Explicit x-ticks for every column
        ax.set_xticks(prevs)
        ax.set_xticklabels(prevs)

        # Zero line
        if zero_line:
            ax.axhline(0, color="black", linestyle=":", linewidth=2)

        if use_symlog:
            ax.set_yscale("symlog", linthresh=0.5)

    # Shared y-label
    axes[0].set_ylabel("Δ AUROC (%)")

    # Global y-limits already computed
    pad = 0.05 * (global_max - global_min)
    for ax in axes:
        ax.set_ylim(global_min - pad, global_max + pad)

        # Explicit y-axis ticks every 10 units
        ymin, ymax = ax.get_ylim()
        yticks = np.arange(np.floor(ymin / 10) * 10, np.ceil(ymax / 10) * 10 + 1, 10)
        ax.set_yticks(yticks)

    # -----------------------
    # Build legend handles in desired order
    legend_order = [
        ["Baseline", "Rebalancing"],
        ["AdvCl", "AdvCl+Rebal"],
        ["dCor", "dCor+Rebal"],
        ["MINE", "MINE+Rebal"],
        ["MMD", "MMD+Rebal"],
    ]

    legend_elements = []

    for group in legend_order:
        for m in group:
            if m in df["Method"].values:
                if m == "Baseline":
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            color="black",
                            linestyle=":",
                            linewidth=2,
                            label="Baseline",
                        )
                    )
                else:
                    base_method = m.replace("+Rebal", "")
                    linestyle = "--" if "+Rebal" in m else "-"
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            color=color_dict[base_method],
                            marker=marker_dict[base_method],
                            linestyle=linestyle,
                            linewidth=2,
                            markersize=9,
                            label=m,
                        )
                    )

    # Create figure legend slightly lower
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,  # one column per group
        frameon=False,
        fontsize=plt.rcParams["axes.labelsize"],
        bbox_to_anchor=(0.5, -0.08),  # lower y-position
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    save_figure(outdir, "relative_drop", formats)
    plt.close()


def main():
    setup_fonts_and_style()

    parser = argparse.ArgumentParser(description="Plot correlation strength robustness")
    parser.add_argument("--csvs", nargs="+", required=True, help="CSV files")
    parser.add_argument("--labels", nargs="+", help="Optional dataset labels")
    parser.add_argument(
        "--outdir", default="figures/prevalence_ablation", help="Output directory"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="File formats (e.g. png pdf)",
    )

    args = parser.parse_args()
    datasets = load_datasets(args.csvs, args.labels)

    plot_relative_drop(datasets, args.outdir, args.formats)


if __name__ == "__main__":
    main()
