"""
Disentanglement vs Convergence-Time Analysis

Example usage:

python src/shortcut/visualization/plot_disent_vs_time.py \
    --datasets morpho_mnist chexpert oct \
    --save figures/disent_vs_time.pdf

"""

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import brokenaxes
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from shortcut.utils import scale_rcparams, setup_fonts_and_style

# Map internal method names to figure labels
METHOD_NAME_MAP = {
    "baseline": "Baseline",
    "rebalancing": "Rebalancing",
    "adv_cl": "AdvCl",
    "adv_cl_reb": "AdvCl+Rebal",
    "mmd": "MMD",
    "mmd_reb": "MMD+Rebal",
    "dcor": "dCor",
    "dcor_reb": "dCor+Rebal",
    "mine": "MINE",
    "mine_reb": "MINE+Rebal",
}

# Map dataset folder names to figure titles
DATASET_NAME_MAP = {
    "morpho_mnist": "Morpho-MNIST",
    "chexpert": "CheXpert",
    "oct": "OCT",
}

# Define per-dataset x-axis limits and ticks
xlims_per_dataset = {
    "morpho_mnist": (-25, 500),
    "chexpert": (-2.5, 55),
    "oct": (0, 70),
}

xticks_per_dataset = {
    "morpho_mnist": np.arange(0, 500, 100),
    "chexpert": np.arange(0, 55, 10),
    "oct": np.arange(0, 70, 10),
}


def log(msg: str):
    print(f"[INFO] {msg}", flush=True)


def compute_disentanglement(confusion_csv: Path, chance_level: float = 0.5) -> dict:
    df = pd.read_csv(confusion_csv, index_col=0)
    values = df.values

    # subtract chance and take absolute value
    values = np.abs(values - chance_level)

    k = min(values.shape)

    diag = np.diag(values[:k, :k])

    mask = np.ones_like(values, dtype=bool)
    mask[:k, :k] &= ~np.eye(k, dtype=bool)
    off_diag = values[mask]

    mean_diag = diag.mean()
    mean_off = off_diag.mean()

    # avoid division by zero (pure chance everywhere)
    if mean_diag + mean_off == 0:
        diag_dominance = 0.5
    else:
        diag_dominance = mean_diag / (mean_diag + mean_off)

    return {
        "mean_diag_abs": mean_diag,
        "mean_off_abs": mean_off,
        "diag_dominance": diag_dominance,
    }


def pareto_frontier(x: np.ndarray, y: np.ndarray, maximize_y: bool) -> List[int]:
    idxs = []
    for i in range(len(x)):
        dominated = False
        for j in range(len(x)):
            if i == j:
                continue
            better_x = x[j] <= x[i]
            better_y = y[j] >= y[i] if maximize_y else y[j] <= y[i]
            strictly_better = (x[j] < x[i]) or (
                y[j] > y[i] if maximize_y else y[j] < y[i]
            )
            if better_x and better_y and strictly_better:
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot disentanglement performance vs convergence time"
    )
    parser.add_argument("--root", type=Path, default=Path("out"))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(18, 4),
    )
    parser.add_argument("--show_pareto_front", type=bool, default=False)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--latex_table", type=Path, default=None)
    return parser.parse_args()


def get_family_and_marker(method: str):
    """
    Determine method family for color and marker for +Rebal distinction.
    """
    method_lower = method.lower()
    if "+rebal" in method_lower or "_reb" in method_lower:
        marker = "s"  # square for +Rebal
        base = method.replace("+Rebal", "").replace("_reb", "")
    elif "rebalancing" in method:
        marker = "s"  # square for +Rebal
        base = method
    else:
        marker = "o"  # circle for normal
        base = method
    return base, marker


def main():
    args = parse_args()
    setup_fonts_and_style()

    log("Loading experiments")

    records = []
    dataset_paths: Dict[str, Path] = {Path(ds): args.root / ds for ds in args.datasets}

    for dataset_label, dataset_dir in dataset_paths.items():
        log(f"Scanning dataset '{dataset_label}' at {dataset_dir}")

        eval_dirs = list(dataset_dir.rglob("eval-version0"))
        log(f"Found {len(eval_dirs)} eval-version0 directories")

        for eval_dir in eval_dirs:
            confusion_csv = eval_dir / "subspace_confusion_matrix_mean.csv"
            if not confusion_csv.exists():
                log(f"Skipping {eval_dir}, confusion CSV not found")
                continue

            if eval_dir.parent.name == "cross_val":
                method_name = eval_dir.parent.parent.name
                aggregated_epochs_file = eval_dir.parent / "aggregated_epochs.txt"
                benchmark_csv = eval_dir.parent / "benchmark_summary.csv"
            else:
                method_name = eval_dir.parent.name
                aggregated_epochs_file = eval_dir.parent / "aggregated_epochs.txt"
                benchmark_csv = eval_dir.parent / "benchmark_summary.csv"

            if not aggregated_epochs_file.exists() or not benchmark_csv.exists():
                log(f"Skipping {eval_dir}, missing files")
                continue

            # Read number of epochs
            try:
                with open(aggregated_epochs_file, "r") as f:
                    n_epochs = float(f.read().strip())
            except Exception:
                log(f"Skipping {eval_dir}, cannot parse aggregated_epochs.txt")
                continue

            # Read time per epoch
            try:
                time_per_epoch = pd.read_csv(benchmark_csv)["mean_epoch_time_sec"].iloc[
                    0
                ]
            except Exception:
                log(f"Skipping {eval_dir}, cannot read benchmark_summary.csv")
                continue

            convergence_time = n_epochs * time_per_epoch

            # Compute disentanglement
            disent = compute_disentanglement(confusion_csv)

            records.append(
                {
                    "dataset": str(dataset_label),
                    "method": method_name,
                    "n_epochs": n_epochs,
                    "time_per_epoch": time_per_epoch / 60.0,
                    "convergence_time": convergence_time / 60.0,
                    **disent,
                }
            )

    if not records:
        raise RuntimeError("No valid experiments found.")

    df = pd.DataFrame(records)

    datasets = args.datasets  # preserve order
    all_methods = sorted(df["method"].unique())

    # Custom colors
    colors = [
        "#000000",
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#D55E00",
    ]

    # Assign color per method-family
    families = sorted(set(get_family_and_marker(m)[0] for m in all_methods))

    # Separate baseline and rebalancing
    special = ["baseline", "rebalancing"]
    other_families = [f for f in families if f not in special]

    # Combine: baseline, rebalancing, then alphabetically sorted others
    sorted_families = special + sorted(other_families)
    family_color = {
        fam: colors[i % len(colors)] for i, fam in enumerate(sorted_families)
    }

    # Horizontal subplots
    fig_width = args.figsize[0]
    fig_height = args.figsize[1]
    fig = plt.figure(figsize=(fig_width, fig_height))

    n = len(datasets)
    gs = gridspec.GridSpec(1, n, figure=fig, wspace=0.1)  # 1 row, n columns
    ymin, ymax = 0.5, 1.01

    axes = []
    dataset_dfs = {}
    ylabel = "Disentanglement (↑)"


    for i, dataset in enumerate(datasets):
        dfd = df[df["dataset"] == dataset].reset_index(drop=True)
        dataset_dfs[dataset] = dfd

        y = dfd["diag_dominance"].values
        maximize_y = True

        x = dfd["convergence_time"].values

        if args.show_pareto_front:
            dfd["is_pareto"] = False
            pareto_idx = pareto_frontier(x, y, maximize_y)
            dfd.loc[pareto_idx, "is_pareto"] = True
        
        display_name = DATASET_NAME_MAP.get(dataset, dataset)

        if i == 0:
            # First subplot: broken x-axis
            bax = brokenaxes.brokenaxes(
                xlims=((-25, 200), (400, 600)),
                hspace=0.05,
                wspace=0.3,
                despine=False,
                subplot_spec=gs[i]
            )

            for j, row in dfd.iterrows():
                family, marker = get_family_and_marker(row["method"])
                bax.scatter(
                    row["convergence_time"],
                    y[j],
                    color=family_color[family],
                    marker=marker,
                    alpha=0.75,
                    s=200,
                )
                if args.show_pareto_front and row["is_pareto"]:
                    bax.scatter(
                        row["convergence_time"],
                        row["diag_dominance"],
                        facecolors="none",
                        edgecolors="black",
                        linewidths=2.0,
                        marker=marker,
                        s=200,
                    )

            bax.set_xlabel("Convergence time (minutes ↓)", labelpad=31)

            for k, subax in enumerate(bax.axs):
                # Remove top/right spines
                subax.spines["top"].set_visible(False)
                subax.spines["right"].set_visible(False)
                subax.spines["left"].set_linewidth(1.2)
                subax.spines["bottom"].set_linewidth(1.2)

                # Grid & ticks
                subax.grid(True, which="major", color="gray", alpha=0.3)
                subax.minorticks_off()
                subax.tick_params(axis="both", which="major", direction="out", length=6)

                # Shared y-limits
                subax.set_ylim(ymin, ymax)
                subax.yaxis.set_major_locator(MultipleLocator(0.1))
                if k > 0:
                    subax.tick_params(axis="y", labelleft=False)

            bax.axs[0].set_ylabel(ylabel, labelpad=15)
            bax.axs[0].set_title(display_name)
            axes.append(bax)
        else:
            # Normal subplot
            ax = fig.add_subplot(gs[i])
            for j, row in dfd.iterrows():
                family, marker = get_family_and_marker(row["method"])
                ax.scatter(
                    row["convergence_time"],
                    y[j],
                    color=family_color[family],
                    marker=marker,
                    alpha=0.75,
                    s=200,
                )
                if args.show_pareto_front and row["is_pareto"]:
                    ax.scatter(
                        row["convergence_time"],
                        row["diag_dominance"],
                        facecolors="none",
                        edgecolors="black",
                        linewidths=2.0,
                        marker=marker,
                        s=200,
                    )

            ax.set_xlabel("Convergence time (minutes ↓)")
            ax.set_ylim(ymin, ymax)
            ax.grid(True, which="major", color="gray", alpha=0.3)
            ax.minorticks_off()
            ax.tick_params(axis="both", which="major", direction="out", length=6)
            ax.tick_params(axis="y", which="major", labelleft=False, left=True)

            xlim = xlims_per_dataset.get(dataset, (0, max(dfd["convergence_time"])*1.05))
            xticks = xticks_per_dataset.get(dataset, np.linspace(0, xlim[1], 6))
            ax.set_xlim(*xlim)
            ax.set_xticks(xticks)

            # Remove top/right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

            ax.set_title(display_name)
            axes.append(ax)
        
    # Baseline -> Rebalancing -> others alphabetically
    legend_order = []
    for m in ["baseline", "rebalancing"]:
        if m in all_methods:
            legend_order.append(m)
    for m in all_methods:
        if m not in legend_order:
            legend_order.append(m)

    handles = []
    table_rows = []
    for method in legend_order:
        family, marker = get_family_and_marker(method)
        # Map method name to figure label
        method_fig_name = METHOD_NAME_MAP.get(method, method)
        row = {"Method": METHOD_NAME_MAP.get(method, method)}
        label = method_fig_name
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color=family_color[family],
                linestyle="",
                linewidth=2,
                markersize=12,
                alpha=0.8,
                label=label,
            )
        )
        for dataset_key, dfd in dataset_dfs.items():
            match = dfd[dfd["method"] == method]

            if len(match) == 0:
                # method missing for this dataset
                row[(dataset_key, "Epochs")] = "--"
                row[(dataset_key, "Time")] = "--"
            else:
                row[(dataset_key, "Epochs")] = math.ceil((match["n_epochs"].iloc[0]))
                row[(dataset_key, "Time")] = math.ceil(
                    (match["convergence_time"].iloc[0])
                )

        table_rows.append(row)

    scale_rcparams(fig_width, fig_height, scale_factor=1.8)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, len(handles)),
        frameon=False,
        fontsize=plt.rcParams["axes.labelsize"],
        bbox_to_anchor=(0.5, -0.3),
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    if args.save:
        plt.savefig(args.save, bbox_inches="tight")
        log(f"Saved figure to {args.save}")
    else:
        plt.show()

    table_df = pd.DataFrame(table_rows)

    # Create a MultiIndex for columns: first "Method", then dataset/metric pairs
    table_df.columns = pd.MultiIndex.from_tuples(
        [("Method", "")]
        + [(ds, col) for ds in dataset_dfs.keys() for col in ["Epochs", "Time"]]
    )

    latex_str = table_df.to_latex(
        index=False,
        multicolumn=True,
        multicolumn_format="c",
        escape=False,
        column_format="l" + "rr" * len(dataset_dfs),
    )

    # Save to file
    if args.latex_table is not None:
        latex_table = f"""
        \\begin{{table}}[htbp]
            \\centering
            \\caption{{Comparison of number of epochs and compute time in minutes until convergence.}}
            \\label{{tab:compute-time}}
        {latex_str}
        \\end{{table}}
        """
        with open(args.latex_table, "w") as f:
            f.write(latex_table)

    log("Done.")


if __name__ == "__main__":
    main()
