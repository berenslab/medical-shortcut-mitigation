import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from lightning.pytorch import Trainer
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from shortcut.config import load_yaml_config
from shortcut.data.datamodule import get_test_dataset
from shortcut.training import LightningWrapper
from shortcut.utils import setup_fonts_and_style, scale_rcparams

"""Example usage:
Single dataset:
python src/shortcut/visualization/scatterplots_disentanglement.py \
  --exp_roots data/exp1 \
  --versions 0 \
  --folds 0 \
  --data_roots path

Multiple datasets:
python src/shortcut/visualization/scatterplots_disentanglement.py \
  --exp_roots data/exp1 data/exp2 data/exp3 \
  --versions 0 1 2 \
  --folds 0 0 1 \
  --data_roots pathA pathB pathC
"""


METHOD_MAP = {
    "Baseline": ["baseline"],
    "Rebalancing": ["rebalancing"],
    "AdvCl": ["adv_cl", "advcl"],
    "AdvCl+Reb": ["adv_cl_reb", "advcl_reb"],
    "dCor": ["dcor"],
    "dCor+Reb": ["dcor_reb"],
    "MINE": ["mine"],
    "MINE+Reb": ["mine_reb"],
    "MMD": ["mmd"],
    "MMD+Reb": ["mmd_reb"],
}

def generate_scatter(
    embedding,
    labels,
    split_indices,
    ax,
    subspace_id=0,
    label_id=1,
):
    start, end = split_indices[subspace_id], split_indices[subspace_id + 1]
    sub_emb = embedding[:, start:end]

    if sub_emb.shape[1] < 2:
        ax.text(0.5, 0.5, "Subspace < 2 dims", ha="center", va="center")
        ax.axis("off")
        return None, None

    sub_xy = sub_emb[:, :2].cpu().numpy()
    y = labels[:, label_id]

    colors = [(0.96, 0.06, 0.58), (0, 0.59, 0.8)]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=2)

    sc = ax.scatter(
        sub_xy[:, 0],
        sub_xy[:, 1],
        c=y,
        cmap=cmap,
        s=20,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.axis("off")

    return sc, y


def process_method(method_folder, version, fold, data_path):
    exp_root = method_folder
    if not os.path.isdir(os.path.join(exp_root, f"fold{fold}")):
        for sub in os.listdir(method_folder):
            cand = os.path.join(method_folder, sub)
            if os.path.isdir(os.path.join(cand, f"fold{fold}")):
                exp_root = cand
                break

    fold_dir = os.path.join(
        exp_root,
        f"fold{fold}",
        "lightning_logs",
        f"version_{version}",
    )

    hparams_path = os.path.join(fold_dir, "hparams.yaml")
    if not os.path.exists(hparams_path):
        print(f"Missing hparams.yaml in {fold_dir}")
        return None

    cfg = OmegaConf.create(load_yaml_config(hparams_path))

    if data_path is not None:
        cfg.data.dataset_path = data_path

    _, test_dataset, _ = get_test_dataset(cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_data.batch_size,
        shuffle=False,
        num_workers=cfg.test_data.num_workers,
        prefetch_factor=cfg.test_data.prefetch_factor,
    )

    ckpt_file = os.path.join(fold_dir, "best_ckpt.txt")
    if not os.path.exists(ckpt_file):
        print(f"Missing best_ckpt.txt in {fold_dir}")
        return None

    ckpt_path = open(ckpt_file).read().strip()
    model = LightningWrapper.load_from_checkpoint(ckpt_path)
    trainer = Trainer(accelerator="auto", devices=1, logger=False)

    preds = trainer.predict(model, test_loader)
    embedding = torch.cat(preds)

    labels = torch.stack(
        [test_dataset[i][1] for i in range(len(test_dataset))],
        dim=0,
    ).numpy()

    subspace_dims = list(cfg.model.subspace_dims)
    split_indices = np.cumsum([0] + subspace_dims)

    return embedding, labels, split_indices


def plot_grid_multi(
    exp_roots,
    versions,
    folds,
    data_roots,
    row_titles,
    output,
    subspace_id=0,
    label_id=1,
    label_names=(r"$y_1$", r"$y_2$"),
):
    n_rows = len(exp_roots)
    n_cols = len(METHOD_MAP)

    fig_width = 4 * n_cols
    fig_height = 4 * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    scale_rcparams(fig_width, fig_height, scale_factor=3.5)

    shared_scatter = None

    for row_idx, (dataset_root, version, fold, data_path) in enumerate(
        zip(exp_roots, versions, folds, data_roots)
    ):
        print(f"Prepare scatterplots for {dataset_root}.")
        for col_idx, (display, folder_aliases) in enumerate(METHOD_MAP.items()):
            ax = axes[row_idx, col_idx]

            method_folder = None
            for name in folder_aliases:
                cand = os.path.join(dataset_root, name)
                if os.path.exists(cand):
                    method_folder = cand
                    break

            if method_folder is None:
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(display)
                continue

            data = process_method(
                method_folder,
                version=version,
                fold=fold,
                data_path=data_path,
            )

            if data is None:
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(display)
                continue

            emb, labels, split_indices = data

            sc, _ = generate_scatter(
                emb,
                labels,
                split_indices,
                ax,
                subspace_id=subspace_id,
                label_id=label_id,
            )

            if row_idx == 0:
                ax.set_title(display)

            if col_idx == 0:
                ax.annotate(
                    row_titles[row_idx],
                    xy=(0, 0.5),
                    xycoords="axes fraction",
                    xytext=(-200, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    rotation=0,
                    fontsize=plt.rcParams["axes.titlesize"],
                )

            if shared_scatter is None and sc is not None:
                shared_scatter = sc

    if shared_scatter is not None:
        legend_labels = [
            rf"${label_names[label_id].strip('$')} = 0$",
            rf"${label_names[label_id].strip('$')} = 1$",
        ]
        handles, _ = shared_scatter.legend_elements()
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
            markerscale=0.5,
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved grid: {output}")


if __name__ == "__main__":
    setup_fonts_and_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_roots", nargs="+", required=True)
    parser.add_argument("--versions", nargs="+", type=int, required=True)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--data_roots", nargs="+", default=None)
    parser.add_argument("--output", default="figures/combined_scatter.pdf")
    parser.add_argument(
        "--row_titles",
        nargs="+",
        default=None,
        help="Optional custom row titles (one per dataset)"
    )


    args = parser.parse_args()

    n = len(args.exp_roots)

    if not (len(args.versions) == len(args.folds) == n):
        raise ValueError("exp_roots, versions, and folds must have same length")

    if args.data_roots is None:
        data_roots = [None] * n
    else:
        if len(args.data_roots) != n:
            raise ValueError("data_roots must match exp_roots length")
        data_roots = [
            None if (p is None or p.lower() == "none" or p == "")
            else p
            for p in args.data_roots
        ]

    if args.row_titles is None:
        row_titles = [
            os.path.basename(root.rstrip("/"))
            for root in args.exp_roots
        ]
    else:
        if len(args.row_titles) != n:
            raise ValueError("row_titles must match exp_roots length")
        row_titles = args.row_titles


    plot_grid_multi(
        exp_roots=args.exp_roots,
        versions=args.versions,
        folds=args.folds,
        data_roots=data_roots,
        row_titles=row_titles,
        output=args.output,
    )