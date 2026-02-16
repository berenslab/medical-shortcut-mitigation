import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from shortcut.config import load_yaml_config
from shortcut.data.datamodule import get_test_dataset, get_train_val_dataset
from shortcut.training import LightningWrapper


def plot_2d_scatter(
    embedding,
    labels,
    split_indices,
    subspace_id,
    label_id,
    output_dir,
    fold_idx,
    test: bool = True,
):
    colors = [(0.96, 0.06, 0.58), (0, 0.59, 0.8)]  # R -> G -> B
    custom_colors = LinearSegmentedColormap.from_list("custom_colors", colors, N=2)

    subspace_names = [r"$z_1$", r"$z_2$"]
    label_names = [r"$y_1$", r"$y_2$"]

    start, end = split_indices[subspace_id], split_indices[subspace_id + 1]
    if (end - start) != 2:
        return  # Only plot 2D subspaces
    test_sub = embedding[:, start:end].numpy()

    # if i == j:
    #    return  # skip diagonal (correct subspace-label pair)
    y_labels = labels[:, label_id]
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        test_sub[:, 0],
        test_sub[:, 1],
        c=y_labels,
        cmap=custom_colors,
        s=0.4,
    )
    handles, _ = scatter.legend_elements()
    plt.legend(
        handles=handles,
        labels=[f"{label_names[label_id]}=0", f"{label_names[label_id]}=1"],
    )
    plt.title(
        f"Subspace {subspace_names[subspace_id]} (2D) colored by Label {label_names[label_id]}"
    )
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    name = "scatter"
    if test:
        name = "test_" + name
    else:
        name = "train_" + name
    fname = os.path.join(
        output_dir, f"{name}_subspace{subspace_id}_label{label_id}_fold{fold_idx}.png"
    )
    plt.savefig(fname)
    plt.close()
    print(f"[✓] Saved scatter plot: {fname}")


def evaluate_knn(cfg: dict, version: int, output_csv: str, k_neighbors: int):
    folds = cfg.training.num_folds
    base_dir = cfg.out_dir
    save_dir = os.path.join(base_dir, f"eval-version{version}")
    os.makedirs(save_dir, exist_ok=True)

    train_dataset, _ = get_train_val_dataset(cfg)
    _, test_dataset, _ = get_test_dataset(cfg)  # only the balanced dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.test_data.num_workers,
        prefetch_factor=cfg.test_data.prefetch_factor,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_data.batch_size,
        shuffle=False,
        num_workers=cfg.test_data.num_workers,
        prefetch_factor=cfg.test_data.prefetch_factor,
    )

    subspace_dims = cfg.model.subspace_dims
    num_labels = len(cfg.model.class_dims)
    num_subspaces = len(subspace_dims)
    split_indices = np.cumsum([0] + subspace_dims)
    confusion_tensor = np.zeros((folds, num_labels, num_subspaces))  # label x subspace

    results = []

    for fold_idx, fold in enumerate(range(folds)):
        with open(
            os.path.join(
                base_dir,
                f"fold{fold}",
                "lightning_logs",
                f"version_{version}",
                "best_ckpt.txt",
            ),
            "r",
        ) as file:
            ckpt_path = file.read().rstrip()

        print(f"[+] Evaluating Fold {fold} | Checkpoint: {ckpt_path}")

        model = LightningWrapper.load_from_checkpoint(ckpt_path)
        trainer = Trainer(accelerator="auto", devices=1, logger=False)

        train_emb = torch.cat(trainer.predict(model, train_loader))
        test_emb = torch.cat(trainer.predict(model, test_loader))

        train_labels_matrix = torch.stack(
            [train_dataset[i][1] for i in range(len(train_dataset))], dim=0
        ).numpy()
        test_labels_matrix = torch.stack(
            [test_dataset[i][1] for i in range(len(test_dataset))], dim=0
        ).numpy()

        for j in range(num_labels):  # label index (row)
            y_train = train_labels_matrix[:, j]
            y_test = test_labels_matrix[:, j]
            for i in range(num_subspaces):  # subspace index (column)
                start, end = split_indices[i], split_indices[i + 1]
                train_sub = train_emb[:, start:end].numpy()
                test_sub = test_emb[:, start:end].numpy()

                knn = KNeighborsClassifier(n_neighbors=k_neighbors)
                knn.fit(train_sub, y_train)
                y_pred = knn.predict(test_sub)
                acc = accuracy_score(y_true=y_test, y_pred=y_pred)
                balanced_acc = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
                confusion_tensor[fold_idx, j, i] = balanced_acc

                results.append(
                    {
                        "fold": fold,
                        "subspace": i,
                        "label": j,
                        "version": version,
                        "checkpoint_path": ckpt_path,
                        "accuracy": acc,
                        "balanced_acc": balanced_acc,
                    }
                )

                plot_2d_scatter(
                    embedding=train_emb,
                    labels=train_labels_matrix,
                    split_indices=split_indices,
                    subspace_id=i,
                    label_id=j,
                    output_dir=save_dir,
                    fold_idx=fold_idx,
                    test=False,
                )
                plot_2d_scatter(
                    embedding=test_emb,
                    labels=test_labels_matrix,
                    split_indices=split_indices,
                    subspace_id=i,
                    label_id=j,
                    output_dir=save_dir,
                    fold_idx=fold_idx,
                    test=True,
                )

    # Compute mean and stddev
    confusion_mean = np.mean(confusion_tensor, axis=0)
    confusion_std = np.std(confusion_tensor, axis=0)

    index_labels = [f"label_{j}" for j in range(num_labels)]
    col_labels = [f"subspace_{i}" for i in range(num_subspaces)]

    pd.DataFrame(confusion_mean, index=index_labels, columns=col_labels).to_csv(
        os.path.join(save_dir, "subspace_confusion_matrix_mean.csv")
    )
    pd.DataFrame(confusion_std, index=index_labels, columns=col_labels).to_csv(
        os.path.join(save_dir, "subspace_confusion_matrix_std.csv")
    )

    print("\n[✓] Saved mean confusion matrix to: subspace_confusion_matrix_mean.csv")
    print("\n[✓] Saved stddev confusion matrix to: subspace_confusion_matrix_std.csv")

    if results:
        diag_accs = [r["accuracy"] for r in results if r["subspace"] == r["label"]]
        mean_acc = np.mean(diag_accs)
        std_acc = np.std(diag_accs)
        diag_bal_accs = [r["balanced_acc"] for r in results if r["subspace"] == r["label"]]
        mean_bal_acc = np.mean(diag_bal_accs)
        std_bal_acc = np.std(diag_bal_accs)
        print(f"\n[✓] Average Diagonal Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"\n[✓] Average Diagonal Balanced Accuracy: {mean_bal_acc:.4f} ± {std_bal_acc:.4f}")
    else:
        mean_acc = std_acc = mean_bal_acc = std_bal_acc = None

    output_csv = os.path.join(save_dir, output_csv)
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "fold",
                "subspace",
                "label",
                "version",
                "checkpoint_path",
                "accuracy",
                "balanced_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
        if mean_acc is not None:
            writer.writerow(
                {
                    "fold": "average diagonal accuracy",
                    "subspace": "-",
                    "label": "-",
                    "version": version,
                    "checkpoint_path": "-",
                    "accuracy": mean_acc,
                    "balanced_acc": mean_bal_acc,
                }
            )

    print(f"\n[✓] Saved per-fold results to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate kNN classifier on subspace embeddings."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory with fold*/lightning_logs/version_x/",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0,
        help="Version of fold.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="knn_results.csv",
        help="Output CSV for per-fold results",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to dataset.",
    )
    parser.add_argument("--k", type=int, default=30, help="Number of neighbors for kNN")

    args = parser.parse_args()
    cfg = OmegaConf.create(
        load_yaml_config(
            os.path.join(
                args.root_dir,
                f"fold0/lightning_logs/version_{args.version}/hparams.yaml",
            ),
        ),
    )
    if args.dataset_dir is not None:
        cfg.data.dataset_path = args.dataset_dir

    evaluate_knn(
        cfg=cfg,
        version=args.version,
        output_csv=args.output_csv,
        k_neighbors=args.k,
    )
