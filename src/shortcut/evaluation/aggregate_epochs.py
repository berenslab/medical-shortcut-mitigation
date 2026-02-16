import os
import re
import argparse

def extract_epoch(ckpt_path: str) -> int:
    """Extracts the epoch number from a checkpoint filename of the form epoch=17.ckpt.
    
    Returns an integer epoch.
    """
    match = re.search(r"epoch=(\d+)\.ckpt", ckpt_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not parse epoch from checkpoint path: {ckpt_path}")


def get_mean_epoch(experiment_root: str) -> float:
    """Given an experiment directory containing fold subfolders, compute the mean of
    best checkpoint epochs across folds.
    """

    experiment_root = experiment_root.rstrip("/")
    fold_epochs = []

    # Detect fold directories (fold0, fold1, ...)
    for item in sorted(os.listdir(experiment_root)):
        fold_dir = os.path.join(experiment_root, item)
        if not os.path.isdir(fold_dir) or not item.startswith("fold"):
            continue

        print(f"Processing {fold_dir} ...")

        latest_log_file = os.path.join(fold_dir, "latest_log_dir.txt")
        if not os.path.exists(latest_log_file):
            raise FileNotFoundError(f"latest_log_dir.txt not found in {fold_dir}")

        with open(latest_log_file, "r") as f:
            latest_log_dir = f.read().strip()

        best_ckpt_file = os.path.join(latest_log_dir, "best_ckpt.txt")
        if not os.path.exists(best_ckpt_file):
            raise FileNotFoundError(f"best_ckpt.txt not found in {latest_log_dir}")

        with open(best_ckpt_file, "r") as f:
            best_ckpt_path = f.read().strip()

        epoch_num = extract_epoch(best_ckpt_path) + 1
        fold_epochs.append(epoch_num)

    if len(fold_epochs) == 0:
        raise RuntimeError("No folds found or no epochs extracted.")

    mean_epoch = sum(fold_epochs) / len(fold_epochs)
    return mean_epoch


def save_epoch(experiment_root: str, mean_epoch: float):
    """Saves the mean epoch to aggregated_epochs.txt inside the experiment root."""
    output_path = os.path.join(experiment_root, "aggregated_epochs.txt")
    with open(output_path, "w") as f:
        f.write(f"{mean_epoch:.4f}\n")

    print(f"\nSaved mean epoch to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean epoch of best_ckpt across folds.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory with fold*/lightning_logs/version_x/",
    )
    args = parser.parse_args()

    mean_epoch = get_mean_epoch(args.root_dir)
    print(f"\nMean best checkpoint epoch: {mean_epoch:.2f}")

    save_epoch(args.root_dir, mean_epoch)
