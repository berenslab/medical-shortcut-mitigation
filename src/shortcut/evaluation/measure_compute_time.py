import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml


def prepare_config_for_benchmarking(
    hparams_file,
    fold_index,
    max_epochs=5,
    devices="0",
    dataset_path=None,
):
    with open(hparams_file) as f:
        hparams = yaml.safe_load(f)

    cfg_content = hparams.get("cfg", {})

    cfg_content["training"]["fold"] = fold_index
    cfg_content["max_epochs"] = max_epochs
    cfg_content["devices"] = [int(d) for d in str(devices).split(",")]
    cfg_content["benchmark_timing"] = True
    cfg_content["training"]["check_val_every_n_epoch"] = 999999
    cfg_content["training"]["warmup_epochs"] = 0
    cfg_content["training"]["start_saving_epoch"] = 0

    if dataset_path is not None:
        if "data" not in cfg_content:
            cfg_content["data"] = {}
        cfg_content["data"]["dataset_path"] = dataset_path

    fold_path = Path(cfg_content["out_dir"])
    benchmark_folder = fold_path / "benchmark_logs"
    benchmark_folder.mkdir(parents=True, exist_ok=True)

    cfg_content["default_root_dir"] = str(benchmark_folder)
    cfg_content["out_dir"] = str(benchmark_folder)

    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml")
    yaml.safe_dump(cfg_content, temp_file)
    temp_file.close()
    return temp_file.name, benchmark_folder


def run_training(
    fold_path,
    train_script="train.py",
    max_epochs=5,
    devices="0",
    dataset_path=None,
):
    latest_file = fold_path / "latest_log_dir.txt"
    if not latest_file.exists():
        raise FileNotFoundError(f"{latest_file} missing for {fold_path}")

    version_path = Path(latest_file.read_text().strip())
    hparams_file = version_path / "hparams.yaml"
    if not hparams_file.exists():
        raise FileNotFoundError(f"{hparams_file} missing")

    temp_config, benchmark_folder = prepare_config_for_benchmarking(
        hparams_file,
        fold_index=int(fold_path.name[-1]),
        max_epochs=max_epochs,
        devices=devices,
        dataset_path=dataset_path,
    )

    cmd = ["python", train_script, "--cfg", temp_config]
    print(f"Running training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    benchmark_folder = fold_path.parent / "benchmark_logs" / fold_path.name
    version_parent = benchmark_folder / "lightning_logs"
    if not version_parent.exists():
        raise FileNotFoundError(f"No lightning_logs folder found in {benchmark_folder}")

    versions = sorted(version_parent.glob("version_*"), key=lambda x: x.name)
    benchmark_version_path = versions[-1]

    os.remove(temp_config)

    return benchmark_version_path


def load_epoch_times(version_path, benchmark_epochs_file="epoch_times.txt"):
    epoch_file = version_path / benchmark_epochs_file
    if not epoch_file.exists():
        raise FileNotFoundError(f"{epoch_file} missing in {version_path}")

    times = np.loadtxt(epoch_file)
    if len(times) > 1:
        times = times[1:]
    return times.mean()


def get_method_name(version_path, fallback_name):
    hparams_file = version_path / "hparams.yaml"
    if hparams_file.exists():
        with open(hparams_file) as f:
            hparams = yaml.safe_load(f)
            return hparams.get("cfg", {}).get("model", {}).get("method", fallback_name)
    return fallback_name


def discover_folds(experiment_path):
    experiment_path = Path(experiment_path)
    fold_dirs = sorted(
        [p for p in experiment_path.iterdir() if p.is_dir() and p.name.startswith("fold")]
    )
    if not fold_dirs:
        raise ValueError(f"No fold folders found in {experiment_path}")
    return fold_dirs


def benchmark_experiment_folder(
    experiment_path,
    train_script="train.py",
    max_epochs=5,
    benchmark_epochs_file="epoch_times.txt",
    devices="0",
    dataset_path=None,
):
    experiment_path = Path(experiment_path)
    fold_dirs = discover_folds(experiment_path)
    method_name_fallback = experiment_path.name
    fold_means = []

    version_path_first_fold = None

    for fold_path in fold_dirs:
        version_path = run_training(
            fold_path,
            train_script=train_script,
            max_epochs=max_epochs,
            devices=devices,
            dataset_path=dataset_path,
        )
        mean_time = load_epoch_times(version_path, benchmark_epochs_file)
        fold_means.append(mean_time)
        print(f"Fold {fold_path.name} mean epoch time: {mean_time:.4f} s")

        if version_path_first_fold is None:
            version_path_first_fold = version_path

    method_name = get_method_name(version_path_first_fold, method_name_fallback)
    method_mean = float(np.mean(fold_means))
    print(f"Mean epoch time for {method_name}: {method_mean:.4f} s")

    experiment_path.mkdir(parents=True, exist_ok=True)
    with open(experiment_path / "aggregated_epoch_times.txt", "w") as f:
        f.write(f"{method_mean:.6f}\n")

    return method_name, method_mean


def benchmark_path(
    path,
    train_script="train.py",
    max_epochs=5,
    benchmark_epochs_file="epoch_times.txt",
    devices="0",
    dataset_path=None,
):
    path = Path(path)
    results = []

    if any(p.name.startswith("fold") for p in path.iterdir() if p.is_dir()):
        result = benchmark_experiment_folder(
            path,
            train_script=train_script,
            max_epochs=max_epochs,
            benchmark_epochs_file=benchmark_epochs_file,
            devices=devices,
            dataset_path=dataset_path,
        )
        results.append(result)
    else:
        methods = [p for p in path.iterdir() if p.is_dir()]
        for method_path in methods:
            result = benchmark_experiment_folder(
                method_path,
                train_script=train_script,
                max_epochs=max_epochs,
                benchmark_epochs_file=benchmark_epochs_file,
                devices=devices,
                dataset_path=dataset_path,
            )
            results.append(result)

    summary_file = path / "benchmark_summary.csv"
    with open(summary_file, "w") as f:
        f.write("method,mean_epoch_time_sec\n")
        for m, v in results:
            f.write(f"{m},{v:.6f}\n")

    print(f"\n=== Benchmarking complete ===")
    print(f"Global summary saved to {summary_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark experiments or single experiment folder."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Experiment folder or root folder containing multiple experiments",
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default="train.py",
        help="Path to the training script",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5,
        help="Number of epochs to run per fold for benchmarking",
    )
    parser.add_argument(
        "--benchmark_epochs_file",
        type=str,
        default="epoch_times.txt",
        help="File containing per-epoch durations",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Comma-separated list of GPU device indices",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override cfg.data.dataset_path in the loaded config",
    )

    args = parser.parse_args()

    benchmark_path(
        args.path,
        train_script=args.train_script,
        max_epochs=args.max_epochs,
        benchmark_epochs_file=args.benchmark_epochs_file,
        devices=args.devices,
        dataset_path=args.dataset_path,
    )
