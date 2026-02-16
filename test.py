import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar

from shortcut.config import get_config
from shortcut.data import get_test_dataloaders
from shortcut.training import LightningWrapper
from shortcut.utils import LitProgressBar, TestMetricsLoggerCallback

warnings.simplefilter("ignore", FutureWarning)

PATH_TO_DEFAULT_CFG = "configs/default.yaml"


def test(cfg: dict, fold: int, version: Optional[int] = None):
    exp_folder = Path(cfg.out_dir) / f"fold{fold}"
    if version is not None:
        log_dir = os.path.join(exp_folder, f"version_{version}")
    else:
        # Read the latest log dir.
        latest_path_file = f"{exp_folder}/latest_log_dir.txt"
        if not os.path.exists(latest_path_file):
            raise ValueError("No latest_log_dir.txt found for this fold!")
        with open(latest_path_file, "r") as f:
            log_dir = f.read().strip()
        version = os.path.basename(log_dir)

    best_ckpt_file = os.path.join(log_dir, "best_ckpt.txt")
    if not os.path.exists(best_ckpt_file):
        raise FileNotFoundError(f"No best_ckpt.txt found in {log_dir}")

    with open(best_ckpt_file, "r") as f:
        best_ckpt_path = f.read().strip()

    if not sys.stdout.isatty() or not os.isatty(0):
        ProgressBar = TQDMProgressBar()
    else:
        ProgressBar = LitProgressBar()

    # Evaluate best model on test data distributions.
    print("Model evaluation on shifted test distributions.")
    best_model = LightningWrapper.load_from_checkpoint(best_ckpt_path, cfg=cfg)
    best_model.eval()

    # Create a new trainer with only 1 GPU -- only test on single GPU.
    test_logger = pl.loggers.TensorBoardLogger(
        save_dir=exp_folder,
        version=version,  # reuse the same version number
        default_hp_metric=False,
    )
    devices = cfg.devices
    test_trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=[devices[0]],
        logger=test_logger,
        callbacks=[ProgressBar, TestMetricsLoggerCallback(fold=fold)],
    )
    test_loader, test_loader_balanced, test_loader_inverse = get_test_dataloaders(cfg)
    for test_suffix, dataloader in zip(
        ["original", "balanced", "inverse"],
        [test_loader, test_loader_balanced, test_loader_inverse],
    ):
        best_model.test_suffix = test_suffix
        test_trainer.test(best_model, dataloaders=dataloader)


if __name__ == "__main__":
    cfg = get_config(PATH_TO_DEFAULT_CFG)
    test(cfg, cfg.training.fold, cfg.version)
