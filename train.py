import os
import random
import sys
import warnings
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar

from shortcut.config import get_config
from shortcut.data import get_datamodule
from shortcut.training import LightningWrapper
from shortcut.utils import (BestValMetricsLoggerCallback, DelayedCheckpoint,
                            EpochTimerCallback, BenchmarkEpochTimer, LitProgressBar)

warnings.simplefilter("ignore", FutureWarning)

PATH_TO_DEFAULT_CFG = "configs/default.yaml"


def train(cfg: dict, fold: int):
    module = LightningWrapper(cfg)
    if cfg.seed is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    exp_folder = Path(cfg.out_dir) / f"fold{fold}"
    try:
        # Creating logging directory.
        Path(exp_folder).mkdir(parents=False, exist_ok=True)
    except:
        pass

    # Don't use LitProgressBar for background processes (e.g. nohup).
    if not sys.stdout.isatty() or not os.isatty(0):
        ProgressBar = TQDMProgressBar()
    else:
        ProgressBar = LitProgressBar()

    model_checkpoint = DelayedCheckpoint(
        start_saving_epoch=cfg.training.start_saving_epoch,
        save_top_k=cfg.training.checkpoints.save_top_k,
        monitor=cfg.training.checkpoints.monitor,
        mode=cfg.training.checkpoints.mode,
        filename=cfg.training.checkpoints.filename,
    )
    logger = pl.loggers.TensorBoardLogger(
        save_dir=exp_folder,
        default_hp_metric=False,
    )
    # Use custom sampler instead.
    use_distributed_sampler = (
        (cfg.data.weighted_bucket_sampler is None)
        and (cfg.data.factor_bucket_sampler is None)
        and (not cfg.data.weighted_random_sampler)
    )

    callbacks = [
        ProgressBar,
        model_checkpoint,
        BestValMetricsLoggerCallback(
            fold=fold,
            start_saving_epoch=cfg.training.start_saving_epoch,
        ),
    ]

    # Switch based on config flag.
    if getattr(cfg, "benchmark_timing", True):
        callbacks.append(BenchmarkEpochTimer())
    else:
        callbacks.append(EpochTimerCallback())

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=(
            pl.strategies.DDPStrategy(find_unused_parameters=True)
            if len(cfg.devices) > 1
            else "auto"
        ),
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,  #2 if use_distributed_sampler else 0,
        log_every_n_steps=1,
        val_check_interval=None,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        limit_train_batches=1.0,
        max_steps=-1,
        precision=cfg.training.precision,
        default_root_dir=exp_folder,
        use_distributed_sampler=use_distributed_sampler,
    )

    print(f"Model training. Method {cfg.model.method}, fold {fold}.")
    trainer.fit(
        module,
        datamodule=get_datamodule(cfg, fold),
    )

    # Only rank 0 writes the info file.
    if trainer.global_rank == 0:
        log_dir = trainer.logger.log_dir  # e.g., lightning_logs/version_3
        with open(f"{exp_folder}/latest_log_dir.txt", "w") as f:
            f.write(log_dir)
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        with open(os.path.join(logger.log_dir, "best_ckpt.txt"), "w") as text_file:
            text_file.write(best_ckpt_path)


if __name__ == "__main__":
    cfg = get_config(PATH_TO_DEFAULT_CFG)
    train(cfg, fold=cfg.training.fold)
