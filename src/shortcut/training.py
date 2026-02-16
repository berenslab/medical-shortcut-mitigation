import lightning.pytorch as pl
import numpy as np
import torch

from .metrics import get_metrics
from .models import get_model


class LightningWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningWrapper, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.method, self.encoder = get_model(cfg)
        self.method_name = cfg.model.method
        self.warmup_epochs = cfg.training.warmup_epochs

        self.modes = ["train", "val", "test"]
        self.test_suffix = None  # Set in terms of multiple test sets.

        self.overall_losses = {mode: [] for mode in self.modes}
        self.dependence_estimate = {mode: [] for mode in self.modes}
        self.subspace_classifiers_loss = {mode: [] for mode in self.modes}

        # levels of abstraction:
        # modes -> classifiers -> metrics
        self.num_classifiers = len(cfg.model.class_dims)

        metric_names = cfg.metrics
        if self.method_name == "adv_cl":
            if "Accuracy" not in metric_names:
                metric_names += ["Accuracy"]

        metrics = []
        for classifier_id in range(self.num_classifiers):
            for _ in self.modes:
                metrics.append(
                    get_metrics(
                        cfg.model.class_dims[classifier_id],
                        metric_names,
                    )
                )
        self.metrics = torch.nn.ModuleList(metrics)

        self.mode_to_metrics = {}
        for mode_id, mode in enumerate(self.modes):
            self.mode_to_metrics[mode] = {}
            for classifier_id in range(self.num_classifiers):
                self.mode_to_metrics[mode][classifier_id] = self.metrics[
                    len(self.modes) * classifier_id + mode_id
                ]

    def configure_optimizers(self):
        return self.method.configure_optimizers(self.encoder)

    def on_train_epoch_start(self) -> None:
        self.__reset_metrics("train")

    def on_validation_epoch_start(self) -> None:
        self.__reset_metrics("val")

    def on_test_epoch_start(self) -> None:
        self.__reset_metrics("test")

    def __reset_metrics(self, mode: str) -> None:
        self.overall_losses[mode] = []
        self.dependence_estimate[mode] = []
        self.subspace_classifiers_loss[mode] = []
        for classifier_id in range(self.num_classifiers):
            metrics = self.mode_to_metrics[mode][classifier_id]
            metrics.reset()

    def training_step(self, batch, batch_idx):
        self.__step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self.__step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.__step(batch, batch_idx, "test")

    def forward(self, batch):
        images, _ = batch
        z = self.encoder(images)
        return z

    def __step(self, batch, batch_idx, mode):  # mode is one of ['train','test','val']
        images, labels = batch
        z = self(batch)
        if mode == "train":
            total_steps = self.trainer.estimated_stepping_batches
            start_steps = self.current_epoch * self.trainer.num_training_batches
            p = float(batch_idx + start_steps) / total_steps
            subspace_classifiers_loss, dependence_estimate, loss, logits = (
                self.method.gradient_step(
                    lightning_module=self,
                    z=z,
                    labels=labels,
                    batch_idx=batch_idx,
                    warmup=self.current_epoch < self.warmup_epochs,
                    p=p,
                )
            )
        else:
            subspace_classifiers_loss, dependence_estimate, loss, logits = (
                self.method.get_losses(
                    z,
                    labels,
                )
            )
        if torch.isnan(loss.cpu()):
            raise Exception("loss is Nan.")
        else:
            self.overall_losses[mode].append(loss.detach().cpu().item())
            self.dependence_estimate[mode].append(
                dependence_estimate.detach().cpu().item()
            )
            self.subspace_classifiers_loss[mode].append(
                subspace_classifiers_loss.detach().cpu().item()
            )
            with torch.no_grad():
                for classifier_id in range(self.num_classifiers):
                    metrics = self.mode_to_metrics[mode][classifier_id]
                    metrics.update(
                        images, labels[:, classifier_id], logits[classifier_id]
                    )

    def on_train_epoch_end(self) -> None:
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        self.__log_metrics(mode="train")

    def on_validation_epoch_end(self) -> None:
        self.__log_metrics(mode="val")

    def on_test_epoch_end(self) -> None:
        self.__log_metrics(mode="test")

    def __log_metrics(self, mode):
        if mode == "test" and self.test_suffix is not None:
            mode_suffix = f"_{self.test_suffix}"
        else:
            mode_suffix = ""
        logs = {}
        for classifier_id in range(self.num_classifiers):
            metrics = self.mode_to_metrics[mode][classifier_id]
            metrics_out = metrics.get_out_dict()
            logs.update(
                {
                    f"{mode}{mode_suffix} {classifier_id} {key}": val
                    for key, val in metrics_out.items()
                }
            )
        logs[f"{mode}{mode_suffix} loss"] = np.mean(self.overall_losses[mode])
        logs[f"{mode}{mode_suffix} dependence estimate"] = np.mean(
            self.dependence_estimate[mode]
        )
        logs[f"{mode}{mode_suffix} subspace classifiers"] = np.mean(
            self.subspace_classifiers_loss[mode]
        )
        logs["step"] = float(self.current_epoch)

        if (self.method_name == "adv_cl") and (mode == "val"):
            mean_acc = logs[f"{mode} 0 Accuracy"] - logs[f"{mode} 1 Accuracy"]
            logs[f"{mode}_acc_difference"] = mean_acc

        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)