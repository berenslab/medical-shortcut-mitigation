from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from shortcut.criterion import get_criterion
from shortcut.models.classification_heads import LinearClassifier
from shortcut.models.dependence_measures import (AdvClassifier,
                                                 LinearMMDEstimator,
                                                 MIEstimator, MMDEstimator,
                                                 dCorEstimator)
from shortcut.optimizers import get_optimizer
from shortcut.utils import flatten_list


class SubspaceDistanglementMethod(ABC, torch.nn.Module):
    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def gradient_step(self):
        pass

    @abstractmethod
    def predict(self, z) -> List[Tensor]:
        pass

    @abstractmethod
    def get_losses(self):
        pass


class Baseline(SubspaceDistanglementMethod):
    """Baseline method, only classification heads, no dependence measure minimization."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.subspace_dims = cfg.model.subspace_dims
        self.class_dims = cfg.model.class_dims

        self.optimizer_name = cfg.optimizer.name
        self.optimizer_kwargs = cfg.optimizer.kwargs

        self.classifiers = self._get_classifiers()
        self.criterion = get_criterion(
            cfg.training.criterion.name,
            kwargs=cfg.training.criterion.kwargs,
        )

    def configure_optimizers(self, encoder):
        classifier_parameters = [
            list(classifier.parameters()) for classifier in self.classifiers
        ]
        parameters = list(encoder.parameters()) + flatten_list(classifier_parameters)
        optimizer = get_optimizer(self.optimizer_name)

        return optimizer(parameters, **self.optimizer_kwargs)

    def gradient_step(
        self,
        lightning_module,
        z,
        labels,
        batch_idx: int,
        p: float,
        warmup: bool = False,
    ):
        encoder_opt = lightning_module.optimizers()

        logits = self.predict(z)
        ce_loss = self._subspace_classifiers_loss(logits, labels)

        # Update encoder model.
        encoder_opt.zero_grad()
        lightning_module.manual_backward(ce_loss)
        encoder_opt.step()

        return ce_loss, torch.tensor(0.0), ce_loss, logits

    def predict(
        self,
        z,
    ):
        y_hat_logits_list = []
        for i in range(len(self.subspace_dims)):
            subspace = z[
                :, sum(self.subspace_dims[:i]) : sum(self.subspace_dims[: i + 1])
            ]
            y_hat_logits = self.classifiers[i](subspace).squeeze()
            y_hat_logits_list.append(y_hat_logits)

        return y_hat_logits_list

    def get_losses(
        self,
        z,
        labels,
    ):
        logits = self.predict(z)
        ce_loss = self._subspace_classifiers_loss(logits, labels)
        overall_loss = ce_loss
        return ce_loss, torch.tensor(0.0), overall_loss, logits

    def _get_classifiers(self):
        num_subspaces = len(self.subspace_dims)
        Cs = []
        for i in range(num_subspaces):
            Cs.append(
                LinearClassifier(
                    z_shape=self.subspace_dims[i],
                    c_shape=self.class_dims[i],
                )
            )
        return torch.nn.ModuleList(Cs)

    def _subspace_classifiers_loss(self, logits, labels):
        assert labels.shape[1] == len(
            self.class_dims
        ), "Number of subspaces and number of labels/tasks in the dataset (labels_dict) don't match."
        per_subspace_losses = []
        for i, subspace_logits in enumerate(logits):
            subspace_labels = labels[:, i]
            if self.class_dims[i] == 1:
                subspace_logits = subspace_logits.squeeze(-1)  # safe squeeze
                subspace_labels = subspace_labels.to(torch.float32)
                assert subspace_logits.shape == subspace_labels.shape
            subspace_loss = self.criterion(subspace_logits, subspace_labels)
            per_subspace_losses.append(subspace_loss)
        mean_loss = sum(per_subspace_losses) / len(per_subspace_losses)
        return mean_loss


class MINE(Baseline):
    """MINE method: classification heads, MI minimization between subspaces."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.optimizer_name_mi = cfg.optimizer_mine.name
        self.optimizer_kwargs_mi = cfg.optimizer_mine.kwargs
        self.mine_update_interval = cfg.training.mine_update_interval
        self.lambda_mi = cfg.training.lambda_dmeasure

        self.mi_estimator = MIEstimator(
            feature_dim=sum(cfg.model.subspace_dims),
        )

    def configure_optimizers(self, encoder):
        classifier_parameters = [
            list(classifier.parameters()) for classifier in self.classifiers
        ]
        encoder_params = list(encoder.parameters()) + flatten_list(
            classifier_parameters
        )
        optimizer_enc = get_optimizer(self.optimizer_name)
        optimizer_enc = optimizer_enc(encoder_params, **self.optimizer_kwargs)

        optimizer_mine = get_optimizer(self.optimizer_name_mi)
        optimizer_mine = optimizer_mine(
            self.mi_estimator.parameters(),
            **self.optimizer_kwargs_mi,
        )
        return optimizer_enc, optimizer_mine

    def gradient_step(
        self,
        lightning_module,
        z,
        labels,
        batch_idx: int,
        p: float,
        warmup: bool = False,
    ):
        encoder_opt, mine_opt = lightning_module.optimizers()

        logits = self.predict(z)
        ce_loss = self._subspace_classifiers_loss(logits, labels)

        mi_estimator_update = batch_idx % self.mine_update_interval != 0

        if mi_estimator_update:
            # Update MI estimation model.
            mi_estimate = self._estimate_mi(z.detach())
            mine_opt.zero_grad()
            lightning_module.manual_backward(-mi_estimate)
            mine_opt.step()
            overall_loss = ce_loss + self.lambda_mi * mi_estimate

        else:
            # Update encoder model.
            mi_estimate = self._estimate_mi(z)
            overall_loss = ce_loss
            if not warmup:
                overall_loss = overall_loss + self.lambda_mi * mi_estimate

            encoder_opt.zero_grad()
            lightning_module.manual_backward(overall_loss)
            encoder_opt.step()

        return ce_loss, mi_estimate, overall_loss, logits

    def _estimate_mi(self, z):
        z1 = z[:, 0 : self.subspace_dims[0]]
        z2 = z[
            :,
            self.subspace_dims[0] : self.subspace_dims[0] + self.subspace_dims[1],
        ]
        return self.mi_estimator(z1, z2)

    def get_losses(
        self,
        z,
        labels,
    ):
        logits = self.predict(z)
        ce_loss = self._subspace_classifiers_loss(logits, labels)
        mi_estimate = torch.clamp(self._estimate_mi(z), min=0)
        overall_loss = ce_loss + self.lambda_mi * mi_estimate
        return ce_loss, mi_estimate, overall_loss, logits


class dCor(Baseline):
    """dCor method: classification heads, dCor minimization between subspaces."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lambda_dmeasure = cfg.training.lambda_dmeasure
        self.gamma = cfg.training.gamma

    def gradient_step(
        self,
        lightning_module,
        z,
        labels,
        batch_idx: int,
        p: float,
        warmup: bool = False,
    ):
        encoder_opt = lightning_module.optimizers()

        logits = self.predict(z)
        ce_loss = self._subspace_classifiers_loss(logits, labels)

        if self.gamma is not None:
            lambda_dmeasure = (
                2.0 / (1.0 + np.exp(-self.gamma * p)) - 1
            ) * self.lambda_dmeasure
        else:
            lambda_dmeasure = self.lambda_dmeasure

        # Update encoder model.
        overall_loss = ce_loss
        dcor_estimate = self._estimate_dcor(z)
        if not warmup:
            overall_loss = overall_loss + lambda_dmeasure * dcor_estimate

        encoder_opt.zero_grad()
        lightning_module.manual_backward(overall_loss)
        encoder_opt.step()

        return ce_loss, dcor_estimate, overall_loss, logits

    def _estimate_dcor(
        self,
        z,
    ):
        z1 = z[:, : self.subspace_dims[0]]
        z2 = z[:, self.subspace_dims[0] : sum(self.subspace_dims)]
        return dCorEstimator(z1, z2)

    def get_losses(
        self,
        z,
        labels,
    ):
        logits = self.predict(z)
        ce_loss = self._subspace_classifiers_loss(logits, labels)
        dcor_estimate = self._estimate_dcor(z)
        overall_loss = ce_loss + self.lambda_dmeasure * dcor_estimate
        return ce_loss, dcor_estimate, overall_loss, logits


class AdversarialClassifierGRL(Baseline):
    """Adv. classifier: classification heads, maximization of adv. classifier criterion."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.gamma = cfg.training.gamma
        self.alpha_scale = cfg.training.alpha_scale
        self.latent_dim = sum(cfg.model.subspace_dims)
        self.learning_rate = cfg.optimizer.kwargs.lr
        self.total_epochs = cfg.max_epochs

        self.classifiers = self._get_classifiers_adv()

    def configure_optimizers(self, encoder):
        classifier_parameters = [
            list(classifier.parameters()) for classifier in self.classifiers
        ]
        parameters = list(encoder.parameters()) + flatten_list(classifier_parameters)
        optimizer = get_optimizer(self.optimizer_name)
        optimizer = optimizer(parameters, **self.optimizer_kwargs)

        def lr_lambda(current_epoch):
            x = current_epoch / self.total_epochs
            return 1.0 / ((1 + 10 * x) ** 0.75)

        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            ),
        }

    def predict(
        self,
        z,
        alpha=None,
    ):
        y_hat_logits_list = []
        y_hat_logits_list.append(self.classifiers[0](z).squeeze())
        y_hat_logits_list.append(self.classifiers[1](z, alpha).squeeze())

        return y_hat_logits_list

    def gradient_step(
        self,
        lightning_module,
        z,
        labels,
        batch_idx: int,
        p: float,
        warmup: bool = False,
    ):
        encoder_opt = lightning_module.optimizers()
        alpha = (2.0 / (1.0 + np.exp(-self.gamma * p)) - 1) * self.alpha_scale

        logits = self.predict(z, alpha)
        ce_loss_cl, ce_loss_adv_cl = self._subspace_classifiers_loss(logits, labels)

        overall_loss = ce_loss_cl + ce_loss_adv_cl

        encoder_opt.zero_grad()
        lightning_module.manual_backward(overall_loss)
        encoder_opt.step()

        return ce_loss_cl, ce_loss_adv_cl, overall_loss, logits

    def _get_classifiers_adv(self):
        self.classifier = LinearClassifier(
            z_shape=self.latent_dim,
            c_shape=self.class_dims[0],
        )

        self.adv_classifier = AdvClassifier(
            z_shape=self.latent_dim,
            c_shape=self.class_dims[1],
        )
        return torch.nn.ModuleList([self.classifier, self.adv_classifier])

    def _subspace_classifiers_loss(self, logits, labels):
        assert labels.shape[1] == len(
            self.class_dims
        ), "Number of subspaces and number of labels/tasks in the dataset (labels_dict) don't match."
        ce_losses = []
        for i, subspace_logits in enumerate(logits):
            subspace_labels = labels[:, i]
            if self.class_dims[i] == 1:
                subspace_logits = subspace_logits.squeeze()
                subspace_labels = subspace_labels.to(torch.float32)
                subspace_loss = self.criterion(subspace_logits, subspace_labels)
            else:
                subspace_loss = self.criterion(subspace_logits, subspace_labels)
            ce_losses.append(subspace_loss)
        return tuple(ce_losses)

    def get_losses(
        self,
        z,
        labels,
    ):
        logits = self.predict(z)
        ce_loss_cl, ce_loss_adv_cl = self._subspace_classifiers_loss(logits, labels)
        overall_loss = ce_loss_cl + ce_loss_adv_cl
        return ce_loss_cl, ce_loss_adv_cl, overall_loss, logits


class MMD(Baseline):
    """MMD method: classification heads, MMD minimization between subspaces."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lambda_dmeasure = cfg.training.lambda_dmeasure
        self.gamma = cfg.training.gamma
        self.linear = cfg.training.linear_time_estimator
        self.biased_estimate = cfg.training.biased_estimate  # always positive
        self.soft_clamping = cfg.training.soft_clamping

        if self.linear:
            self.mmd_estimator = LinearMMDEstimator()
        else:
            self.beta = cfg.training.beta
            self.mmd_estimator = MMDEstimator(self.biased_estimate)

    def gradient_step(
        self,
        lightning_module,
        z,
        labels,
        batch_idx: int,
        p: float,
        warmup: bool = False,
    ):
        encoder_opt = lightning_module.optimizers()

        logits = self.predict(z)
        ce_loss = self._subspace_classifiers_loss(logits, labels)

        if self.gamma is not None:
            lambda_dmeasure = (
                2.0 / (1.0 + np.exp(-self.gamma * p)) - 1
            ) * self.lambda_dmeasure
        else:
            lambda_dmeasure = self.lambda_dmeasure

        # Update encoder model.
        overall_loss = ce_loss
        mmd_estimate = self._estimate_mmd(z)
        # The unbiased estimators (Gretton et al., 2012) can yield negative values.
        if not self.biased_estimate and self.soft_clamping:
            mmd_estimate = F.softplus(mmd_estimate - self.beta)
        if not warmup:
            overall_loss = overall_loss + lambda_dmeasure * mmd_estimate

        encoder_opt.zero_grad()
        lightning_module.manual_backward(overall_loss)
        encoder_opt.step()

        return ce_loss, mmd_estimate, overall_loss, logits

    def _estimate_mmd(
        self,
        z,
    ):
        z1 = z[:, : self.subspace_dims[0]]
        z2 = z[:, self.subspace_dims[0] : sum(self.subspace_dims)]
        return self.mmd_estimator(z1, z2)

    def get_losses(
        self,
        z,
        labels,
    ):
        logits = self.predict(z)
        ce_loss = self._subspace_classifiers_loss(logits, labels)
        mmd_estimate = self._estimate_mmd(z)
        overall_loss = ce_loss + self.lambda_dmeasure * mmd_estimate
        return ce_loss, mmd_estimate, overall_loss, logits
