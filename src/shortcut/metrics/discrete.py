from torchmetrics.classification import Accuracy as TorchAcc
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.classification import CohenKappa as TorchCohenKappa
from torchmetrics.classification import (MulticlassAUROC,
                                         MulticlassAveragePrecision)

from .abstract import Metric


class Accuracy(Metric):
    def __init__(self, num_classes):
        super().__init__()
        if num_classes == 1:
            self._acc = TorchAcc(
                task="binary",
                sync_on_compute=True,
            )
        else:
            self._acc = TorchAcc(
                task="multiclass", num_classes=num_classes, sync_on_compute=True
            )

    def reset(self):
        self._acc.reset()

    def update(self, _, groundtruth, prediction):
        self._acc.update(preds=prediction, target=groundtruth)

    def get_acc(self):
        return self._acc.compute()

    def _get_key_to_eval_func(self):
        return {
            "Accuracy": self.get_acc,
        }

    @classmethod
    def get_keys(cls, num_classes):
        instance = cls(num_classes)
        return list(instance._get_key_to_eval_func().keys())

    @classmethod
    def from_cfg(cls, num_classes):
        return cls(num_classes)


class BalancedAcc(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Use Accuracy with macro averaging for balanced accuracy.
        self._balanced_acc = TorchAcc(
            task="multiclass" if num_classes > 1 else "binary",
            num_classes=num_classes if num_classes > 1 else 2,
            average="macro",
            sync_on_compute=True,
        )

    def reset(self):
        self._balanced_acc.reset()

    def update(self, _, groundtruth, prediction):
        self._balanced_acc.update(preds=prediction, target=groundtruth)

    def get_balanced_accuracy(self):
        return self._balanced_acc.compute()

    def _get_key_to_eval_func(self):
        return {"Balanced_Acc": self.get_balanced_accuracy}

    @classmethod
    def get_keys(cls, num_classes):
        instance = cls(num_classes)
        return list(instance._get_key_to_eval_func().keys())

    @classmethod
    def from_cfg(cls, num_classes):
        return cls(num_classes)


class Kappa(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self._kappa = TorchCohenKappa(
            num_classes=num_classes if num_classes > 1 else 2,
            task="multiclass" if num_classes > 1 else "binary",
            sync_on_compute=True,
        )

    def reset(self):
        self._kappa.reset()

    def update(self, _, groundtruth, prediction):
        self._kappa.update(preds=prediction, target=groundtruth)

    def get_kappa(self):
        return self._kappa.compute()

    def _get_key_to_eval_func(self):
        return {"Kappa": self.get_kappa}

    @classmethod
    def get_keys(cls, num_classes):
        instance = cls(num_classes)
        return list(instance._get_key_to_eval_func().keys())

    @classmethod
    def from_cfg(cls, num_classes):
        return cls(num_classes)


class AUROC(Metric):
    def __init__(self, num_classes):
        super().__init__()
        if num_classes == 1:
            self._auroc = BinaryAUROC(sync_on_compute=True)
        else:
            self._auroc = MulticlassAUROC(
                num_classes=num_classes, average="macro", sync_on_compute=True
            )

    def reset(self):
        self._auroc.reset()

    def update(self, _, groundtruth, prediction):
        self._auroc.update(preds=prediction, target=groundtruth)

    def get_auroc(self):
        return self._auroc.compute()

    def _get_key_to_eval_func(self):
        return {
            "AUROC": self.get_auroc,
        }

    @classmethod
    def get_keys(cls, num_classes):
        instance = cls(num_classes)
        return list(instance._get_key_to_eval_func().keys())

    @classmethod
    def from_cfg(cls, num_classes):
        return cls(num_classes)


class AUPRC(Metric):
    def __init__(self, num_classes):
        super().__init__()
        if num_classes == 1:
            self._auprc = BinaryAveragePrecision(sync_on_compute=True)
        else:
            self._auprc = MulticlassAveragePrecision(
                num_classes=num_classes,
                average="macro",
                sync_on_compute=True,
            )

    def reset(self):
        self._auprc.reset()

    def update(self, _, groundtruth, prediction):
        self._auprc.update(preds=prediction, target=groundtruth)

    def get_auprc(self):
        return self._auprc.compute()

    def _get_key_to_eval_func(self):
        return {
            "AUPRC": self.get_auprc,
        }

    @classmethod
    def get_keys(cls, num_classes):
        instance = cls(num_classes)
        return list(instance._get_key_to_eval_func().keys())

    @classmethod
    def from_cfg(cls, num_classes):
        return cls(num_classes)
