import torch
from torch.nn import functional as F


class IOU_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        n = targets.numel()
        pred = torch.sigmoid(logits)

        a = (pred * targets).sum()
        b = ((1 - pred) * (1 - targets)).sum()
        iou_true = a / (n - b)

        return 1.0 - iou_true


class MSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        pred = torch.sigmoid(logits)
        return ((pred - targets) ** 2).mean()


class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=0.2, reduction="mean", eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )

        # Probabilities clamped to avoid log(0).
        p = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)

        pt = torch.where(targets == 1, p, 1 - p)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiClassFocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [batch_size, num_classes]
        # targets: [batch_size] with class indices
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)  # class weights
            loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_criterion(criterion, kwargs):
    criteria = {
        "BCE": torch.nn.BCEWithLogitsLoss,
        "CCE": torch.nn.CrossEntropyLoss,
        "IoU": IOU_loss,
        "MSE": MSE,
        "binary_focal": BinaryFocalLoss,
        "multi_focal": MultiClassFocalLoss,
    }
    if criterion == "BCE" and "pos_weight" in kwargs:
        kwargs = dict(kwargs)
        pos_weight = float(kwargs["pos_weight"])
        kwargs["pos_weight"] = torch.tensor([pos_weight], 
                                            dtype=torch.float32, 
                                            device="cuda")


    return criteria[criterion](**kwargs)
