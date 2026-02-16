import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNetEncoder(nn.Module):
    """Image encoder using a ResNet backbone, with options for pretrained weights,
    input channels, latent dimension, and layer freezing.

    Attributes:
        cfg: Configuration dictionary containing model parameters:
            - cfg.model.resnet_backbone (int): ResNet version to use (18, 34, or 50).
            - cfg.model.in_channels (int): Number of input channels for the first
                convolutional layer.
            - cfg.model.subspace_dims (list[int]): List of integers specifying the
                output latent dimensions.
            - cfg.model.imagenet_pretrained (bool): Whether to load ImageNet
                pretrained weights.
            - cfg.model.freeze_layers (int): Number of ResNet blocks to freeze (0–4):
                - 0: freeze nothing
                - 1: freeze layer1
                - 2: freeze layer1 + layer2
                - 3: freeze layer1 + layer2 + layer3
                - 4: freeze layer1 + layer2 + layer3 + layer4 (freeze everything except
                    final FC)
            cfg.model.partial_layer4 (bool): If True, only freeze paty of layer 4.
            cfg.model.nonlinear_class_head (bool): If True, include nonlinearity in
                classification head.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        resnet_backbone = cfg.model.resnet_backbone
        in_channels = cfg.model.in_channels
        latent_dim = sum(cfg.model.subspace_dims)
        freeze_layers = cfg.model.freeze_layers
        partial_layer4 = cfg.model.partial_layer4
        nonlinear_class_head = cfg.model.nonlinear_class_head

        # Load pretrained backbone.
        if resnet_backbone == 18:
            weights = (
                torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                if cfg.model.imagenet_pretrained
                else None
            )
            backbone = torchvision.models.resnet18(weights=weights)
        elif resnet_backbone == 34:
            weights = (
                torchvision.models.ResNet34_Weights.IMAGENET1K_V1
                if cfg.model.imagenet_pretrained
                else None
            )
            backbone = torchvision.models.resnet34(weights=weights)
        elif resnet_backbone == 50:
            weights = (
                torchvision.models.ResNet50_Weights.IMAGENET1K_V1
                if cfg.model.imagenet_pretrained
                else None
            )
            backbone = torchvision.models.resnet50(weights=weights)

        # Customize input channels.
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Freeze layers.
        if freeze_layers is not None:
            layers = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
            for i in range(freeze_layers):
                if i < 3:  # layer1, layer2, layer3
                    for param in layers[i].parameters():
                        param.requires_grad = False
                elif i == 3:  # layer4
                    if partial_layer4:
                        # Freeze all blocks except last.
                        for block in layers[i][:-1]:
                            for param in block.parameters():
                                param.requires_grad = False
                    else:
                        # Freeze entire layer4.
                        for param in layers[i].parameters():
                            param.requires_grad = False

        # Replace the final FC layer.
        num_features = backbone.fc.in_features
        if nonlinear_class_head:
            backbone.fc = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, latent_dim, bias=True)
            )
        else:
            backbone.fc = nn.Linear(num_features, latent_dim, bias=True)

        self.encoder = backbone

    def forward(self, batch):
        return self.encoder(batch)


class EfficientNetB1(nn.Module):
    """Image encoder with a efficientnet_b1 backbone."""

    def __init__(self, cfg: dict):
        super().__init__()
        in_channels = cfg.model.in_channels
        latent_dim = sum(cfg.model.subspace_dims)

        if cfg.model.imagenet_pretrained:
            weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
        else:
            weights = None

        backbone = torchvision.models.efficientnet_b1(weights=weights)

        if in_channels != 3:
            backbone.features[0][0] = nn.Conv2d(
                in_channels,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )

        backbone.classifier[1] = nn.Linear(
            in_features=1280, out_features=latent_dim, bias=False
        )
        self.model = backbone

    def forward(self, batch):
        return self.model(batch)


class SimpleEncoder(nn.Module):
    """Simple image encoder with 3 conv layers."""

    def __init__(self, cfg: dict):
        super().__init__()
        in_channels = cfg.model.in_channels  # 1 or 3
        latent_dim = sum(cfg.model.subspace_dims)
        image_size = cfg.data.image_size  # square images

        # Conv backbone.
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 6, 3, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        image_size_out = (((image_size - 3) // 1 + 1) // 2 - 3) // 1 + 1  # after conv2
        image_size_out = (image_size_out - 3) // 1 + 1  # after conv3
        image_size_out = image_size_out // 2  # after pool2
        flattened_size = 16 * image_size_out * image_size_out

        # Fully connected layers.
        self.fc = nn.Linear(flattened_size, 256)
        self.fc_pt_sc = nn.Linear(256, latent_dim)

    def forward(self, X):
        X = self.model(X)
        X = X.view(X.size(0), -1)  # Flatten
        X = nn.functional.relu(self.fc(X))
        return self.fc_pt_sc(X)


class ShallowCNNEncoder(nn.Module):
    """Shallow CNN encoder for small datasets.

    Args:
        cfg (dict): Configuration dictionary containing model parameters:
            - cfg.model.in_channels (int): Number of input channels.
            - cfg.model.subspace_dims (list[int]): List of integers specifying the output latent dimensions.
        dropout (float, default=0.5): Dropout probability before the final linear layer.
    """

    def __init__(self, cfg: dict, dropout: float = 0.5):
        super().__init__()
        in_channels = cfg.model.in_channels
        latent_dim = sum(cfg.model.subspace_dims)

        # Convolutional blocks
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Adaptive pooling to handle variable input size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout before final linear layer
        self.dropout = nn.Dropout(p=dropout)

        # Final linear layer
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Latent representation of shape (B, latent_dim)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Global average pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)
        return x