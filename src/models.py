from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBackbone(nn.Module):
    """Practical 3 convolutional backbone used as the shared SSL teacher/student stem."""

    def __init__(
        self,
        num_features: int,
        input_channels: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.features = nn.Linear(128, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.relu(self.features(x))

    def intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Conventional feature-KD location: post-conv pooled representation."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.avg_pool(x)
        return torch.flatten(x, 1)


class ClassifierHead(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.projection = nn.Linear(input_size, projection_size)
        self.classifier = nn.Linear(projection_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return self.classifier(F.relu(x))


class ClassificationModel(nn.Module):
    def __init__(self, backbone: ConvBackbone, head: ClassifierHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.intermediate_features(x)


def build_pretext_model(cfg: dict, num_pretext_classes: int = 4) -> nn.Sequential:
    backbone = ConvBackbone(num_features=cfg["backbone_output_size"])
    head = ClassifierHead(
        input_size=cfg["backbone_output_size"],
        projection_size=cfg["head_projection_size"],
        num_classes=num_pretext_classes,
    )
    return nn.Sequential(backbone, head)


def build_classifier(cfg: dict, num_classes: int) -> ClassificationModel:
    backbone = ConvBackbone(num_features=cfg["backbone_output_size"])
    head = ClassifierHead(
        input_size=cfg["backbone_output_size"],
        projection_size=cfg["head_projection_size"],
        num_classes=num_classes,
    )
    return ClassificationModel(backbone, head)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
