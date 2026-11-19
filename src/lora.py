from __future__ import annotations

import torch
from torch import nn

from .models import ClassificationModel, count_parameters


class LoraLayer(nn.Module):
    """Low-rank adapter wrapping an existing Linear layer (Practical 4).

    Forward:  y = x W^T + scale * dropout(x) A^T B^T + b
    Ax is the bottleneck; BAx is the adapter output used in RQ2/RQ3.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.2,
        verbose: bool = False,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")

        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(p=dropout)
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        self.W = original_layer.weight
        self.b = original_layer.bias
        self.W.requires_grad = False
        if self.b is not None:
            self.b.requires_grad = False

        self.A = nn.Parameter(torch.randn(r, self.in_features, device=self.W.device, dtype=self.W.dtype))
        self.B = nn.Parameter(torch.zeros(self.out_features, r, device=self.W.device, dtype=self.W.dtype))

        if verbose:
            n_full = self.W.numel()
            n_lora = self.A.numel() + self.B.numel()
            print(f"- W {tuple(self.W.shape)} ({n_full} frozen), A {tuple(self.A.shape)} B {tuple(self.B.shape)} ({n_lora})")

    def ax(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.dropout(x), self.A.T)

    def bax(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.ax(x), self.B.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_frozen = torch.matmul(x, self.W.T)
        z = z_frozen + self.scale * self.bax(x)
        if self.b is not None:
            z = z + self.b
        return z


def inject_lora(
    model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    module_names: tuple[str, ...] | None = None,
) -> list[str]:
    """Replace selected Linear modules with LoraLayer. Defaults to backbone fc layers."""
    replaced: list[str] = []
    target = model.backbone if isinstance(model, ClassificationModel) else model
    names = module_names or ("fc1", "fc2", "features")
    for name in names:
        layer = getattr(target, name, None)
        if isinstance(layer, nn.Linear):
            setattr(target, name, LoraLayer(layer, r=r, alpha=alpha, dropout=dropout))
            replaced.append(name)
    return replaced


def collect_lora_layers(model: nn.Module) -> list[LoraLayer]:
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def freeze_non_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if ".A" not in name and ".B" not in name:
            param.requires_grad = False


def apply_lora_from_config(model: nn.Module, cfg: dict, freeze_base: bool = True) -> nn.Module:
    lora_cfg = cfg["lora"]
    inject_lora(model, r=lora_cfg["rank"], alpha=lora_cfg["alpha"], dropout=lora_cfg["dropout"])
    if freeze_base:
        freeze_non_lora(model)
    return model


def report_trainable(model: nn.Module, label: str = "") -> tuple[int, int]:
    total, trainable = count_parameters(model)
    prefix = f"{label}: " if label else ""
    print(f"{prefix}{trainable:,} / {total:,} trainable ({trainable / max(total, 1):.1%})")
    return total, trainable
