from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .lora import LoraLayer, collect_lora_layers


def output_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = temperature
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def feature_kd_loss(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"Feature KD shape mismatch: student {tuple(student_features.shape)} "
            f"vs teacher {tuple(teacher_features.shape)}. Align architectures or add a projector."
        )
    return F.mse_loss(student_features, teacher_features.detach())


def capture_adapter_reps(model: nn.Module, images: torch.Tensor, kind: str) -> torch.Tensor:
    """Run a forward pass and concatenate Ax or BAx from every LoraLayer (RQ2/RQ3)."""
    captured: list[torch.Tensor] = []
    hooks = []

    def make_hook(layer: LoraLayer):
        def hook(mod: LoraLayer, inputs, _output):
            x = inputs[0]
            captured.append(mod.ax(x) if kind == "ax" else mod.bax(x))

        return hook

    layers = collect_lora_layers(model)
    if not layers:
        raise RuntimeError("LoRA-space KD requires injected LoraLayer modules.")
    for layer in layers:
        hooks.append(layer.register_forward_hook(make_hook(layer)))
    try:
        model(images)
    finally:
        for hook in hooks:
            hook.remove()
    return torch.cat([rep.flatten(1) for rep in captured], dim=1)


def distillation_loss(
    student: nn.Module,
    teacher: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    kd_type: str,
    temperature: float,
    kd_alpha: float,
    feature_weight: float,
    ce_criterion: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_logits = student(images)
    with torch.no_grad():
        teacher_logits = teacher(images)

    ce = ce_criterion(student_logits, labels)
    logs = {"ce": float(ce.detach().cpu()), "student_logits": student_logits}
    loss = (1.0 - kd_alpha) * ce

    if kd_type == "output":
        kd = output_kd_loss(student_logits, teacher_logits, temperature)
        loss = loss + kd_alpha * kd
        logs["kd_output"] = float(kd.detach().cpu())
    elif kd_type == "feature":
        if not (hasattr(student, "features") and hasattr(teacher, "features")):
            raise RuntimeError("Feature KD requires models with a .features() method.")
        kd = feature_kd_loss(student.features(images), teacher.features(images))
        loss = loss + feature_weight * kd
        logs["kd_feature"] = float(kd.detach().cpu())
    elif kd_type in {"lora_space", "lora_ax", "lora_bax", "lora_ax_bax"}:
        parts = []
        if kd_type in {"lora_space", "lora_ax", "lora_ax_bax"}:
            parts.append(
                feature_kd_loss(
                    capture_adapter_reps(student, images, "ax"),
                    capture_adapter_reps(teacher, images, "ax"),
                )
            )
        if kd_type in {"lora_space", "lora_bax", "lora_ax_bax"}:
            parts.append(
                feature_kd_loss(
                    capture_adapter_reps(student, images, "bax"),
                    capture_adapter_reps(teacher, images, "bax"),
                )
            )
        kd = sum(parts) / len(parts)
        loss = loss + feature_weight * kd
        logs["kd_lora"] = float(kd.detach().cpu())
    else:
        raise ValueError(f"Unknown kd_type: {kd_type}")

    logs["loss"] = float(loss.detach().cpu())
    return loss, logs
