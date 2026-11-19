from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .kd import distillation_loss
from .plots import TrainingMetrics


def top1_acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred.argmax(dim=1) == y).float().mean().item())


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses, accs = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses.append(float(criterion(logits, labels).item()))
        accs.append(top1_acc(logits, labels))
    return float(sum(losses) / max(len(losses), 1)), float(sum(accs) / max(len(accs), 1))


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    title: str = "",
    curve_path: Path | None = None,
) -> TrainingMetrics:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    metrics = TrainingMetrics(title=title)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            metrics.log_train(float(loss.item()), top1_acc(logits.detach(), labels))
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        metrics.log_val(val_loss, val_acc)
        tr_loss, va_loss = metrics.epoch_loss
        tr_acc, va_acc = metrics.epoch_acc
        print(f"epoch {epoch + 1:>3}/{epochs} | loss {tr_loss:.3f}/{va_loss:.3f} | acc {tr_acc:.1%}/{va_acc:.1%}")

    if curve_path is not None:
        metrics.plot(title=title, save_path=curve_path)
    return metrics


def train_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    kd_type: str,
    epochs: int,
    lr: float,
    temperature: float,
    kd_alpha: float,
    feature_weight: float,
    device: torch.device,
    title: str = "",
    curve_path: Path | None = None,
) -> TrainingMetrics:
    student.to(device)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=lr)
    metrics = TrainingMetrics(title=title)

    for epoch in range(epochs):
        student.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss, logs = distillation_loss(
                student,
                teacher,
                images,
                labels,
                kd_type=kd_type,
                temperature=temperature,
                kd_alpha=kd_alpha,
                feature_weight=feature_weight,
                ce_criterion=criterion,
            )
            loss.backward()
            optimizer.step()
            metrics.log_train(float(loss.item()), top1_acc(logs["student_logits"].detach(), labels))
        val_loss, val_acc = evaluate_model(student, val_loader, criterion, device)
        metrics.log_val(val_loss, val_acc)
        tr_loss, va_loss = metrics.epoch_loss
        tr_acc, va_acc = metrics.epoch_acc
        print(f"epoch {epoch + 1:>3}/{epochs} | loss {tr_loss:.3f}/{va_loss:.3f} | acc {tr_acc:.1%}/{va_acc:.1%}")

    if curve_path is not None:
        metrics.plot(title=title, save_path=curve_path)
    return metrics
