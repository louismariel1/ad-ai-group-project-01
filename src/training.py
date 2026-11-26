from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .kd import distillation_loss
from .models import count_parameters
from .plots import TrainingMetrics


def top1_acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred.argmax(dim=1) == y).float().mean().item())


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        batch_size = labels.size(0)
        loss_sum += float(criterion(logits, labels).item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Cannot evaluate a model with an empty data loader.")
    return loss_sum / sample_count, correct / sample_count


def _start_measurement(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _record_measurement(
    metrics: TrainingMetrics,
    model: nn.Module,
    device: torch.device,
    started_at: float,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        metrics.peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    metrics.training_time_seconds = time.perf_counter() - started_at
    metrics.total_parameters, metrics.trainable_parameters = count_parameters(model)


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
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    metrics = TrainingMetrics(title=title)
    started_at = _start_measurement(device)
    best_val_acc = float("-inf")
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            metrics.log_train(float(loss.item()), top1_acc(logits.detach(), labels))
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        metrics.log_val(val_loss, val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        tr_loss, va_loss = metrics.epoch_loss
        tr_acc, va_acc = metrics.epoch_acc
        print(f"epoch {epoch + 1:>3}/{epochs} | loss {tr_loss:.3f}/{va_loss:.3f} | acc {tr_acc:.1%}/{va_acc:.1%}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    _record_measurement(metrics, model, device, started_at)
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
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    student.to(device)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=lr)
    metrics = TrainingMetrics(title=title)
    started_at = _start_measurement(device)
    best_val_acc = float("-inf")
    best_state_dict = None

    for epoch in range(epochs):
        student.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            }
        tr_loss, va_loss = metrics.epoch_loss
        tr_acc, va_acc = metrics.epoch_acc
        print(f"epoch {epoch + 1:>3}/{epochs} | loss {tr_loss:.3f}/{va_loss:.3f} | acc {tr_acc:.1%}/{va_acc:.1%}")

    if best_state_dict is not None:
        student.load_state_dict(best_state_dict)
    _record_measurement(metrics, student, device, started_at)
    if curve_path is not None:
        metrics.plot(title=title, save_path=curve_path)
    return metrics
