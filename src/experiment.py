from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from .checkpoints import load_checkpoint, save_checkpoint
from .config import load_pipelines, load_project_config
from .data import target_dataloaders
from .lora import apply_lora_from_config, report_trainable
from .models import build_classifier
from .paths import FIGURES_DIR, METRICS_DIR, STUDENT_CKPT_DIR, TEACHER_CKPT_DIR
from .repro import get_device, set_seed
from .training import evaluate_model, train_distillation, train_supervised


def _metrics_path(rq: str, name: str) -> Path:
    return METRICS_DIR / rq / f"{name}.json"


def write_run_metrics(
    rq: str,
    name: str,
    payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    path = _metrics_path(rq, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Experiment metrics already exist: {path}")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def load_ssl_backbone(cfg: dict, ssl_ckpt: Path, device: torch.device):
    model = build_classifier(cfg, num_classes=len(cfg["target_classes"]))
    payload = load_checkpoint(ssl_ckpt, map_location=device)
    model.backbone.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def adapt_teacher(
    kind: str,
    cfg: dict,
    ssl_ckpt: Path,
    epochs: int,
    *,
    overwrite: bool = False,
) -> Path:
    set_seed(cfg["seed"])
    device = get_device()
    out_path = TEACHER_CKPT_DIR / f"teacher_{kind}.ckpt"
    metrics_path = _metrics_path("teachers", kind)
    if not overwrite and (out_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"Teacher output already exists ({out_path} or {metrics_path}); "
            "pass --overwrite to replace it."
        )
    train_loader, val_loader, _test_loader = target_dataloaders(cfg)
    model = load_ssl_backbone(cfg, ssl_ckpt, device)
    if kind == "lora":
        apply_lora_from_config(model, cfg, freeze_base=True)
    report_trainable(model, f"teacher-{kind}")
    started_at = time.perf_counter()
    metrics = train_supervised(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=cfg["training"]["lr"],
        device=device,
        title=f"Teacher {kind} fine-tune",
        curve_path=FIGURES_DIR / "teachers" / f"teacher_{kind}_curve.png",
    )
    save_checkpoint(
        model,
        out_path,
        config=cfg,
        training_info={
            "kind": kind,
            "best_val_acc": metrics.best_val_acc,
            "epochs": epochs,
            "seed": cfg["seed"],
            "training_time_seconds": metrics.training_time_seconds,
            "peak_cuda_memory_bytes": metrics.peak_cuda_memory_bytes,
            "total_parameters": metrics.total_parameters,
            "trainable_parameters": metrics.trainable_parameters,
        },
        overwrite=overwrite,
    )
    write_run_metrics(
        "teachers",
        kind,
        {
            "kind": kind,
            "seed": cfg["seed"],
            "config": cfg,
            "checkpoint": str(out_path),
            "checkpoint_size_bytes": out_path.stat().st_size,
            "training": metrics.to_dict(),
            "wall_time_seconds": time.perf_counter() - started_at,
        },
        overwrite=overwrite,
    )
    return out_path


def build_student(cfg: dict, student_kind: str, num_classes: int) -> torch.nn.Module:
    student = build_classifier(cfg, num_classes=num_classes)
    if student_kind == "lora":
        apply_lora_from_config(student, cfg, freeze_base=True)
    report_trainable(student, f"student-{student_kind}")
    return student


def run_kd_pipeline(
    rq: str,
    name: str,
    *,
    teacher_kind: str,
    student_kind: str,
    kd_type: str,
    ssl_ckpt: Path,
    epochs: int,
    teacher_ckpt: Path | None = None,
    overwrite: bool = False,
) -> Path:
    cfg = load_project_config()
    set_seed(cfg["seed"])
    device = get_device()
    student_ckpt_path = STUDENT_CKPT_DIR / f"{rq}_{name}.ckpt"
    metrics_path = _metrics_path(rq, name)
    if not overwrite and (student_ckpt_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"Experiment output already exists ({student_ckpt_path} or {metrics_path}); "
            "pass --overwrite to replace it."
        )
    train_loader, val_loader, test_loader = target_dataloaders(cfg)

    teacher_path = teacher_ckpt or (TEACHER_CKPT_DIR / f"teacher_{teacher_kind}.ckpt")
    if not teacher_path.exists():
        teacher_path = adapt_teacher(
            teacher_kind,
            cfg,
            ssl_ckpt,
            epochs=cfg["training"]["teacher_epochs"],
            overwrite=overwrite,
        )

    teacher = build_student(cfg, teacher_kind, len(cfg["target_classes"]))
    teacher_payload = load_checkpoint(teacher_path, teacher, map_location=device)

    student = load_ssl_backbone(cfg, ssl_ckpt, device)
    if student_kind == "lora":
        apply_lora_from_config(student, cfg, freeze_base=True)
    report_trainable(student, f"student-{student_kind}")

    metrics = train_distillation(
        student,
        teacher,
        train_loader,
        val_loader,
        kd_type=kd_type,
        epochs=epochs,
        lr=cfg["training"]["lr"],
        temperature=cfg["training"]["kd_temperature"],
        kd_alpha=cfg["training"]["kd_alpha"],
        feature_weight=cfg["training"]["feature_kd_weight"],
        device=device,
        title=f"{rq} {name} ({kd_type})",
        curve_path=FIGURES_DIR / rq / f"{name}_curve.png",
    )
    test_loss, test_acc = evaluate_model(student, test_loader, torch.nn.CrossEntropyLoss(), device)
    save_checkpoint(
        student,
        STUDENT_CKPT_DIR / f"{rq}_{name}.ckpt",
        config=cfg,
        training_info={
            "rq": rq,
            "name": name,
            "kd_type": kd_type,
            "test_acc": test_acc,
            "seed": cfg["seed"],
            "epochs": epochs,
            "training_time_seconds": metrics.training_time_seconds,
            "peak_cuda_memory_bytes": metrics.peak_cuda_memory_bytes,
            "total_parameters": metrics.total_parameters,
            "trainable_parameters": metrics.trainable_parameters,
        },
        overwrite=overwrite,
    )
    return write_run_metrics(
        rq,
        name,
        {
            "pipeline": name,
            "teacher": teacher_kind,
            "student": student_kind,
            "kd": kd_type,
            "seed": cfg["seed"],
            "config": cfg,
            "best_val_acc": metrics.best_val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "checkpoint": str(student_ckpt_path),
            "teacher_checkpoint": str(teacher_path),
            "resources": {
                "teacher_adaptation_time_seconds": teacher_payload["training_info"].get("training_time_seconds"),
                "teacher_trainable_parameters": teacher_payload["training_info"].get("trainable_parameters"),
                "teacher_total_parameters": teacher_payload["training_info"].get("total_parameters"),
                "teacher_checkpoint_bytes": teacher_path.stat().st_size,
                "student_training_time_seconds": metrics.training_time_seconds,
                "student_trainable_parameters": metrics.trainable_parameters,
                "student_total_parameters": metrics.total_parameters,
                "student_checkpoint_bytes": student_ckpt_path.stat().st_size,
                "peak_cuda_memory_bytes": metrics.peak_cuda_memory_bytes,
            },
            "history": metrics.to_dict(),
        },
        overwrite=overwrite,
    )


def pipeline_spec(rq: str, name: str) -> dict[str, Any]:
    return load_pipelines()[rq]["pipelines"][name]
