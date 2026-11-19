from __future__ import annotations

import datetime
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    config: dict[str, Any],
    training_info: dict[str, Any],
    extra: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Frozen checkpoint already exists: {path}")

    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config,
        "training_info": training_info,
        "torch_version": torch.__version__,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )
    return path


def load_checkpoint(path: Path, model: nn.Module | None = None, map_location: str | torch.device = "cpu"):
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload
