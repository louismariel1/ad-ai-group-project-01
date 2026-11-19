from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import PIPELINES_PATH, PROJECT_CONFIG_PATH


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        existing = load_json(path)
        if existing != payload:
            raise ValueError(
                f"Existing file differs and overwrite is False: {path}. "
                "Coordinate with the group before changing frozen configs."
            )
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_project_config() -> dict[str, Any]:
    return load_json(PROJECT_CONFIG_PATH)


def load_pipelines() -> dict[str, Any]:
    return load_json(PIPELINES_PATH)
