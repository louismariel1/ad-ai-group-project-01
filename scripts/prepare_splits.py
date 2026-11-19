"""Freeze CIFAR prototype splits. Re-run after changing the dataset in project_config.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_project_config
from src.data import write_frozen_splits
from src.paths import ensure_layout
from src.repro import set_seed

if __name__ == "__main__":
    ensure_layout()
    cfg = load_project_config()
    set_seed(cfg["seed"])
    target_path, ssl_path = write_frozen_splits(cfg)
    print("Wrote", target_path)
    print("Wrote", ssl_path)
    print(json.dumps({"prototype": cfg["prototype"], "source": cfg["source_dataset"], "target": cfg["target_dataset"]}, indent=2))
