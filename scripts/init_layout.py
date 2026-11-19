"""Create the shared GitHub folder layout (config, checkpoints, experiments, results, files)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import PROJECT_ROOT, ensure_layout

if __name__ == "__main__":
    ensure_layout()
    print("Project root:", PROJECT_ROOT)
    print("Layout ready: config, checkpoints, experiments/RQ1-RQ3, results, files")
