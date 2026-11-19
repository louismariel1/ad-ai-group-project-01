"""RQ1 pipelines A–E. Member 1: A,B. Member 2: C,D,E."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_project_config
from src.experiment import pipeline_spec, run_kd_pipeline
from src.paths import SSL_CKPT_DIR, ensure_layout

PIPELINES = ["A", "B", "C", "D", "E"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", nargs="+", default=PIPELINES, choices=PIPELINES)
    parser.add_argument("--ssl-ckpt", type=Path, default=SSL_CKPT_DIR / "teacher_ssl_v1.ckpt")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    ensure_layout()
    cfg = load_project_config()
    epochs = args.epochs or cfg["training"]["student_epochs"]
    for name in args.pipelines:
        spec = pipeline_spec("RQ1", name)
        path = run_kd_pipeline(
            "RQ1",
            name,
            teacher_kind=spec["teacher"],
            student_kind=spec["student"],
            kd_type=spec["kd"],
            ssl_ckpt=args.ssl_ckpt,
            epochs=epochs,
        )
        print("Wrote", path)


if __name__ == "__main__":
    main()
