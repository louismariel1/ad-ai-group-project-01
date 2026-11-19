"""Fine-tune dense and LoRA teachers once, then share the checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_project_config
from src.experiment import adapt_teacher
from src.paths import SSL_CKPT_DIR, ensure_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-ckpt", type=Path, default=SSL_CKPT_DIR / "teacher_ssl_v1.ckpt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--kinds", nargs="+", default=["dense", "lora"])
    args = parser.parse_args()

    ensure_layout()
    cfg = load_project_config()
    epochs = args.epochs or cfg["training"]["teacher_epochs"]
    if not args.ssl_ckpt.exists():
        raise FileNotFoundError(f"Train SSL first: missing {args.ssl_ckpt}")
    for kind in args.kinds:
        path = adapt_teacher(kind, cfg, args.ssl_ckpt, epochs=epochs)
        print("Saved teacher:", path)


if __name__ == "__main__":
    main()
