"""Train the shared SSL rotation teacher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpoints import save_checkpoint
from src.config import load_project_config
from src.data import ssl_dataloaders
from src.models import build_pretext_model
from src.paths import FIGURES_DIR, SSL_CKPT_DIR, ensure_layout
from src.repro import get_device, set_seed
from src.training import train_supervised


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_layout()
    cfg = load_project_config()
    set_seed(cfg["seed"])
    device = get_device()
    epochs = args.epochs or cfg["training"]["ssl_epochs"]

    train_loader, val_loader = ssl_dataloaders(cfg)
    model = build_pretext_model(cfg)
    metrics = train_supervised(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=cfg["training"]["lr"],
        device=device,
        title="SSL rotation teacher",
        curve_path=FIGURES_DIR / "ssl" / "teacher_ssl_curve.png",
    )
    backbone = model[0]
    save_checkpoint(
        backbone,
        SSL_CKPT_DIR / "teacher_ssl_v1.ckpt",
        config=cfg,
        training_info={"pretext": cfg["ssl_pretext"], "epochs": epochs, "best_val_acc": metrics.best_val_acc},
        extra={"history": metrics.to_dict()},
        overwrite=args.overwrite,
    )
    print("Saved SSL backbone:", SSL_CKPT_DIR / "teacher_ssl_v1.ckpt")


if __name__ == "__main__":
    main()
