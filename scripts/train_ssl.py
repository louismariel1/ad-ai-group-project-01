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
from src.experiment import write_run_metrics
from src.models import build_pretext_model
from src.paths import FIGURES_DIR, METRICS_DIR, SSL_CKPT_DIR, ensure_layout
from src.repro import get_device, set_seed
from src.training import train_supervised


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_layout()
    checkpoint_path = SSL_CKPT_DIR / "teacher_ssl_v1.ckpt"
    metrics_path = METRICS_DIR / "ssl" / "teacher_ssl.json"
    if not args.overwrite and (checkpoint_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"SSL output already exists ({checkpoint_path} or {metrics_path}); "
            "pass --overwrite to replace it."
        )
    cfg = load_project_config()
    set_seed(cfg["seed"])
    device = get_device()
    epochs = cfg["training"]["ssl_epochs"] if args.epochs is None else args.epochs

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
        checkpoint_path,
        config=cfg,
        training_info={
            "pretext": cfg["ssl_pretext"],
            "epochs": epochs,
            "seed": cfg["seed"],
            "best_val_acc": metrics.best_val_acc,
            "training_time_seconds": metrics.training_time_seconds,
            "peak_cuda_memory_bytes": metrics.peak_cuda_memory_bytes,
            "total_parameters": metrics.total_parameters,
            "trainable_parameters": metrics.trainable_parameters,
        },
        overwrite=args.overwrite,
    )
    write_run_metrics(
        "ssl",
        "teacher_ssl",
        {
            "seed": cfg["seed"],
            "config": cfg,
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "training": metrics.to_dict(),
        },
        overwrite=args.overwrite,
    )
    print("Saved SSL backbone:", checkpoint_path)


if __name__ == "__main__":
    main()
