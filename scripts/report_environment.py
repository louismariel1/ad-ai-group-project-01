"""Write a per-member environment report under files/ (shared Colab bootstrap Cell 9)."""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torchvision

from src.checkpoints import sha256_file
from src.paths import FILES_DIR, PROJECT_CONFIG_PATH, SSL_CKPT_DIR, SSL_SPLIT_PATH, TARGET_SPLIT_PATH, ensure_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", required=True, help='e.g. "Eric Perez"')
    args = parser.parse_args()

    ensure_layout()
    ckpt = SSL_CKPT_DIR / "teacher_ssl_v1.ckpt"
    member_id = re.sub(r"[^a-z0-9]+", "_", args.member.lower()).strip("_")
    report = {
        "member": args.member,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_sha256": sha256_file(PROJECT_CONFIG_PATH) if PROJECT_CONFIG_PATH.exists() else None,
        "target_split_sha256": sha256_file(TARGET_SPLIT_PATH) if TARGET_SPLIT_PATH.exists() else None,
        "ssl_split_sha256": sha256_file(SSL_SPLIT_PATH) if SSL_SPLIT_PATH.exists() else None,
        "ssl_checkpoint_exists": ckpt.is_file(),
        "ssl_checkpoint_sha256": sha256_file(ckpt) if ckpt.is_file() else None,
    }
    out = FILES_DIR / f"{member_id}_environment.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
