from __future__ import annotations

import json
import logging
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set global random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def setup_experiment_dirs(base_dir: Path, exp_name: str) -> Dict[str, Path]:
    """Create sequential experiment directory tree (exp_001 style)."""

    base_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("exp_")])
    next_idx = 1
    if existing:
        try:
            next_idx = max(int(p.name.split("_")[1]) for p in existing if len(p.name.split("_")) > 1) + 1
        except Exception:
            next_idx = len(existing) + 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"exp_{next_idx:03d}_{exp_name}_{ts}"
    ckpt_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    preds_dir = exp_dir / "predictions"
    vis_dir = exp_dir / "visuals"

    for d in [exp_dir, ckpt_dir, logs_dir, preds_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "exp_dir": exp_dir,
        "ckpt_dir": ckpt_dir,
        "logs_dir": logs_dir,
        "preds_dir": preds_dir,
        "vis_dir": vis_dir,
    }


def setup_logger(log_file: Path) -> logging.Logger:
    """Create a logger that writes both console and file output."""

    logger = logging.getLogger(f"baseline_seg_{log_file.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Serialize dictionary to JSON with pretty formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_env_info() -> Dict[str, Any]:
    """Collect runtime environment details for reproducibility."""

    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_info_error"] = str(exc)

    try:
        import monai

        info["monai_version"] = monai.__version__
    except Exception as exc:
        info["monai_info_error"] = str(exc)

    return info
