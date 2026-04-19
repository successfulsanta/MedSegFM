from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ConfigError(RuntimeError):
    """Raised when the experiment config is invalid."""


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON/YAML configuration for baseline experiments."""

    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        cfg = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ConfigError("YAML config requires pyyaml. Install it or use JSON.") from exc
        cfg = yaml.safe_load(text)
    else:
        raise ConfigError("Unsupported config extension. Use .json, .yaml, or .yml")

    return validate_config(cfg)


def _parse_override_value(raw: str) -> Any:
    """Parse CLI override values into native Python types when possible."""

    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null" or lower == "none":
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except Exception:
        return raw


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply dot-path CLI overrides, e.g. train.lr=2e-4."""

    updated = json.loads(json.dumps(cfg))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}'. Use key=value format.")
        key_path, raw_val = item.split("=", 1)
        key_parts = [p for p in key_path.split(".") if p]
        if not key_parts:
            raise ConfigError(f"Invalid override key path: '{key_path}'")

        cursor: Dict[str, Any] = updated
        for part in key_parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[key_parts[-1]] = _parse_override_value(raw_val)

    return validate_config(updated)


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate mandatory keys and normalize common defaults."""

    required_sections = ["data", "model", "train", "augment", "loss", "infer", "logging"]
    for section in required_sections:
        if section not in cfg:
            raise ConfigError(f"Missing config section: {section}")

    cfg.setdefault("seed", 42)
    cfg.setdefault("deterministic", True)
    cfg.setdefault("device", "auto")
    cfg.setdefault("pre_clip_min", -200)
    cfg.setdefault("pre_clip_max", 300)

    cfg["train"].setdefault("max_epochs", 200)
    cfg["train"].setdefault("val_interval", 1)
    cfg["train"].setdefault("early_stopping_patience", 30)
    cfg["train"].setdefault("amp", True)
    cfg["train"].setdefault("grad_accum_steps", 1)

    cfg.setdefault("optimizer", {"name": "adamw", "lr": cfg["train"].get("lr", 1e-4), "weight_decay": cfg["train"].get("weight_decay", 1e-5)})
    cfg.setdefault("scheduler", {"name": "cosine", "t_max": cfg["train"].get("max_epochs", 200), "eta_min": 1e-6})

    cfg["augment"].setdefault("flip_prob", 0.5)
    cfg["augment"].setdefault("rotate_prob", 0.2)
    cfg["augment"].setdefault("scale_prob", 0.2)
    cfg["augment"].setdefault("noise_prob", 0.15)
    cfg["augment"].setdefault("elastic_prob", 0.0)

    cfg["logging"].setdefault("tensorboard", False)
    cfg["logging"].setdefault("save_epoch_metrics_json", True)
    cfg["infer"].setdefault("save_probabilities", False)
    cfg["data"].setdefault("combine_train_val_for_train", False)

    cfg.setdefault(
        "standard_recipe",
        {
            "recipe_name": "default_standard_recipe",
            "comparison_group": "baseline_family",
            "enforce": True,
            "notes": "Keep this section unchanged across experiments for fair comparison.",
        },
    )

    cfg.setdefault(
        "pretraining",
        {
            "mode": "scratch",
            "method": "masked_recon",
            "pretrained_path": None,
            "freeze_backbone": False,
            "ssl_epochs": 50,
            "ssl_batch_size": 1,
            "ssl_lr": 1e-4,
            "ssl_weight_decay": 1e-5,
            "mask_ratio": 0.4,
            "patch_size": [8, 8, 8],
            "temperature": 0.1,
        },
    )

    cfg.setdefault(
        "adaptation",
        {
            "mode": "full_ft",
            "pretrained_path": None,
            "freeze_backbone": False,
            "lora_rank": 4,
            "lora_alpha": 1.0,
            "lora_on_conv": True,
            "adapter_size": 8,
            "adapter_on_linear": True,
            "train_head": True,
        },
    )

    cfg.setdefault(
        "prompting",
        {
            "mode": "none",
            "num_points": 3,
            "use_bboxes": True,
            "point_radius": 1,
            "text_embedding_dim": 8,
            "max_text_organs": 4,
            "train_prompt_level": "full",
            "val_prompt_level": "minimal",
            "inference_prompt_level": "minimal",
            "prompt_generation_strategy": "from_ground_truth",
        },
    )

    cfg.setdefault(
        "final_hybrid",
        {
            "enabled": False,
            "selected_components": {},
            "selection_notes": "",
        },
    )

    p_mode = str(cfg["prompting"].get("mode", "none")).lower()
    if p_mode not in {"none", "spatial", "text"}:
        raise ConfigError("prompting.mode must be one of: none, spatial, text")

    for key in ["train_prompt_level", "val_prompt_level", "inference_prompt_level"]:
        level = str(cfg["prompting"].get(key, "minimal")).lower()
        if level not in {"none", "minimal", "full"}:
            raise ConfigError(f"prompting.{key} must be one of: none, minimal, full")

    adaptation_mode = str(cfg["adaptation"].get("mode", "full_ft")).lower()
    if adaptation_mode not in {"full_ft", "lora", "adapters", "partial_decoder"}:
        raise ConfigError("adaptation.mode must be one of: full_ft, lora, adapters, partial_decoder")

    mode = str(cfg["pretraining"].get("mode", "scratch")).lower()
    if mode not in {"scratch", "supervised", "ssl"}:
        raise ConfigError("pretraining.mode must be one of: scratch, supervised, ssl")

    method = str(cfg["pretraining"].get("method", "masked_recon")).lower()
    if method not in {"masked_recon", "denoise", "contrastive"}:
        raise ConfigError("pretraining.method must be one of: masked_recon, denoise, contrastive")

    if mode in {"supervised", "ssl"} and not cfg["pretraining"].get("pretrained_path"):
        # Pretrained path can be injected at runtime for resume/evaluation runs.
        pass

    roi_size = cfg["model"].get("roi_size", [128, 128, 128])
    if len(roi_size) != 3:
        raise ConfigError("model.roi_size must be length 3")

    target_workers = int(cfg["data"].get("num_workers", 4))
    if target_workers < 0:
        raise ConfigError("data.num_workers must be >= 0")

    return cfg


def fairness_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Create a hashable fairness signature for comparison across experiments."""

    fairness_payload = {
        "data_root": cfg["data"].get("root_dir"),
        "label_map": cfg["data"].get("label_map_path"),
        "pre_clip": [cfg.get("pre_clip_min", -200), cfg.get("pre_clip_max", 300)],
        "augment": cfg.get("augment", {}),
        "loss": cfg.get("loss", {}),
        "optimizer": cfg.get("optimizer", {}),
        "scheduler": cfg.get("scheduler", {}),
        "adaptation": cfg.get("adaptation", {}),
        "prompting": cfg.get("prompting", {}),
        "train_core": {
            "batch_size": cfg["train"].get("batch_size"),
            "max_epochs": cfg["train"].get("max_epochs"),
            "val_interval": cfg["train"].get("val_interval"),
            "early_stopping_patience": cfg["train"].get("early_stopping_patience"),
        },
        "roi_size": cfg["model"].get("roi_size"),
        "recipe": cfg.get("standard_recipe", {}),
    }
    encoded = json.dumps(fairness_payload, sort_keys=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return {"fingerprint": digest, "payload": fairness_payload}


def load_label_map(label_map_path: Path) -> Dict[int, str]:
    """Load label ID to organ name mapping from JSON."""

    if not label_map_path.exists():
        raise ConfigError(f"Label map file does not exist: {label_map_path}")

    raw = json.loads(label_map_path.read_text(encoding="utf-8"))
    out: Dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return dict(sorted(out.items(), key=lambda item: item[0]))


def resolve_num_classes(cfg: Dict[str, Any], label_map: Dict[int, str]) -> int:
    """Resolve class count from config and label map consistency."""

    cfg_classes = int(cfg["model"].get("num_classes", 0))
    implied_classes = max(label_map.keys()) + 1 if label_map else 0

    if cfg_classes <= 0 and implied_classes <= 0:
        raise ConfigError("Unable to resolve num_classes from config or label map")

    if cfg_classes <= 0:
        return implied_classes

    if implied_classes > 0 and cfg_classes != implied_classes:
        raise ConfigError(
            f"model.num_classes={cfg_classes} does not match label map implied classes={implied_classes}"
        )

    return cfg_classes


def resolve_device(device_name: str) -> str:
    """Resolve device string with optional auto mode."""

    if device_name == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device_name


def split_dirs(root_dir: Path, split: str) -> Tuple[Path, Path]:
    """Map logical split name to image/label directories."""

    mapping = {
        "train": (root_dir / "imagesTr", root_dir / "labelsTr"),
        "val": (root_dir / "imagesVal", root_dir / "labelsVal"),
        "test": (root_dir / "imagesTs", root_dir / "labelsTs"),
    }
    if split not in mapping:
        raise ConfigError(f"Unsupported split: {split}")
    return mapping[split]
