from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


LOCKED_KEYS = [
    "seed",
    "data.root_dir",
    "data.split_manifest",
    "data.label_map_path",
    "pre_clip_min",
    "pre_clip_max",
    "augment",
    "loss",
    "optimizer",
    "scheduler",
    "train.batch_size",
    "train.grad_accum_steps",
    "train.max_epochs",
    "train.early_stopping_patience",
    "train.val_interval",
]


def _get_by_path(payload: Dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    cur: Any = payload
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _set_by_path(payload: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = payload
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def locked_settings_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the fairness-critical training settings from config."""

    out: Dict[str, Any] = {}
    for key in LOCKED_KEYS:
        _set_by_path(out, key, _get_by_path(cfg, key))
    return out


def model_variant_descriptor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the variant knobs that are allowed to differ in comparisons."""

    return {
        "backbone": cfg.get("model", {}).get("backbone", "unknown"),
        "pretraining_mode": cfg.get("pretraining", {}).get("mode", "scratch"),
        "pretraining_method": cfg.get("pretraining", {}).get("method", "none"),
        "adaptation_mode": cfg.get("adaptation", {}).get("mode", "full_ft"),
        "prompt_mode": cfg.get("prompting", {}).get("mode", "none"),
    }


def _dict_diff(a: Dict[str, Any], b: Dict[str, Any], prefix: str = "") -> Dict[str, Dict[str, Any]]:
    diffs: Dict[str, Dict[str, Any]] = {}
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        p = f"{prefix}.{k}" if prefix else k
        av = a.get(k, None)
        bv = b.get(k, None)
        if isinstance(av, dict) and isinstance(bv, dict):
            diffs.update(_dict_diff(av, bv, prefix=p))
        elif av != bv:
            diffs[p] = {"reference": av, "current": bv}
    return diffs


def enforce_fairness_lock(
    cfg: Dict[str, Any],
    output_root: Path,
    exp_name: str,
    logger,
) -> Dict[str, Any]:
    """Register and validate locked settings for a comparison group."""

    recipe = cfg.get("standard_recipe", {})
    enforce = bool(recipe.get("enforce", True))
    group = str(recipe.get("comparison_group", "default_group"))

    if not enforce:
        logger.info("Fairness enforcement disabled by standard_recipe.enforce=false")
        return {
            "group": group,
            "valid": True,
            "differences": {},
            "variant_differences": {},
            "variant_diff_count": 0,
            "registry": "",
            "initialized": False,
            "enforced": False,
        }
    registry_dir = output_root / "_fairness"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / f"{group}.json"

    current_locked = locked_settings_snapshot(cfg)
    current_variant = model_variant_descriptor(cfg)

    if not registry_path.exists():
        payload = {
            "comparison_group": group,
            "reference_experiment": exp_name,
            "locked_reference": current_locked,
            "variant_reference": current_variant,
            "runs": [],
        }
        registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Fairness lock initialized for group '%s' with experiment '%s'", group, exp_name)
        return {
            "group": group,
            "valid": True,
            "differences": {},
            "variant_differences": {},
            "variant_diff_count": 0,
            "registry": str(registry_path),
            "initialized": True,
            "enforced": True,
        }

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    reference = payload.get("locked_reference", {})
    variant_reference = payload.get("variant_reference", {})
    diffs = _dict_diff(reference, current_locked)
    variant_diffs = _dict_diff(variant_reference, current_variant)
    variant_diff_count = len(variant_diffs)

    shared_valid = len(diffs) == 0
    variant_valid = variant_diff_count <= 1
    valid = shared_valid and variant_valid

    if shared_valid:
        logger.info("Fairness lock check passed for group '%s'", group)
    else:
        logger.warning("Fairness lock mismatch for group '%s':", group)
        for k, dv in diffs.items():
            logger.warning("  %s | ref=%s | current=%s", k, dv.get("reference"), dv.get("current"))

    if variant_valid:
        logger.info("Variant control check passed for group '%s' (changed knobs=%d)", group, variant_diff_count)
    else:
        logger.warning("Variant control mismatch for group '%s': changed knobs=%d (must be <= 1)", group, variant_diff_count)
        for k, dv in variant_diffs.items():
            logger.warning("  %s | ref=%s | current=%s", k, dv.get("reference"), dv.get("current"))

    payload.setdefault("runs", []).append(
        {
            "experiment": exp_name,
            "valid": valid,
            "differences": diffs,
            "variant_differences": variant_diffs,
            "variant_diff_count": variant_diff_count,
        }
    )
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "group": group,
        "valid": valid,
        "differences": diffs,
        "variant_differences": variant_diffs,
        "variant_diff_count": variant_diff_count,
        "registry": str(registry_path),
        "initialized": False,
        "enforced": True,
    }


def save_run_summary(
    output_root: Path,
    exp_dir: Path,
    cfg: Dict[str, Any],
    fairness: Dict[str, Any],
    val_metrics: Optional[Dict[str, Any]] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save per-run summary JSON and append to global CSV index."""

    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    variant = model_variant_descriptor(cfg)
    locked = locked_settings_snapshot(cfg)

    row: Dict[str, Any] = {
        "experiment_name": exp_dir.name,
        "experiment_path": str(exp_dir),
        "comparison_group": fairness.get("group"),
        "fairness_valid": bool(fairness.get("valid", False)),
        "fairness_diff_count": int(len(fairness.get("differences", {}))),
        "variant_diff_count": int(fairness.get("variant_diff_count", 0)),
        "fairness_differences": json.dumps(fairness.get("differences", {})),
        "variant_differences": json.dumps(fairness.get("variant_differences", {})),
        "model_variant": json.dumps(variant),
        "locked_settings": json.dumps(locked),
        "val_dice": float("nan"),
        "test_dice": float("nan"),
        "test_hd95": float("nan"),
    }

    if val_metrics is not None:
        row["val_dice"] = float(val_metrics.get("val_dice", float("nan")))
    if test_metrics is not None:
        row["test_dice"] = float(test_metrics.get("dice_mean", float("nan")))
        row["test_hd95"] = float(test_metrics.get("hd95_mean", float("nan")))

    run_json = logs_dir / "run_summary.json"
    run_json.write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary_csv = output_root / "run_summaries.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        df = df[df["experiment_name"] != row["experiment_name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(summary_csv, index=False)
    return run_json


def build_fair_comparison_table(summary_csv: Path, comparison_group: str, output_csv: Path) -> pd.DataFrame:
    """Build comparison table using only fairness-valid runs in a group."""

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    subset = df[(df["comparison_group"] == comparison_group) & (df["fairness_valid"] == True)].copy()

    if subset.empty:
        out = pd.DataFrame()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        return out

    subset["variant_backbone"] = subset["model_variant"].apply(
        lambda x: json.loads(x).get("backbone", "unknown") if isinstance(x, str) else "unknown"
    )
    subset["variant_pretraining"] = subset["model_variant"].apply(
        lambda x: json.loads(x).get("pretraining_mode", "unknown") if isinstance(x, str) else "unknown"
    )
    subset["variant_adaptation"] = subset["model_variant"].apply(
        lambda x: json.loads(x).get("adaptation_mode", "unknown") if isinstance(x, str) else "unknown"
    )
    subset["variant_prompt"] = subset["model_variant"].apply(
        lambda x: json.loads(x).get("prompt_mode", "unknown") if isinstance(x, str) else "unknown"
    )

    keep = [
        "experiment_name",
        "comparison_group",
        "variant_backbone",
        "variant_pretraining",
        "variant_adaptation",
        "variant_prompt",
        "val_dice",
        "test_dice",
        "test_hd95",
        "fairness_valid",
        "fairness_diff_count",
        "variant_diff_count",
    ]

    out = subset[keep].sort_values(by=["test_dice", "val_dice"], ascending=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def best_val_from_history(logs_dir: Path) -> Dict[str, Any]:
    """Extract best validation metrics from training history CSV when available."""

    hist = logs_dir / "training_history.csv"
    if not hist.exists():
        return {"val_dice": float("nan")}

    df = pd.read_csv(hist)
    if "val_dice" not in df.columns or df["val_dice"].dropna().empty:
        return {"val_dice": float("nan")}

    idx = int(df["val_dice"].astype(float).idxmax())
    row = df.loc[idx]
    return {
        "val_dice": float(row.get("val_dice", np.nan)),
        "val_iou": float(row.get("val_iou", np.nan)) if "val_iou" in row else float("nan"),
        "val_hd95": float(row.get("val_hd95", np.nan)) if "val_hd95" in row else float("nan"),
        "epoch": int(row.get("epoch", -1)) if "epoch" in row else -1,
    }
