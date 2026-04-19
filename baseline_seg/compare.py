from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import json

import numpy as np
import pandas as pd


AGG_METRIC_COLUMNS = [
    "dice_mean",
    "iou_mean",
    "hd95_mean",
    "surface_dice_mean",
    "precision_mean",
    "recall_mean",
    "volume_error_mean",
    "ece",
    "mean_confidence",
    "mean_entropy",
    "mean_variation_ratio",
]


def _aggregate_eval_dataframe(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in AGG_METRIC_COLUMNS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").astype(float).values
            vals = vals[np.isfinite(vals)]
            out[col] = float(np.mean(vals)) if vals.size > 0 else float("nan")
        else:
            out[col] = float("nan")
    return out


def build_mode_comparison_table(mode_csv: Dict[str, Path], output_csv: Path) -> pd.DataFrame:
    """Aggregate evaluation summaries across scratch/supervised/ssl modes."""

    rows = []
    for mode, path in mode_csv.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        row = {
            "mode": mode,
            "n_cases": int(len(df)),
        }
        row.update(_aggregate_eval_dataframe(df))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(by="mode") if rows else pd.DataFrame()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _load_efficiency(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "trainable_params": float(payload.get("trainable_params", float("nan"))),
            "total_params": float(payload.get("total_params", float("nan"))),
            "trainable_ratio": float(payload.get("trainable_ratio", float("nan"))),
            "training_time_sec": float(payload.get("training_time_sec", float("nan"))),
            "gpu_max_memory_mb": float(payload.get("gpu_max_memory_mb", float("nan"))),
        }
    except Exception:
        return {}


def build_adaptation_comparison_table(
    eval_csvs: Dict[str, Path],
    efficiency_jsons: Dict[str, Optional[Path]],
    output_csv: Path,
) -> pd.DataFrame:
    """Aggregate adaptation method comparison table (accuracy + efficiency)."""

    rows = []
    for mode in ["full_ft", "lora", "adapters"]:
        csv_path = eval_csvs.get(mode)
        if csv_path is None or not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        row = {
            "mode": mode,
            "n_cases": int(len(df)),
        }
        row.update(_aggregate_eval_dataframe(df))
        row.update(_load_efficiency(efficiency_jsons.get(mode)))
        rows.append(row)

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def build_prompt_comparison_table(prompt_csvs: Dict[str, Path], output_csv: Path) -> pd.DataFrame:
    """Aggregate prompt mode performance and compute gains over no-prompt baseline."""

    rows = []
    baseline = None
    for mode in ["none", "spatial", "text"]:
        path = prompt_csvs.get(mode)
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path)
        row = {
            "prompt_mode": mode,
            "n_cases": int(len(df)),
        }
        row.update(_aggregate_eval_dataframe(df))
        if mode == "none":
            baseline = row["dice_mean"]
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and baseline is not None:
        out["dice_gain_vs_none"] = out["dice_mean"] - float(baseline)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def build_split_comparison_table(eval_csvs: Dict[str, Path], split_name: str, output_csv: Path) -> pd.DataFrame:
    """Compare multiple variants on a given split using standardized metric definitions."""

    rows = []
    for variant, csv_path in eval_csvs.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        row = {
            "variant": variant,
            "split": split_name,
            "n_cases": int(len(df)),
        }
        row.update(_aggregate_eval_dataframe(df))
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["dice_mean", "iou_mean"], ascending=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def build_per_organ_comparison_table(per_organ_csvs: Dict[str, Path], output_csv: Path) -> pd.DataFrame:
    """Build consistent per-organ comparison table across experiment variants."""

    rows = []
    for variant, csv_path in per_organ_csvs.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "organ" not in df.columns:
            continue
        for _, rec in df.iterrows():
            rows.append(
                {
                    "variant": variant,
                    "organ": rec.get("organ"),
                    "size_category": rec.get("size_category", "unknown"),
                    "is_small_or_rare": rec.get("is_small_or_rare", False),
                    "dice_mean": rec.get("dice_mean", np.nan),
                    "iou_mean": rec.get("iou_mean", np.nan),
                    "hd95_mean": rec.get("hd95_mean", np.nan),
                    "dice_rank": rec.get("dice_rank", np.nan),
                    "hd95_rank": rec.get("hd95_rank", np.nan),
                    "low_dice_flag": rec.get("low_dice_flag", False),
                    "high_hd95_flag": rec.get("high_hd95_flag", False),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["organ", "dice_mean"], ascending=[True, False])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
