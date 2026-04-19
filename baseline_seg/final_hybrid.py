from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _finite_or_nan(v: Any) -> float:
    try:
        fv = float(v)
    except Exception:
        return float("nan")
    return fv if np.isfinite(fv) else float("nan")


def _metric_columns(df: pd.DataFrame) -> Tuple[str, str]:
    dice_candidates = ["test_dice", "dice_mean", "val_dice"]
    hd95_candidates = ["test_hd95", "hd95_mean", "val_hd95"]

    dice_col = next((c for c in dice_candidates if c in df.columns), "")
    hd95_col = next((c for c in hd95_candidates if c in df.columns), "")
    return dice_col, hd95_col


def _safe_mean(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).values
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _score_row(row: pd.Series, dice_col: str, hd95_col: str) -> Tuple[float, float]:
    dice = _finite_or_nan(row.get(dice_col, np.nan)) if dice_col else float("nan")
    hd95 = _finite_or_nan(row.get(hd95_col, np.nan)) if hd95_col else float("nan")

    dice_score = dice if np.isfinite(dice) else -1e9
    hd95_score = -hd95 if np.isfinite(hd95) else -1e9
    return dice_score, hd95_score


def _select_best_value(df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in dataframe.")

    dice_col, hd95_col = _metric_columns(df)
    if not dice_col and not hd95_col:
        raise ValueError("No metric columns found for selection.")

    best_idx = None
    best_score = (-1e18, -1e18)
    for idx, row in df.iterrows():
        score = _score_row(row, dice_col=dice_col, hd95_col=hd95_col)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        raise RuntimeError("Unable to select best row.")

    row = df.loc[best_idx]
    return {
        "value": str(row[value_col]),
        "dice_col": dice_col,
        "hd95_col": hd95_col,
        "dice": _finite_or_nan(row.get(dice_col, np.nan)) if dice_col else float("nan"),
        "hd95": _finite_or_nan(row.get(hd95_col, np.nan)) if hd95_col else float("nan"),
    }


def select_components_from_tables(
    pretraining_csv: Optional[Path] = None,
    adaptation_csv: Optional[Path] = None,
    prompting_csv: Optional[Path] = None,
    architecture_csv: Optional[Path] = None,
    run_summaries_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """Select best pretraining/adaptation/prompting/backbone from available tables."""

    selected: Dict[str, Any] = {
        "pretraining_mode": "scratch",
        "pretraining_method": "masked_recon",
        "adaptation_mode": "full_ft",
        "prompt_mode": "none",
        "backbone": "unet",
        "sources": {},
        "compatibility_warnings": [],
    }

    if pretraining_csv and pretraining_csv.exists():
        df = pd.read_csv(pretraining_csv)
        if "mode" in df.columns:
            best = _select_best_value(df, value_col="mode")
            selected["pretraining_mode"] = best["value"]
            selected["sources"]["pretraining"] = best

    if adaptation_csv and adaptation_csv.exists():
        df = pd.read_csv(adaptation_csv)
        mode_col = "mode" if "mode" in df.columns else "variant"
        best = _select_best_value(df, value_col=mode_col)
        selected["adaptation_mode"] = best["value"]
        selected["sources"]["adaptation"] = best

    if prompting_csv and prompting_csv.exists():
        df = pd.read_csv(prompting_csv)
        mode_col = "prompt_mode" if "prompt_mode" in df.columns else "mode"
        best = _select_best_value(df, value_col=mode_col)
        selected["prompt_mode"] = best["value"]
        selected["sources"]["prompting"] = best

    if architecture_csv and architecture_csv.exists():
        df = pd.read_csv(architecture_csv)
        if "backbone" in df.columns:
            value_col = "backbone"
        elif "variant_backbone" in df.columns:
            value_col = "variant_backbone"
        elif "variant" in df.columns:
            value_col = "variant"
        else:
            value_col = "mode" if "mode" in df.columns else ""
        if value_col:
            best = _select_best_value(df, value_col=value_col)
            selected["backbone"] = best["value"]
            selected["sources"]["architecture"] = best

    if run_summaries_csv and run_summaries_csv.exists():
        rs = pd.read_csv(run_summaries_csv)
        if "fairness_valid" in rs.columns:
            rs = rs[rs["fairness_valid"] == True].copy()  # noqa: E712
        if not rs.empty and "model_variant" in rs.columns:
            parsed = rs["model_variant"].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip() else {})
            parsed_df = pd.DataFrame(list(parsed.values))
            merged = pd.concat([rs.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

            for comp_col, target_key in [
                ("pretraining_mode", "pretraining_mode"),
                ("adaptation_mode", "adaptation_mode"),
                ("prompt_mode", "prompt_mode"),
                ("backbone", "backbone"),
            ]:
                if comp_col not in merged.columns:
                    continue
                grouped_rows = []
                for comp_val, grp in merged.groupby(comp_col):
                    if pd.isna(comp_val):
                        continue
                    dice_col, hd95_col = _metric_columns(grp)
                    row = {
                        "candidate": str(comp_val),
                        "dice_mean": _safe_mean(grp[dice_col]) if dice_col else float("nan"),
                        "hd95_mean": _safe_mean(grp[hd95_col]) if hd95_col else float("nan"),
                    }
                    grouped_rows.append(row)

                if grouped_rows:
                    agg = pd.DataFrame(grouped_rows)
                    best = _select_best_value(agg, value_col="candidate")
                    selected[target_key] = best["value"]
                    selected["sources"][f"run_summaries_{target_key}"] = best

            # Pretraining method may be available only in run summary variants.
            if "pretraining_method" in merged.columns:
                grouped_rows = []
                for comp_val, grp in merged.groupby("pretraining_method"):
                    if pd.isna(comp_val):
                        continue
                    dice_col, hd95_col = _metric_columns(grp)
                    grouped_rows.append(
                        {
                            "candidate": str(comp_val),
                            "dice_mean": _safe_mean(grp[dice_col]) if dice_col else float("nan"),
                            "hd95_mean": _safe_mean(grp[hd95_col]) if hd95_col else float("nan"),
                        }
                    )
                if grouped_rows:
                    agg = pd.DataFrame(grouped_rows)
                    best = _select_best_value(agg, value_col="candidate")
                    selected["pretraining_method"] = best["value"]
                    selected["sources"]["run_summaries_pretraining_method"] = best

    return selected


def build_final_hybrid_config(
    base_cfg: Dict[str, Any],
    selected: Dict[str, Any],
    output_config_path: Path,
    pretrained_weights_path: Optional[Path] = None,
    experiment_name: str = "final_hybrid",
) -> Dict[str, Any]:
    """Create a final hybrid config by injecting selected best components."""

    cfg = json.loads(json.dumps(base_cfg))

    cfg.setdefault("pretraining", {})
    cfg.setdefault("adaptation", {})
    cfg.setdefault("prompting", {})
    cfg.setdefault("model", {})
    cfg.setdefault("logging", {})

    cfg["pretraining"]["mode"] = selected.get("pretraining_mode", cfg["pretraining"].get("mode", "scratch"))
    cfg["pretraining"]["method"] = selected.get("pretraining_method", cfg["pretraining"].get("method", "masked_recon"))
    cfg["adaptation"]["mode"] = selected.get("adaptation_mode", cfg["adaptation"].get("mode", "full_ft"))
    cfg["prompting"]["mode"] = selected.get("prompt_mode", cfg["prompting"].get("mode", "none"))
    cfg["model"]["backbone"] = selected.get("backbone", cfg["model"].get("backbone", "unet"))

    if pretrained_weights_path is not None:
        pw = str(pretrained_weights_path.resolve())
        cfg["pretraining"]["pretrained_path"] = pw
        cfg["adaptation"]["pretrained_path"] = pw

    cfg["logging"]["experiment_name"] = str(experiment_name)

    cfg.setdefault("final_hybrid", {})
    cfg["final_hybrid"].update(
        {
            "selected_components": selected,
            "selection_notes": "Auto-selected best components from comparison artifacts.",
        }
    )

    cfg["data"].setdefault("combine_train_val_for_train", False)

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def final_results_table(
    final_eval_csv: Path,
    baseline_eval_csv: Optional[Path],
    output_csv: Path,
    model_name: str = "final_hybrid",
    baseline_name: str = "baseline",
    notes: str = "",
) -> pd.DataFrame:
    """Build final comparison table: model | Dice | HD95 | notes."""

    def _aggregate(path: Path) -> Tuple[float, float, float]:
        if not path.exists():
            return float("nan"), float("nan"), 0.0
        df = pd.read_csv(path)
        dice = _safe_mean(df.get("dice_mean", pd.Series(dtype=float)))
        hd95 = _safe_mean(df.get("hd95_mean", pd.Series(dtype=float)))
        n = float(len(df))
        return dice, hd95, n

    rows = []
    f_dice, f_hd95, f_n = _aggregate(final_eval_csv)
    rows.append({"model": model_name, "dice": f_dice, "hd95": f_hd95, "n_cases": int(f_n), "notes": notes})

    if baseline_eval_csv is not None:
        b_dice, b_hd95, b_n = _aggregate(baseline_eval_csv)
        rows.append({"model": baseline_name, "dice": b_dice, "hd95": b_hd95, "n_cases": int(b_n), "notes": "reference"})

        rows.append(
            {
                "model": f"delta_{model_name}_minus_{baseline_name}",
                "dice": f_dice - b_dice if np.isfinite(f_dice) and np.isfinite(b_dice) else float("nan"),
                "hd95": f_hd95 - b_hd95 if np.isfinite(f_hd95) and np.isfinite(b_hd95) else float("nan"),
                "n_cases": int(min(f_n, b_n)),
                "notes": "positive dice and negative hd95 indicate improvement",
            }
        )

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def export_final_package(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    package_name: str = "final_hybrid_model",
) -> Dict[str, str]:
    """Package final model checkpoint/config and emit reusable inference helpers."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = output_dir / f"{package_name}.pt"
    cfg_out = output_dir / f"{package_name}_config.json"

    shutil.copy2(checkpoint_path, ckpt_out)
    shutil.copy2(config_path, cfg_out)

    py_script = output_dir / "run_final_infer.py"
    py_script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import subprocess",
                "",
                f"cfg = Path(r'{str(cfg_out)}')",
                f"ckpt = Path(r'{str(ckpt_out)}')",
                "out_dir = Path('predictions_final')",
                "cmd = [",
                "    'python', '-m', 'baseline_seg.cli', 'infer',",
                "    '--config', str(cfg),",
                "    '--checkpoint', str(ckpt),",
                "    '--split', 'test',",
                "    '--output-dir', str(out_dir),",
                "]",
                "subprocess.run(cmd, check=True)",
                "print({'predictions_dir': str(out_dir.resolve())})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ps_script = output_dir / "run_final_infer.ps1"
    ps_script.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"$cfg = '{str(cfg_out)}'",
                f"$ckpt = '{str(ckpt_out)}'",
                "$out = 'predictions_final'",
                "python -m baseline_seg.cli infer --config $cfg --checkpoint $ckpt --split test --output-dir $out",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "checkpoint": str(ckpt_out),
        "config": str(cfg_out),
        "python_infer_script": str(py_script),
        "powershell_infer_script": str(ps_script),
    }


def export_ablation_insights(
    selection_json: Path,
    output_json: Path,
    output_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """Export component-wise ablation insights from final component selection metadata."""

    if not selection_json.exists():
        raise FileNotFoundError(f"Selection JSON not found: {selection_json}")

    payload = json.loads(selection_json.read_text(encoding="utf-8"))
    selected = payload.get("selected_components", {})
    sources = selected.get("sources", {}) if isinstance(selected, dict) else {}

    rows: List[Dict[str, Any]] = []
    for key in ["pretraining_mode", "pretraining_method", "adaptation_mode", "prompt_mode", "backbone"]:
        val = selected.get(key, "")
        src_key = next((k for k in sources.keys() if k.endswith(key) or key in k), "")
        src = sources.get(src_key, {}) if src_key else {}
        rows.append(
            {
                "component": key,
                "selected_value": val,
                "source_key": src_key,
                "source_dice": _finite_or_nan(src.get("dice", np.nan)) if isinstance(src, dict) else float("nan"),
                "source_hd95": _finite_or_nan(src.get("hd95", np.nan)) if isinstance(src, dict) else float("nan"),
                "notes": "best available from comparison artifacts",
            }
        )

    out_payload = {
        "selected_components": {
            "pretraining_mode": selected.get("pretraining_mode", ""),
            "pretraining_method": selected.get("pretraining_method", ""),
            "adaptation_mode": selected.get("adaptation_mode", ""),
            "prompt_mode": selected.get("prompt_mode", ""),
            "backbone": selected.get("backbone", ""),
        },
        "ablation_rows": rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    return out_payload