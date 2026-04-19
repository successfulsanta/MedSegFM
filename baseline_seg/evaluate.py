from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from baseline_seg.failure_analysis import analyze_case_organ_failures, summarize_failure_patterns
from baseline_seg.metrics import case_metrics
from baseline_seg.visualization import save_evaluation_panel


CORE_MEAN_COLUMNS = [
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


def _nanmean(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).values
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _nanstd(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).values
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def _nanmedian(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).values
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.strip().lower())


def _label_name(label_map: Dict[int, str], class_idx: int) -> str:
    return _sanitize_name(label_map.get(class_idx, f"class_{class_idx}"))


def _load_probability_tensor(probability_dir: Optional[Path], case_id: str, num_classes: int) -> Optional[np.ndarray]:
    if probability_dir is None:
        return None

    npz_path = probability_dir / f"{case_id}.npz"
    npy_path = probability_dir / f"{case_id}.npy"
    arr: Optional[np.ndarray] = None

    if npz_path.exists():
        payload = np.load(npz_path, allow_pickle=False)
        for key in ["probs", "prob", "logits", "arr_0"]:
            if key in payload:
                arr = payload[key]
                break
    elif npy_path.exists():
        arr = np.load(npy_path, allow_pickle=False)

    if arr is None or arr.ndim != 4:
        return None

    if arr.shape[0] == num_classes:
        chan_first = arr
    elif arr.shape[-1] == num_classes:
        chan_first = np.moveaxis(arr, -1, 0)
    else:
        return None

    chan_first = chan_first.astype(np.float32)
    csum = np.sum(chan_first, axis=0)
    is_prob = np.all((chan_first >= 0.0) & (chan_first <= 1.0)) and np.all(np.abs(csum - 1.0) < 1e-2)
    if is_prob:
        return chan_first

    # Fallback: treat as logits and softmax over class channel.
    max_logits = np.max(chan_first, axis=0, keepdims=True)
    exp_logits = np.exp(chan_first - max_logits)
    denom = np.sum(exp_logits, axis=0, keepdims=True) + 1e-8
    return exp_logits / denom


def _build_case_row(
    case_id: str,
    metrics: Dict[str, Any],
    label_map: Dict[int, str],
    num_classes: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"case_id": case_id}
    for key in CORE_MEAN_COLUMNS:
        row[key] = metrics.get(key, float("nan"))

    for class_idx in range(1, num_classes):
        organ = _label_name(label_map, class_idx)
        col_suffix = f"organ_{organ}"
        pos = class_idx - 1
        row[f"dice_{col_suffix}"] = metrics["dice_per_class"][pos]
        row[f"iou_{col_suffix}"] = metrics["iou_per_class"][pos]
        row[f"hd95_{col_suffix}"] = metrics["hd95_per_class"][pos]
        row[f"surface_dice_{col_suffix}"] = metrics["surface_dice_per_class"][pos]
        row[f"precision_{col_suffix}"] = metrics["precision_per_class"][pos]
        row[f"recall_{col_suffix}"] = metrics["recall_per_class"][pos]
        row[f"volume_error_{col_suffix}"] = metrics["volume_error_per_class"][pos]
    return row


def _organ_aggregate(
    per_case: pd.DataFrame,
    label_map: Dict[int, str],
    num_classes: int,
    organ_presence_counts: Dict[int, int],
    organ_voxel_fracs: Dict[int, float],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_cases = max(1, int(len(per_case)))
    rare_threshold_cases = max(2, int(0.1 * n_cases))
    rare_threshold_vox = 0.005
    low_dice_threshold = 0.5
    high_hd95_threshold = 20.0

    for class_idx in range(1, num_classes):
        organ = _label_name(label_map, class_idx)
        cprefix = f"organ_{organ}"
        record: Dict[str, Any] = {
            "class_id": class_idx,
            "organ": organ,
            "present_cases": int(organ_presence_counts.get(class_idx, 0)),
            "present_case_ratio": float(organ_presence_counts.get(class_idx, 0) / float(n_cases)),
            "gt_voxel_fraction": float(organ_voxel_fracs.get(class_idx, 0.0)),
        }

        for metric in ["dice", "iou", "hd95", "surface_dice", "precision", "recall", "volume_error"]:
            col = f"{metric}_{cprefix}"
            if col in per_case.columns:
                vals = pd.to_numeric(per_case[col], errors="coerce").values
                record[f"{metric}_mean"] = _nanmean(vals)
                record[f"{metric}_std"] = _nanstd(vals)
                record[f"{metric}_median"] = _nanmedian(vals)
            else:
                record[f"{metric}_mean"] = float("nan")
                record[f"{metric}_std"] = float("nan")
                record[f"{metric}_median"] = float("nan")

        is_rare = (
            record["present_cases"] <= rare_threshold_cases
            or record["gt_voxel_fraction"] <= rare_threshold_vox
        )
        record["is_small_or_rare"] = bool(is_rare)
        if record["gt_voxel_fraction"] <= 0.005:
            record["size_category"] = "small"
        elif record["gt_voxel_fraction"] <= 0.02:
            record["size_category"] = "medium"
        else:
            record["size_category"] = "large"
        record["low_dice_flag"] = bool(np.isfinite(record["dice_mean"]) and record["dice_mean"] < low_dice_threshold)
        record["high_hd95_flag"] = bool(np.isfinite(record["hd95_mean"]) and record["hd95_mean"] > high_hd95_threshold)
        rows.append(record)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["dice_rank"] = out["dice_mean"].rank(method="min", ascending=True)
        out["hd95_rank"] = out["hd95_mean"].rank(method="min", ascending=False)
        out["worstness_score"] = (
            pd.to_numeric(out["dice_rank"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["hd95_rank"], errors="coerce").fillna(0.0)
        )
        out = out.sort_values(by=["is_small_or_rare", "dice_mean"], ascending=[False, True])
    return out


def _size_category_report(per_organ: pd.DataFrame) -> pd.DataFrame:
    if per_organ.empty or "size_category" not in per_organ.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for category in ["small", "medium", "large"]:
        sub = per_organ[per_organ["size_category"] == category]
        if sub.empty:
            continue
        rows.append(
            {
                "size_category": category,
                "n_organs": int(len(sub)),
                "dice_mean": _nanmean(sub["dice_mean"].values),
                "iou_mean": _nanmean(sub["iou_mean"].values),
                "hd95_mean": _nanmean(sub["hd95_mean"].values),
            }
        )
    return pd.DataFrame(rows)


def _compact_organ_report(per_organ: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if per_organ.empty:
        return pd.DataFrame()

    cols = [
        "organ",
        "size_category",
        "is_small_or_rare",
        "dice_mean",
        "iou_mean",
        "hd95_mean",
        "low_dice_flag",
        "high_hd95_flag",
        "dice_rank",
        "hd95_rank",
        "worstness_score",
    ]
    available_cols = [c for c in cols if c in per_organ.columns]
    core = per_organ[available_cols].copy()
    core = core.sort_values(by=["worstness_score", "dice_mean"], ascending=[False, True])

    worst = core.head(max(1, int(top_k))).copy()
    worst.insert(0, "section", "worst_organs")

    macro_row = {
        "section": "macro_average",
        "organ": "macro_average",
        "size_category": "all",
        "is_small_or_rare": False,
        "dice_mean": _nanmean(per_organ["dice_mean"].values),
        "iou_mean": _nanmean(per_organ["iou_mean"].values),
        "hd95_mean": _nanmean(per_organ["hd95_mean"].values),
        "low_dice_flag": False,
        "high_hd95_flag": False,
        "dice_rank": float("nan"),
        "hd95_rank": float("nan"),
        "worstness_score": float("nan"),
    }

    return pd.concat([pd.DataFrame([macro_row]), worst], ignore_index=True)


def _organ_focus_visuals(
    per_case: pd.DataFrame,
    per_organ: pd.DataFrame,
    vis_candidates: List[Tuple[float, str, np.ndarray, np.ndarray, np.ndarray]],
    vis_dir: Path,
    split_name: str,
) -> List[str]:
    if per_case.empty or per_organ.empty or not vis_candidates:
        return []

    case_to_arrays = {cid: (image, gt, pred) for _, cid, image, gt, pred in vis_candidates}
    emitted: List[str] = []

    weak_organ_row = per_organ.sort_values(by=["dice_mean"], ascending=True).head(1)
    strong_organ_row = per_organ.sort_values(by=["dice_mean"], ascending=False).head(1)
    for tag, organ_row in [("weak", weak_organ_row), ("strong", strong_organ_row)]:
        if organ_row.empty:
            continue
        organ_name = str(organ_row.iloc[0]["organ"])
        dice_col = f"dice_organ_{organ_name}"
        if dice_col not in per_case.columns:
            continue

        ranked = per_case[["case_id", dice_col]].copy()
        ranked[dice_col] = pd.to_numeric(ranked[dice_col], errors="coerce")
        ranked = ranked[np.isfinite(ranked[dice_col])]
        if ranked.empty:
            continue

        ranked = ranked.sort_values(by=[dice_col], ascending=(tag == "weak"))
        case_id = str(ranked.iloc[0]["case_id"])
        if case_id not in case_to_arrays:
            continue

        image, gt, pred = case_to_arrays[case_id]
        out_path = vis_dir / f"{split_name}_{tag}_organ_{organ_name}_{case_id}.png"
        save_evaluation_panel(
            image=image,
            gt=gt,
            pred=pred,
            out_path=out_path,
            title=f"{split_name} | {tag} organ: {organ_name} | case: {case_id}",
            score_text=f"OrganDice={float(ranked.iloc[0][dice_col]):.4f}",
        )
        emitted.append(str(out_path))

    return emitted


def _hard_case_table(per_case: pd.DataFrame, failure_k: int, suspicious_cases: Optional[List[str]] = None) -> pd.DataFrame:
    if per_case.empty:
        return pd.DataFrame()

    suspicious = set(s.strip() for s in (suspicious_cases or []) if str(s).strip())
    out = per_case.copy()
    out["dice_rank_hard"] = pd.to_numeric(out.get("dice_mean", np.nan), errors="coerce").rank(method="min", ascending=True)
    out["hd95_rank_hard"] = pd.to_numeric(out.get("hd95_mean", np.nan), errors="coerce").rank(method="min", ascending=False)
    out["hardness_score"] = (
        pd.to_numeric(out["dice_rank_hard"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["hd95_rank_hard"], errors="coerce").fillna(0.0)
    )
    out["is_suspicious_manual"] = out["case_id"].astype(str).apply(lambda x: x in suspicious)

    worst = out.sort_values(by=["hardness_score", "dice_mean"], ascending=[False, True]).head(max(1, int(failure_k)))
    manual = out[out["is_suspicious_manual"] == True]
    merged = pd.concat([worst, manual], ignore_index=True)
    merged = merged.drop_duplicates(subset=["case_id"])
    return merged.sort_values(by=["is_suspicious_manual", "hardness_score"], ascending=[False, False])


def _representative_bad_examples_per_organ(
    failure_df: pd.DataFrame,
    vis_candidates: List[Tuple[float, str, np.ndarray, np.ndarray, np.ndarray]],
    vis_dir: Path,
    split_name: str,
    max_organs: int = 5,
) -> List[str]:
    if failure_df.empty or not vis_candidates:
        return []

    case_to_arrays = {cid: (image, gt, pred) for _, cid, image, gt, pred in vis_candidates}
    emit: List[str] = []

    organ_counts = (
        failure_df["organ"].value_counts().head(max(1, int(max_organs))).index.tolist()
        if "organ" in failure_df.columns
        else []
    )

    for organ in organ_counts:
        sub = failure_df[failure_df["organ"] == organ].copy()
        if sub.empty:
            continue
        sub["_rank"] = pd.to_numeric(sub.get("severity_score", np.nan), errors="coerce").fillna(0.0)
        sub = sub.sort_values(by=["_rank", "dice"], ascending=[False, True])
        row = sub.iloc[0]
        case_id = str(row.get("case_id", ""))
        if case_id not in case_to_arrays:
            continue

        image, gt, pred = case_to_arrays[case_id]
        out_path = vis_dir / f"{split_name}_failure_organ_{_sanitize_name(str(organ))}_{case_id}.png"
        save_evaluation_panel(
            image=image,
            gt=gt,
            pred=pred,
            out_path=out_path,
            title=f"{split_name} | failure organ: {organ} | case: {case_id}",
            score_text=f"Dice={float(row.get('dice', np.nan)):.4f} | HD95={float(row.get('hd95', np.nan)):.2f}",
        )
        emit.append(str(out_path))

    return emit


def _summary_payload(
    per_case: pd.DataFrame,
    per_organ: pd.DataFrame,
    split_name: str,
    csv_path: Path,
    per_organ_csv: Path,
    failure_csv: Path,
    compact_organ_csv: Path,
    size_category_csv: Path,
    hard_cases_csv: Path,
    failure_details_csv: Path,
    failure_patterns_csv: Path,
    failure_patterns_json: Path,
    vis_dir: Optional[Path],
    organ_focus_visuals: List[str],
    representative_failure_visuals: List[str],
    suspicious_cases: Optional[List[str]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "split": split_name,
        "n_cases": int(len(per_case)),
        "per_case_csv": str(csv_path),
        "per_organ_csv": str(per_organ_csv),
        "failure_cases_csv": str(failure_csv),
        "compact_organ_csv": str(compact_organ_csv),
        "size_category_csv": str(size_category_csv),
        "hard_cases_csv": str(hard_cases_csv),
        "failure_details_csv": str(failure_details_csv),
        "failure_patterns_csv": str(failure_patterns_csv),
        "failure_patterns_json": str(failure_patterns_json),
        "visualization_dir": str(vis_dir) if vis_dir is not None else "",
        "organ_focus_visuals": organ_focus_visuals,
        "representative_failure_visuals": representative_failure_visuals,
        "manual_suspicious_cases": suspicious_cases or [],
    }

    for col in CORE_MEAN_COLUMNS:
        if col in per_case.columns:
            payload[col] = _nanmean(per_case[col].values)

    if not per_organ.empty:
        payload["macro_dice"] = _nanmean(per_organ["dice_mean"].values)
        payload["macro_iou"] = _nanmean(per_organ["iou_mean"].values)
        payload["macro_hd95"] = _nanmean(per_organ["hd95_mean"].values)
        payload["macro_surface_dice"] = _nanmean(per_organ["surface_dice_mean"].values)
        payload["macro_precision"] = _nanmean(per_organ["precision_mean"].values)
        payload["macro_recall"] = _nanmean(per_organ["recall_mean"].values)
        payload["macro_volume_error"] = _nanmean(per_organ["volume_error_mean"].values)
        payload["small_or_rare_organs"] = per_organ[per_organ["is_small_or_rare"] == True]["organ"].tolist()
    else:
        payload["macro_dice"] = float("nan")
        payload["macro_iou"] = float("nan")
        payload["macro_hd95"] = float("nan")
        payload["macro_surface_dice"] = float("nan")
        payload["macro_precision"] = float("nan")
        payload["macro_recall"] = float("nan")
        payload["macro_volume_error"] = float("nan")
        payload["small_or_rare_organs"] = []

    return payload


def evaluate_prediction_dir(
    pred_dir: Path,
    gt_dir: Path,
    num_classes: int,
    output_csv: Path,
    label_map: Optional[Dict[int, str]] = None,
    image_dir: Optional[Path] = None,
    probability_dir: Optional[Path] = None,
    vis_dir: Optional[Path] = None,
    max_visual_cases: int = 6,
    failure_cases_k: int = 10,
    split_name: str = "test",
    suspicious_cases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate predicted NIfTI masks and export case/organ summaries and visual checks."""

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    effective_label_map = label_map or {i: f"class_{i}" for i in range(num_classes)}

    rows: List[Dict[str, Any]] = []
    pred_paths = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_paths:
        raise RuntimeError(f"No prediction files found: {pred_dir}")

    organ_presence_counts: Dict[int, int] = {i: 0 for i in range(1, num_classes)}
    organ_voxel_sums: Dict[int, float] = {i: 0.0 for i in range(1, num_classes)}
    total_voxels = 0.0

    vis_candidates: List[Tuple[float, str, np.ndarray, np.ndarray, np.ndarray]] = []
    failure_rows: List[Dict[str, Any]] = []

    for pred_path in tqdm(pred_paths, desc="Evaluating predictions", unit="case"):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue

        pred_img = nib.load(str(pred_path))
        gt_img = nib.load(str(gt_path))
        pred = np.asanyarray(pred_img.dataobj).astype(np.int32)
        gt = np.asanyarray(gt_img.dataobj).astype(np.int32)

        case_id = pred_path.name[:-7] if pred_path.name.endswith(".nii.gz") else pred_path.stem
        probs = _load_probability_tensor(probability_dir, case_id, num_classes=num_classes)

        metrics = case_metrics(
            pred=pred,
            target=gt,
            num_classes=num_classes,
            spacing=list(gt_img.header.get_zooms()[:3]),
            probs=probs,
        )

        failure_rows.extend(
            analyze_case_organ_failures(
                case_id=case_id,
                pred=pred,
                target=gt,
                label_map=effective_label_map,
                num_classes=num_classes,
                metrics=metrics,
                probs=probs,
            )
        )

        row = _build_case_row(case_id=case_id, metrics=metrics, label_map=effective_label_map, num_classes=num_classes)

        rows.append(row)

        gt_vox = float(np.prod(gt.shape))
        total_voxels += gt_vox
        for class_idx in range(1, num_classes):
            gt_count = int(np.sum(gt == class_idx))
            if gt_count > 0:
                organ_presence_counts[class_idx] += 1
            organ_voxel_sums[class_idx] += float(gt_count)

        if image_dir is not None:
            image_path = image_dir / pred_path.name
            if image_path.exists():
                image = np.asanyarray(nib.load(str(image_path)).dataobj)
                if image.ndim > 3:
                    image = image[..., 0]
                if gt.ndim > 3:
                    gt = gt[..., 0]
                if pred.ndim > 3:
                    pred = pred[..., 0]
                vis_candidates.append((float(metrics["dice_mean"]), case_id, image, gt, pred))

    per_case_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    per_case_df.to_csv(output_csv, index=False)

    organ_voxel_fracs = {
        class_idx: (organ_voxel_sums[class_idx] / max(total_voxels, 1.0)) for class_idx in range(1, num_classes)
    }
    per_organ_df = _organ_aggregate(
        per_case=per_case_df,
        label_map=effective_label_map,
        num_classes=num_classes,
        organ_presence_counts=organ_presence_counts,
        organ_voxel_fracs=organ_voxel_fracs,
    )

    per_organ_csv = output_csv.with_name(f"{output_csv.stem}_per_organ.csv")
    per_organ_df.to_csv(per_organ_csv, index=False)

    failure_csv = output_csv.with_name(f"{output_csv.stem}_failure_cases.csv")
    hard_cases_df = _hard_case_table(per_case_df, failure_k=failure_cases_k, suspicious_cases=suspicious_cases)
    hard_cases_df.to_csv(failure_csv, index=False)

    failure_details_csv = output_csv.with_name(f"{output_csv.stem}_failure_details.csv")
    failure_details_df = pd.DataFrame(failure_rows)
    failure_details_df.to_csv(failure_details_csv, index=False)

    failure_pattern_payload = summarize_failure_patterns(failure_details_df)
    failure_patterns_csv = output_csv.with_name(f"{output_csv.stem}_failure_patterns.csv")
    pd.DataFrame(failure_pattern_payload.get("pattern_counts", [])).to_csv(failure_patterns_csv, index=False)
    failure_patterns_json = output_csv.with_name(f"{output_csv.stem}_failure_patterns.json")
    failure_patterns_json.write_text(json.dumps(failure_pattern_payload, indent=2), encoding="utf-8")

    compact_organ_csv = output_csv.with_name(f"{output_csv.stem}_compact_organs.csv")
    compact_df = _compact_organ_report(per_organ_df, top_k=max(3, int(failure_cases_k)))
    compact_df.to_csv(compact_organ_csv, index=False)

    size_category_csv = output_csv.with_name(f"{output_csv.stem}_size_categories.csv")
    size_df = _size_category_report(per_organ_df)
    size_df.to_csv(size_category_csv, index=False)

    saved_vis_dir: Optional[Path] = None
    if vis_dir is not None and vis_candidates:
        saved_vis_dir = vis_dir
        saved_vis_dir.mkdir(parents=True, exist_ok=True)
        sorted_cases = sorted(vis_candidates, key=lambda x: x[0])
        k = max(1, int(max_visual_cases))
        selected = sorted_cases[: min(k // 2 + 1, len(sorted_cases))]
        remaining = [c for c in sorted_cases if c not in selected]
        selected += remaining[-min(k - len(selected), len(remaining)) :]

        for dice_val, case_id, image, gt, pred in selected:
            out_path = saved_vis_dir / f"{split_name}_{case_id}.png"
            save_evaluation_panel(
                image=image,
                gt=gt,
                pred=pred,
                out_path=out_path,
                title=f"{split_name} | {case_id}",
                score_text=f"Dice={dice_val:.4f}",
            )

    organ_focus_paths: List[str] = []
    rep_failure_paths: List[str] = []
    if saved_vis_dir is not None:
        organ_focus_paths = _organ_focus_visuals(
            per_case=per_case_df,
            per_organ=per_organ_df,
            vis_candidates=vis_candidates,
            vis_dir=saved_vis_dir,
            split_name=split_name,
        )
        rep_failure_paths = _representative_bad_examples_per_organ(
            failure_df=failure_details_df,
            vis_candidates=vis_candidates,
            vis_dir=saved_vis_dir,
            split_name=split_name,
            max_organs=max(3, int(failure_cases_k)),
        )

    summary = _summary_payload(
        per_case=per_case_df,
        per_organ=per_organ_df,
        split_name=split_name,
        csv_path=output_csv,
        per_organ_csv=per_organ_csv,
        failure_csv=failure_csv,
        compact_organ_csv=compact_organ_csv,
        size_category_csv=size_category_csv,
        hard_cases_csv=failure_csv,
        failure_details_csv=failure_details_csv,
        failure_patterns_csv=failure_patterns_csv,
        failure_patterns_json=failure_patterns_json,
        vis_dir=saved_vis_dir,
        organ_focus_visuals=organ_focus_paths,
        representative_failure_visuals=rep_failure_paths,
        suspicious_cases=suspicious_cases,
    )

    summary_json = output_csv.with_name(f"{output_csv.stem}_summary.json")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json)
    summary["csv"] = str(output_csv)
    return summary
