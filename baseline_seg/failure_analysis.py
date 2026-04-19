from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import ndimage


def _component_count(mask: np.ndarray) -> int:
    if int(np.sum(mask)) == 0:
        return 0
    _, ncomp = ndimage.label(mask.astype(np.uint8))
    return int(ncomp)


def _slice_inconsistency(mask: np.ndarray) -> float:
    if mask.ndim != 3:
        return float("nan")
    areas = np.sum(mask, axis=(0, 1)).astype(np.float32)
    if np.sum(areas) <= 0:
        return float("nan")
    diffs = np.abs(np.diff(areas))
    return float(np.mean(diffs) / (np.mean(areas) + 1e-6))


def _dominant_confused_class(pred_c: np.ndarray, target: np.ndarray) -> Optional[int]:
    wrong = pred_c & (target > 0) & (target != 0)
    if int(np.sum(wrong)) == 0:
        return None
    vals, counts = np.unique(target[wrong], return_counts=True)
    if vals.size == 0:
        return None
    return int(vals[int(np.argmax(counts))])


def analyze_case_organ_failures(
    case_id: str,
    pred: np.ndarray,
    target: np.ndarray,
    label_map: Dict[int, str],
    num_classes: int,
    metrics: Dict[str, Any],
    probs: Optional[np.ndarray] = None,
    hd95_high_threshold: float = 20.0,
    dice_low_threshold: float = 0.5,
    overseg_ratio_threshold: float = 1.5,
) -> List[Dict[str, Any]]:
    """Analyze organ-level failure categories for a single case."""

    conf_map = None
    if probs is not None and probs.ndim == 4:
        conf_map = np.max(probs, axis=0)

    out: List[Dict[str, Any]] = []
    for class_idx in range(1, num_classes):
        organ_name = label_map.get(class_idx, f"class_{class_idx}")
        pred_c = pred == class_idx
        tgt_c = target == class_idx

        gt_vox = int(np.sum(tgt_c))
        pred_vox = int(np.sum(pred_c))
        tp = int(np.sum(pred_c & tgt_c))
        fp = int(np.sum(pred_c & (~tgt_c)))
        fn = int(np.sum((~pred_c) & tgt_c))

        if gt_vox == 0 and pred_vox == 0:
            continue

        idx = class_idx - 1
        dice = float(metrics["dice_per_class"][idx])
        iou = float(metrics["iou_per_class"][idx])
        hd95 = float(metrics["hd95_per_class"][idx])
        precision = float(metrics["precision_per_class"][idx])
        recall = float(metrics["recall_per_class"][idx])

        failure_types: List[str] = []

        if gt_vox > 0 and tp == 0:
            failure_types.append("missed_organ")

        if pred_vox > 0 and gt_vox > 0:
            pred_gt_ratio = pred_vox / float(gt_vox + 1e-6)
            fp_ratio = fp / float(pred_vox + 1e-6)
            if pred_gt_ratio >= overseg_ratio_threshold and fp_ratio >= 0.5:
                failure_types.append("over_segmentation")

        if np.isfinite(hd95) and hd95 >= hd95_high_threshold and np.isfinite(dice) and dice < 0.8:
            failure_types.append("boundary_error")

        ncomp = _component_count(pred_c)
        if pred_vox > 0 and ncomp >= 3:
            failure_types.append("disconnected_masks")

        inconsistency = _slice_inconsistency(pred_c.astype(np.uint8))
        if np.isfinite(inconsistency) and inconsistency >= 0.8:
            failure_types.append("slice_inconsistency")

        confused_with = None
        dom = _dominant_confused_class(pred_c, target)
        if dom is not None and dom != class_idx:
            confusion_vox = int(np.sum(pred_c & (target == dom)))
            wrong_vox = int(np.sum(pred_c & (target != class_idx)))
            if wrong_vox > 0 and (confusion_vox / float(wrong_vox)) >= 0.3:
                confusion_name = label_map.get(dom, f"class_{dom}")
                confused_with = confusion_name
                failure_types.append("wrong_organ_confusion")

        wrong_region = (pred != target) & (pred_c | tgt_c)
        wrong_count = int(np.sum(wrong_region))
        wrong_confidence_mean = float("nan")
        overconfident_wrong = False
        if conf_map is not None and wrong_count > 0:
            wrong_confidence_mean = float(np.mean(conf_map[wrong_region]))
            if wrong_confidence_mean >= 0.8:
                overconfident_wrong = True
                failure_types.append("overconfident_error")

        if not failure_types and np.isfinite(dice) and dice < dice_low_threshold:
            failure_types.append("low_dice_unspecified")

        severity = 0.0
        if np.isfinite(dice):
            severity += max(0.0, 1.0 - dice)
        if np.isfinite(hd95):
            severity += min(1.0, hd95 / 50.0)
        severity += 0.2 * max(0, len(failure_types) - 1)

        out.append(
            {
                "case_id": case_id,
                "class_id": class_idx,
                "organ": organ_name,
                "dice": dice,
                "iou": iou,
                "hd95": hd95,
                "precision": precision,
                "recall": recall,
                "gt_voxels": gt_vox,
                "pred_voxels": pred_vox,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "num_components": ncomp,
                "slice_inconsistency": inconsistency,
                "wrong_voxels": wrong_count,
                "wrong_confidence_mean": wrong_confidence_mean,
                "overconfident_wrong": bool(overconfident_wrong),
                "confused_with": confused_with or "",
                "failure_types": ";".join(sorted(set(failure_types))),
                "severity_score": float(severity),
            }
        )

    return out


def summarize_failure_patterns(failure_df) -> Dict[str, Any]:
    """Aggregate failure pattern counts and ranked distributions."""

    if failure_df is None or len(failure_df) == 0:
        return {
            "pattern_counts": [],
            "organ_pattern_counts": [],
            "most_common_patterns": [],
            "most_affected_organs": [],
        }

    pattern_counter: Counter = Counter()
    organ_pattern_counter: Counter = Counter()
    organ_counter: Counter = Counter()

    for _, row in failure_df.iterrows():
        organ = str(row.get("organ", "unknown"))
        types = [t for t in str(row.get("failure_types", "")).split(";") if t]
        if not types:
            continue
        organ_counter[organ] += 1
        for t in types:
            pattern_counter[t] += 1
            organ_pattern_counter[(organ, t)] += 1

    pattern_counts = [
        {"failure_type": k, "count": int(v)}
        for k, v in pattern_counter.most_common()
    ]
    organ_pattern_counts = [
        {"organ": org, "failure_type": typ, "count": int(cnt)}
        for (org, typ), cnt in sorted(organ_pattern_counter.items(), key=lambda x: x[1], reverse=True)
    ]
    most_affected_organs = [
        {"organ": k, "count": int(v)}
        for k, v in organ_counter.most_common()
    ]

    return {
        "pattern_counts": pattern_counts,
        "organ_pattern_counts": organ_pattern_counts,
        "most_common_patterns": pattern_counts[:10],
        "most_affected_organs": most_affected_organs[:10],
    }
