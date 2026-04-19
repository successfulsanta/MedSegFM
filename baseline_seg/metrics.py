from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


def dice_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int) -> List[float]:
    """Compute class-wise Dice score from integer masks."""

    scores: List[float] = []
    eps = 1e-6
    for c in range(1, num_classes):
        pred_c = pred == c
        tgt_c = target == c
        denom = 2 * np.sum(pred_c & tgt_c) + np.sum(pred_c ^ tgt_c)
        if np.sum(pred_c) == 0 and np.sum(tgt_c) == 0:
            scores.append(float("nan"))
        else:
            inter = np.sum(pred_c & tgt_c)
            scores.append(float((2 * inter + eps) / (np.sum(pred_c) + np.sum(tgt_c) + eps)))
    return scores


def iou_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int) -> List[float]:
    """Compute class-wise IoU/Jaccard from integer masks."""

    scores: List[float] = []
    eps = 1e-6
    for c in range(1, num_classes):
        pred_c = pred == c
        tgt_c = target == c
        union = np.sum(pred_c | tgt_c)
        if union == 0:
            scores.append(float("nan"))
        else:
            inter = np.sum(pred_c & tgt_c)
            scores.append(float((inter + eps) / (union + eps)))
    return scores


def precision_recall_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, List[float]]:
    """Compute class-wise precision and recall from integer masks."""

    eps = 1e-6
    precision: List[float] = []
    recall: List[float] = []

    for c in range(1, num_classes):
        pred_c = pred == c
        tgt_c = target == c
        tp = float(np.sum(pred_c & tgt_c))
        fp = float(np.sum(pred_c & (~tgt_c)))
        fn = float(np.sum((~pred_c) & tgt_c))

        if np.sum(pred_c) == 0 and np.sum(tgt_c) == 0:
            precision.append(float("nan"))
            recall.append(float("nan"))
            continue

        precision.append(float((tp + eps) / (tp + fp + eps)))
        recall.append(float((tp + eps) / (tp + fn + eps)))

    return {"precision_per_class": precision, "recall_per_class": recall}


def volume_error_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int) -> List[float]:
    """Compute relative volume error per class: (pred - gt) / gt."""

    errs: List[float] = []
    for c in range(1, num_classes):
        pred_vol = float(np.sum(pred == c))
        tgt_vol = float(np.sum(target == c))
        if tgt_vol <= 0 and pred_vol <= 0:
            errs.append(float("nan"))
        elif tgt_vol <= 0:
            errs.append(float("nan"))
        else:
            errs.append(float((pred_vol - tgt_vol) / tgt_vol))
    return errs


def surface_dice_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    spacing: Optional[List[float]] = None,
    tolerance_mm: float = 1.0,
) -> List[float]:
    """Compute class-wise surface Dice using MONAI when available."""

    try:
        import torch
        from monai.metrics import compute_surface_dice
    except Exception:
        return [float("nan")] * (num_classes - 1)

    spacing_arr = spacing if spacing is not None else [1.0, 1.0, 1.0]
    vals: List[float] = []

    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.uint8)
        tgt_c = (target == c).astype(np.uint8)

        if np.sum(pred_c) == 0 and np.sum(tgt_c) == 0:
            vals.append(float("nan"))
            continue

        pred_t = torch.from_numpy(pred_c[None, None]).float()
        tgt_t = torch.from_numpy(tgt_c[None, None]).float()
        try:
            out = compute_surface_dice(
                y_pred=pred_t,
                y=tgt_t,
                class_thresholds=[float(tolerance_mm)],
                include_background=True,
                spacing=spacing_arr,
            )
            vals.append(float(out.cpu().numpy().reshape(-1)[0]))
        except Exception:
            vals.append(float("nan"))

    return vals


def hd95_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    spacing: Optional[List[float]] = None,
) -> List[float]:
    """Compute class-wise HD95 using MONAI when available."""

    try:
        import torch
        from monai.metrics import compute_hausdorff_distance
    except Exception:
        return [float("nan")] * (num_classes - 1)

    spacing_arr = spacing if spacing is not None else [1.0, 1.0, 1.0]
    hd_vals: List[float] = []
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.uint8)
        tgt_c = (target == c).astype(np.uint8)

        if np.sum(pred_c) == 0 and np.sum(tgt_c) == 0:
            hd_vals.append(float("nan"))
            continue

        pred_t = torch.from_numpy(pred_c[None, None]).float()
        tgt_t = torch.from_numpy(tgt_c[None, None]).float()
        try:
            val = compute_hausdorff_distance(
                y_pred=pred_t,
                y=tgt_t,
                include_background=True,
                percentile=95.0,
                spacing=spacing_arr,
            )
            hd_vals.append(float(val.cpu().numpy().reshape(-1)[0]))
        except Exception:
            hd_vals.append(float("nan"))

    return hd_vals


def summarize_scores(class_scores: List[float]) -> float:
    """Return nan-safe mean score across classes."""

    arr = np.asarray(class_scores, dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def calibration_from_probabilities(
    probs: np.ndarray,
    target: np.ndarray,
    num_bins: int = 10,
    max_samples: int = 250000,
) -> Dict[str, float]:
    """Compute confidence, ECE, and uncertainty statistics from class probabilities."""

    if probs.ndim != 4:
        return {
            "ece": float("nan"),
            "mean_confidence": float("nan"),
            "mean_entropy": float("nan"),
            "mean_variation_ratio": float("nan"),
        }

    conf = np.max(probs, axis=0).reshape(-1)
    pred = np.argmax(probs, axis=0).reshape(-1)
    tgt = target.reshape(-1)

    valid = (tgt >= 0) & (tgt < probs.shape[0])
    if not np.any(valid):
        return {
            "ece": float("nan"),
            "mean_confidence": float("nan"),
            "mean_entropy": float("nan"),
            "mean_variation_ratio": float("nan"),
        }

    conf = conf[valid]
    pred = pred[valid]
    tgt = tgt[valid]

    n = conf.shape[0]
    if n > max_samples:
        step = int(math.ceil(n / float(max_samples)))
        conf = conf[::step]
        pred = pred[::step]
        tgt = tgt[::step]

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    correct = (pred == tgt).astype(np.float32)

    ece = 0.0
    for i in range(num_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == num_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(correct[mask]))
        c = float(np.mean(conf[mask]))
        w = float(np.mean(mask.astype(np.float32)))
        ece += abs(acc - c) * w

    safe_probs = np.clip(probs.astype(np.float32), 1e-8, 1.0)
    entropy = -np.sum(safe_probs * np.log(safe_probs), axis=0)
    var_ratio = 1.0 - np.max(safe_probs, axis=0)

    return {
        "ece": float(ece),
        "mean_confidence": float(np.mean(conf)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_variation_ratio": float(np.mean(var_ratio)),
    }


def case_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    spacing: Optional[List[float]] = None,
    probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute all per-case metrics used in reports."""

    dice = dice_per_class(pred, target, num_classes=num_classes)
    iou = iou_per_class(pred, target, num_classes=num_classes)
    hd95 = hd95_per_class(pred, target, num_classes=num_classes, spacing=spacing)
    surface_dice = surface_dice_per_class(pred, target, num_classes=num_classes, spacing=spacing)
    pr = precision_recall_per_class(pred, target, num_classes=num_classes)
    volume_err = volume_error_per_class(pred, target, num_classes=num_classes)

    calibration = {
        "ece": float("nan"),
        "mean_confidence": float("nan"),
        "mean_entropy": float("nan"),
        "mean_variation_ratio": float("nan"),
    }
    if probs is not None:
        calibration = calibration_from_probabilities(probs=probs, target=target)

    return {
        "dice_per_class": dice,
        "dice_mean": summarize_scores(dice),
        "iou_per_class": iou,
        "iou_mean": summarize_scores(iou),
        "hd95_per_class": hd95,
        "hd95_mean": summarize_scores(hd95),
        "surface_dice_per_class": surface_dice,
        "surface_dice_mean": summarize_scores(surface_dice),
        "precision_per_class": pr["precision_per_class"],
        "precision_mean": summarize_scores(pr["precision_per_class"]),
        "recall_per_class": pr["recall_per_class"],
        "recall_mean": summarize_scores(pr["recall_per_class"]),
        "volume_error_per_class": volume_err,
        "volume_error_mean": summarize_scores(volume_err),
        "ece": calibration["ece"],
        "mean_confidence": calibration["mean_confidence"],
        "mean_entropy": calibration["mean_entropy"],
        "mean_variation_ratio": calibration["mean_variation_ratio"],
    }
