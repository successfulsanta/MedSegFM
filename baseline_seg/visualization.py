from __future__ import annotations

from pathlib import Path

import numpy as np


def _pick_slice(label_or_pred: np.ndarray) -> int:
    """Select an axial slice with maximal foreground area when possible."""

    if label_or_pred.ndim != 3:
        return 0
    sums = np.sum(label_or_pred > 0, axis=(0, 1))
    if np.max(sums) <= 0:
        return label_or_pred.shape[2] // 2
    return int(np.argmax(sums))


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim <= 3:
        return arr
    return arr[..., 0]


def save_evaluation_panel(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    title: str,
    score_text: str = "",
) -> None:
    """Save a 4-panel qualitative figure: image, GT, prediction, and error overlay."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for visualization.") from exc

    image = _ensure_3d(image)
    gt = _ensure_3d(gt)
    pred = _ensure_3d(pred)

    z = _pick_slice(gt if np.any(gt > 0) else pred)
    img_s = image[:, :, z].T
    gt_s = gt[:, :, z].T
    pred_s = pred[:, :, z].T

    tp = (gt_s > 0) & (pred_s > 0)
    fp = (gt_s == 0) & (pred_s > 0)
    fn = (gt_s > 0) & (pred_s == 0)

    overlay = np.zeros((*gt_s.shape, 3), dtype=np.float32)
    overlay[..., 1] = tp.astype(np.float32)  # green
    overlay[..., 2] = fp.astype(np.float32)  # blue
    overlay[..., 0] = fn.astype(np.float32)  # red

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    subtitle = f"{title} | {score_text}" if score_text else title
    fig.suptitle(subtitle)

    axes[0].imshow(img_s, cmap="gray", origin="lower")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(img_s, cmap="gray", origin="lower")
    axes[1].imshow(np.ma.masked_where(gt_s == 0, gt_s), cmap="tab20", alpha=0.45, origin="lower")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(img_s, cmap="gray", origin="lower")
    axes[2].imshow(np.ma.masked_where(pred_s == 0, pred_s), cmap="tab20", alpha=0.45, origin="lower")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(img_s, cmap="gray", origin="lower")
    axes[3].imshow(overlay, alpha=0.55, origin="lower")
    axes[3].set_title("Overlay (R=FN, G=TP, B=FP)")
    axes[3].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_triptych(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: Path, title: str) -> None:
    """Save side-by-side image, ground truth, and prediction overlays."""
    save_evaluation_panel(image=image, gt=gt, pred=pred, out_path=out_path, title=title)
