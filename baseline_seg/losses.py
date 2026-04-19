from __future__ import annotations

from typing import Any, Dict


def build_loss(cfg: Dict[str, Any]):
    """Build Dice + Cross Entropy combined segmentation loss."""

    try:
        from monai.losses import DiceCELoss
    except Exception as exc:
        raise RuntimeError("MONAI is required for loss functions. Install monai.") from exc

    loss_cfg = cfg["loss"]
    return DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=float(loss_cfg.get("dice_weight", 1.0)),
        lambda_ce=float(loss_cfg.get("ce_weight", 1.0)),
    )
