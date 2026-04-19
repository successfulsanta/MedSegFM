from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import torch

from baseline_seg.models import build_model_with_out_channels


def _extract_state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state" in payload and isinstance(payload["model_state"], dict):
        return payload["model_state"]
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: Path, logger) -> Dict[str, int]:
    """Load checkpoint weights with shape-safe partial matching."""

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported checkpoint payload in {checkpoint_path}")

    incoming = _extract_state_dict(payload)
    current = model.state_dict()

    matched = {}
    skipped = 0
    for k, v in incoming.items():
        if k in current and current[k].shape == v.shape:
            matched[k] = v
        else:
            skipped += 1

    current.update(matched)
    model.load_state_dict(current, strict=False)

    logger.info(
        "Loaded pretrained weights from %s | matched=%d | skipped=%d",
        checkpoint_path,
        len(matched),
        skipped,
    )
    return {"matched": len(matched), "skipped": skipped}


def freeze_backbone(model: torch.nn.Module, logger) -> Dict[str, int]:
    """Freeze backbone parameters and keep head-like layers trainable via heuristics."""

    for p in model.parameters():
        p.requires_grad = False

    trainable_patterns = ["out", "final", "head", "seg"]
    for name, param in model.named_parameters():
        if any(token in name.lower() for token in trainable_patterns):
            param.requires_grad = True

    total = sum(1 for _ in model.parameters())
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info("Freeze backbone enabled | trainable params groups=%d/%d", trainable, total)
    return {"total": total, "trainable": trainable}


def apply_pretraining_mode(model: torch.nn.Module, cfg: Dict[str, Any], logger) -> None:
    """Apply scratch/supervised/ssl initialization and optional freezing."""

    p_cfg = cfg.get("pretraining", {})
    mode = str(p_cfg.get("mode", "scratch")).lower()

    if mode == "scratch":
        logger.info("Pretraining mode: scratch (random initialization)")
    elif mode in {"supervised", "ssl"}:
        ckpt_path = p_cfg.get("pretrained_path")
        if not ckpt_path:
            raise RuntimeError(f"Pretraining mode '{mode}' requires pretraining.pretrained_path")
        load_pretrained_weights(model, Path(ckpt_path).resolve(), logger=logger)
        logger.info("Pretraining mode: %s", mode)
    else:
        raise RuntimeError(f"Unsupported pretraining mode: {mode}")

    if bool(p_cfg.get("freeze_backbone", False)):
        freeze_backbone(model, logger=logger)


def build_ssl_pretrain_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Build SSL model that mirrors segmentation backbone with a 1-channel reconstruction head."""

    method = str(cfg.get("pretraining", {}).get("method", "masked_recon")).lower()
    if method in {"masked_recon", "denoise"}:
        return build_model_with_out_channels(cfg, out_channels=1)

    if method == "contrastive":
        base = build_model_with_out_channels(cfg, out_channels=64)
        return base

    raise RuntimeError(f"Unsupported SSL method: {method}")


def masked_patch_input(x: torch.Tensor, mask_ratio: float, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Apply MAE-style random patch masking on 3D volumes."""

    if mask_ratio <= 0:
        return x

    out = x.clone()
    b, c, d, h, w = out.shape
    pd, ph, pw = patch_size

    nd = max(1, d // pd)
    nh = max(1, h // ph)
    nw = max(1, w // pw)
    total = nd * nh * nw
    nmask = max(1, int(total * mask_ratio))

    for bi in range(b):
        perm = torch.randperm(total, device=out.device)[:nmask]
        for idx in perm.tolist():
            z = idx // (nh * nw)
            rem = idx % (nh * nw)
            y = rem // nw
            xk = rem % nw
            z0, y0, x0 = z * pd, y * ph, xk * pw
            out[bi, :, z0 : min(z0 + pd, d), y0 : min(y0 + ph, h), x0 : min(x0 + pw, w)] = 0.0

    return out


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Simple symmetric InfoNCE loss for paired 3D embeddings."""

    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    logits = (z1 @ z2.t()) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss1 = torch.nn.functional.cross_entropy(logits, labels)
    loss2 = torch.nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss1 + loss2)
