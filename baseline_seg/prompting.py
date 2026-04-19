from __future__ import annotations

import hashlib
from typing import Dict, Optional

import torch


def get_prompt_mode(cfg: Dict) -> str:
    return str(cfg.get("prompting", {}).get("mode", "none")).lower()


def get_prompt_channel_count(cfg: Dict) -> int:
    """Return number of extra channels injected as prompts."""

    p_cfg = cfg.get("prompting", {})
    mode = get_prompt_mode(cfg)
    if mode == "none":
        return 0
    if mode == "spatial":
        return 2  # point map + bbox map
    if mode == "text":
        return int(p_cfg.get("text_embedding_dim", 8))
    raise RuntimeError(f"Unsupported prompt mode: {mode}")


def _text_embedding(name: str, dim: int, device: torch.device) -> torch.Tensor:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    vals = []
    for i in range(dim):
        vals.append((digest[i % len(digest)] / 255.0) * 2.0 - 1.0)
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _sample_present_organs(label_3d: torch.Tensor, max_organs: int) -> torch.Tensor:
    labels = torch.unique(label_3d)
    labels = labels[labels > 0]
    if labels.numel() == 0:
        return labels
    if labels.numel() <= max_organs:
        return labels
    perm = torch.randperm(labels.numel(), device=labels.device)
    return labels[perm[:max_organs]]


def _build_spatial_prompts(
    label: torch.Tensor,
    num_points: int,
    use_boxes: bool,
    radius: int,
    level: str,
) -> torch.Tensor:
    b, _, d, h, w = label.shape
    out = torch.zeros((b, 2, d, h, w), dtype=torch.float32, device=label.device)

    eff_points = max(1, int(num_points))
    if level == "minimal":
        eff_points = 1
    elif level == "none":
        eff_points = 0

    for bi in range(b):
        lab = label[bi, 0]
        fg = torch.nonzero(lab > 0, as_tuple=False)
        if fg.numel() == 0:
            continue

        if eff_points > 0:
            pick = torch.randperm(fg.shape[0], device=label.device)[: min(eff_points, fg.shape[0])]
            for idx in pick:
                z, y, x = fg[idx]
                z0, z1 = max(0, int(z) - radius), min(d, int(z) + radius + 1)
                y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
                x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
                out[bi, 0, z0:z1, y0:y1, x0:x1] = 1.0

        if use_boxes and level != "none":
            minc = torch.amin(fg, dim=0)
            maxc = torch.amax(fg, dim=0)
            z0, y0, x0 = [int(v) for v in minc.tolist()]
            z1, y1, x1 = [int(v) + 1 for v in maxc.tolist()]
            out[bi, 1, z0:z1, y0:y1, x0:x1] = 1.0

    return out


def _build_text_prompts(
    image: torch.Tensor,
    label: Optional[torch.Tensor],
    class_names: Dict[int, str],
    text_dim: int,
    max_organs: int,
    level: str,
) -> torch.Tensor:
    b, _, d, h, w = image.shape
    out = torch.zeros((b, text_dim, d, h, w), dtype=torch.float32, device=image.device)

    if level == "none":
        return out

    for bi in range(b):
        if label is None:
            organs = torch.tensor([], device=image.device)
        else:
            organs = _sample_present_organs(label[bi, 0], max_organs=max_organs)

        if organs.numel() == 0:
            organs = torch.tensor([1], device=image.device)

        if level == "minimal" and organs.numel() > 1:
            organs = organs[:1]

        embeds = []
        for oid in organs.tolist():
            name = class_names.get(int(oid), f"organ_{int(oid)}")
            embeds.append(_text_embedding(name, dim=text_dim, device=image.device))

        vec = torch.stack(embeds, dim=0).mean(dim=0)
        out[bi] = vec.view(text_dim, 1, 1, 1).expand(text_dim, d, h, w)

    return out


def build_prompt_tensor(
    cfg: Dict,
    image: torch.Tensor,
    label: Optional[torch.Tensor],
    class_names: Dict[int, str],
    stage: str = "train",
) -> Optional[torch.Tensor]:
    """Create prompt channels for train/val/infer under selected prompt mode."""

    mode = get_prompt_mode(cfg)
    if mode == "none":
        return None

    p_cfg = cfg.get("prompting", {})

    if stage == "train":
        level = str(p_cfg.get("train_prompt_level", "full")).lower()
    elif stage == "val":
        level = str(p_cfg.get("val_prompt_level", "minimal")).lower()
    else:
        level = str(p_cfg.get("inference_prompt_level", "minimal")).lower()

    if mode == "spatial":
        return _build_spatial_prompts(
            label=image.new_zeros((image.shape[0], 1, *image.shape[2:])) if label is None else label,
            num_points=int(p_cfg.get("num_points", 3)),
            use_boxes=bool(p_cfg.get("use_bboxes", True)),
            radius=int(p_cfg.get("point_radius", 1)),
            level=level,
        )

    if mode == "text":
        return _build_text_prompts(
            image=image,
            label=label,
            class_names=class_names,
            text_dim=int(p_cfg.get("text_embedding_dim", 8)),
            max_organs=int(p_cfg.get("max_text_organs", 4)),
            level=level,
        )

    raise RuntimeError(f"Unsupported prompting mode: {mode}")


def apply_prompt_to_image(image: torch.Tensor, prompt: Optional[torch.Tensor]) -> torch.Tensor:
    """Concatenate prompt channels with image channels when prompts are enabled."""

    if prompt is None:
        return image
    return torch.cat([image, prompt], dim=1)
