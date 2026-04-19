from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from baseline_seg.pretraining import load_pretrained_weights


@dataclass
class AdaptationStats:
    mode: str
    total_params: int
    trainable_params: int


class LoRALinear(nn.Module):
    """LoRA wrapper for Linear layers: y = base(x) + scale * B(A(x))."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        rank = max(1, int(rank))
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_b(self.lora_a(x))


class LoRAConv3d(nn.Module):
    """LoRA-style low-rank residual branch for Conv3d layers."""

    def __init__(self, base: nn.Conv3d, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        rank = max(1, int(rank))
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Conv3d(base.in_channels, rank, kernel_size=1, bias=False)
        self.lora_b = nn.Conv3d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_b(self.lora_a(x))


class ConvAdapter(nn.Module):
    """Residual bottleneck adapter for Conv3d features."""

    def __init__(self, base: nn.Conv3d, bottleneck: int = 8) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        b = max(1, int(bottleneck))
        self.down = nn.Conv3d(base.out_channels, b, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.up = nn.Conv3d(b, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.up(self.act(self.down(y)))


class LinearAdapter(nn.Module):
    """Residual bottleneck adapter for Linear layers."""

    def __init__(self, base: nn.Linear, bottleneck: int = 8) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        b = max(1, int(bottleneck))
        self.down = nn.Linear(base.out_features, b, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(b, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.up(self.act(self.down(y)))


def _replace_modules_with_lora(module: nn.Module, rank: int, alpha: float, use_conv: bool) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced += 1
            continue
        if use_conv and isinstance(child, nn.Conv3d):
            setattr(module, name, LoRAConv3d(child, rank=rank, alpha=alpha))
            replaced += 1
            continue
        replaced += _replace_modules_with_lora(child, rank=rank, alpha=alpha, use_conv=use_conv)
    return replaced


def _replace_modules_with_adapters(module: nn.Module, size: int, use_linear: bool) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d):
            setattr(module, name, ConvAdapter(child, bottleneck=size))
            replaced += 1
            continue
        if use_linear and isinstance(child, nn.Linear):
            setattr(module, name, LinearAdapter(child, bottleneck=size))
            replaced += 1
            continue
        replaced += _replace_modules_with_adapters(child, size=size, use_linear=use_linear)
    return replaced


def _set_requires_grad_all(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def _enable_decoder_or_head(model: nn.Module) -> int:
    tokens = ["up", "decoder", "out", "final", "head", "seg"]
    enabled = 0
    for name, p in model.named_parameters():
        if any(t in name.lower() for t in tokens):
            p.requires_grad = True
            enabled += 1
    return enabled


def _param_counts(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def apply_adaptation_mode(model: nn.Module, cfg: Dict[str, Any], logger) -> AdaptationStats:
    """Apply adaptation strategy: full_ft, lora, adapters with freeze options."""

    a_cfg = cfg.get("adaptation", {})
    mode = str(a_cfg.get("mode", "full_ft")).lower()

    pretrained_path = a_cfg.get("pretrained_path") or cfg.get("pretraining", {}).get("pretrained_path")
    if pretrained_path:
        load_pretrained_weights(model, Path(pretrained_path).resolve(), logger=logger)

    if mode == "full_ft":
        _set_requires_grad_all(model, True)
    elif mode == "partial_decoder":
        _set_requires_grad_all(model, False)
        enabled = _enable_decoder_or_head(model)
        logger.info("Partial freeze mode: enabled decoder/head parameter groups=%d", enabled)
    elif mode == "lora":
        _set_requires_grad_all(model, False)
        replaced = _replace_modules_with_lora(
            model,
            rank=int(a_cfg.get("lora_rank", 4)),
            alpha=float(a_cfg.get("lora_alpha", 1.0)),
            use_conv=bool(a_cfg.get("lora_on_conv", True)),
        )
        logger.info("LoRA mode: replaced modules=%d", replaced)
        if bool(a_cfg.get("train_head", True)):
            _enable_decoder_or_head(model)
    elif mode == "adapters":
        _set_requires_grad_all(model, False)
        replaced = _replace_modules_with_adapters(
            model,
            size=int(a_cfg.get("adapter_size", 8)),
            use_linear=bool(a_cfg.get("adapter_on_linear", True)),
        )
        logger.info("Adapter mode: replaced modules=%d", replaced)
        if bool(a_cfg.get("train_head", True)):
            _enable_decoder_or_head(model)
    else:
        raise RuntimeError("adaptation.mode must be one of: full_ft, lora, adapters, partial_decoder")

    if bool(a_cfg.get("freeze_backbone", False)):
        _set_requires_grad_all(model, False)
        _enable_decoder_or_head(model)

    total, trainable = _param_counts(model)
    logger.info("Adaptation mode=%s | trainable params=%d / total=%d", mode, trainable, total)
    return AdaptationStats(mode=mode, total_params=total, trainable_params=trainable)
