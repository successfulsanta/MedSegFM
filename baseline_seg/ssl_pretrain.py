from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from baseline_seg.dataset import build_unlabeled_dataloader
from baseline_seg.pretraining import build_ssl_pretrain_model, info_nce_loss, masked_patch_input


def run_ssl_pretraining(cfg: Dict[str, Any], device: str, exp_dirs: Dict[str, Path], logger) -> Path:
    """Run SSL pretraining on unlabeled CT volumes and save transferable weights."""

    p_cfg = cfg.get("pretraining", {})
    method = str(p_cfg.get("method", "masked_recon")).lower()

    model = build_ssl_pretrain_model(cfg).to(device)
    model.train()

    loader = build_unlabeled_dataloader(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(p_cfg.get("ssl_lr", 1e-4)),
        weight_decay=float(p_cfg.get("ssl_weight_decay", 1e-5)),
    )

    max_epochs = int(p_cfg.get("ssl_epochs", 50))
    use_amp = bool(cfg["train"].get("amp", True)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        losses = []
        pbar = tqdm(loader, desc=f"SSL epoch {epoch}", unit="batch")
        for batch in pbar:
            x = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if method == "masked_recon":
                    x_in = masked_patch_input(
                        x,
                        mask_ratio=float(p_cfg.get("mask_ratio", 0.4)),
                        patch_size=tuple(int(v) for v in p_cfg.get("patch_size", [8, 8, 8])),
                    )
                    out = model(x_in)
                    loss = torch.nn.functional.l1_loss(out, x)
                elif method == "denoise":
                    noise = torch.randn_like(x) * float(cfg["augment"].get("noise_std", 0.02))
                    out = model(x + noise)
                    loss = torch.nn.functional.mse_loss(out, x)
                elif method == "contrastive":
                    noise_std = float(cfg["augment"].get("noise_std", 0.02))
                    x1 = x + torch.randn_like(x) * noise_std
                    x2 = x + torch.randn_like(x) * noise_std
                    f1 = model(x1)
                    f2 = model(x2)
                    z1 = torch.mean(f1, dim=(2, 3, 4))
                    z2 = torch.mean(f2, dim=(2, 3, 4))
                    loss = info_nce_loss(z1, z2, temperature=float(p_cfg.get("temperature", 0.1)))
                else:
                    raise RuntimeError(f"Unsupported SSL method: {method}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lv = float(loss.detach().cpu().item())
            losses.append(lv)
            pbar.set_postfix({"loss": f"{lv:.4f}"})

        mean_loss = float(sum(losses) / max(1, len(losses)))
        rows.append({"epoch": epoch, "ssl_loss": mean_loss})
        logger.info("SSL epoch %d | loss=%.5f", epoch, mean_loss)

    pd.DataFrame(rows).to_csv(exp_dirs["logs_dir"] / "ssl_pretrain_history.csv", index=False)
    ckpt_path = exp_dirs["ckpt_dir"] / "ssl_pretrained.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "ssl_method": method,
            "ssl_epochs": max_epochs,
            "config": cfg,
            "training_time_sec": float(time.perf_counter() - t0),
        },
        ckpt_path,
    )
    logger.info("Saved SSL pretrained weights: %s", ckpt_path)
    return ckpt_path
