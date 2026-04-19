from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import nullcontext
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from baseline_seg.dataset import build_dataloader
from baseline_seg.inference import remove_small_components, sliding_window_predict
from baseline_seg.losses import build_loss
from baseline_seg.metrics import case_metrics
from baseline_seg.models import build_model
from baseline_seg.adaptation import apply_adaptation_mode
from baseline_seg.pretraining import apply_pretraining_mode
from baseline_seg.prompting import apply_prompt_to_image, build_prompt_tensor
from baseline_seg.visualization import save_triptych


class Trainer:
    """Baseline trainer with validation, AMP, early stopping, and checkpointing."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        class_names: Dict[int, str],
        device: str,
        exp_dirs: Dict[str, Path],
        logger,
        resume_checkpoint: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.class_names = class_names
        self.device = device
        self.exp_dirs = exp_dirs
        self.logger = logger

        self.model = build_model(cfg).to(self.device)
        apply_pretraining_mode(self.model, cfg=cfg, logger=logger)
        self.adaptation_stats = apply_adaptation_mode(self.model, cfg=cfg, logger=logger)
        self.loss_fn = build_loss(cfg)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.use_amp = bool(cfg["train"].get("amp", True)) and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.train_loader = build_dataloader(cfg, split="train", require_labels=True)
        self.val_loader = build_dataloader(cfg, split="val", require_labels=True)

        self.best_dice = -1.0
        self.best_epoch = -1
        self.bad_epochs = 0
        self.start_epoch = 1
        self.training_started_at = 0.0

        self.tb_writer = None
        if bool(cfg["logging"].get("tensorboard", False)):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(self.exp_dirs["logs_dir"] / "tb"))
            except Exception as exc:
                self.logger.warning("TensorBoard disabled: %s", exc)

        if resume_checkpoint is not None:
            self._load_checkpoint(resume_checkpoint)

    def _build_optimizer(self):
        opt_cfg = self.cfg.get("optimizer", {})
        name = str(opt_cfg.get("name", "adamw")).lower()
        lr = float(opt_cfg.get("lr", self.cfg["train"].get("lr", 1e-4)))
        weight_decay = float(opt_cfg.get("weight_decay", self.cfg["train"].get("weight_decay", 1e-5)))

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found after adaptation setup.")

        if name == "adamw":
            return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            momentum = float(opt_cfg.get("momentum", 0.9))
            return torch.optim.SGD(
                trainable_params,
                lr=lr,
                momentum=momentum,
                nesterov=bool(opt_cfg.get("nesterov", True)),
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {name}")

    def _build_scheduler(self):
        sch_cfg = self.cfg.get("scheduler", {})
        name = str(sch_cfg.get("name", "cosine")).lower()

        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(sch_cfg.get("t_max", self.cfg["train"].get("max_epochs", 200))),
                eta_min=float(sch_cfg.get("eta_min", 1e-6)),
            )

        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(sch_cfg.get("step_size", 50)),
                gamma=float(sch_cfg.get("gamma", 0.1)),
            )

        if name == "none":
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)

        raise ValueError(f"Unsupported scheduler: {name}")

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt and self.use_amp:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_dice = float(ckpt.get("best_dice", -1.0))
        self.best_epoch = int(ckpt.get("best_epoch", -1))
        self.bad_epochs = int(ckpt.get("bad_epochs", 0))
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.logger.info("Resumed from checkpoint %s at epoch %d", checkpoint_path, self.start_epoch)

    def _save_checkpoint(self, epoch: int, is_best: bool) -> Path:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_dice": self.best_dice,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "config": self.cfg,
        }
        path = self.exp_dirs["ckpt_dir"] / ("best.pt" if is_best else "latest.pt")
        torch.save(ckpt, path)
        return path

    def _run_val_epoch(self, epoch: int, save_visuals: bool = True) -> Dict[str, Any]:
        self.model.eval()

        roi = tuple(int(x) for x in self.cfg["model"]["roi_size"])
        sw_batch = int(self.cfg["infer"].get("sw_batch_size", 1))
        overlap = float(self.cfg["infer"].get("overlap", 0.5))
        min_size = int(self.cfg["infer"].get("min_component_size", 0))
        apply_postprocess = bool(self.cfg["infer"].get("apply_postprocess", True))

        case_rows: List[Dict[str, Any]] = []
        save_count = 0

        for batch in tqdm(self.val_loader, desc=f"Val epoch {epoch}", unit="case"):
            image = batch["image"].to(self.device)
            label = batch["label"].to(self.device)
            case_id = str(batch["case_id"][0])

            prompt = build_prompt_tensor(
                cfg=self.cfg,
                image=image,
                label=label,
                class_names=self.class_names,
                stage="val",
            )
            model_in = apply_prompt_to_image(image, prompt)

            with torch.no_grad():
                amp_ctx = torch.amp.autocast("cuda", enabled=True) if self.use_amp else nullcontext()
                with amp_ctx:
                    logits = sliding_window_predict(
                        self.model,
                        model_in,
                        roi_size=roi,
                        sw_batch_size=sw_batch,
                        overlap=overlap,
                    )

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
            gt = label.squeeze(0).squeeze(0).cpu().numpy().astype(np.int32)

            if apply_postprocess:
                pred = remove_small_components(pred.astype(np.int32), min_size=min_size)

            try:
                spacing = list(batch["image_meta_dict"]["pixdim"][0][1:4].cpu().numpy())
            except Exception:
                spacing = [1.0, 1.0, 1.0]

            metrics = case_metrics(
                pred=pred,
                target=gt,
                num_classes=int(self.cfg["model"]["num_classes"]),
                spacing=spacing,
            )

            row = {
                "epoch": epoch,
                "case_id": case_id,
                "dice_mean": metrics["dice_mean"],
                "iou_mean": metrics["iou_mean"],
                "hd95_mean": metrics["hd95_mean"],
            }

            for i, score in enumerate(metrics["dice_per_class"], start=1):
                row[f"dice_class_{i}"] = score
            for i, score in enumerate(metrics["iou_per_class"], start=1):
                row[f"iou_class_{i}"] = score
            for i, score in enumerate(metrics["hd95_per_class"], start=1):
                row[f"hd95_class_{i}"] = score

            case_rows.append(row)

            if save_visuals and save_count < 3:
                img_np = image.squeeze(0).squeeze(0).cpu().numpy()
                out_path = self.exp_dirs["vis_dir"] / f"epoch_{epoch:03d}_{case_id}.png"
                save_triptych(img_np, gt, pred, out_path=out_path, title=f"Epoch {epoch} | {case_id}")
                save_count += 1

        val_df = pd.DataFrame(case_rows)
        if val_df.empty:
            return {"val_dice": float("nan"), "val_iou": float("nan"), "val_hd95": float("nan"), "df": val_df}

        val_dice = float(np.nanmean(val_df["dice_mean"].values))
        val_iou = float(np.nanmean(val_df["iou_mean"].values))
        val_hd95 = float(np.nanmean(val_df["hd95_mean"].values))
        return {"val_dice": val_dice, "val_iou": val_iou, "val_hd95": val_hd95, "df": val_df}

    def train(self) -> Path:
        """Run full training loop and return best checkpoint path."""

        max_epochs = int(self.cfg["train"].get("max_epochs", 200))
        val_interval = max(1, int(self.cfg["train"].get("val_interval", 1)))
        patience = int(self.cfg["train"].get("early_stopping_patience", 30))

        history_rows: List[Dict[str, Any]] = []

        self.training_started_at = time.perf_counter()
        for epoch in range(self.start_epoch, max_epochs + 1):
            self.model.train()
            epoch_losses: List[float] = []
            accum_steps = max(1, int(self.cfg["train"].get("grad_accum_steps", 1)))

            pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}", unit="batch")
            self.optimizer.zero_grad(set_to_none=True)
            for step_idx, batch in enumerate(pbar, start=1):
                image = batch["image"].to(self.device)
                label = batch["label"].to(self.device)

                prompt = build_prompt_tensor(
                    cfg=self.cfg,
                    image=image,
                    label=label,
                    class_names=self.class_names,
                    stage="train",
                )
                model_in = apply_prompt_to_image(image, prompt)

                amp_ctx = torch.amp.autocast("cuda", enabled=True) if self.use_amp else nullcontext()
                with amp_ctx:
                    logits = self.model(model_in)
                    loss = self.loss_fn(logits, label) / float(accum_steps)

                self.scaler.scale(loss).backward()
                do_step = (step_idx % accum_steps == 0) or (step_idx == len(self.train_loader))
                if do_step:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                loss_val = float((loss.detach().cpu().item()) * float(accum_steps))
                epoch_losses.append(loss_val)
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            self.scheduler.step()

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            row = {"epoch": epoch, "train_loss": train_loss}

            if epoch % val_interval == 0:
                val_out = self._run_val_epoch(epoch, save_visuals=True)
                val_df: pd.DataFrame = val_out["df"]
                row.update(
                    {
                        "val_dice": val_out["val_dice"],
                        "val_iou": val_out["val_iou"],
                        "val_hd95": val_out["val_hd95"],
                    }
                )

                val_csv = self.exp_dirs["logs_dir"] / f"val_epoch_{epoch:03d}.csv"
                val_df.to_csv(val_csv, index=False)

                if val_out["val_dice"] > self.best_dice:
                    self.best_dice = val_out["val_dice"]
                    self.best_epoch = epoch
                    self.bad_epochs = 0
                    best_path = self._save_checkpoint(epoch, is_best=True)
                    self.logger.info(
                        "New best checkpoint at epoch %d | val_dice=%.4f | %s",
                        epoch,
                        self.best_dice,
                        best_path,
                    )
                else:
                    self.bad_epochs += 1

                self.logger.info(
                    "Epoch %d | train_loss=%.4f | val_dice=%.4f | val_iou=%.4f | val_hd95=%.4f",
                    epoch,
                    train_loss,
                    val_out["val_dice"],
                    val_out["val_iou"],
                    val_out["val_hd95"],
                )
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("val/dice_mean", float(val_out["val_dice"]), epoch)
                    self.tb_writer.add_scalar("val/iou_mean", float(val_out["val_iou"]), epoch)
                    self.tb_writer.add_scalar("val/hd95_mean", float(val_out["val_hd95"]), epoch)
            else:
                self.logger.info("Epoch %d | train_loss=%.4f", epoch, train_loss)

            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/loss", float(train_loss), epoch)
                self.tb_writer.add_scalar("train/lr", float(self.optimizer.param_groups[0]["lr"]), epoch)

            self._save_checkpoint(epoch, is_best=False)
            history_rows.append(row)

            if self.bad_epochs >= patience:
                self.logger.info("Early stopping triggered at epoch %d", epoch)
                break

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(self.exp_dirs["logs_dir"] / "training_history.csv", index=False)
        if bool(self.cfg["logging"].get("save_epoch_metrics_json", True)):
            (self.exp_dirs["logs_dir"] / "training_history.json").write_text(
                json.dumps(history_rows, indent=2),
                encoding="utf-8",
            )

        training_seconds = float(time.perf_counter() - self.training_started_at)
        self.logger.info("Training wall time (s): %.2f", training_seconds)
        efficiency = {
            "adaptation_mode": str(self.cfg.get("adaptation", {}).get("mode", "full_ft")),
            "trainable_params": int(self.adaptation_stats.trainable_params),
            "total_params": int(self.adaptation_stats.total_params),
            "trainable_ratio": float(self.adaptation_stats.trainable_params / max(1, self.adaptation_stats.total_params)),
            "training_time_sec": training_seconds,
            "device": self.device,
        }
        if self.device.startswith("cuda"):
            try:
                efficiency["gpu_max_memory_mb"] = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            except Exception:
                pass

        (self.exp_dirs["logs_dir"] / "efficiency.json").write_text(
            json.dumps(efficiency, indent=2),
            encoding="utf-8",
        )
        if self.tb_writer is not None:
            self.tb_writer.add_text("run/training_time_seconds", str(training_seconds))
            self.tb_writer.add_text("run/trainable_params", str(self.adaptation_stats.trainable_params))
            self.tb_writer.close()

        best_path = self.exp_dirs["ckpt_dir"] / "best.pt"
        if not best_path.exists():
            best_path = self.exp_dirs["ckpt_dir"] / "latest.pt"
        return best_path

    def validate_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Evaluate a checkpoint on validation set and return summary metrics."""

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        result = self._run_val_epoch(epoch=int(ckpt.get("epoch", 0)), save_visuals=True)

        out_csv = self.exp_dirs["logs_dir"] / "validation_metrics.csv"
        result["df"].to_csv(out_csv, index=False)
        self.logger.info(
            "Validation summary | dice=%.4f | iou=%.4f | hd95=%.4f | csv=%s",
            result["val_dice"],
            result["val_iou"],
            result["val_hd95"],
            out_csv,
        )
        return result
