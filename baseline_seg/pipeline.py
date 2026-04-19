from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from baseline_seg.dataset import build_dataloader, build_pair_list
from baseline_seg.evaluate import evaluate_prediction_dir
from baseline_seg.inference import remove_small_components, save_prediction, sliding_window_predict
from baseline_seg.metrics import case_metrics
from baseline_seg.models import build_model
from baseline_seg.prompting import apply_prompt_to_image, build_prompt_tensor, get_prompt_mode
from baseline_seg.trainer import Trainer
from baseline_seg.visualization import save_triptych


def train_experiment(
    cfg: Dict[str, Any],
    class_names: Dict[int, str],
    device: str,
    exp_dirs: Dict[str, Path],
    logger,
    resume_checkpoint: Optional[Path] = None,
) -> Path:
    """Run baseline training and return best checkpoint path."""

    trainer = Trainer(
        cfg,
        class_names=class_names,
        device=device,
        exp_dirs=exp_dirs,
        logger=logger,
        resume_checkpoint=resume_checkpoint,
    )
    best_ckpt = trainer.train()
    logger.info("Training complete. Best checkpoint: %s", best_ckpt)
    return best_ckpt


def validate_experiment(
    cfg: Dict[str, Any],
    class_names: Dict[int, str],
    device: str,
    exp_dirs: Dict[str, Path],
    logger,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Validate a checkpoint and save per-case metrics."""

    trainer = Trainer(cfg, class_names=class_names, device=device, exp_dirs=exp_dirs, logger=logger)
    return trainer.validate_checkpoint(checkpoint_path)


def infer_split(
    cfg: Dict[str, Any],
    device: str,
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
    logger,
) -> Path:
    """Run sliding-window inference over a split and export NIfTI predictions."""

    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from baseline_seg.config import load_label_map

    prompt_mode = get_prompt_mode(cfg)
    simulate_prompts = bool(cfg.get("prompting", {}).get("prompt_generation_strategy", "from_ground_truth") == "from_ground_truth")
    want_labels = prompt_mode in {"spatial", "text"} and simulate_prompts
    try:
        loader = build_dataloader(cfg, split=split, require_labels=want_labels)
    except Exception:
        loader = build_dataloader(cfg, split=split, require_labels=False)

    label_map = load_label_map(Path(cfg["data"]["label_map_path"]).resolve())
    roi = tuple(int(x) for x in cfg["model"]["roi_size"])
    sw_batch = int(cfg["infer"].get("sw_batch_size", 1))
    overlap = float(cfg["infer"].get("overlap", 0.5))
    min_component_size = int(cfg["infer"].get("min_component_size", 0))
    apply_postprocess = bool(cfg["infer"].get("apply_postprocess", True))
    save_probabilities = bool(cfg["infer"].get("save_probabilities", False))
    use_amp = bool(cfg["train"].get("amp", True)) and device.startswith("cuda")

    pred_dir = output_dir / f"pred_{split}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = output_dir / f"prob_{split}"
    if save_probabilities:
        prob_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=f"Infer {split}", unit="case"):
        if isinstance(batch, list):
            if not batch:
                continue
            batch = batch[0]

        image = batch["image"]
        if isinstance(image, list):
            image = image[0]
        image = image.to(device)

        label = batch.get("label")
        if label is not None:
            if isinstance(label, list):
                label = label[0]
            if hasattr(label, "to"):
                label = label.to(device)
            else:
                label = None

        case_id_value = batch.get("case_id")
        image_path_value = batch.get("image_path")
        if isinstance(case_id_value, list):
            case_id_value = case_id_value[0]
        if isinstance(image_path_value, list):
            image_path_value = image_path_value[0]

        case_id = str(case_id_value)
        ref_path = Path(str(image_path_value))

        prompt = build_prompt_tensor(
            cfg=cfg,
            image=image,
            label=label,
            class_names=label_map,
            stage="infer",
        )
        model_in = apply_prompt_to_image(image, prompt)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = sliding_window_predict(model, model_in, roi_size=roi, sw_batch_size=sw_batch, overlap=overlap)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.int16)
        if apply_postprocess:
            pred = remove_small_components(pred, min_size=min_component_size).astype(np.int16)

        out_path = pred_dir / f"{case_id}.nii.gz"
        save_prediction(pred, ref_nifti_path=ref_path, output_path=out_path)
        if save_probabilities:
            prob_path = prob_dir / f"{case_id}.npz"
            np.savez_compressed(prob_path, probs=probs.squeeze(0).detach().cpu().numpy().astype(np.float16))

    logger.info("Inference complete for split=%s. Predictions at %s", split, pred_dir)
    return pred_dir


def evaluate_split(
    cfg: Dict[str, Any],
    split: str,
    pred_dir: Path,
    out_csv: Path,
    label_map: Dict[int, str],
    gt_dir: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    probability_dir: Optional[Path] = None,
    vis_dir: Optional[Path] = None,
    max_visual_cases: int = 6,
    failure_cases_k: int = 10,
    suspicious_cases: Optional[list] = None,
) -> Dict[str, Any]:
    """Evaluate predicted masks against available split labels."""

    split_to_label_dir = {
        "train": "labelsTr",
        "val": "labelsVal",
        "test": "labelsTs",
    }
    root = Path(cfg["data"]["root_dir"]).resolve()
    resolved_gt_dir = gt_dir if gt_dir is not None else root / split_to_label_dir[split]
    split_to_image_dir = {
        "train": root / "imagesTr",
        "val": root / "imagesVal",
        "test": root / "imagesTs",
    }
    resolved_image_dir = image_dir if image_dir is not None else split_to_image_dir.get(split, None)

    return evaluate_prediction_dir(
        pred_dir=pred_dir,
        gt_dir=resolved_gt_dir,
        num_classes=int(cfg["model"]["num_classes"]),
        output_csv=out_csv,
        label_map=label_map,
        image_dir=resolved_image_dir,
        probability_dir=probability_dir,
        vis_dir=vis_dir,
        max_visual_cases=max_visual_cases,
        failure_cases_k=failure_cases_k,
        split_name=split,
        suspicious_cases=suspicious_cases,
    )


def visualize_predictions(
    cfg: Dict[str, Any],
    split: str,
    pred_dir: Path,
    output_dir: Path,
    max_cases: int = 3,
) -> Path:
    """Create side-by-side visualizations of CT, GT, and prediction."""

    root = Path(cfg["data"]["root_dir"]).resolve()
    split_map = {
        "train": (root / "imagesTr", root / "labelsTr"),
        "val": (root / "imagesVal", root / "labelsVal"),
        "test": (root / "imagesTs", root / "labelsTs"),
    }
    image_dir, label_dir = split_map[split]

    vis_dir = output_dir / f"vis_{split}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for pred_path in sorted(pred_dir.glob("*.nii.gz")):
        if count >= max_cases:
            break
        image_path = image_dir / pred_path.name
        label_path = label_dir / pred_path.name
        if not image_path.exists() or not label_path.exists():
            continue

        image = np.asanyarray(nib.load(str(image_path)).dataobj)
        gt = np.asanyarray(nib.load(str(label_path)).dataobj)
        pred = np.asanyarray(nib.load(str(pred_path)).dataobj)

        if image.ndim > 3:
            image = image[..., 0]
        if gt.ndim > 3:
            gt = gt[..., 0]
        if pred.ndim > 3:
            pred = pred[..., 0]

        out_path = vis_dir / f"{pred_path.stem}.png"
        save_triptych(image, gt, pred, out_path=out_path, title=f"{split} | {pred_path.name}")
        count += 1

    return vis_dir
