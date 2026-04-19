from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def seed_worker(worker_id: int, base_seed: int) -> None:
    import random

    import numpy as np

    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_pair_list(root_dir: Path, split: str, require_labels: bool = True) -> List[Dict[str, Any]]:
    """Build image/label file lists for a given split."""

    split_map = {
        "train": ("imagesTr", "labelsTr"),
        "val": ("imagesVal", "labelsVal"),
        "test": ("imagesTs", "labelsTs"),
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split: {split}")

    image_dir = root_dir / split_map[split][0]
    label_dir = root_dir / split_map[split][1]

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if require_labels and not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    items: List[Dict[str, Any]] = []
    for image_path in sorted(image_dir.glob("*.nii.gz")):
        case_id = image_path.name[:-7] if image_path.name.endswith(".nii.gz") else image_path.stem
        record: Dict[str, Any] = {
            "image": str(image_path),
            "case_id": case_id,
            "image_path": str(image_path),
        }

        label_path = label_dir / image_path.name
        if require_labels:
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for case {case_id}: {label_path}")
            record["label"] = str(label_path)
            record["label_path"] = str(label_path)
        elif label_path.exists():
            record["label"] = str(label_path)
            record["label_path"] = str(label_path)

        items.append(record)

    if not items:
        raise RuntimeError(f"No NIfTI files found in {image_dir}")

    return items


def _import_monai_transforms():
    try:
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            EnsureTyped,
            LoadImaged,
            Orientationd,
            RandAdjustContrastd,
            RandAffined,
            RandCropByPosNegLabeld,
            Rand3DElasticd,
            RandSpatialCropd,
            RandFlipd,
            RandGaussianNoised,
            ScaleIntensityRanged,
        )
    except Exception as exc:
        raise RuntimeError("MONAI is required for dataset transforms. Install monai.") from exc

    return {
        "Compose": Compose,
        "LoadImaged": LoadImaged,
        "EnsureChannelFirstd": EnsureChannelFirstd,
        "Orientationd": Orientationd,
        "ScaleIntensityRanged": ScaleIntensityRanged,
        "RandCropByPosNegLabeld": RandCropByPosNegLabeld,
        "Rand3DElasticd": Rand3DElasticd,
        "RandSpatialCropd": RandSpatialCropd,
        "RandFlipd": RandFlipd,
        "RandAffined": RandAffined,
        "RandGaussianNoised": RandGaussianNoised,
        "RandAdjustContrastd": RandAdjustContrastd,
        "EnsureTyped": EnsureTyped,
    }


def build_transforms(cfg: Dict[str, Any], split: str):
    """Create MONAI transforms for training or evaluation."""

    tr = _import_monai_transforms()
    model_cfg = cfg["model"]
    aug_cfg = cfg["augment"]

    roi_size = tuple(int(x) for x in model_cfg["roi_size"])
    rotate_prob = float(aug_cfg.get("rotate_prob", 0.2))
    scale_prob = float(aug_cfg.get("scale_prob", 0.2))
    elastic_prob = float(aug_cfg.get("elastic_prob", 0.0))
    noise_prob = float(aug_cfg.get("noise_prob", 0.15))
    flip_prob = float(aug_cfg.get("flip_prob", 0.5))
    rotate_range = tuple(float(x) for x in aug_cfg.get("rotate_range", [0.15, 0.15, 0.15]))
    scale_range = tuple(float(x) for x in aug_cfg.get("scale_range", [0.1, 0.1, 0.1]))

    common = [
        tr["LoadImaged"](keys=["image", "label"] if split != "test" else ["image"]),
        tr["EnsureChannelFirstd"](keys=["image", "label"] if split != "test" else ["image"]),
        tr["Orientationd"](keys=["image", "label"] if split != "test" else ["image"], axcodes="RAS"),
        # HU clip window aligned with preprocessing defaults; safe if already normalized.
        tr["ScaleIntensityRanged"](
            keys=["image"],
            a_min=float(cfg.get("pre_clip_min", -200)),
            a_max=float(cfg.get("pre_clip_max", 300)),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    if split == "train":
        train_aug = [
            tr["RandCropByPosNegLabeld"](
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=int(cfg["train"].get("num_samples_per_volume", 2)),
            ),
            tr["RandFlipd"](keys=["image", "label"], prob=flip_prob, spatial_axis=0),
            tr["RandFlipd"](keys=["image", "label"], prob=flip_prob, spatial_axis=1),
            tr["RandFlipd"](keys=["image", "label"], prob=flip_prob, spatial_axis=2),
            tr["RandAffined"](
                keys=["image", "label"],
                prob=max(rotate_prob, scale_prob),
                rotate_range=rotate_range,
                scale_range=scale_range,
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            tr["Rand3DElasticd"](
                keys=["image", "label"],
                prob=elastic_prob,
                sigma_range=tuple(float(x) for x in aug_cfg.get("elastic_sigma_range", [4.0, 6.0])),
                magnitude_range=tuple(float(x) for x in aug_cfg.get("elastic_magnitude_range", [50.0, 100.0])),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            tr["RandGaussianNoised"](keys=["image"], prob=noise_prob, std=float(aug_cfg.get("noise_std", 0.02))),
            tr["RandAdjustContrastd"](keys=["image"], prob=0.15, gamma=(0.9, 1.1)),
            tr["EnsureTyped"](keys=["image", "label"]),
        ]
        return tr["Compose"](common + train_aug)

    out_keys = ["image", "label"] if split != "test" else ["image"]
    return tr["Compose"](common + [tr["EnsureTyped"](keys=out_keys)])


def build_dataloader(cfg: Dict[str, Any], split: str, require_labels: bool = True):
    """Build MONAI DataLoader for a given split."""

    try:
        from monai.data import CacheDataset, DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("MONAI is required for dataloaders. Install monai.") from exc

    root_dir = Path(cfg["data"]["root_dir"]).resolve()
    items = build_pair_list(root_dir, split=split, require_labels=require_labels)
    if split == "train" and bool(cfg.get("data", {}).get("combine_train_val_for_train", False)):
        try:
            val_items = build_pair_list(root_dir, split="val", require_labels=require_labels)
            items = items + val_items
        except Exception:
            pass
    transforms = build_transforms(cfg, split=split)

    cache_rate = float(cfg["data"].get("cache_rate", 0.0))
    num_workers = int(cfg["data"].get("num_workers", 4))

    if cache_rate > 0:
        ds = CacheDataset(data=items, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
    else:
        ds = Dataset(data=items, transform=transforms)

    seed = int(cfg.get("seed", 42))

    from functools import partial

    worker_init = partial(seed_worker, base_seed=seed)

    generator = None
    try:
        import torch

        generator = torch.Generator()
        generator.manual_seed(seed)
    except Exception:
        generator = None

    if split == "train":
        return DataLoader(
            ds,
            batch_size=int(cfg["train"].get("batch_size", 1)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init,
            generator=generator,
        )

    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=generator,
    )


def build_unlabeled_dataloader(cfg: Dict[str, Any]):
    """Build MONAI dataloader for SSL pretraining using unlabeled CT volumes."""

    try:
        from monai.data import DataLoader, Dataset
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            EnsureTyped,
            LoadImaged,
            Orientationd,
            RandAdjustContrastd,
            RandAffined,
            Rand3DElasticd,
            RandFlipd,
            RandGaussianNoised,
            RandSpatialCropd,
            ScaleIntensityRanged,
        )
    except Exception as exc:
        raise RuntimeError("MONAI is required for SSL dataloaders. Install monai.") from exc

    root_dir = Path(cfg["data"]["root_dir"]).resolve()
    p_cfg = cfg.get("pretraining", {})
    roi_size = tuple(int(x) for x in cfg["model"].get("roi_size", [128, 128, 128]))

    image_roots = p_cfg.get("unlabeled_dirs") or ["imagesTr", "imagesVal", "imagesTs"]
    image_paths = []
    for sub in image_roots:
        image_paths.extend(sorted((root_dir / sub).glob("*.nii.gz")))

    if not image_paths:
        raise RuntimeError(f"No unlabeled CT volumes found under {root_dir}")

    items = [{"image": str(p), "case_id": p.name[:-7] if p.name.endswith('.nii.gz') else p.stem} for p in image_paths]

    flip_prob = float(cfg["augment"].get("flip_prob", 0.5))
    rotate_prob = float(cfg["augment"].get("rotate_prob", 0.2))
    scale_prob = float(cfg["augment"].get("scale_prob", 0.2))
    elastic_prob = float(cfg["augment"].get("elastic_prob", 0.0))
    noise_prob = float(cfg["augment"].get("noise_prob", 0.15))

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(cfg.get("pre_clip_min", -200)),
                a_max=float(cfg.get("pre_clip_max", 300)),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandSpatialCropd(keys=["image"], roi_size=roi_size, random_size=False),
            RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=0),
            RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=1),
            RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=2),
            RandAffined(
                keys=["image"],
                prob=max(rotate_prob, scale_prob),
                rotate_range=tuple(float(x) for x in cfg["augment"].get("rotate_range", [0.15, 0.15, 0.15])),
                scale_range=tuple(float(x) for x in cfg["augment"].get("scale_range", [0.1, 0.1, 0.1])),
                mode=("bilinear",),
                padding_mode="border",
            ),
            Rand3DElasticd(
                keys=["image"],
                prob=elastic_prob,
                sigma_range=tuple(float(x) for x in cfg["augment"].get("elastic_sigma_range", [4.0, 6.0])),
                magnitude_range=tuple(float(x) for x in cfg["augment"].get("elastic_magnitude_range", [50.0, 100.0])),
                mode=("bilinear",),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=["image"], prob=noise_prob, std=float(cfg["augment"].get("noise_std", 0.02))),
            RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.9, 1.1)),
            EnsureTyped(keys=["image"]),
        ]
    )

    ds = Dataset(data=items, transform=transforms)

    return DataLoader(
        ds,
        batch_size=int(p_cfg.get("ssl_batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
    )
