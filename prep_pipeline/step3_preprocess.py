from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_to_output

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - lightweight fallback for minimal environments
    def tqdm(iterable, *args, **kwargs):
        return iterable


IMAGE_SPLITS = ("imagesTr", "imagesVal", "imagesTs")
LABEL_SPLITS = ("labelsTr", "labelsVal", "labelsTs")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_TO_SUFFIX = {"train": "Tr", "val": "Val", "test": "Ts"}


class PreprocessError(RuntimeError):
    """Raised when preprocessing configuration or execution fails."""


@dataclass
class VolumeRecord:
    """Metadata collected for each processed 3D case."""

    split: str
    case_id: str
    image_path: str
    label_path: str
    original_shape: Tuple[int, ...]
    original_spacing: Tuple[float, ...]
    original_dtype: str
    new_shape: Tuple[int, ...]
    new_spacing: Tuple[float, ...]
    new_dtype: str
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    label_values: List[int]
    warnings: List[str]
    errors: List[str]


# -----------------------------
# Configuration / I/O helpers
# -----------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON or YAML config for Step 3 preprocessing."""

    if not config_path.exists():
        raise PreprocessError(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise PreprocessError(
                "YAML config requested but PyYAML is not installed. Use JSON or install pyyaml."
            ) from exc
        return yaml.safe_load(text)

    raise PreprocessError("Unsupported config format. Use .json, .yaml, or .yml")


def ensure_output_structure(output_dir: Path) -> Dict[str, Path]:
    """Create the standardized preprocessed output structure."""

    split_dirs = [
        "imagesTr",
        "labelsTr",
        "imagesVal",
        "labelsVal",
        "imagesTs",
        "labelsTs",
        "manifests",
        "visuals",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Path] = {}
    for name in split_dirs:
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        result[name] = path
    return result


def _find_split_dir(root: Path, split_dir_name: str) -> Path:
    path = root / split_dir_name
    if not path.exists():
        raise PreprocessError(f"Missing expected directory: {path}")
    return path


def _iter_case_pairs(input_dir: Path) -> Iterable[Tuple[str, str, Path, Path]]:
    """Yield (split, case_id, image_path, label_path) tuples from Step 2 outputs."""

    split_mapping = (
        ("train", "imagesTr", "labelsTr"),
        ("val", "imagesVal", "labelsVal"),
        ("test", "imagesTs", "labelsTs"),
    )
    for split_name, image_subdir, label_subdir in split_mapping:
        image_dir = _find_split_dir(input_dir, image_subdir)
        label_dir = _find_split_dir(input_dir, label_subdir)

        for image_path in sorted(image_dir.glob("*.nii.gz")):
            if image_path.name.startswith("._"):
                continue
            case_id = image_path.name[:-7] if image_path.name.endswith(".nii.gz") else image_path.stem
            label_path = label_dir / image_path.name
            yield split_name, case_id, image_path, label_path


# -----------------------------
# Core image operations
# -----------------------------


def load_nifti(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI image with explicit error handling."""

    try:
        return nib.load(str(path))
    except Exception as exc:
        raise PreprocessError(f"Failed to load NIfTI file: {path} ({exc})") from exc


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient a NIfTI image to RAS canonical orientation."""

    try:
        return nib.as_closest_canonical(img)
    except Exception as exc:
        raise PreprocessError(f"Failed to reorient volume to RAS: {exc}") from exc


def get_spacing(img: nib.Nifti1Image) -> Tuple[float, ...]:
    """Return voxel spacing for the spatial dimensions."""

    zooms = img.header.get_zooms()
    ndim = min(len(zooms), len(img.shape))
    return tuple(float(z) for z in zooms[:ndim])


def resample_volume(
    img: nib.Nifti1Image,
    target_spacing: Sequence[float],
    is_label: bool,
    order: Optional[int] = None,
) -> nib.Nifti1Image:
    """Resample a 3D volume to a new voxel spacing.

    Images use linear interpolation (order=1). Masks use nearest-neighbor (order=0)
    so integer label IDs are preserved and not blended into invalid fractional values.
    """

    img = reorient_to_ras(img)
    target_spacing = tuple(float(v) for v in target_spacing)
    if len(target_spacing) != 3:
        raise PreprocessError(f"target_spacing must have 3 values, got {target_spacing}")

    interp_order = 0 if is_label else 1 if order is None else int(order)
    try:
        return resample_to_output(img, voxel_sizes=target_spacing, order=interp_order)
    except Exception as exc:
        raise PreprocessError(f"Failed to resample volume to spacing {target_spacing}: {exc}") from exc


def normalize_ct_intensity(
    image_data: np.ndarray,
    hu_min: float,
    hu_max: float,
    mode: str = "zscore",
) -> np.ndarray:
    """Clip CT Hounsfield units and normalize intensities."""

    clipped = np.clip(image_data.astype(np.float32), hu_min, hu_max)

    if mode.lower() == "zscore":
        mean = float(clipped.mean())
        std = float(clipped.std())
        if std < 1e-8:
            return clipped - mean
        return (clipped - mean) / std

    if mode.lower() == "minmax":
        min_val = float(clipped.min())
        max_val = float(clipped.max())
        if np.isclose(min_val, max_val):
            return np.zeros_like(clipped, dtype=np.float32)
        return (clipped - min_val) / (max_val - min_val)

    raise PreprocessError(f"Unsupported normalization mode: {mode}")


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    """Compute a foreground bounding box from a non-zero label mask."""

    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    mins = [int(np.min(c)) for c in coords[:3]]
    maxs = [int(np.max(c)) + 1 for c in coords[:3]]
    return tuple(slice(mi, ma) for mi, ma in zip(mins, maxs))  # type: ignore[return-value]


def apply_bbox(data: np.ndarray, bbox: Tuple[slice, slice, slice]) -> np.ndarray:
    """Crop a 3D array by the provided bounding box."""

    if data.ndim != 3:
        raise PreprocessError(f"Expected a 3D volume for cropping, got shape {data.shape}")
    return data[bbox[0], bbox[1], bbox[2]]


def center_crop_or_pad(data: np.ndarray, target_size: Sequence[int], pad_value: float = 0.0) -> np.ndarray:
    """Center crop or pad a 3D array to a fixed output size."""

    target = np.asarray(list(target_size), dtype=int)
    if target.shape[0] != 3:
        raise PreprocessError(f"target_size must have 3 elements, got {target_size}")

    result = data
    for axis in range(3):
        current = result.shape[axis]
        desired = int(target[axis])
        if current > desired:
            start = (current - desired) // 2
            end = start + desired
            slc = [slice(None)] * 3
            slc[axis] = slice(start, end)
            result = result[tuple(slc)]

    pad_width = []
    for axis in range(3):
        current = result.shape[axis]
        desired = int(target[axis])
        before = max((desired - current) // 2, 0)
        after = max(desired - current - before, 0)
        pad_width.append((before, after))

    if any(before > 0 or after > 0 for before, after in pad_width):
        result = np.pad(result, pad_width, mode="constant", constant_values=pad_value)
    return result


def match_image_label_shapes(image: np.ndarray, label: np.ndarray, case_id: str) -> None:
    """Validate that image and label arrays have matching shapes."""

    if image.shape != label.shape:
        raise PreprocessError(
            f"Shape mismatch after preprocessing for {case_id}: image={image.shape}, label={label.shape}"
        )


# -----------------------------
# Metadata, saving, reporting
# -----------------------------


def save_nifti(data: np.ndarray, template_img: nib.Nifti1Image, output_path: Path) -> None:
    """Save a NIfTI volume using the template affine/header."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = template_img.header.copy()
    header.set_data_dtype(data.dtype)
    nib.save(nib.Nifti1Image(data, affine=template_img.affine, header=header), str(output_path))


def collect_intensity_stats(image_data: np.ndarray) -> Dict[str, float]:
    """Compute common intensity statistics."""

    arr = image_data.astype(np.float32)
    return {
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),
        "min_intensity": float(arr.min()),
        "max_intensity": float(arr.max()),
    }


# -----------------------------
# Main pipeline
# -----------------------------


def preprocess_case(
    split: str,
    case_id: str,
    image_path: Path,
    label_path: Path,
    output_root: Path,
    target_spacing: Sequence[float],
    target_size: Optional[Sequence[int]],
    crop_mode: str,
    hu_min: float,
    hu_max: float,
    normalization: str,
    save_mode: str,
) -> VolumeRecord:
    """Preprocess a single image-mask pair and save outputs."""

    errors: List[str] = []
    notes: List[str] = []

    image_img = load_nifti(image_path)
    label_img = load_nifti(label_path)

    image_img = reorient_to_ras(image_img)
    label_img = reorient_to_ras(label_img)

    original_shape = tuple(int(x) for x in np.asanyarray(image_img.dataobj).shape[:3])
    original_spacing = get_spacing(image_img)
    original_dtype = str(image_img.get_data_dtype())

    image_rs = resample_volume(image_img, target_spacing=target_spacing, is_label=False)
    label_rs = resample_volume(label_img, target_spacing=target_spacing, is_label=True)

    image_arr = np.asanyarray(image_rs.dataobj).astype(np.float32)
    label_arr = np.asanyarray(label_rs.dataobj)

    if image_arr.ndim > 3:
        image_arr = image_arr[..., 0]
    if label_arr.ndim > 3:
        label_arr = label_arr[..., 0]

    if crop_mode == "foreground":
        bbox = bbox_from_mask(label_arr)
        if bbox is None:
            notes.append("foreground_crop_skipped_empty_mask")
        else:
            image_arr = apply_bbox(image_arr, bbox)
            label_arr = apply_bbox(label_arr, bbox)
    elif crop_mode == "center":
        if target_size is None:
            raise PreprocessError("center crop mode requires --target_size")
        image_arr = center_crop_or_pad(image_arr, target_size=target_size, pad_value=hu_min)
        label_arr = center_crop_or_pad(label_arr, target_size=target_size, pad_value=0)
    elif crop_mode == "none":
        pass
    else:
        raise PreprocessError(f"Unsupported crop_mode: {crop_mode}")

    if crop_mode == "foreground" and target_size is not None:
        image_arr = center_crop_or_pad(image_arr, target_size=target_size, pad_value=hu_min)
        label_arr = center_crop_or_pad(label_arr, target_size=target_size, pad_value=0)

    match_image_label_shapes(image_arr, label_arr, case_id)

    image_norm = normalize_ct_intensity(image_arr, hu_min=hu_min, hu_max=hu_max, mode=normalization)
    stats = collect_intensity_stats(image_norm)
    new_shape = tuple(int(x) for x in image_norm.shape)
    new_spacing = tuple(float(v) for v in target_spacing)
    new_dtype = str(image_norm.dtype)

    split_suffix = SPLIT_TO_SUFFIX.get(split)
    if split_suffix is None:
        raise PreprocessError(f"Unsupported split name: {split}")

    image_out = output_root / f"images{split_suffix}" / f"{case_id}.nii.gz"
    label_out = output_root / f"labels{split_suffix}" / f"{case_id}.nii.gz"

    image_template = nib.Nifti1Image(image_norm.astype(np.float32), affine=image_rs.affine, header=image_rs.header)
    label_template = nib.Nifti1Image(label_arr.astype(np.int16), affine=label_rs.affine, header=label_rs.header)

    if save_mode == "save":
        save_nifti(image_norm.astype(np.float32), image_template, image_out)
        save_nifti(label_arr.astype(np.int16), label_template, label_out)
    elif save_mode == "dry-run":
        notes.append("dry_run_no_files_written")
    else:
        raise PreprocessError(f"Unsupported save_mode: {save_mode}")

    try:
        label_values = sorted({int(x) for x in np.unique(label_arr)})
    except Exception as exc:
        label_values = []
        errors.append(f"label_value_inspection_failed: {exc}")

    return VolumeRecord(
        split=split,
        case_id=case_id,
        image_path=str(image_path),
        label_path=str(label_path),
        original_shape=original_shape,
        original_spacing=tuple(float(x) for x in original_spacing),
        original_dtype=original_dtype,
        new_shape=new_shape,
        new_spacing=new_spacing,
        new_dtype=new_dtype,
        mean_intensity=stats["mean_intensity"],
        std_intensity=stats["std_intensity"],
        min_intensity=stats["min_intensity"],
        max_intensity=stats["max_intensity"],
        label_values=label_values,
        warnings=notes,
        errors=errors,
    )


def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    target_spacing: Sequence[float],
    target_size: Optional[Sequence[int]],
    hu_min: float,
    hu_max: float,
    normalization: str,
    crop_mode: str,
    seed: int,
    save_mode: str = "save",
) -> pd.DataFrame:
    """Process all Step 2 cases into a standardized deep-learning-ready layout."""

    np.random.seed(seed)
    output_dirs = ensure_output_structure(output_dir)

    records: List[VolumeRecord] = []
    pairs = list(_iter_case_pairs(input_dir))
    if not pairs:
        raise PreprocessError(f"No NIfTI pairs were found under: {input_dir}")

    for split, case_id, image_path, label_path in tqdm(pairs, desc="Preprocessing cases", unit="case"):
        try:
            record = preprocess_case(
                split=split,
                case_id=case_id,
                image_path=image_path,
                label_path=label_path,
                output_root=output_dir,
                target_spacing=target_spacing,
                target_size=target_size,
                crop_mode=crop_mode,
                hu_min=hu_min,
                hu_max=hu_max,
                normalization=normalization,
                save_mode=save_mode,
            )
            records.append(record)
        except Exception as exc:
            warnings.warn(f"Skipping corrupted case {case_id}: {exc}", RuntimeWarning)
            records.append(
                VolumeRecord(
                    split=split,
                    case_id=case_id,
                    image_path=str(image_path),
                    label_path=str(label_path),
                    original_shape=(),
                    original_spacing=(),
                    original_dtype="",
                    new_shape=(),
                    new_spacing=tuple(float(v) for v in target_spacing),
                    new_dtype="",
                    mean_intensity=float("nan"),
                    std_intensity=float("nan"),
                    min_intensity=float("nan"),
                    max_intensity=float("nan"),
                    label_values=[],
                    warnings=[],
                    errors=[str(exc)],
                )
            )

    df = pd.DataFrame([record.__dict__ for record in records])
    if not df.empty:
        df["has_error"] = df["errors"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        df["has_warning"] = df["warnings"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    return df


def save_metadata(df: pd.DataFrame, manifests_dir: Path) -> Tuple[Path, Path]:
    """Save preprocessing metadata to CSV and JSON."""

    manifests_dir.mkdir(parents=True, exist_ok=True)
    csv_path = manifests_dir / "preprocessing_metadata.csv"
    json_path = manifests_dir / "preprocessing_metadata.json"

    out = df.copy()
    for col in ["original_shape", "original_spacing", "new_shape", "new_spacing", "label_values", "warnings", "errors"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, tuple)) else x)
    out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    return csv_path, json_path


def print_report(df: pd.DataFrame) -> str:
    """Print a concise report of preprocessing outcomes."""

    total = len(df)
    errors = int(df["has_error"].sum()) if "has_error" in df.columns else 0
    warnings_count = int(df["has_warning"].sum()) if "has_warning" in df.columns else 0
    lines = [
        "=" * 80,
        "Step 3 Report: Standardized Preprocessing",
        "=" * 80,
        f"Total cases: {total}",
        f"Cases with errors: {errors}",
        f"Cases with warnings: {warnings_count}",
        "-" * 80,
    ]
    if "split" in df.columns:
        lines.append("Split summary:")
        for split_name in SPLIT_NAMES:
            count = int((df["split"] == split_name).sum())
            lines.append(f"- {split_name}: {count}")
        lines.append("-" * 80)

    problematic = df[df["has_error"]] if "has_error" in df.columns else pd.DataFrame()
    if not problematic.empty:
        lines.append("Problem cases:")
        for row in problematic.itertuples(index=False):
            lines.append(f"- {row.case_id}: {'; '.join(row.errors)}")
    else:
        lines.append("No preprocessing errors found.")

    report = "\n".join(lines)
    print(report)
    return report


# -----------------------------
# Visualization utility
# -----------------------------


def visualize_random_cases(
    output_dir: Path,
    n_cases: int = 3,
    seed: int = 42,
    alpha: float = 0.35,
    save_png: Optional[Path] = None,
) -> None:
    """Visual sanity check for 2-3 random preprocessed cases with mask overlays."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise PreprocessError("matplotlib is required for visualization") from exc

    image_paths = []
    for split_dir in [output_dir / "imagesTr", output_dir / "imagesVal", output_dir / "imagesTs"]:
        image_paths.extend(sorted(split_dir.glob("*.nii.gz")))
    if not image_paths:
        raise PreprocessError(f"No preprocessed images found under: {output_dir}")

    rng = np.random.default_rng(seed)
    take = min(int(n_cases), len(image_paths))
    sampled = rng.choice(image_paths, size=take, replace=False)

    fig, axes = plt.subplots(take, 2, figsize=(12, 4 * take))
    if take == 1:
        axes = np.array([axes])

    for idx, img_path in enumerate(sampled):
        img_path = Path(img_path)
        case_id = img_path.name[:-7] if img_path.name.endswith(".nii.gz") else img_path.stem
        split = "Tr" if "imagesTr" in str(img_path) else "Val" if "imagesVal" in str(img_path) else "Ts"
        label_path = output_dir / f"labels{split}" / img_path.name

        image = np.asanyarray(load_nifti(img_path).dataobj)
        label = np.asanyarray(load_nifti(label_path).dataobj)
        if image.ndim > 3:
            image = image[..., 0]
        if label.ndim > 3:
            label = label[..., 0]
        z = image.shape[2] // 2

        axes[idx, 0].imshow(image[:, :, z].T, cmap="gray", origin="lower")
        axes[idx, 0].set_title(f"Image | {case_id}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(image[:, :, z].T, cmap="gray", origin="lower")
        mask = np.ma.masked_where(label[:, :, z].T == 0, label[:, :, z].T)
        axes[idx, 1].imshow(mask, cmap="tab20", alpha=alpha, origin="lower")
        axes[idx, 1].set_title("Overlay")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_png, dpi=150)
        print(f"Saved sanity-check figure to: {save_png}")
    else:
        plt.show()
