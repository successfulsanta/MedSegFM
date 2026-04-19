from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


def sliding_window_predict(model, image, roi_size: Tuple[int, int, int], sw_batch_size: int, overlap: float):
    """Run MONAI sliding-window inference for large 3D volumes."""

    try:
        from monai.inferers import sliding_window_inference
    except Exception as exc:
        raise RuntimeError("MONAI is required for sliding-window inference. Install monai.") from exc

    return sliding_window_inference(
        inputs=image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
    )


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove tiny disconnected prediction components per class."""

    if min_size <= 0:
        return mask

    output = np.zeros_like(mask, dtype=mask.dtype)
    classes = [int(c) for c in np.unique(mask) if c != 0]
    for c in classes:
        binary = (mask == c).astype(np.uint8)
        labeled, ncomp = ndimage.label(binary)
        for comp_id in range(1, ncomp + 1):
            comp = labeled == comp_id
            if int(comp.sum()) >= min_size:
                output[comp] = c
    return output


def save_prediction(mask: np.ndarray, ref_nifti_path: Path, output_path: Path) -> None:
    """Save predicted mask with reference affine/header geometry."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref = nib.load(str(ref_nifti_path))
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(mask.astype(np.int16), ref.affine, hdr), str(output_path))
