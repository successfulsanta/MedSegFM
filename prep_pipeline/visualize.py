from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd


def visualize_random_cases(
    manifest_csv: Path,
    n_cases: int = 3,
    seed: int = 42,
    alpha: float = 0.35,
    output_png: Optional[Path] = None,
) -> None:
    """Load random cases and visualize axial slice overlays for sanity checks."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install matplotlib to use visual-check."
        ) from exc

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    if "split" in df.columns:
        df = df[df["split"].isin(["train", "val", "test"])]

    image_col = "processed_image_path" if "processed_image_path" in df.columns else "image_path"
    label_col = "processed_label_path" if "processed_label_path" in df.columns else "label_path"

    df = df[df[image_col].notna() & df[label_col].notna()]
    if df.empty:
        raise RuntimeError("No cases with image/label paths were found in manifest.")

    rng = np.random.default_rng(seed)
    take = min(n_cases, len(df))
    sampled_idx = rng.choice(df.index.to_numpy(), size=take, replace=False)
    sampled = df.loc[sampled_idx]

    fig, axes = plt.subplots(take, 2, figsize=(12, 4 * take))
    if take == 1:
        axes = np.array([axes])

    for i, row in enumerate(sampled.itertuples(index=False)):
        img_path = Path(getattr(row, image_col))
        lbl_path = Path(getattr(row, label_col))

        img = nib.load(str(img_path))
        lbl = nib.load(str(lbl_path))

        img_arr = np.asanyarray(img.dataobj)
        lbl_arr = np.asanyarray(lbl.dataobj)

        if img_arr.ndim > 3:
            img_arr = img_arr[..., 0]
        if lbl_arr.ndim > 3:
            lbl_arr = lbl_arr[..., 0]

        z = img_arr.shape[2] // 2
        img_slice = img_arr[:, :, z]
        lbl_slice = lbl_arr[:, :, z]

        axes[i, 0].imshow(img_slice.T, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"Image | {getattr(row, 'case_uid', 'unknown_case')}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_slice.T, cmap="gray", origin="lower")
        overlay = np.ma.masked_where(lbl_slice.T == 0, lbl_slice.T)
        axes[i, 1].imshow(overlay, cmap="tab20", alpha=alpha, origin="lower")
        axes[i, 1].set_title("Mask Overlay")
        axes[i, 1].axis("off")

    plt.tight_layout()
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, dpi=150)
        print(f"Saved sanity-check image to: {output_png}")
    else:
        plt.show()

