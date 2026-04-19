from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_PROCESSED_DIRS = [
    "imagesTr",
    "labelsTr",
    "imagesVal",
    "labelsVal",
    "imagesTs",
    "labelsTs",
    "manifests",
]


@dataclass
class CaseRecord:
    """Container for per-case metadata and validation state."""

    dataset: str
    case_id: str
    image_path: str
    label_path: str
    shape: Optional[Tuple[int, ...]]
    spacing: Optional[Tuple[float, ...]]
    image_dtype: Optional[str]
    label_dtype: Optional[str]
    image_read_ok: bool
    label_read_ok: bool
    missing_label: bool
    shape_mismatch: bool
    unique_labels: Optional[List[int]]
    unexpected_labels: Optional[List[int]]
    errors: List[str]


class PipelineError(RuntimeError):
    """Raised for fatal pipeline configuration or execution failures."""


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON or YAML config with graceful fallback when YAML is unavailable."""

    if not config_path.exists():
        raise PipelineError(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix in {".json"}:
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise PipelineError(
                "YAML config requested but PyYAML is not installed. "
                "Install pyyaml or use JSON config."
            ) from exc
        return yaml.safe_load(text)

    raise PipelineError("Unsupported config format. Use .json, .yaml, or .yml")


def ensure_processed_structure(processed_dir: Path) -> Dict[str, Path]:
    """Create standardized processed directory structure."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for dname in REQUIRED_PROCESSED_DIRS:
        dpath = processed_dir / dname
        dpath.mkdir(parents=True, exist_ok=True)
        out[dname] = dpath
    return out


def normalize_case_id(path: Path, image_suffix: str = ".nii.gz") -> str:
    """Derive a stable case ID from file name."""

    name = path.name
    if image_suffix and name.endswith(image_suffix):
        return name[: -len(image_suffix)]
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def _iter_image_files(image_root: Path, recursive: bool, glob_pattern: str) -> Iterable[Path]:
    if not image_root.exists():
        return []
    if recursive:
        return sorted(image_root.rglob(glob_pattern))
    return sorted(image_root.glob(glob_pattern))


def _safe_load_nifti(path: Path) -> Tuple[bool, Optional[nib.Nifti1Image], Optional[str]]:
    try:
        img = nib.load(str(path))
        return True, img, None
    except Exception as exc:
        return False, None, str(exc)


def check_label_values(label_array: np.ndarray, expected_labels: Optional[set[int]]) -> Tuple[List[int], List[int]]:
    """Return unique labels and labels unexpected under the provided schema."""

    uniques = sorted({int(x) for x in np.unique(label_array)})
    if expected_labels is None:
        return uniques, []
    unexpected = sorted(set(uniques) - set(expected_labels))
    return uniques, unexpected


def scan_dataset_cases(dataset_cfg: Dict[str, Any], workspace_root: Path) -> List[CaseRecord]:
    """Scan one dataset definition and build validated case records."""

    dataset_name = dataset_cfg["name"]
    raw_dir = (workspace_root / dataset_cfg["raw_dir"]).resolve()
    image_suffix = dataset_cfg.get("image_suffix", ".nii.gz")
    label_suffix = dataset_cfg.get("label_suffix", ".nii.gz")
    label_map_cfg = dataset_cfg.get("label_mapping", {})
    expected_source_labels: Optional[set[int]] = None
    if label_map_cfg:
        expected_source_labels = {int(k) for k in label_map_cfg.keys()}

    records: List[CaseRecord] = []

    for pair in dataset_cfg.get("pair_directories", []):
        image_dir = pair["image_dir"]
        label_dir = pair["label_dir"]
        recursive = bool(pair.get("recursive", False))
        image_glob = pair.get("image_glob", f"*{image_suffix}")

        image_root = raw_dir / image_dir
        label_root = raw_dir / label_dir

        for image_path in _iter_image_files(image_root, recursive=recursive, glob_pattern=image_glob):
            if image_path.name.startswith("._"):
                # Skip macOS metadata sidecar files.
                continue

            rel = image_path.relative_to(image_root)
            label_rel_name = rel.name
            if image_suffix and label_suffix and label_rel_name.endswith(image_suffix):
                label_rel_name = f"{label_rel_name[: -len(image_suffix)]}{label_suffix}"
            label_rel = rel.with_name(label_rel_name)
            label_path = label_root / label_rel

            case_id = normalize_case_id(image_path, image_suffix=image_suffix)
            errors: List[str] = []
            missing_label = not label_path.exists()

            image_ok, image_obj, image_err = _safe_load_nifti(image_path)
            if not image_ok:
                errors.append(f"image_read_error: {image_err}")

            label_ok = False
            label_obj: Optional[nib.Nifti1Image] = None
            label_err: Optional[str] = None
            if missing_label:
                errors.append("missing_label")
            else:
                label_ok, label_obj, label_err = _safe_load_nifti(label_path)
                if not label_ok:
                    errors.append(f"label_read_error: {label_err}")

            shape: Optional[Tuple[int, ...]] = None
            spacing: Optional[Tuple[float, ...]] = None
            image_dtype: Optional[str] = None
            label_dtype: Optional[str] = None
            shape_mismatch = False
            unique_labels: Optional[List[int]] = None
            unexpected_labels: Optional[List[int]] = None

            if image_obj is not None:
                shape = tuple(int(x) for x in image_obj.shape)
                zooms = image_obj.header.get_zooms()
                spacing = tuple(float(x) for x in zooms[: len(shape)])
                image_dtype = str(image_obj.get_data_dtype())

            if label_obj is not None:
                label_dtype = str(label_obj.get_data_dtype())
                if image_obj is not None and tuple(label_obj.shape) != tuple(image_obj.shape):
                    shape_mismatch = True
                    errors.append(
                        f"shape_mismatch: image={tuple(image_obj.shape)}, label={tuple(label_obj.shape)}"
                    )

                try:
                    label_data = np.asanyarray(label_obj.dataobj)
                    unique_labels, unexpected_labels = check_label_values(
                        label_data,
                        expected_source_labels,
                    )
                    if unexpected_labels:
                        errors.append(
                            f"unexpected_labels: {unexpected_labels} (expected subset of {sorted(expected_source_labels)})"
                        )
                except Exception as exc:
                    errors.append(f"label_inspection_error: {exc}")

            records.append(
                CaseRecord(
                    dataset=dataset_name,
                    case_id=case_id,
                    image_path=str(image_path),
                    label_path=str(label_path),
                    shape=shape,
                    spacing=spacing,
                    image_dtype=image_dtype,
                    label_dtype=label_dtype,
                    image_read_ok=image_ok,
                    label_read_ok=label_ok,
                    missing_label=missing_label,
                    shape_mismatch=shape_mismatch,
                    unique_labels=unique_labels,
                    unexpected_labels=unexpected_labels,
                    errors=errors,
                )
            )

    return records


def collect_records(config: Dict[str, Any], workspace_root: Path) -> pd.DataFrame:
    """Collect records from all datasets in config into a single dataframe."""

    all_records: List[CaseRecord] = []
    for dataset_cfg in config.get("datasets", []):
        all_records.extend(scan_dataset_cases(dataset_cfg, workspace_root))

    rows: List[Dict[str, Any]] = []
    for rec in all_records:
        rows.append(
            {
                "dataset": rec.dataset,
                "case_id": rec.case_id,
                "case_uid": f"{rec.dataset}::{rec.case_id}",
                "image_path": rec.image_path,
                "label_path": rec.label_path,
                "shape": rec.shape,
                "spacing": rec.spacing,
                "image_dtype": rec.image_dtype,
                "label_dtype": rec.label_dtype,
                "image_read_ok": rec.image_read_ok,
                "label_read_ok": rec.label_read_ok,
                "missing_label": rec.missing_label,
                "shape_mismatch": rec.shape_mismatch,
                "unique_labels": rec.unique_labels,
                "unexpected_labels": rec.unexpected_labels,
                "errors": rec.errors,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise PipelineError("No cases were discovered. Check config paths and glob patterns.")

    df["is_usable"] = (
        (~df["missing_label"])
        & (df["image_read_ok"])
        & (df["label_read_ok"])
        & (~df["shape_mismatch"])
    )
    return df


def split_cases(df: pd.DataFrame, split_ratios: Dict[str, float], seed: int) -> Dict[str, str]:
    """Create reproducible train/val/test split by case UID."""

    train_ratio = float(split_ratios.get("train", 0.7))
    val_ratio = float(split_ratios.get("val", 0.15))
    test_ratio = float(split_ratios.get("test", 0.15))
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise PipelineError(
            f"Split ratios must sum to 1.0, got {total:.6f}. "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    usable_uids = sorted(df.loc[df["is_usable"], "case_uid"].tolist())
    if len(usable_uids) < 3:
        raise PipelineError("Need at least 3 usable cases to create train/val/test split.")

    assignments: Dict[str, str] = {}

    if test_ratio > 0:
        train_val, test = train_test_split(
            usable_uids,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_val, test = usable_uids, []

    if val_ratio > 0:
        denom = train_ratio + val_ratio
        val_relative = val_ratio / denom
        train, val = train_test_split(
            train_val,
            test_size=val_relative,
            random_state=seed,
            shuffle=True,
        )
    else:
        train, val = train_val, []

    for uid in train:
        assignments[uid] = "train"
    for uid in val:
        assignments[uid] = "val"
    for uid in test:
        assignments[uid] = "test"

    return assignments


def remap_label_image(label_img: nib.Nifti1Image, label_mapping: Dict[str, Any]) -> nib.Nifti1Image:
    """Remap raw label IDs to target schema IDs."""

    if not label_mapping:
        return label_img

    src_to_dst = {int(src): int(dst) for src, dst in label_mapping.items()}
    data = np.asanyarray(label_img.dataobj)
    remapped = np.zeros_like(data, dtype=np.int16)

    for src, dst in src_to_dst.items():
        remapped[data == src] = dst

    return nib.Nifti1Image(remapped, affine=label_img.affine, header=label_img.header)


def _materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError:
            # Windows often blocks symlink creation without proper permissions.
            shutil.copy2(src, dst)
            return

    if mode == "none":
        return

    raise PipelineError(f"Unsupported materialize mode: {mode}")


def materialize_dataset(
    df: pd.DataFrame,
    config: Dict[str, Any],
    processed_dirs: Dict[str, Path],
    assignments: Dict[str, str],
    mode: str,
    apply_label_mapping: bool,
) -> pd.DataFrame:
    """Copy/symlink and optionally remap labels into standardized processed structure."""

    dataset_cfg_map = {d["name"]: d for d in config.get("datasets", [])}
    image_split_dir = {"train": "imagesTr", "val": "imagesVal", "test": "imagesTs"}
    label_split_dir = {"train": "labelsTr", "val": "labelsVal", "test": "labelsTs"}

    out_rows: List[Dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        row = dict(row)
        uid = row["case_uid"]
        split = assignments.get(uid, "excluded")
        row["split"] = split
        row["processed_image_path"] = None
        row["processed_label_path"] = None

        if split == "excluded":
            out_rows.append(row)
            continue

        img_src = Path(row["image_path"])
        lbl_src = Path(row["label_path"])

        safe_case = f"{row['dataset']}__{row['case_id']}"
        img_dst = processed_dirs[image_split_dir[split]] / f"{safe_case}.nii.gz"
        lbl_dst = processed_dirs[label_split_dir[split]] / f"{safe_case}.nii.gz"

        _materialize_file(img_src, img_dst, mode)

        dcfg = dataset_cfg_map[row["dataset"]]
        label_mapping = dcfg.get("label_mapping", {})

        if mode != "none":
            if apply_label_mapping and label_mapping:
                ok, label_img, err = _safe_load_nifti(lbl_src)
                if not ok or label_img is None:
                    row["errors"] = list(row["errors"]) + [f"label_remap_read_error: {err}"]
                else:
                    remapped = remap_label_image(label_img, label_mapping)
                    nib.save(remapped, str(lbl_dst))
            else:
                _materialize_file(lbl_src, lbl_dst, mode)

            row["processed_image_path"] = str(img_dst)
            row["processed_label_path"] = str(lbl_dst)

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def save_manifests(df: pd.DataFrame, manifests_dir: Path) -> Tuple[Path, Path]:
    """Write per-case manifest to CSV and JSON."""

    csv_path = manifests_dir / "cases_manifest.csv"
    json_path = manifests_dir / "cases_manifest.json"

    df_out = df.copy()
    for col in ["shape", "spacing", "unique_labels", "unexpected_labels", "errors"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, tuple)) else x)

    df_out.to_csv(csv_path, index=False)

    records = df.to_dict(orient="records")
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    return csv_path, json_path


def report_issues(df: pd.DataFrame) -> str:
    """Generate and print a clean report of quality and pairing issues."""

    total = len(df)
    usable = int(df["is_usable"].sum())
    missing = int(df["missing_label"].sum())
    image_fail = int((~df["image_read_ok"]).sum())
    label_fail = int((~df["label_read_ok"] & ~df["missing_label"]).sum())
    shape_mismatch = int(df["shape_mismatch"].sum())

    with_unexpected = df["unexpected_labels"].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()

    lines = [
        "=" * 80,
        "Step 2 Report: Dataset Assembly and Cleaning",
        "=" * 80,
        f"Total discovered cases: {total}",
        f"Usable cases: {usable}",
        f"Missing image-mask pairs: {missing}",
        f"Unreadable images: {image_fail}",
        f"Unreadable masks: {label_fail}",
        f"Image/label shape mismatches: {shape_mismatch}",
        f"Cases with unexpected labels: {int(with_unexpected)}",
        "-" * 80,
    ]

    problem_df = df[df["errors"].apply(lambda errs: bool(errs))]
    if not problem_df.empty:
        lines.append("Problem cases:")
        for row in problem_df.itertuples(index=False):
            lines.append(f"- {row.case_uid}: {'; '.join(row.errors)}")
    else:
        lines.append("No problems found.")

    split_counts = (
        df[df["split"].isin(["train", "val", "test"])]
        .groupby("split")["case_uid"]
        .count()
        .to_dict()
        if "split" in df.columns
        else {}
    )
    if split_counts:
        lines.append("-" * 80)
        lines.append("Split summary:")
        for split_name in ["train", "val", "test"]:
            lines.append(f"- {split_name}: {split_counts.get(split_name, 0)}")

    report = "\n".join(lines)
    print(report)
    return report
