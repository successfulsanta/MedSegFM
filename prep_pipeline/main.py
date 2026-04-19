from __future__ import annotations

import argparse
from pathlib import Path

from prep_pipeline.pipeline import (
    PipelineError,
    collect_records,
    ensure_processed_structure,
    load_config,
    materialize_dataset,
    report_issues,
    save_manifests,
    split_cases,
)
from prep_pipeline.visualize import visualize_random_cases
from prep_pipeline.step3_preprocess import (
    PreprocessError,
    load_config as load_preprocess_config,
    print_report as print_preprocess_report,
    run_preprocessing,
    save_metadata,
    visualize_random_cases as visualize_preprocessed_cases,
)


def run_pipeline(config_path: Path) -> None:
    """Execute Step 2: assemble and clean datasets."""

    workspace_root = Path.cwd()
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    split_ratios = config.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    materialize_mode = str(config.get("materialize_mode", "copy")).lower()
    apply_mapping = bool(config.get("apply_label_mapping", True))

    processed_dir = (workspace_root / config.get("processed_dir", "processed")).resolve()
    processed_dirs = ensure_processed_structure(processed_dir)

    print("[1/5] Scanning and validating image/mask pairs...")
    df = collect_records(config, workspace_root)

    print("[2/5] Creating deterministic train/val/test split...")
    assignments = split_cases(df, split_ratios=split_ratios, seed=seed)

    print(f"[3/5] Materializing dataset structure (mode={materialize_mode})...")
    df = materialize_dataset(
        df,
        config=config,
        processed_dirs=processed_dirs,
        assignments=assignments,
        mode=materialize_mode,
        apply_label_mapping=apply_mapping,
    )

    print("[4/5] Saving manifests...")
    csv_path, json_path = save_manifests(df, processed_dirs["manifests"])

    print("[5/5] Printing quality report...")
    report_issues(df)

    print("Done.")
    print(f"CSV manifest: {csv_path}")
    print(f"JSON manifest: {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2 and Step 3 pipeline for multi-organ CT segmentation data preparation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run", help="Run assembly and cleaning pipeline.")
    run_cmd.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON/YAML config describing datasets and mappings.",
    )

    vis_cmd = subparsers.add_parser("visual-check", help="Show random image/mask overlays.")
    vis_cmd.add_argument(
        "--manifest-csv",
        type=Path,
        required=True,
        help="Path to cases_manifest.csv",
    )
    vis_cmd.add_argument("--num-cases", type=int, default=3, help="Number of random cases to inspect.")
    vis_cmd.add_argument("--seed", type=int, default=42, help="Random seed for case sampling.")
    vis_cmd.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, plots are shown interactively.",
    )

    prep_cmd = subparsers.add_parser(
        "preprocess",
        help="Run standardized preprocessing on cleaned Step 2 outputs.",
    )
    prep_cmd.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional Step 3 preprocessing config (JSON or YAML). CLI flags override config values.",
    )
    prep_cmd.add_argument("--input_dir", type=Path, default=None, help="Input directory from Step 2.")
    prep_cmd.add_argument("--output_dir", type=Path, default=None, help="Output directory for Step 3.")
    prep_cmd.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Target voxel spacing in mm, e.g. 1.5 1.5 1.5.",
    )
    prep_cmd.add_argument(
        "--target_size",
        type=int,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional fixed 3D size, e.g. 128 128 128.",
    )
    prep_cmd.add_argument("--hu_min", type=float, default=None, help="Lower HU clip bound.")
    prep_cmd.add_argument("--hu_max", type=float, default=None, help="Upper HU clip bound.")
    prep_cmd.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["zscore", "minmax"],
        help="Intensity normalization mode.",
    )
    prep_cmd.add_argument(
        "--crop_mode",
        type=str,
        default=None,
        choices=["foreground", "center", "none"],
        help="Cropping strategy.",
    )
    prep_cmd.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    prep_cmd.add_argument(
        "--save_mode",
        type=str,
        default=None,
        choices=["save", "dry-run"],
        help="Whether to write output files or only simulate the run.",
    )

    prep_vis_cmd = subparsers.add_parser(
        "visual-check-preprocessed",
        help="Inspect random cases from preprocessed output with overlay plots.",
    )
    prep_vis_cmd.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to preprocessed output directory.",
    )
    prep_vis_cmd.add_argument("--num-cases", type=int, default=3, help="Number of random cases to inspect.")
    prep_vis_cmd.add_argument("--seed", type=int, default=42, help="Random seed for case sampling.")
    prep_vis_cmd.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, plots are shown interactively.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run":
            run_pipeline(args.config)
        elif args.command == "visual-check":
            visualize_random_cases(
                manifest_csv=args.manifest_csv,
                n_cases=args.num_cases,
                seed=args.seed,
                output_png=args.output_png,
            )
        elif args.command == "preprocess":
            config = load_preprocess_config(args.config) if args.config is not None else {}
            input_dir = Path(args.input_dir or config.get("input_dir", "processed")).resolve()
            output_dir = Path(args.output_dir or config.get("output_dir", "preprocessed")).resolve()
            target_spacing = args.target_spacing or config.get("target_spacing", [1.5, 1.5, 1.5])
            target_size = args.target_size or config.get("target_size", [128, 128, 128])
            hu_min = float(args.hu_min if args.hu_min is not None else config.get("hu_min", -200))
            hu_max = float(args.hu_max if args.hu_max is not None else config.get("hu_max", 300))
            normalization = str(args.normalization or config.get("normalization", "zscore"))
            crop_mode = str(args.crop_mode or config.get("crop_mode", "foreground"))
            seed = int(args.seed if args.seed is not None else config.get("seed", 42))
            save_mode = str(args.save_mode or config.get("save_mode", "save"))

            print("[1/4] Running standardized preprocessing...")
            df = run_preprocessing(
                input_dir=input_dir,
                output_dir=output_dir,
                target_spacing=target_spacing,
                target_size=target_size,
                hu_min=hu_min,
                hu_max=hu_max,
                normalization=normalization,
                crop_mode=crop_mode,
                seed=seed,
                save_mode=save_mode,
            )

            print("[2/4] Saving metadata...")
            manifests_dir = output_dir / "manifests"
            csv_path, json_path = save_metadata(df, manifests_dir)

            print("[3/4] Printing report...")
            print_preprocess_report(df)

            print("[4/4] Done.")
            print(f"CSV metadata: {csv_path}")
            print(f"JSON metadata: {json_path}")
        elif args.command == "visual-check-preprocessed":
            visualize_preprocessed_cases(
                output_dir=args.output_dir,
                n_cases=args.num_cases,
                seed=args.seed,
                save_png=args.output_png,
            )
        else:
            raise PipelineError(f"Unsupported command: {args.command}")
    except PipelineError as exc:
        print(f"PipelineError: {exc}")
        raise SystemExit(2) from exc
    except PreprocessError as exc:
        print(f"PreprocessError: {exc}")
        raise SystemExit(3) from exc
    except Exception as exc:
        print(f"Fatal error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
