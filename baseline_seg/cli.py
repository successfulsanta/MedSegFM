from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline_seg.config import (
    apply_overrides,
    fairness_fingerprint,
    load_config,
    load_label_map,
    resolve_device,
    resolve_num_classes,
)
from baseline_seg.compare import (
    build_adaptation_comparison_table,
    build_per_organ_comparison_table,
    build_split_comparison_table,
    build_mode_comparison_table,
    build_prompt_comparison_table,
)
from baseline_seg.experiment_manager import (
    best_val_from_history,
    build_fair_comparison_table,
    enforce_fairness_lock,
    save_run_summary,
)
from baseline_seg.final_hybrid import (
    build_final_hybrid_config,
    export_ablation_insights,
    export_final_package,
    final_results_table,
    select_components_from_tables,
)
from baseline_seg.pipeline import (
    evaluate_split,
    infer_split,
    train_experiment,
    validate_experiment,
    visualize_predictions,
)
from baseline_seg.review_failures import write_failure_review_markdown
from baseline_seg.ssl_pretrain import run_ssl_pretraining
from baseline_seg.utils import collect_env_info, save_json, set_seed, setup_experiment_dirs, setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline multi-organ CT segmentation pipeline (train/val/infer/evaluate/visualize)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_cmd = sub.add_parser("train", help="Train baseline model.")
    train_cmd.add_argument("--config", type=Path, required=True, help="Path to experiment config.")
    train_cmd.add_argument("--resume", type=Path, default=None, help="Optional checkpoint path to resume training.")
    train_cmd.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values using dot-path key=value, e.g. train.max_epochs=100",
    )

    val_cmd = sub.add_parser("validate", help="Validate a trained checkpoint.")
    val_cmd.add_argument("--config", type=Path, required=True)
    val_cmd.add_argument("--checkpoint", type=Path, required=True)
    val_cmd.add_argument("--override", nargs="*", default=[])

    infer_cmd = sub.add_parser("infer", help="Run inference for a data split.")
    infer_cmd.add_argument("--config", type=Path, required=True)
    infer_cmd.add_argument("--checkpoint", type=Path, required=True)
    infer_cmd.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    infer_cmd.add_argument("--output-dir", type=Path, default=None)
    infer_cmd.add_argument("--override", nargs="*", default=[])

    eval_cmd = sub.add_parser("evaluate", help="Evaluate a predictions directory.")
    eval_cmd.add_argument("--config", type=Path, required=True)
    eval_cmd.add_argument("--pred-dir", type=Path, required=True)
    eval_cmd.add_argument("--split", type=str, default="test")
    eval_cmd.add_argument("--gt-dir", type=Path, default=None, help="Optional GT directory for custom/cross-dataset evaluation.")
    eval_cmd.add_argument("--image-dir", type=Path, default=None, help="Optional image directory used for qualitative figures.")
    eval_cmd.add_argument("--prob-dir", type=Path, default=None, help="Optional probability directory (.npz/.npy) for calibration metrics.")
    eval_cmd.add_argument("--vis-dir", type=Path, default=None, help="Optional output directory for qualitative evaluation panels.")
    eval_cmd.add_argument("--max-vis-cases", type=int, default=6)
    eval_cmd.add_argument("--failure-k", type=int, default=10)
    eval_cmd.add_argument(
        "--suspicious-cases",
        nargs="*",
        default=[],
        help="Optional list of case IDs to force-include in hard-case review.",
    )
    eval_cmd.add_argument("--output-csv", type=Path, required=True)
    eval_cmd.add_argument("--override", nargs="*", default=[])

    vis_cmd = sub.add_parser("visualize", help="Visualize predictions against GT.")
    vis_cmd.add_argument("--config", type=Path, required=True)
    vis_cmd.add_argument("--pred-dir", type=Path, required=True)
    vis_cmd.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    vis_cmd.add_argument("--output-dir", type=Path, required=True)
    vis_cmd.add_argument("--max-cases", type=int, default=3)
    vis_cmd.add_argument("--override", nargs="*", default=[])

    pre_cmd = sub.add_parser("pretrain-ssl", help="Run SSL pretraining on unlabeled CT volumes.")
    pre_cmd.add_argument("--config", type=Path, required=True)
    pre_cmd.add_argument("--override", nargs="*", default=[])

    cmp_cmd = sub.add_parser("compare-table", help="Build comparison table from mode evaluation CSV files.")
    cmp_cmd.add_argument("--scratch-csv", type=Path, required=True)
    cmp_cmd.add_argument("--supervised-csv", type=Path, required=True)
    cmp_cmd.add_argument("--ssl-csv", type=Path, required=True)
    cmp_cmd.add_argument("--output-csv", type=Path, required=True)

    ada_cmd = sub.add_parser("compare-adaptation", help="Build adaptation comparison table with efficiency stats.")
    ada_cmd.add_argument("--full-ft-csv", type=Path, required=True)
    ada_cmd.add_argument("--lora-csv", type=Path, required=True)
    ada_cmd.add_argument("--adapters-csv", type=Path, required=True)
    ada_cmd.add_argument("--full-ft-eff", type=Path, default=None)
    ada_cmd.add_argument("--lora-eff", type=Path, default=None)
    ada_cmd.add_argument("--adapters-eff", type=Path, default=None)
    ada_cmd.add_argument("--output-csv", type=Path, required=True)

    prm_cmd = sub.add_parser("compare-prompting", help="Build prompt strategy comparison table.")
    prm_cmd.add_argument("--none-csv", type=Path, required=True)
    prm_cmd.add_argument("--spatial-csv", type=Path, required=True)
    prm_cmd.add_argument("--text-csv", type=Path, required=True)
    prm_cmd.add_argument("--output-csv", type=Path, required=True)

    fair_cmd = sub.add_parser("fair-compare", help="Build comparison table using only fairness-valid runs.")
    fair_cmd.add_argument("--summary-csv", type=Path, required=True)
    fair_cmd.add_argument("--comparison-group", type=str, required=True)
    fair_cmd.add_argument("--output-csv", type=Path, required=True)

    split_cmp_cmd = sub.add_parser("compare-eval", help="Build split-wise comparison tables (val/test/cross).")
    split_cmp_cmd.add_argument("--val-csvs", nargs="*", default=[], help="List of variant=csv_path items for validation set.")
    split_cmp_cmd.add_argument("--test-csvs", nargs="*", default=[], help="List of variant=csv_path items for test set.")
    split_cmp_cmd.add_argument("--cross-csvs", nargs="*", default=[], help="List of variant=csv_path items for cross-dataset test set.")
    split_cmp_cmd.add_argument("--output-dir", type=Path, required=True)

    organ_cmp_cmd = sub.add_parser("compare-organs", help="Build per-organ comparison table across variants.")
    organ_cmp_cmd.add_argument(
        "--organ-csvs",
        nargs="+",
        required=True,
        help="List of variant=per_organ_csv items (e.g. unet=..._per_organ.csv)",
    )
    organ_cmp_cmd.add_argument("--output-csv", type=Path, required=True)

    sel_final_cmd = sub.add_parser("select-final", help="Select best components and build final hybrid config.")
    sel_final_cmd.add_argument("--base-config", type=Path, required=True)
    sel_final_cmd.add_argument("--output-config", type=Path, required=True)
    sel_final_cmd.add_argument("--output-json", type=Path, required=True)
    sel_final_cmd.add_argument("--pretraining-csv", type=Path, default=None)
    sel_final_cmd.add_argument("--adaptation-csv", type=Path, default=None)
    sel_final_cmd.add_argument("--prompting-csv", type=Path, default=None)
    sel_final_cmd.add_argument("--architecture-csv", type=Path, default=None)
    sel_final_cmd.add_argument("--run-summaries-csv", type=Path, default=None)
    sel_final_cmd.add_argument("--pretrained-path", type=Path, default=None)
    sel_final_cmd.add_argument("--experiment-name", type=str, default="final_hybrid")

    cmp_final_cmd = sub.add_parser("compare-final", help="Build final-vs-baseline summary table.")
    cmp_final_cmd.add_argument("--final-csv", type=Path, required=True)
    cmp_final_cmd.add_argument("--baseline-csv", type=Path, default=None)
    cmp_final_cmd.add_argument("--output-csv", type=Path, required=True)
    cmp_final_cmd.add_argument("--model-name", type=str, default="final_hybrid")
    cmp_final_cmd.add_argument("--baseline-name", type=str, default="baseline")
    cmp_final_cmd.add_argument("--notes", type=str, default="")
    cmp_final_cmd.add_argument("--selection-json", type=Path, default=None)

    pkg_cmd = sub.add_parser("package-final", help="Package final model checkpoint/config and inference scripts.")
    pkg_cmd.add_argument("--checkpoint", type=Path, required=True)
    pkg_cmd.add_argument("--config", type=Path, required=True)
    pkg_cmd.add_argument("--output-dir", type=Path, required=True)
    pkg_cmd.add_argument("--package-name", type=str, default="final_hybrid_model")

    review_cmd = sub.add_parser("review-failures", help="Generate markdown review report from evaluation summary JSON.")
    review_cmd.add_argument("--summary-json", type=Path, required=True)
    review_cmd.add_argument("--output-md", type=Path, required=True)
    review_cmd.add_argument("--top-n", type=int, default=20)

    return parser


def _parse_named_csv_items(items):
    mapping = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid CSV mapping '{raw}'. Use variant=path.")
        key, value = raw.split("=", 1)
        mapping[str(key).strip()] = Path(value).resolve()
    return mapping


def _bootstrap(config_path: Path, overrides):
    cfg = load_config(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    label_map = load_label_map(Path(cfg["data"]["label_map_path"]).resolve())
    cfg["model"]["num_classes"] = resolve_num_classes(cfg, label_map)

    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))
    try:
        from monai.utils import set_determinism

        set_determinism(seed=int(cfg.get("seed", 42)))
    except Exception:
        pass

    adapt_mode = str(cfg.get("adaptation", {}).get("mode", "full_ft")).lower()
    prompt_mode = str(cfg.get("prompting", {}).get("mode", "none")).lower()
    prompt_dir = {"none": "no_prompt", "spatial": "spatial_prompt", "text": "text_prompt"}.get(prompt_mode, prompt_mode)
    exp_root = Path(cfg["logging"]["output_dir"]).resolve() / adapt_mode / prompt_dir

    exp_dirs = setup_experiment_dirs(
        exp_root,
        exp_name=str(cfg["logging"].get("experiment_name", "baseline")),
    )
    logger = setup_logger(exp_dirs["logs_dir"] / "run.log")
    save_json(exp_dirs["logs_dir"] / "resolved_config.json", cfg)

    device = resolve_device(str(cfg.get("device", "auto")))
    logger.info("Using device: %s", device)
    logger.info("Random seed: %d | deterministic=%s", int(cfg.get("seed", 42)), bool(cfg.get("deterministic", True)))

    env_info = collect_env_info()
    save_json(exp_dirs["logs_dir"] / "environment.json", env_info)
    logger.info("Environment: %s", env_info)

    fairness = fairness_fingerprint(cfg)
    save_json(exp_dirs["logs_dir"] / "fairness_fingerprint.json", fairness)
    logger.info("Fairness fingerprint: %s", fairness["fingerprint"])

    fairness = enforce_fairness_lock(
        cfg=cfg,
        output_root=Path(cfg["logging"]["output_dir"]).resolve(),
        exp_name=exp_dirs["exp_dir"].name,
        logger=logger,
    )
    save_json(exp_dirs["logs_dir"] / "fairness_check.json", fairness)

    return cfg, label_map, device, exp_dirs, logger, fairness


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "compare-table":
            table = build_mode_comparison_table(
                {
                    "scratch": args.scratch_csv.resolve(),
                    "supervised": args.supervised_csv.resolve(),
                    "ssl": args.ssl_csv.resolve(),
                },
                output_csv=args.output_csv.resolve(),
            )
            print(table.to_string(index=False) if not table.empty else "No rows to compare.")
            return

        if args.command == "compare-adaptation":
            table = build_adaptation_comparison_table(
                eval_csvs={
                    "full_ft": args.full_ft_csv.resolve(),
                    "lora": args.lora_csv.resolve(),
                    "adapters": args.adapters_csv.resolve(),
                },
                efficiency_jsons={
                    "full_ft": args.full_ft_eff.resolve() if args.full_ft_eff else None,
                    "lora": args.lora_eff.resolve() if args.lora_eff else None,
                    "adapters": args.adapters_eff.resolve() if args.adapters_eff else None,
                },
                output_csv=args.output_csv.resolve(),
            )
            print(table.to_string(index=False) if not table.empty else "No rows to compare.")
            return

        if args.command == "compare-prompting":
            table = build_prompt_comparison_table(
                prompt_csvs={
                    "none": args.none_csv.resolve(),
                    "spatial": args.spatial_csv.resolve(),
                    "text": args.text_csv.resolve(),
                },
                output_csv=args.output_csv.resolve(),
            )
            print(table.to_string(index=False) if not table.empty else "No rows to compare.")
            return

        if args.command == "fair-compare":
            table = build_fair_comparison_table(
                summary_csv=args.summary_csv.resolve(),
                comparison_group=args.comparison_group,
                output_csv=args.output_csv.resolve(),
            )
            print(table.to_string(index=False) if not table.empty else "No valid fair runs found.")
            return

        if args.command == "compare-eval":
            out_dir = args.output_dir.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            split_inputs = {
                "val": _parse_named_csv_items(args.val_csvs),
                "test": _parse_named_csv_items(args.test_csvs),
                "cross": _parse_named_csv_items(args.cross_csvs),
            }

            built = []
            for split_name, csvs in split_inputs.items():
                if not csvs:
                    continue
                table = build_split_comparison_table(
                    eval_csvs=csvs,
                    split_name=split_name,
                    output_csv=out_dir / f"compare_{split_name}.csv",
                )
                built.append((split_name, table))

            if not built:
                print("No split CSV mappings provided.")
            else:
                for split_name, table in built:
                    print(f"[{split_name}]")
                    print(table.to_string(index=False) if not table.empty else "No rows to compare.")
            return

        if args.command == "compare-organs":
            table = build_per_organ_comparison_table(
                per_organ_csvs=_parse_named_csv_items(args.organ_csvs),
                output_csv=args.output_csv.resolve(),
            )
            print(table.to_string(index=False) if not table.empty else "No per-organ rows to compare.")
            return

        if args.command == "select-final":
            base_cfg = load_config(args.base_config.resolve())
            selected = select_components_from_tables(
                pretraining_csv=args.pretraining_csv.resolve() if args.pretraining_csv else None,
                adaptation_csv=args.adaptation_csv.resolve() if args.adaptation_csv else None,
                prompting_csv=args.prompting_csv.resolve() if args.prompting_csv else None,
                architecture_csv=args.architecture_csv.resolve() if args.architecture_csv else None,
                run_summaries_csv=args.run_summaries_csv.resolve() if args.run_summaries_csv else None,
            )
            cfg = build_final_hybrid_config(
                base_cfg=base_cfg,
                selected=selected,
                output_config_path=args.output_config.resolve(),
                pretrained_weights_path=args.pretrained_path.resolve() if args.pretrained_path else None,
                experiment_name=args.experiment_name,
            )
            payload = {"selected_components": selected, "output_config": str(args.output_config.resolve()), "final_hybrid": cfg.get("final_hybrid", {})}
            args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
            args.output_json.resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(payload)
            return

        if args.command == "compare-final":
            table = final_results_table(
                final_eval_csv=args.final_csv.resolve(),
                baseline_eval_csv=args.baseline_csv.resolve() if args.baseline_csv else None,
                output_csv=args.output_csv.resolve(),
                model_name=args.model_name,
                baseline_name=args.baseline_name,
                notes=args.notes,
            )
            if args.selection_json:
                out_json = args.output_csv.resolve().with_name(f"{args.output_csv.resolve().stem}_ablation.json")
                out_csv = args.output_csv.resolve().with_name(f"{args.output_csv.resolve().stem}_ablation.csv")
                export_ablation_insights(
                    selection_json=args.selection_json.resolve(),
                    output_json=out_json,
                    output_csv=out_csv,
                )
            print(table.to_string(index=False) if not table.empty else "No rows to compare.")
            return

        if args.command == "package-final":
            out = export_final_package(
                checkpoint_path=args.checkpoint.resolve(),
                config_path=args.config.resolve(),
                output_dir=args.output_dir.resolve(),
                package_name=str(args.package_name),
            )
            print(out)
            return

        if args.command == "review-failures":
            out = write_failure_review_markdown(
                summary_json=args.summary_json.resolve(),
                output_md=args.output_md.resolve(),
                top_n=int(args.top_n),
            )
            print(out)
            return

        cfg, label_map, device, exp_dirs, logger, fairness = _bootstrap(args.config, getattr(args, "override", []))

        if args.command == "train":
            best = train_experiment(
                cfg,
                class_names=label_map,
                device=device,
                exp_dirs=exp_dirs,
                logger=logger,
                resume_checkpoint=args.resume.resolve() if args.resume else None,
            )
            logger.info("Best checkpoint: %s", best)
            val_metrics = best_val_from_history(exp_dirs["logs_dir"])
            save_run_summary(
                output_root=Path(cfg["logging"]["output_dir"]).resolve(),
                exp_dir=exp_dirs["exp_dir"],
                cfg=cfg,
                fairness=fairness,
                val_metrics=val_metrics,
                test_metrics=None,
            )
            return

        if args.command == "pretrain-ssl":
            p_mode = str(cfg.get("pretraining", {}).get("mode", "scratch")).lower()
            if p_mode != "ssl":
                logger.warning("pretrain-ssl called with mode=%s. Overriding to ssl for this run.", p_mode)
                cfg["pretraining"]["mode"] = "ssl"
            ckpt = run_ssl_pretraining(cfg=cfg, device=device, exp_dirs=exp_dirs, logger=logger)
            logger.info("SSL pretraining checkpoint: %s", ckpt)
            return

        if args.command == "validate":
            out = validate_experiment(
                cfg,
                class_names=label_map,
                device=device,
                exp_dirs=exp_dirs,
                logger=logger,
                checkpoint_path=args.checkpoint,
            )
            logger.info("Validation mean Dice: %.4f", out["val_dice"])
            return

        if args.command == "infer":
            output_dir = args.output_dir.resolve() if args.output_dir else exp_dirs["preds_dir"]
            pred_dir = infer_split(
                cfg,
                device=device,
                checkpoint_path=args.checkpoint,
                split=args.split,
                output_dir=output_dir,
                logger=logger,
            )
            logger.info("Inference output directory: %s", pred_dir)
            return

        if args.command == "evaluate":
            summary = evaluate_split(
                cfg,
                split=args.split,
                pred_dir=args.pred_dir.resolve(),
                out_csv=args.output_csv.resolve(),
                label_map=label_map,
                gt_dir=args.gt_dir.resolve() if args.gt_dir else None,
                image_dir=args.image_dir.resolve() if args.image_dir else None,
                probability_dir=args.prob_dir.resolve() if args.prob_dir else None,
                vis_dir=args.vis_dir.resolve() if args.vis_dir else None,
                max_visual_cases=int(args.max_vis_cases),
                failure_cases_k=int(args.failure_k),
                suspicious_cases=[str(x) for x in args.suspicious_cases],
            )
            logger.info("Evaluation summary: %s", summary)
            val_metrics = best_val_from_history(exp_dirs["logs_dir"])
            save_run_summary(
                output_root=Path(cfg["logging"]["output_dir"]).resolve(),
                exp_dir=exp_dirs["exp_dir"],
                cfg=cfg,
                fairness=fairness,
                val_metrics=val_metrics,
                test_metrics=summary,
            )
            return

        if args.command == "visualize":
            vis_dir = visualize_predictions(
                cfg,
                split=args.split,
                pred_dir=args.pred_dir.resolve(),
                output_dir=args.output_dir.resolve(),
                max_cases=int(args.max_cases),
            )
            logger.info("Saved visualizations to: %s", vis_dir)
            return

        raise RuntimeError(f"Unsupported command: {args.command}")
    except Exception as exc:
        raise SystemExit(f"Pipeline failed: {exc}") from exc


if __name__ == "__main__":
    main()
