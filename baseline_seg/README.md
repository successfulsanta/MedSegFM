# Baseline 3D CT Segmentation Pipeline

This package provides a reproducible baseline for multi-organ CT segmentation on
NIfTI datasets prepared in Steps 2 and 3.

## Expected input layout

```
preprocessed/
  imagesTr/
  labelsTr/
  imagesVal/
  labelsVal/
  imagesTs/
  labelsTs/
  manifests/
```

## Why nearest-neighbor for masks?

Segmentation masks store integer class IDs. During geometric transforms,
nearest-neighbor interpolation preserves these discrete values. Linear
interpolation would create invalid fractional labels.

## Install dependencies

```bash
pip install torch monai nibabel numpy pandas scipy tqdm matplotlib pyyaml
```

## Standardized Recipe Features

- Centralized JSON/YAML config loaded at runtime
- Fixed seed + deterministic mode + environment capture
- Reusable MONAI augmentation recipe across experiments
- Configurable optimizer/scheduler/loss weights
- Early stopping + fixed validation cadence
- Best/last checkpoints with full training state
- Resume training from checkpoint
- CSV + JSON per-epoch metrics
- Optional TensorBoard logging
- Fairness fingerprint logging to compare run conditions

## Commands

```bash
python -m baseline_seg.cli train --config baseline_seg/config_example.json
python -m baseline_seg.cli train --config baseline_seg/config_standard.yaml --override train.max_epochs=100 optimizer.lr=2e-4
python -m baseline_seg.cli train --config baseline_seg/config_standard.yaml --resume experiments/exp_001_xxx/checkpoints/latest.pt
python -m baseline_seg.cli validate --config baseline_seg/config_example.json --checkpoint <best.pt>
python -m baseline_seg.cli infer --config baseline_seg/config_example.json --checkpoint <best.pt> --split test
python -m baseline_seg.cli evaluate --config baseline_seg/config_example.json --pred-dir <pred_dir> --split test --output-csv <metrics.csv>
python -m baseline_seg.cli visualize --config baseline_seg/config_example.json --pred-dir <pred_dir> --split test --output-dir <vis_dir> --max-cases 3
python -m baseline_seg.cli pretrain-ssl --config baseline_seg/config_compare_example.yaml --override pretraining.mode=ssl pretraining.method=masked_recon
python -m baseline_seg.cli compare-table --scratch-csv <scratch_eval.csv> --supervised-csv <supervised_eval.csv> --ssl-csv <ssl_eval.csv> --output-csv <comparison.csv>
```

Compatibility shortcut requested in the spec:

```bash
python train.py --config baseline_seg/config_standard.yaml
python train.py --config baseline_seg/config_standard.yaml --override train.max_epochs=50
python pretrain_ssl.py --config baseline_seg/config_compare_example.yaml --override pretraining.mode=ssl pretraining.method=denoise
python finetune.py --config baseline_seg/config_compare_example.yaml --override pretraining.mode=supervised pretraining.pretrained_path=path/to/supervised.pt
```

## Report

You can view the full detailed architectural and numerical findings in the attached PDF: [MedSegFM_Comprehensive_Detailed_Report.pdf](../MedSegFM_Comprehensive_Detailed_Report.pdf).

## Notes

- Reproducibility controls are enabled via seed + deterministic flags.
- Experiment folders follow `experiments/exp_001_...`, `experiments/exp_002_...`.
- Backbone can be swapped using `model.backbone` (`unet`, `unetr`, `swinunetr`).
- For fair comparisons, keep `standard_recipe`, `augment`, `optimizer`, `scheduler`, and `loss` unchanged unless that component is the variable under test.

## Pretraining Strategy Comparison

Modes are configured via `pretraining.mode`:

- `scratch`: random initialization.
- `supervised`: load `pretraining.pretrained_path` before fine-tuning.
- `ssl`: run SSL pretraining, then fine-tune using the saved SSL weights.

SSL methods (`pretraining.method`):

- `masked_recon`: MAE-style patch masking + reconstruction.
- `denoise`: noisy input reconstruction.
- `contrastive`: two-view InfoNCE using shared 3D backbone.

Experiment directories are grouped by mode automatically:

```
experiments/
  scratch/
  supervised/
  ssl/
```

Suggested fair-comparison sequence:

1. Scratch fine-tuning:
`python finetune.py --config baseline_seg/config_compare_example.yaml --override pretraining.mode=scratch`
2. Supervised transfer fine-tuning:
`python finetune.py --config baseline_seg/config_compare_example.yaml --override pretraining.mode=supervised pretraining.pretrained_path=path/to/supervised_weights.pt`
3. SSL pretraining:
`python pretrain_ssl.py --config baseline_seg/config_compare_example.yaml --override pretraining.mode=ssl pretraining.method=masked_recon`
4. SSL fine-tuning:
`python finetune.py --config baseline_seg/config_compare_example.yaml --override pretraining.mode=ssl pretraining.pretrained_path=path/to/ssl_pretrained.pt`
5. Build comparison table:
`python -m baseline_seg.cli compare-table --scratch-csv <scratch_eval.csv> --supervised-csv <supervised_eval.csv> --ssl-csv <ssl_eval.csv> --output-csv <comparison.csv>`

## Adaptation Method Comparison

Adaptation is controlled via `adaptation.mode`:

- `full_ft`: full fine-tuning (all parameters trainable)
- `lora`: LoRA-style low-rank residual updates with frozen backbone
- `adapters`: bottleneck adapters with frozen backbone
- `partial_decoder`: train decoder/head-like parameters only

Outputs are grouped by adaptation mode:

```
experiments/
  full_ft/
  lora/
  adapters/
```

Run examples:

1. Full fine-tuning:
`python finetune.py --config baseline_seg/config_adaptation_example.yaml --override adaptation.mode=full_ft adaptation.pretrained_path=path/to/pretrained.pt`
2. LoRA:
`python finetune.py --config baseline_seg/config_adaptation_example.yaml --override adaptation.mode=lora adaptation.lora_rank=4 adaptation.lora_alpha=1.0 adaptation.pretrained_path=path/to/pretrained.pt`
3. Adapters:
`python finetune.py --config baseline_seg/config_adaptation_example.yaml --override adaptation.mode=adapters adaptation.adapter_size=8 adaptation.pretrained_path=path/to/pretrained.pt`

Evaluate each run and then aggregate:

`python -m baseline_seg.cli compare-adaptation --full-ft-csv <fullft_eval.csv> --lora-csv <lora_eval.csv> --adapters-csv <adapters_eval.csv> --full-ft-eff <fullft_efficiency.json> --lora-eff <lora_efficiency.json> --adapters-eff <adapters_efficiency.json> --output-csv <adaptation_comparison.csv>`

Fairness checklist for adaptation studies:

- same dataset split directories
- same preprocessing output directory
- same pretrained initialization
- same augmentation/optimizer/scheduler/loss/training schedule
- only adaptation section changed

## Prompting Strategy Comparison

Prompt modes via `prompting.mode`:

- `none`: prompt-free segmentation.
- `spatial`: point/box prompt channels generated from GT (simulation).
- `text`: organ-name text prompt embeddings as conditioning channels.

Prompt levels:

- `none`: no prompts.
- `minimal`: sparse/low-information prompts.
- `full`: richer prompts.

Run examples (same weights/splits/preprocessing):

1. No prompt:
`python finetune.py --config baseline_seg/config_prompting_example.yaml --override prompting.mode=none adaptation.pretrained_path=path/to/pretrained.pt`
2. Spatial prompts:
`python finetune.py --config baseline_seg/config_prompting_example.yaml --override prompting.mode=spatial prompting.num_points=3 prompting.use_bboxes=true adaptation.pretrained_path=path/to/pretrained.pt`
3. Text prompts:
`python finetune.py --config baseline_seg/config_prompting_example.yaml --override prompting.mode=text prompting.text_embedding_dim=8 adaptation.pretrained_path=path/to/pretrained.pt`

Inference prompt-level testing:

`python -m baseline_seg.cli infer --config baseline_seg/config_prompting_example.yaml --checkpoint <best.pt> --split test --override prompting.mode=spatial prompting.inference_prompt_level=minimal`

Prompt comparison table:

`python -m baseline_seg.cli compare-prompting --none-csv <none_eval.csv> --spatial-csv <spatial_eval.csv> --text-csv <text_eval.csv> --output-csv <prompt_comparison.csv>`

## Fairness Enforcement Across Variants

Use `baseline_seg/config_fairness_template.yaml` as the source template for all runs.

During each run, the pipeline now:

- snapshots locked shared settings (seed/data split/preprocessing/augment/loss/optimizer/scheduler/schedule)
- compares against the reference for `standard_recipe.comparison_group`
- warns and logs exact differences if mismatched
- writes fairness results to each run folder (`logs/fairness_check.json`)
- updates global run summary index (`experiments/run_summaries.csv`)

Per-run summary files include:

- experiment name/path
- model variant descriptor
- locked settings snapshot
- fairness validity flag + diff count
- validation Dice
- test Dice
- test HD95

Build a comparison table using only fairness-valid runs:

`python -m baseline_seg.cli fair-compare --summary-csv experiments/run_summaries.csv --comparison-group all_variants_fair --output-csv experiments/fair_comparison.csv`

If shared settings mismatch, the run is flagged invalid for fair comparison.

## Final Hybrid Model Workflow

Use final-hybrid selection to combine the best pretraining/adaptation/prompting/backbone from prior comparison artifacts.

1. Select components and build final config:

`python -m baseline_seg.cli select-final --base-config baseline_seg/config_final_hybrid_template.yaml --output-config baseline_seg/config_final_hybrid_selected.json --output-json experiments/final_hybrid_selection.json --run-summaries-csv experiments/run_summaries.csv`

2. Train final model (optional: combine train+val):

`python -m baseline_seg.cli train --config baseline_seg/config_final_hybrid_selected.json --override data.combine_train_val_for_train=true`

3. Evaluate final model:

`python -m baseline_seg.cli evaluate --config baseline_seg/config_final_hybrid_selected.json --pred-dir <final_pred_dir> --split test --output-csv experiments/final_hybrid_eval_test.csv`

4. Compare final vs baseline:

`python -m baseline_seg.cli compare-final --final-csv experiments/final_hybrid_eval_test.csv --baseline-csv <baseline_eval.csv> --output-csv experiments/final_results_table.csv --notes "best components hybrid"`

5. Package model for reproducible inference:

`python -m baseline_seg.cli package-final --checkpoint <best.pt> --config baseline_seg/config_final_hybrid_selected.json --output-dir experiments/final_package --package-name final_hybrid_model`

The package exports:

- final checkpoint copy
- frozen config copy
- reusable Python and PowerShell inference scripts
