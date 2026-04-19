from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baseline_seg.cli import main as baseline_main


def main() -> None:
    """Compatibility entrypoint: train.py --config config.yaml."""

    parser = argparse.ArgumentParser(description="Train baseline segmentation model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key=value format, e.g. train.max_epochs=100",
    )
    args = parser.parse_args()

    argv = ["baseline_seg.cli", "train", "--config", str(args.config)]
    if args.resume is not None:
        argv.extend(["--resume", str(args.resume)])
    if args.override:
        argv.append("--override")
        argv.extend(args.override)

    sys.argv = argv
    baseline_main()


if __name__ == "__main__":
    main()
