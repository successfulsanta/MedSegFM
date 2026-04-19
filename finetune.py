from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baseline_seg.cli import main as baseline_main


def main() -> None:
    """Convenience entrypoint for fine-tuning segmentation with selected pretraining mode."""

    parser = argparse.ArgumentParser(description="Fine-tune segmentation model under standardized recipe.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    argv = ["baseline_seg.cli", "train", "--config", str(args.config)]
    if args.resume:
        argv.extend(["--resume", str(args.resume)])
    if args.override:
        argv.append("--override")
        argv.extend(args.override)
    sys.argv = argv
    baseline_main()


if __name__ == "__main__":
    main()
