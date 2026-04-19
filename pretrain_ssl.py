from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baseline_seg.cli import main as baseline_main


def main() -> None:
    """Convenience entrypoint for SSL pretraining runs."""

    parser = argparse.ArgumentParser(description="SSL pretraining for 3D CT segmentation backbone.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    argv = ["baseline_seg.cli", "pretrain-ssl", "--config", str(args.config)]
    if args.override:
        argv.append("--override")
        argv.extend(args.override)
    sys.argv = argv
    baseline_main()


if __name__ == "__main__":
    main()
