from __future__ import annotations

import argparse
from pathlib import Path

from baseline_seg.review_failures import write_failure_review_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate quick markdown review for failure-case analysis outputs.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = write_failure_review_markdown(
        summary_json=args.summary_json.resolve(),
        output_md=args.output_md.resolve(),
        top_n=int(args.top_n),
    )
    print(out)


if __name__ == "__main__":
    main()
