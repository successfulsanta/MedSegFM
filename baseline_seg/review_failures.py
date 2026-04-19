from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _to_rel(base: Path, target: str) -> str:
    p = Path(target)
    try:
        return str(p.relative_to(base)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def write_failure_review_markdown(summary_json: Path, output_md: Path, top_n: int = 20) -> Dict[str, Any]:
    """Create a quick markdown review report for worst cases and failure patterns."""

    if not summary_json.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_json}")

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    hard_cases_csv = Path(summary.get("hard_cases_csv", ""))
    failure_patterns_csv = Path(summary.get("failure_patterns_csv", ""))
    failure_details_csv = Path(summary.get("failure_details_csv", ""))

    hard_df = pd.read_csv(hard_cases_csv) if hard_cases_csv.exists() else pd.DataFrame()
    pat_df = pd.read_csv(failure_patterns_csv) if failure_patterns_csv.exists() else pd.DataFrame()
    detail_df = pd.read_csv(failure_details_csv) if failure_details_csv.exists() else pd.DataFrame()

    output_md.parent.mkdir(parents=True, exist_ok=True)
    base = output_md.parent.resolve()

    lines = []
    lines.append("# Failure Review")
    lines.append("")
    lines.append(f"- Split: {summary.get('split', 'unknown')}")
    lines.append(f"- Cases: {summary.get('n_cases', 'unknown')}")
    lines.append(f"- Macro Dice: {summary.get('macro_dice', 'nan')}")
    lines.append(f"- Macro HD95: {summary.get('macro_hd95', 'nan')}")
    lines.append("")

    lines.append("## Top Hard Cases")
    if hard_df.empty:
        lines.append("No hard-case rows found.")
    else:
        cols = [c for c in ["case_id", "dice_mean", "hd95_mean", "hardness_score", "is_suspicious_manual"] if c in hard_df.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in hard_df.head(max(1, int(top_n))).iterrows():
            vals = [str(row.get(c, "")) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## Common Failure Patterns")
    if pat_df.empty:
        lines.append("No failure pattern rows found.")
    else:
        cols = [c for c in ["failure_type", "count"] if c in pat_df.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in pat_df.head(max(1, int(top_n))).iterrows():
            vals = [str(row.get(c, "")) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## Representative Failure Visuals")
    vis = summary.get("representative_failure_visuals", []) or []
    if not vis:
        lines.append("No representative failure visuals available.")
    else:
        for v in vis:
            rel = _to_rel(base, v)
            lines.append(f"- [{rel}]({rel})")
    lines.append("")

    lines.append("## Organ-Focused Visuals")
    focus = summary.get("organ_focus_visuals", []) or []
    if not focus:
        lines.append("No organ-focused visuals available.")
    else:
        for v in focus:
            rel = _to_rel(base, v)
            lines.append(f"- [{rel}]({rel})")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "markdown": str(output_md),
        "hard_cases_rows": int(len(hard_df)),
        "patterns_rows": int(len(pat_df)),
        "details_rows": int(len(detail_df)),
    }
