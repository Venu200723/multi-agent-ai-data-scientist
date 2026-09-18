"""ReportAgent — assemble markdown (+ HTML) from agent outputs."""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ReportAgent:
    """Write a human-readable pipeline report."""

    output_path: str | Path = "examples/sample_report.md"
    log: list[str] = field(default_factory=list)

    def run(self, results: dict[str, Any], csv_name: str = "data.csv") -> dict[str, Any]:
        self.log = []
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cleaning = results.get("cleaning", {})
        eda = results.get("eda", {})
        ml = results.get("ml", {})
        explain = results.get("explain", {})

        lines: list[str] = []
        lines.append("# Multi-Agent AI Data Scientist — Pipeline Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
        lines.append(f"**Source CSV:** `{csv_name}`  ")
        lines.append(f"**Target:** `{ml.get('target', 'n/a')}`  ")
        lines.append(f"**Task:** `{ml.get('task', 'n/a')}`")
        lines.append("")
        lines.append("## 1. Cleaning Agent")
        lines.append("")
        lines.append(
            f"- Shape: `{cleaning.get('original_shape')}` → `{cleaning.get('cleaned_shape')}`"
        )
        for a in cleaning.get("actions", []):
            lines.append(f"- {a}")
        lines.append("")
        lines.append("## 2. EDA Agent")
        lines.append("")
        profile = eda.get("profile", {})
        lines.append(
            f"- Rows/cols: **{profile.get('n_rows')}** × **{profile.get('n_cols')}** "
            f"(numeric={profile.get('n_numeric')}, categorical={profile.get('n_categorical')})"
        )
        lines.append(f"- Missing overall: **{profile.get('missing_pct', 0):.2f}%**")
        if eda.get("target_summary"):
            lines.append(f"- Target profile: `{eda['target_summary']}`")
        if eda.get("top_correlations"):
            lines.append("- Top correlations:")
            for a, b, v in eda["top_correlations"][:5]:
                lines.append(f"  - `{a}` ↔ `{b}`: **{v:.3f}**")
        for fig in eda.get("figures", []):
            rel = Path(fig).name
            lines.append(f"- Figure: `{fig}`")
            lines.append(f"  ![{rel}]({fig})")
        lines.append("")
        lines.append("## 3. ML Agent")
        lines.append("")
        lines.append(
            f"- Train/test sizes: **{ml.get('n_train')}** / **{ml.get('n_test')}**"
        )
        metrics = ml.get("metrics", {})
        if metrics:
            lines.append("- Holdout metrics:")
            for k, v in metrics.items():
                lines.append(f"  - **{k}**: `{v:.4f}`")
        lines.append("")
        lines.append("## 4. Explain Agent")
        lines.append("")
        lines.append(explain.get("narrative", "_No narrative_"))
        lines.append("")
        if explain.get("importances"):
            lines.append("| Feature | Importance | Std |")
            lines.append("|---|---:|---:|")
            for row in explain["importances"]:
                lines.append(
                    f"| `{row['feature']}` | {row['importance']:.4f} | {row['std']:.4f} |"
                )
            lines.append("")
        for fig in explain.get("figures", []):
            rel = Path(fig).name
            lines.append(f"- Figure: `{fig}`")
            lines.append(f"  ![{rel}]({fig})")
        lines.append("")
        lines.append("## Agent action logs")
        lines.append("")
        for name in ("cleaning", "eda", "ml", "explain"):
            block = results.get(name, {})
            lines.append(f"### {name}")
            for a in block.get("actions", []):
                lines.append(f"- {a}")
            lines.append("")
        lines.append("---")
        lines.append("_Produced by `multi_agent_ds` — Multi-Agent Autonomous AI Data Scientist._")

        text = "\n".join(lines)
        path.write_text(text, encoding="utf-8")
        self.log.append(f"Wrote report → {path}")

        html_path = path.with_suffix(".html")
        html_path.write_text(_markdown_to_simple_html(text), encoding="utf-8")
        self.log.append(f"Wrote HTML report → {html_path}")

        return {
            "agent": "report",
            "markdown_path": str(path),
            "html_path": str(html_path),
            "actions": list(self.log),
        }


def _markdown_to_simple_html(md: str) -> str:
    """Minimal markdown → HTML for offline viewing (no external deps)."""
    body_parts: list[str] = []
    in_table = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if not in_table else "td"
            if not in_table:
                body_parts.append("<table>")
                in_table = True
            row = "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells)
            body_parts.append(f"<tr>{row}</tr>")
            continue
        if in_table and (not line.startswith("|") or "---" in line):
            if "---" in line:
                continue
            body_parts.append("</table>")
            in_table = False

        if not line:
            body_parts.append("<br/>")
            continue
        if line.startswith("# "):
            body_parts.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            body_parts.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("### "):
            body_parts.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("- "):
            body_parts.append(f"<li>{_inline(line[2:])}</li>")
        elif line.startswith("---"):
            body_parts.append("<hr/>")
        else:
            body_parts.append(f"<p>{_inline(line)}</p>")
    if in_table:
        body_parts.append("</table>")

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Pipeline Report</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,sans-serif;max-width:920px;"
        "margin:2rem auto;padding:0 1rem;line-height:1.55;color:#222}"
        "code{background:#f5f5f5;padding:1px 4px;border-radius:3px}"
        "table{border-collapse:collapse;margin:1rem 0;width:100%}"
        "td,th{border:1px solid #ccc;padding:.4rem .65rem;text-align:left}"
        "th{background:#f0f4f8} img{max-width:100%;height:auto;margin:.5rem 0}"
        "li{margin-left:1rem}"
        "</style></head><body>"
        + "\n".join(body_parts)
        + "</body></html>"
    )


def _inline(text: str) -> str:
    """Escape HTML then apply light inline markdown."""
    s = html_lib.escape(text)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2"/>', s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    return s
