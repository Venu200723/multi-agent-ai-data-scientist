"""CLI entrypoint: python -m multi_agent_ds.cli data/sample.csv [--target COL]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import Orchestrator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="multi_agent_ds",
        description="Run the Multi-Agent Autonomous AI Data Scientist pipeline on a CSV.",
    )
    p.add_argument("csv", type=str, help="Path to input CSV")
    p.add_argument("--target", type=str, default=None, help="Target column (auto-detected if omitted)")
    p.add_argument(
        "--report",
        type=str,
        default="examples/sample_report.md",
        help="Markdown report output path",
    )
    p.add_argument(
        "--artifacts",
        type=str,
        default="artifacts",
        help="Directory for plots / intermediate artifacts",
    )
    p.add_argument("--json-summary", action="store_true", help="Print JSON metrics summary")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    orch = Orchestrator(artifacts_dir=args.artifacts, report_path=args.report)
    result = orch.run(csv_path, target=args.target, report_path=args.report)

    metrics = result["ml"]["metrics"]
    report_md = result["report"]["markdown_path"]
    print("=" * 60)
    print("Multi-Agent AI Data Scientist — run complete")
    print("=" * 60)
    print(f"Target : {result['target']}")
    print(f"Task   : {result['ml']['task']}")
    print(f"Metrics: {metrics}")
    print(f"Report : {report_md}")
    print(f"HTML   : {result['report']['html_path']}")
    if args.json_summary:
        print(json.dumps({"target": result["target"], "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
