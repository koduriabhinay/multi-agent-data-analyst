#!/usr/bin/env python3
"""
Run an analysis from the command line — no server, no database.

    python -m app.cli data_samples/employees.csv
    python -m app.cli data.csv --out report.md

Useful for debugging agents and for CI, where spinning up a server just to
check the pipeline still works would be overkill.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.data.ingestion import IngestionError, read_path  # noqa: E402
from app.workflow.pipeline import run_pipeline  # noqa: E402
from app.workflow.state import new_state  # noqa: E402

STEP_LABELS = {
    "planner": "Planning",
    "cleaner": "Cleaning",
    "analyzer": "Analysing",
    "visualizer": "Charting",
    "reporter": "Writing up",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mada",
        description="Analyse a spreadsheet with a pipeline of agents.",
    )
    parser.add_argument("file", help="Path to a .csv, .tsv, .xlsx or .xls file")
    parser.add_argument("-o", "--out", help="Write the report here (default: print to stdout)")
    parser.add_argument("--charts", help="Directory to save chart JSON into")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show agent logs")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s — %(message)s",
    )

    try:
        df = read_path(args.file)
    except IngestionError as exc:
        print(f"Could not read that file: {exc}", file=sys.stderr)
        return 1

    print(f"Read {len(df):,} rows and {len(df.columns)} columns from {Path(args.file).name}\n")

    started = time.perf_counter()

    def on_progress(agent: str, percent: int) -> None:
        bar = "█" * (percent // 5) + "·" * (20 - percent // 5)
        label = STEP_LABELS.get(agent, agent)
        print(f"  {bar}  {percent:3d}%  {label}")

    state = run_pipeline(new_state("cli", Path(args.file).name, df), on_progress=on_progress)

    elapsed = time.perf_counter() - started

    if state.get("status") == "failed":
        print(f"\nAnalysis failed: {state.get('error')}", file=sys.stderr)
        return 1

    print(f"\nFinished in {elapsed:.1f}s — {len(state.get('charts', []))} charts\n")

    markdown = state["report"]["markdown"]

    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print("─" * 70)
        print(markdown)

    if args.charts:
        import json

        chart_dir = Path(args.charts)
        chart_dir.mkdir(parents=True, exist_ok=True)
        for i, chart in enumerate(state.get("charts", []), 1):
            slug = "".join(c if c.isalnum() else "_" for c in chart["title"]).lower()
            (chart_dir / f"{i:02d}_{slug}.json").write_text(
                json.dumps(chart["plotly_json"]), encoding="utf-8"
            )
        print(f"Charts written to {chart_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
