"""Render public and full-context trial JSON files into a single Markdown table and CSV."""

import argparse
import json
from pathlib import Path

from llm_evidence import report_rows, write_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path, help="output filename stem")
    args = parser.parse_args()
    rows = []
    seen = set()
    for source in args.results:
        data = json.loads(source.read_text())
        if data.get("bench") not in {"standard", "full-context"}:
            parser.error(f"{source}: expected a public or full-context LLM benchmark")
        trial = data.get("run", {}).get("id") or str(source.resolve())
        if trial in seen:
            continue
        seen.add(trial)
        rows.extend(report_rows(data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_tables(args.out, rows)


if __name__ == "__main__":
    main()
