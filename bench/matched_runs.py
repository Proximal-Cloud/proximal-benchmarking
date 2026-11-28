"""Alternate two deployments over independent seeds, retaining each burst separately."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from llm_evidence import save_json, write_tables, report_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", required=True, help="JSON file with a list of deployment-specific CLI arguments"
    )
    parser.add_argument(
        "--candidate", required=True, help="JSON file with a list of deployment-specific CLI arguments"
    )
    parser.add_argument("--pairs", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory for this experiment")
    parser.add_argument(
        "shared", nargs=argparse.REMAINDER, help="-- followed by common standard_bench.py options"
    )
    a = parser.parse_args()
    if a.pairs < 1:
        parser.error("--pairs must be positive")
    deployments = {}
    shared = a.shared[1:] if a.shared[:1] == ["--"] else a.shared
    for name in ("baseline", "candidate"):
        args = json.loads(Path(getattr(a, name)).read_text())
        if not isinstance(args, list) or not all(isinstance(s, str) for s in args):
            parser.error(f"{name} must be a JSON list of strings")
        if any(s.split("=")[0] in {"--out", "--seed"} for s in args + shared):
            parser.error("the matched runner assigns output paths and seeds")
        deployments[name] = args
    a.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"requested_pairs": a.pairs, "runs": []}
    table = []
    failed = False
    for pair in range(a.pairs):
        seed = a.seed + pair
        order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        for name in order:
            output = a.out_dir / f"pair{pair + 1}-{name}.json"
            command = [
                sys.executable,
                str(Path(__file__).with_name("standard_bench.py")),
                *shared,
                *deployments[name],
                "--seed",
                str(seed),
                "--out",
                str(output),
            ]
            entry = {
                "pair": pair + 1,
                "deployment": name,
                "seed": seed,
                "output": output.name,
                "command": command,
            }
            manifest["runs"].append(entry)
            save_json(a.out_dir / "manifest.json", manifest)
            entry["exit_code"] = subprocess.run(command).returncode
            failed = failed or bool(entry["exit_code"])
            if output.exists():
                result = json.loads(output.read_text())
                for row in report_rows(result):
                    table.append({"Pair": pair + 1, "Deployment": name, **row})
                write_tables(a.out_dir / "comparison", table)
            save_json(a.out_dir / "manifest.json", manifest)
        if failed:
            break
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
