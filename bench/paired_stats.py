"""Compare the same workload and load level across independent, matched pairs of runs."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def median_interval(values):
    """Two-sided, distribution-free interval for the population median of paired differences."""
    n = len(values)
    tail = 0
    bounds = None
    ordered = sorted(values)
    for k in range((n + 1) // 2):
        tail += math.comb(n, k)
        if 40 * tail > 2 ** n:  # each tail has probability at most 0.025
            break
        bounds = ordered[k], ordered[-k - 1]
    return bounds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", required=True, help="measurement name in each result JSON")
    ap.add_argument("--pair", action="append", nargs=2, required=True, metavar=("BASELINE", "CANDIDATE"))
    ap.add_argument("--metric", choices=["completed_per_min", "joint_passes_per_min", "output_tok_s", "p95_ttft_ms", "p95_e2e_ms"], default="completed_per_min")
    a = ap.parse_args()
    ids = set()
    differences = []
    protocol = None
    for files in a.pair:
        rates = []
        statuses = []
        seeds = []
        fingerprints = []
        for filename in files:
            data = json.loads(Path(filename).read_text())
            run = data.get("run", {})
            matches = [m for m in data.get("measurements", []) if m["name"] == a.level]
            if len(matches) != 1 or not run.get("id") or run["id"] in ids:
                ap.error(f"{filename}: need one matching level and a distinct run ID")
            ids.add(run["id"])
            row = matches[0]
            seeds.append(data.get("seed"))
            fingerprints.append(row.get("input_fingerprint"))
            settings = (data["bench"], run["workload"], run["criteria"], run["stop_condition"], row["unit"],
                        data.get("dataset"), data.get("dataset_sha256"), data.get("serving", {}).get("cache_state"), data.get("request_rate"), data.get("profile"), data.get("think_time_s"), row["planned"])
            if protocol is not None and settings != protocol:
                ap.error(f"{filename}: workload or measurement settings differ across runs")
            protocol = settings
            if not row["duration_s"] or row["completed"] is None:
                ap.error(f"{filename}: rate unavailable ({row['stop_reason']}). Include this failure in the comparison.")
            value = (60 * row["completed"] / row["duration_s"] if a.metric == "completed_per_min"
                     else row.get(a.metric))
            if value is None:
                ap.error(f"{filename}: {a.metric} unavailable")
            rates.append(value)
            statuses.append(row["status"])
        if fingerprints[0] != fingerprints[1]:
            ap.error("the input fingerprints differ within a pair")
        if seeds[0] != seeds[1]:
            ap.error("the baseline and candidate must use the same seed within each pair")
        differences.append(rates[1] - rates[0])
        print(f"Pair {len(differences)}: {rates[0]:.3f} ({statuses[0]}) -> {rates[1]:.3f} ({statuses[1]})")
    print(f"Median paired difference ({a.metric}, candidate minus baseline): {median(differences):+.3f}")
    bounds = median_interval(differences)
    if bounds:
        print(f"95% interval for the median difference: [{bounds[0]:+.3f}, {bounds[1]:+.3f}]")
    else:
        print("Fewer than six independent pairs: a finite two-sided 95% interval is unavailable.")


if __name__ == "__main__":
    main()
