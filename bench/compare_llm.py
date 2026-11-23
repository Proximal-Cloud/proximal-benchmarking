#!/usr/bin/env python3
"""Print a Markdown comparison of paired LLM benchmark runs."""
import argparse
from pathlib import Path

from common import load_yaml
from cost_report import load, session_point


def number(value):
    return f"{value:.2f}" if value is not None else "—"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Result directories in run order")
    parser.add_argument("--target", default="gemma31b-w4")
    args = parser.parse_args()
    spec = load_yaml("spec.yaml")
    public_levels = load_yaml("llm_targets.yaml")["standard_bench"]["concurrency"]
    print("| Run | Engine | Workload | Concurrency | Output tok/s | p95 TTFT (s) | p95 TPOT (ms) | Errors | Cache hit (%) | KV used (%) | Preemptions |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    operating_points = []
    for run in args.runs:
        engine = (run / "llm_engine").read_text().strip()
        quick = (run / "quick").read_text().strip() == "1"
        for workload, filename, concurrency_file, expected in (
            ("Socratic", f"socratic_{args.target}.json", "socratic_concurrency", [2, 4, 8, 16]),
            ("ShareGPT", f"public_{args.target}_sharegpt.json", "public_concurrency", public_levels),
        ):
            data = load(run / filename) or {"levels": []}
            levels = {level["concurrency"]: level for level in data["levels"]}
            socratic = workload == "Socratic"
            configured = run / concurrency_file
            if configured.is_file():
                expected = [int(value) for value in configured.read_text().strip().split(",")]
            elif quick:
                expected = [2, 8]
            for concurrency in expected:
                level = levels.get(concurrency)
                row = f"| {run.name} | {engine} | {workload} | {concurrency} |"
                if level is None:
                    print(f"{row} Missing | — | — | — | — | — | — |")
                    continue
                ttft = level["ttft_s" if socratic else "ttft_ms"]["p95"]
                if ttft is not None and not socratic:
                    ttft /= 1000
                throughput = level["output_tok_per_s" if socratic else "output_throughput"]
                errors = level["errors" if socratic else "failed"]
                cache = level.get("cache_hit_rate_usage")
                if cache is None:
                    cache = level.get("cache_hit_rate_metrics")
                kv = level.get("kv_cache_usage_max")
                values = [throughput, ttft, level["tpot_ms"]["p95"], errors,
                          None if cache is None else cache * 100,
                          None if kv is None else kv * 100, level.get("preemptions")]
                print(f"{row} " + " | ".join(map(number, values)) + " |")
            if socratic:
                point, meets_slo = session_point(data, spec["slo"])
                if meets_slo:
                    throughput = point["output_tok_per_s"]
                    gpu_seconds = spec["student_day"]["llm"]["output_tokens"] * data["target"]["gpus"] / throughput
                    summary = (f"{point['concurrency']} students, {throughput:.2f} output tok/s, "
                               f"{gpu_seconds:.0f} GPU-seconds per student-day")
                else:
                    summary = "no measured level meets the latency and error targets"
                operating_points.append(f"- {run.name} ({engine}) — {summary}.")
    print("\n### Socratic operating points\n")
    print("\n".join(operating_points))


if __name__ == "__main__":
    main()
