#!/usr/bin/env python3
"""Stand-in for `vllm bench serve` in tests: writes a result JSON with the same keys
vLLM 0.23 saves (--save-result), with made-up numbers that scale with concurrency."""
import argparse
import json
import sys
from pathlib import Path

argv = sys.argv[1:]
assert argv[:2] == ["bench", "serve"], argv
ap = argparse.ArgumentParser()
for k in ("--max-concurrency", "--num-prompts", "--random-input-len", "--random-output-len"):
    ap.add_argument(k, type=int, default=None)
for k in ("--result-dir", "--result-filename", "--dataset-name", "--dataset-path", "--metric-percentiles", "--tokenizer", "--model"):
    ap.add_argument(k, default=None)
a, _ = ap.parse_known_args(argv[2:])
assert a.tokenizer and a.model, "tokenizer and model must be passed"
c, n = a.max_concurrency, a.num_prompts
il, ol = (a.random_input_len, a.random_output_len) if a.dataset_name == "random" else (230, 200)
tpot = 25 + c * 0.6                      # ms per token, grows with batch
out_tps = c * 1000 / tpot                # tokens/s across the batch
dur = n * ol / out_tps
ttft = 60 + c * 25                       # ms
r = {"duration": dur, "completed": n, "failed": 0, "total_input_tokens": n * il, "total_output_tokens": n * ol,
     "request_throughput": n / dur, "output_throughput": out_tps, "total_token_throughput": n * (il + ol) / dur,
     "max_concurrent_requests": c, "max_concurrency": c}
for m, base in (("ttft", ttft), ("tpot", tpot), ("itl", tpot), ("e2el", ttft + tpot * ol)):
    r[f"mean_{m}_ms"] = r[f"median_{m}_ms"] = base
    for p in (a.metric_percentiles or "99").split(","):
        r[f"p{p}_{m}_ms"] = base * (1 + int(p) / 200)
Path(a.result_dir).mkdir(parents=True, exist_ok=True)
(Path(a.result_dir) / a.result_filename).write_text(json.dumps(r))
print("Successful requests:", n)
