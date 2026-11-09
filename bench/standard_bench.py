#!/usr/bin/env python3
"""Public benchmark: short requests, the kind behind published throughput numbers.

Runs vLLM's own load generator, `vllm bench serve`, against a model server
that is already up (scripts/serve_llm.sh), at several concurrency levels, on:

  sharegpt  real ChatGPT conversations (ShareGPT_V3_unfiltered_cleaned_split.json).
            vLLM keeps prompts <= 1,024 tokens and prompt + reply <= 2,048.
  random    synthetic, vLLM's default shape: 1,024 tokens in, 128 out.

Each request is independent: no shared system prompt, no history. Compare its
cost per 1M output tokens with the tutoring-session benchmark (llm_sessions.py)
to see how much of the cost comes from our long-context workload rather than
from the hardware or the setup.

Replies are forced to the dataset's length (--ignore-eos) so every model is
measured on the same token counts.

  python bench/standard_bench.py --target gemma31b-w4 --dataset sharegpt --gpus 0
  python bench/standard_bench.py --target gemma31b-w4 --dataset random --concurrency 2,16
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GpuSampler, load_yaml, resolve_model_dir, write_result  # noqa: E402

PCTS = "50,90,95,99"


def default_vllm() -> str:
    venvs = os.environ.get("VENVS", "/data/tokenfactory/venvs")
    for c in (Path(venvs) / "vllm" / "bin" / "vllm", Path(sys.executable).with_name("vllm")):
        if c.exists():
            return str(c)
    return shutil.which("vllm") or "vllm"


def stats(r: dict, metric: str) -> dict:
    out = {"mean": r.get(f"mean_{metric}_ms")}
    for p in PCTS.split(","):
        out[f"p{p}"] = r.get(f"p{p}_{metric}_ms")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="model name under models: in config/llm_targets.yaml")
    ap.add_argument("--dataset", choices=["sharegpt", "random"], default="sharegpt")
    ap.add_argument("--dataset-path", default=None, help="ShareGPT json (default: standard_bench.datasets.sharegpt.path)")
    ap.add_argument("--concurrency", default=None, help="comma list (default: standard_bench.concurrency)")
    ap.add_argument("--num-prompts", type=int, default=None, help="prompts per level (default: concurrency x 25, 100-1200)")
    ap.add_argument("--base-url", default=None, help="override the target's base_url")
    ap.add_argument("--tokenizer", default=None, help="tokenizer folder or repo (default: the target's repo)")
    ap.add_argument("--vllm", default=None, help="command that runs the vllm CLI (default: $VENVS/vllm/bin/vllm)")
    ap.add_argument("--gpus", default=None, help="GPU indices to sample, e.g. 0 or 1,2")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds allowed per level")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_yaml("llm_targets.yaml")
    target = dict(cfg["models"][a.target])
    sb = cfg.get("standard_bench", {})
    if a.base_url:
        target["base_url"] = a.base_url
    base = target["base_url"].rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    levels_c = [int(x) for x in (a.concurrency.split(",") if a.concurrency else sb.get("concurrency", [2, 4, 8, 16]))]
    per_c = sb.get("prompts_per_concurrency", 25)
    ds_cfg = (sb.get("datasets") or {}).get(a.dataset, {})

    ds_args, desc = [], ""
    if a.dataset == "sharegpt":
        path = a.dataset_path or os.path.expandvars(ds_cfg.get("path", ""))
        if not path or not Path(path).exists():
            sys.exit(f"ShareGPT file not found: {path!r}. Copy ShareGPT_V3_unfiltered_cleaned_split.json "
                     "there (see README) or pass --dataset-path.")
        ds_args = ["--dataset-name", "sharegpt", "--dataset-path", path]
        desc = "ShareGPT: real chat conversations, prompts up to 1,024 tokens, replies at the dataset's length"
    else:
        il, ol = ds_cfg.get("input_len", 1024), ds_cfg.get("output_len", 128)
        ds_args = ["--dataset-name", "random", "--random-input-len", str(il), "--random-output-len", str(ol)]
        desc = f"Random: synthetic prompts, {il:,} tokens in, {ol:,} out"

    tok = a.tokenizer or resolve_model_dir(target["repo"])
    vllm = shlex.split(a.vllm) if a.vllm else [default_vllm()]
    gpu = GpuSampler(a.gpus.split(",") if a.gpus else None).start()
    levels = []
    tmp = Path(tempfile.mkdtemp(prefix="stdbench-"))
    try:
        for c in levels_c:
            n = a.num_prompts or min(1200, max(100, c * per_c))
            fn = f"c{c}.json"
            cmd = vllm + ["bench", "serve", "--backend", "openai", "--base-url", base, "--endpoint", "/v1/completions",
                          "--model", target["model"], "--tokenizer", tok, *ds_args,
                          "--num-prompts", str(n), "--max-concurrency", str(c), "--request-rate", "inf",
                          "--ignore-eos", "--temperature", "0", "--seed", "0",
                          "--num-warmups", str(min(c, 16)),
                          "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", PCTS,
                          "--save-result", "--result-dir", str(tmp), "--result-filename", fn, "--disable-tqdm"]
            print(f"\n=== standard:{a.dataset}  {a.target}  concurrency={c}  prompts={n} ===", flush=True)
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, timeout=a.timeout, capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                print(f"  timed out after {a.timeout:.0f}s; stopping the sweep")
                break
            t1 = time.time()
            if proc.returncode != 0 or not (tmp / fn).exists():
                print(f"  vllm bench serve failed (exit {proc.returncode}). Last output:\n"
                      + "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:]))
                if not levels:
                    print("  command was: " + " ".join(shlex.quote(x) for x in cmd))
                break
            r = json.loads((tmp / fn).read_text())
            lv = {"concurrency": c, "num_prompts": n, "duration_s": r.get("duration"),
                  "completed": r.get("completed"), "failed": r.get("failed", 0),
                  "total_input_tokens": r.get("total_input_tokens"), "total_output_tokens": r.get("total_output_tokens"),
                  "request_throughput": r.get("request_throughput"), "output_throughput": r.get("output_throughput"),
                  "total_token_throughput": r.get("total_token_throughput"),
                  "max_concurrent_requests": r.get("max_concurrent_requests"),
                  "ttft_ms": stats(r, "ttft"), "tpot_ms": stats(r, "tpot"), "itl_ms": stats(r, "itl"),
                  "e2el_ms": stats(r, "e2el"), "gpu": gpu.window(t0, t1)}
            levels.append(lv)
            mi = (lv["total_input_tokens"] or 0) / max(1, lv["completed"] or 1)
            mo = (lv["total_output_tokens"] or 0) / max(1, lv["completed"] or 1)
            print(f"  {lv['output_throughput']:.0f} out tok/s | {lv['request_throughput']:.2f} req/s | "
                  f"mean {mi:.0f} in / {mo:.0f} out tokens | TTFT p95 {(lv['ttft_ms']['p95'] or 0) / 1000:.2f}s | "
                  f"TPOT p95 {lv['tpot_ms']['p95'] or 0:.1f}ms | failed {lv['failed']}", flush=True)
    finally:
        gpu.stop()
        shutil.rmtree(tmp, ignore_errors=True)

    if not levels:
        sys.exit("no level completed")
    write_result(a.out or f"results/public_{a.target}_{a.dataset}.json", {
        "bench": "standard", "tool": "vllm bench serve", "target_name": a.target, "target": target,
        "dataset": a.dataset, "dataset_desc": desc, "tokenizer": tok, "levels": levels})


if __name__ == "__main__":
    main()
