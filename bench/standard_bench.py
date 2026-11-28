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
import asyncio
import math
import signal
import re
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GpuSampler, measurement, run_record, load_yaml, resolve_model_dir  # noqa: E402

from cli_runner import model_info, run_cli
from llm_evidence import RunEvidence, SweepBudget, file_digest, public_outcomes

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
    ap.add_argument("--gpu-count", type=int, default=None, help="override how many GPUs the deployment uses (for cost)")
    ap.add_argument("--request-rate", type=float, default=float("inf"),
                    help="offered requests per second (default: saturate the concurrency limit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600, help="seconds allowed per level")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["public", "full-context"], default="public")
    ap.add_argument("--contexts", help="comma-separated total tokens per request for full-context mode")
    ap.add_argument("--output-tokens", type=int, default=512, help="forced decode length in full-context mode")
    ap.add_argument("--wall-budget", type=float, help="total seconds available for the sweep")
    ap.add_argument("--stall-timeout", type=float, help="seconds without observed token progress")
    ap.add_argument("--device-pci", default="", help="comma-separated serving PCI addresses for kernel health")
    ap.add_argument("--kernel-log", type=Path, help="append-only local kernel log in place of journalctl")
    ap.add_argument("--serving-config", type=Path, help="operator-supplied JSON with serving configuration and dtypes")
    ap.add_argument("--retrieval", action="store_true", help="score beginning, middle and end retrieval after a capacity pass")
    a = ap.parse_args()
    started = time.monotonic()
    for duration in (a.timeout, a.stall_timeout, a.wall_budget):
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            ap.error("time limits must be finite positive seconds")
    if a.num_prompts is not None and a.num_prompts < 1:
        ap.error("--num-prompts must be positive")
    a.device_pci = [s.strip().lower() for s in a.device_pci.split(",") if s.strip()]
    if any(not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", d) for d in a.device_pci):
        ap.error("--device-pci requires full PCI addresses")
    serving = json.loads(a.serving_config.read_text()) if a.serving_config else {}
    if not isinstance(serving, dict):
        ap.error("--serving-config must contain a JSON object")
    serving.update(device_pci=a.device_pci)
    if a.gpus:
        serving["device_ids"] = a.gpus
    if a.mode == "public" and (a.contexts or a.retrieval):
        ap.error("--contexts and --retrieval require --mode full-context")

    if not a.request_rate > 0:
        ap.error("--request-rate must be positive")
    run = run_record(a.dataset, {"all_planned_requests_complete": True},
                     {"concurrency": a.concurrency, "timeout_s_per_level": a.timeout})
    cfg = load_yaml("llm_targets.yaml")
    target = dict(cfg["models"][a.target])
    sb = cfg.get("standard_bench", {})
    if a.base_url:
        target["base_url"] = a.base_url
    if a.gpu_count is not None:
        target["gpus"] = a.gpu_count
    base = target["base_url"].rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    levels_c = [int(x) for x in (a.concurrency.split(",") if a.concurrency else sb.get("concurrency", [2, 4, 8, 16]))]
    if any(c < 1 for c in levels_c) or len(set(levels_c)) != len(levels_c):
        ap.error("concurrency levels must be distinct positive integers")
    run["stop_condition"].update(concurrency=levels_c, wall_budget_s=a.wall_budget, stall_timeout_s=a.stall_timeout)
    measurements = []
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

    if a.mode == "full-context":
        from context_bench import run_context
        if not a.contexts or a.dataset != "sharegpt" or a.num_prompts or math.isfinite(a.request_rate):
            ap.error("full-context requires --contexts, ShareGPT and one simultaneous request per user")
        try:
            a.contexts = [int(v) for v in a.contexts.split(",")]
        except ValueError:
            ap.error("--contexts expects comma-separated integer token limits")
        if (len(set(a.contexts)) != len(a.contexts) or a.output_tokens < 2
                or any(c <= a.output_tokens for c in a.contexts)):
            ap.error("contexts must be distinct and exceed --output-tokens (minimum 2)")
        if not asyncio.run(run_context(a, target, levels_c, path, serving)):
            sys.exit("Full-context sweep stopped; see the saved evidence")
        return

    tok = a.tokenizer or resolve_model_dir(target["repo"])
    vllm = shlex.split(a.vllm) if a.vllm else [default_vllm()]
    levels = []
    out = Path(a.out or f"results/public_{a.target}_{a.dataset}.json")
    payload = {"bench": "standard", "run": run, "measurements": measurements,
               "request_rate": str(a.request_rate), "seed": a.seed, "tool": "vllm bench serve",
               "target_name": a.target, "target": target, "serving": serving, "dataset": a.dataset,
               "dataset_desc": desc, "tokenizer": tok, "levels": levels}
    if a.dataset == "sharegpt":
        payload["dataset_sha256"] = file_digest(path)
    payload["server_model"] = model_info(base, target["model"])
    evidence = RunEvidence(out, payload, sys.argv)
    budget = SweepBudget(a.wall_budget, started)
    slo = load_yaml("spec.yaml")["slo"]
    gpu = GpuSampler(a.gpus.split(",") if a.gpus else None).start()
    try:
        for c in levels_c:
            n = a.num_prompts or min(1200, max(100, c * per_c))
            if not budget.allows(n, f"Concurrency {c}"):
                measurements.append(measurement(f"concurrency {c}", 0, 0, 0, planned=n,
                                                checks={"all_completed": None}, stop_reason="wall_budget"))
                break
            tmp = evidence.root / f"c{c}"
            tmp.mkdir()
            fn = "raw.json"
            cmd = vllm + ["bench", "serve", "--backend", "openai", "--base-url", base.removesuffix("/v1"), "--endpoint", "/v1/completions",
                          "--model", target["model"], "--tokenizer", tok, *ds_args,
                          "--num-prompts", str(n), "--max-concurrency", str(c), "--request-rate", str(a.request_rate),
                          "--ignore-eos", "--temperature", "0", "--seed", str(a.seed),
                          "--num-warmups", str(min(c, 16)),
                          "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", PCTS,
                          "--save-result", "--save-detailed", "--result-dir", str(tmp), "--result-filename", fn, "--disable-tqdm"]
            print(f"\n=== standard:{a.dataset}  {a.target}  concurrency={c}  prompts={n} ===", flush=True)
            t0 = time.time()
            reason = run_cli(cmd, tmp, base, target["model"], min(a.timeout, budget.remaining), a.stall_timeout)
            t1 = time.time()
            if reason != "completed" or not (tmp / fn).is_file():
                measurements.append(measurement(f"concurrency {c}", None, None, t1-t0, planned=n,
                                                checks={"all_completed": False},
                                                stop_reason="runner_failure" if reason=="completed" else reason))
                sampled = json.loads((tmp/"scheduler.json").read_text())["last_observed"]
                measurements[-1].update(active_at_last_sample=sampled.get("num_requests_running"),
                                        queued_at_last_sample=sampled.get("num_requests_waiting"))
                print(f"Stopped at {c} users ({reason}); client log: {tmp/'client.log'}")
                break
            budget.completed(n, t1-t0)
            # Some clients emit NaN for undefined percentiles when all requests fail.
            r = json.loads((tmp / fn).read_text(), parse_constant=lambda _: None)
            lv = {"concurrency": c, "num_prompts": n, "duration_s": r.get("duration"),
                  "completed": r.get("completed"), "failed": r.get("failed", 0),
                  "total_input_tokens": r.get("total_input_tokens"), "total_output_tokens": r.get("total_output_tokens"),
                  "request_throughput": r.get("request_throughput"), "output_throughput": r.get("output_throughput"),
                  "total_token_throughput": r.get("total_token_throughput"),
                  "max_concurrent_requests": r.get("max_concurrent_requests"),
                  "ttft_ms": stats(r, "ttft"), "tpot_ms": stats(r, "tpot"), "itl_ms": stats(r, "itl"),
                  "e2el_ms": stats(r, "e2el"), "gpu": gpu.window(t0, t1)}
            lv["stop_reason"] = "load_generator_finished"
            lv["server_limit"] = payload["server_model"].get("max_model_len")
            lv.update(public_outcomes(r, c, n, slo))
            completed, failed = lv["completed"], lv["failed"]
            issued = completed + failed if completed is not None else None
            measurements.append(measurement(f"concurrency {c}", issued, completed, lv["duration_s"], planned=n,
                                            checks={"all_completed": completed == n and failed == 0},
                                            stop_reason="load_generator_finished"))
            measurements[-1].update(output_tok_s=lv["output_throughput"], p95_ttft_ms=lv["ttft_ms"]["p95"],
                                     p95_e2e_ms=lv["e2el_ms"]["p95"])
            if lv["joint_passed"] is not None and lv["duration_s"]:
                measurements[-1]["joint_passes_per_min"] = 60*lv["joint_passed"]/lv["duration_s"]
            levels.append(lv)
            evidence.checkpoint()
            mi = (lv["total_input_tokens"] or 0) / max(1, lv["completed"] or 1)
            mo = (lv["total_output_tokens"] or 0) / max(1, lv["completed"] or 1)
            print(f"  {lv['output_throughput']} out tok/s | {lv['request_throughput']} req/s | "
                  f"mean {mi:.0f} in / {mo:.0f} out tokens | TTFT p95 {(lv['ttft_ms']['p95'] or 0) / 1000:.2f}s | "
                  f"TPOT p95 {lv['tpot_ms']['p95'] or 0:.1f}ms | failed {lv['failed']}", flush=True)
    finally:
        gpu.stop()
        run["finished_at"] = time.time()
        evidence.checkpoint()
    if any(m["status"] != "qualified" for m in measurements):
        sys.exit("Benchmark incomplete: one or more levels did not complete all planned requests")
    if len(levels) != len(levels_c):
        sys.exit(f"Only {len(levels)} of {len(levels_c)} benchmark levels completed")


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    main()
