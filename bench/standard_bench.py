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
from dataclasses import dataclass, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GpuSampler, measurement, run_record, load_yaml, resolve_model_dir  # noqa: E402

from cli_runner import model_info, run_cli
from llm_evidence import RunEvidence, SweepBudget, file_digest, public_outcomes

PERCENTILES = (50, 90, 95, 99)
LATENCY_METRICS = ("ttft", "tpot", "itl", "e2el")
RAW_RESULT = "raw.json"


@dataclass(frozen=True)
class PublicLoadPolicy:
    prompts_per_concurrency: int = 25
    min_prompts: int = 100
    max_prompts: int = 1200
    max_warmups: int = 16

    def __post_init__(self):
        if self.prompts_per_concurrency < 1 or not 1 <= self.min_prompts <= self.max_prompts:
            raise ValueError("public prompt counts must be positive and min_prompts <= max_prompts")
        if self.max_warmups < 0:
            raise ValueError("max_warmups must be nonnegative")

    @classmethod
    def from_config(cls, settings: dict) -> PublicLoadPolicy:
        overrides = {field.name: settings[field.name] for field in fields(cls) if field.name in settings}
        return cls(**overrides)

    def prompt_count(self, users: int) -> int:
        return min(self.max_prompts, max(self.min_prompts, users * self.prompts_per_concurrency))


@dataclass(frozen=True)
class PublicClient:
    executable: list[str]
    base_url: str
    model: str
    tokenizer: str
    dataset_args: list[str]
    request_rate: float
    seed: int

    def command(self, users: int, prompts: int, warmups: int, result_dir: Path) -> list[str]:
        return [
            *self.executable,
            "bench",
            "serve",
            "--backend",
            "openai",
            "--base-url",
            self.base_url,
            "--endpoint",
            "/v1/completions",
            "--model",
            self.model,
            "--tokenizer",
            self.tokenizer,
            *self.dataset_args,
            "--num-prompts",
            str(prompts),
            "--max-concurrency",
            str(users),
            "--request-rate",
            str(self.request_rate),
            "--ignore-eos",
            "--temperature",
            "0",
            "--seed",
            str(self.seed),
            "--num-warmups",
            str(warmups),
            "--percentile-metrics",
            ",".join(LATENCY_METRICS),
            "--metric-percentiles",
            ",".join(map(str, PERCENTILES)),
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(result_dir),
            "--result-filename",
            RAW_RESULT,
            "--disable-tqdm",
        ]


def default_vllm() -> str:
    venvs = os.environ.get("VENVS", "/data/tokenfactory/venvs")
    for c in (Path(venvs) / "vllm" / "bin" / "vllm", Path(sys.executable).with_name("vllm")):
        if c.exists():
            return str(c)
    return shutil.which("vllm") or "vllm"


def stats(r: dict, metric: str) -> dict:
    out = {"mean": r.get(f"mean_{metric}_ms")}
    for p in PERCENTILES:
        out[f"p{p}"] = r.get(f"p{p}_{metric}_ms")
    return out


def public_level(raw, *, users, planned, gpu, server_limit, slo):
    fields = (
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "max_concurrent_requests",
    )
    return {
        **{key: raw.get(key) for key in fields},
        **{f"{metric}_ms": stats(raw, metric) for metric in LATENCY_METRICS},
        **public_outcomes(raw, users, planned, slo),
        "concurrency": users,
        "num_prompts": planned,
        "duration_s": raw.get("duration"),
        "failed": raw.get("failed", 0),
        "gpu": gpu,
        "server_limit": server_limit,
        "stop_reason": "load_generator_finished",
    }


def public_measurement(level):
    completed = level["completed"]
    failed = level["failed"]
    planned = level["num_prompts"]
    seconds = level["duration_s"]
    result = measurement(
        f"concurrency {level['concurrency']}",
        completed + failed if completed is not None else None,
        completed,
        seconds,
        planned=planned,
        checks={"all_completed": completed == planned and failed == 0},
        stop_reason=level["stop_reason"],
    )
    result.update(
        output_tok_s=level["output_throughput"],
        p95_ttft_ms=level["ttft_ms"]["p95"],
        p95_e2e_ms=level["e2el_ms"]["p95"],
    )
    if level["joint_passed"] is not None and seconds:
        result["joint_passes_per_min"] = 60 * level["joint_passed"] / seconds
    return result


def argument_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="model name under models: in config/llm_targets.yaml")
    ap.add_argument("--dataset", choices=["sharegpt", "random"], default="sharegpt")
    ap.add_argument(
        "--dataset-path", default=None, help="ShareGPT json (default: standard_bench.datasets.sharegpt.path)"
    )
    ap.add_argument("--concurrency", default=None, help="comma list (default: standard_bench.concurrency)")
    ap.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="prompts per level (default: standard_bench load policy)",
    )
    ap.add_argument("--base-url", default=None, help="override the target's base_url")
    ap.add_argument("--tokenizer", default=None, help="tokenizer folder or repo (default: the target's repo)")
    ap.add_argument(
        "--vllm", default=None, help="command that runs the vllm CLI (default: $VENVS/vllm/bin/vllm)"
    )
    ap.add_argument("--gpus", default=None, help="GPU indices to sample, e.g. 0 or 1,2")
    ap.add_argument(
        "--gpu-count", type=int, default=None, help="override how many GPUs the deployment uses (for cost)"
    )
    ap.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="offered requests per second (default: saturate the concurrency limit)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600, help="seconds allowed per level")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["public", "full-context"], default="public")
    ap.add_argument("--contexts", help="comma-separated total tokens per request for full-context mode")
    ap.add_argument(
        "--output-tokens", type=int, default=512, help="forced decode length in full-context mode"
    )
    ap.add_argument("--wall-budget", type=float, help="total seconds available for the sweep")
    ap.add_argument("--stall-timeout", type=float, help="seconds without observed token progress")
    ap.add_argument(
        "--device-pci", default="", help="comma-separated serving PCI addresses for kernel health"
    )
    ap.add_argument("--kernel-log", type=Path, help="append-only local kernel log in place of journalctl")
    ap.add_argument(
        "--serving-config", type=Path, help="operator-supplied JSON with serving configuration and dtypes"
    )
    ap.add_argument(
        "--retrieval",
        action="store_true",
        help="score beginning, middle and end retrieval after a capacity pass",
    )
    return ap


def main():
    ap = argument_parser()
    args = ap.parse_args()
    started = time.monotonic()
    for duration in (args.timeout, args.stall_timeout, args.wall_budget):
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            ap.error("time limits must be finite positive seconds")
    if args.num_prompts is not None and args.num_prompts < 1:
        ap.error("--num-prompts must be positive")
    args.device_pci = [s.strip().lower() for s in args.device_pci.split(",") if s.strip()]
    if any(not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", d) for d in args.device_pci):
        ap.error("--device-pci requires full PCI addresses")
    serving = json.loads(args.serving_config.read_text()) if args.serving_config else {}
    if not isinstance(serving, dict):
        ap.error("--serving-config must contain a JSON object")
    serving.update(device_pci=args.device_pci)
    if args.gpus:
        serving["device_ids"] = args.gpus
    if args.mode == "public" and (args.contexts or args.retrieval):
        ap.error("--contexts and --retrieval require --mode full-context")

    if not args.request_rate > 0:
        ap.error("--request-rate must be positive")
    run = run_record(
        args.dataset,
        {"all_planned_requests_complete": True},
        {"concurrency": args.concurrency, "timeout_s_per_level": args.timeout},
    )
    cfg = load_yaml("llm_targets.yaml")
    target = dict(cfg["models"][args.target])
    settings = cfg.get("standard_bench", {})
    if args.base_url:
        target["base_url"] = args.base_url
    if args.gpu_count is not None:
        target["gpus"] = args.gpu_count
    base = target["base_url"].rstrip("/").removesuffix("/v1")
    concurrency_levels = [
        int(x)
        for x in (
            args.concurrency.split(",") if args.concurrency else settings.get("concurrency", [2, 4, 8, 16])
        )
    ]
    if any(users < 1 for users in concurrency_levels) or len(set(concurrency_levels)) != len(
        concurrency_levels
    ):
        ap.error("concurrency levels must be distinct positive integers")
    run["stop_condition"].update(
        concurrency=concurrency_levels, wall_budget_s=args.wall_budget, stall_timeout_s=args.stall_timeout
    )
    measurements = []
    dataset_config = (settings.get("datasets") or {}).get(args.dataset, {})

    dataset_args, dataset_description = [], ""
    if args.dataset == "sharegpt":
        path = args.dataset_path or os.path.expandvars(dataset_config.get("path", ""))
        if not path or not Path(path).exists():
            sys.exit(
                f"ShareGPT file not found: {path!r}. Copy ShareGPT_V3_unfiltered_cleaned_split.json "
                "there (see README) or pass --dataset-path."
            )
        dataset_args = ["--dataset-name", "sharegpt", "--dataset-path", path]
        dataset_description = (
            "ShareGPT: real chat conversations, prompts up to 1,024 tokens, replies at the dataset's length"
        )
    else:
        input_length, output_length = (
            dataset_config.get("input_len", 1024),
            dataset_config.get("output_len", 128),
        )
        dataset_args = [
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_length),
            "--random-output-len",
            str(output_length),
        ]
        dataset_description = f"Random: synthetic prompts, {input_length:,} tokens in, {output_length:,} out"

    if args.mode == "full-context":
        from context_bench import run_context

        if (
            not args.contexts
            or args.dataset != "sharegpt"
            or args.num_prompts
            or math.isfinite(args.request_rate)
        ):
            ap.error("full-context requires --contexts, ShareGPT and one simultaneous request per user")
        try:
            args.contexts = [int(v) for v in args.contexts.split(",")]
        except ValueError:
            ap.error("--contexts expects comma-separated integer token limits")
        if (
            len(set(args.contexts)) != len(args.contexts)
            or args.output_tokens < 2
            or any(context <= args.output_tokens for context in args.contexts)
        ):
            ap.error("contexts must be distinct and exceed --output-tokens (minimum 2)")
        if not asyncio.run(run_context(args, target, concurrency_levels, path, serving)):
            sys.exit("Full-context sweep stopped; see the saved evidence")
        return

    tokenizer = args.tokenizer or resolve_model_dir(target["repo"])
    vllm = shlex.split(args.vllm) if args.vllm else [default_vllm()]
    client = PublicClient(vllm, base, target["model"], tokenizer, dataset_args, args.request_rate, args.seed)
    policy = PublicLoadPolicy.from_config(settings)
    levels = []
    out = Path(args.out or f"results/public_{args.target}_{args.dataset}.json")
    payload = {
        "bench": "standard",
        "run": run,
        "measurements": measurements,
        "request_rate": str(args.request_rate),
        "seed": args.seed,
        "tool": "vllm bench serve",
        "target_name": args.target,
        "target": target,
        "serving": serving,
        "dataset": args.dataset,
        "dataset_desc": dataset_description,
        "tokenizer": tokenizer,
        "levels": levels,
    }
    if args.dataset == "sharegpt":
        payload["dataset_sha256"] = file_digest(path)
    payload["server_model"] = model_info(base, target["model"])
    evidence = RunEvidence(out, payload, sys.argv)
    budget = SweepBudget(args.wall_budget, started)
    slo = load_yaml("spec.yaml")["slo"]
    gpu = GpuSampler(args.gpus.split(",") if args.gpus else None).start()
    try:
        for users in concurrency_levels:
            planned = args.num_prompts or policy.prompt_count(users)
            if not budget.allows(planned, f"Concurrency {users}"):
                measurements.append(
                    measurement(
                        f"concurrency {users}",
                        0,
                        0,
                        0,
                        planned=planned,
                        checks={"all_completed": None},
                        stop_reason="wall_budget",
                    )
                )
                break
            level_dir = evidence.root / f"c{users}"
            level_dir.mkdir()
            cmd = client.command(users, planned, min(users, policy.max_warmups), level_dir)
            print(
                f"\n=== standard:{args.dataset}  {args.target}  concurrency={users}  prompts={planned} ===",
                flush=True,
            )
            started_at = time.time()
            reason = run_cli(
                cmd, level_dir, base, target["model"], min(args.timeout, budget.remaining), args.stall_timeout
            )
            finished_at = time.time()
            if reason != "completed" or not (level_dir / RAW_RESULT).is_file():
                measurements.append(
                    measurement(
                        f"concurrency {users}",
                        None,
                        None,
                        finished_at - started_at,
                        planned=planned,
                        checks={"all_completed": False},
                        stop_reason="runner_failure" if reason == "completed" else reason,
                    )
                )
                sampled = json.loads((level_dir / "scheduler.json").read_text())["last_observed"]
                measurements[-1].update(
                    active_at_last_sample=sampled.get("num_requests_running"),
                    queued_at_last_sample=sampled.get("num_requests_waiting"),
                )
                print(f"Stopped at {users} users ({reason}); client log: {level_dir / 'client.log'}")
                break
            budget.completed(planned, finished_at - started_at)
            # Some clients emit NaN for undefined percentiles when all requests fail.
            raw = json.loads((level_dir / RAW_RESULT).read_text(), parse_constant=lambda _: None)
            level = public_level(
                raw,
                users=users,
                planned=planned,
                gpu=gpu.window(started_at, finished_at),
                server_limit=payload["server_model"].get("max_model_len"),
                slo=slo,
            )
            measurements.append(public_measurement(level))
            levels.append(level)
            evidence.checkpoint()
            mean_input = (level["total_input_tokens"] or 0) / max(1, level["completed"] or 1)
            mean_output = (level["total_output_tokens"] or 0) / max(1, level["completed"] or 1)
            print(
                f"  {level['output_throughput']} out tok/s | {level['request_throughput']} req/s | "
                f"mean {mean_input:.0f} in / {mean_output:.0f} out tokens | TTFT p95 {(level['ttft_ms']['p95'] or 0) / 1000:.2f}s | "
                f"TPOT p95 {level['tpot_ms']['p95'] or 0:.1f}ms | failed {level['failed']}",
                flush=True,
            )
    finally:
        gpu.stop()
        run["finished_at"] = time.time()
        evidence.checkpoint()
    if any(m["status"] != "qualified" for m in measurements):
        sys.exit("Benchmark incomplete: one or more levels did not complete all planned requests")
    if len(levels) != len(concurrency_levels):
        sys.exit(f"Only {len(levels)} of {len(concurrency_levels)} benchmark levels completed")


if __name__ == "__main__":

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    main()
