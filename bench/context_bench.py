"""Exact-length streaming bursts against an existing vLLM-compatible endpoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
import signal
import sys
import time
from itertools import product

import aiohttp

from common import load_yaml, measurement, run_record, summarize
from context_inputs import ShareGPT, fingerprint, retrieval_cases
from device_journal import DeviceJournal
from llm_stream import Endpoint, burst
from llm_evidence import RunEvidence, SweepBudget, decode_peak, joint_latency, save_json, verdict


def summarize_burst(requests, inputs, outputs, seconds, reason, before, after, health, slo):
    users = len(requests)
    completed = [r for r in requests if r.state == "completed"]
    exact = [r for r in completed if r.exact(inputs, outputs)]
    peak = decode_peak((r.first, r.last) for r in exact)
    overlap = min(r.last for r in exact) - max(r.first for r in exact) if len(exact) == users else None
    delta = None
    if "num_preemptions_total" in before and "num_preemptions_total" in after:
        delta = after["num_preemptions_total"] - before["num_preemptions_total"]
    completion = verdict([len(completed) == users])
    token_ids_verified = all(r.streamed_tokens == outputs for r in exact) if exact else None
    capacity = verdict(
        [
            True if token_ids_verified else None,
            len(exact) == users,
            peak == users,
            None if delta is None else delta == 0,
            *health,
        ]
    )
    samples = [(r.ttft, r.tpot(outputs)) for r in exact]
    return dict(
        concurrency=users,
        num_prompts=users,
        context=inputs + outputs,
        duration_s=seconds,
        streamed_ids_verified=token_ids_verified,
        issued=sum(r.state != "unissued" for r in requests),
        completed=len(completed),
        failed=sum(r.state == "failed" for r in requests),
        interrupted=sum(r.state == "interrupted" for r in requests),
        exact_lengths=len(exact),
        total_input_tokens=sum(r.usage.get("prompt_tokens", 0) for r in completed),
        total_output_tokens=sum(r.usage.get("completion_tokens", 0) for r in completed),
        output_throughput=sum(r.usage.get("completion_tokens", 0) for r in completed) / seconds,
        ttft_ms=summarize(1000 * r.ttft for r in exact),
        tpot_ms=summarize(r.tpot(outputs) for r in exact),
        e2el_ms=summarize(1000 * (r.end - r.start) for r in completed),
        itl_ms=summarize(gap for r in exact for gap in r.gaps_ms)
        if all(r.single_token_chunks for r in exact)
        else {},
        verified_decode_concurrency=peak,
        simultaneous_decode_s=max(0, overlap) if overlap is not None else None,
        preemptions=delta,
        endpoint_healthy=health[0],
        devices_healthy=health[1],
        stop_reason=reason,
        completion_status=completion,
        capacity_status=capacity,
        **joint_latency(samples, users, slo),
    )


async def retrieval(endpoint, filler, seed, outputs, folder):
    results = []
    async for position, code, prompt in retrieval_cases(filler, seed, endpoint.tokenize):
        data = await endpoint.json(
            "/v1/completions", dict(model=endpoint.model, prompt=prompt, temperature=0, max_tokens=outputs)
        )
        answer = data["choices"][0]["text"].strip()
        results.append(dict(position=position, expected=code, answer=answer, passed=answer == code))
        save_json(folder / "retrieval.json", results)
    return verdict([r["passed"] for r in results])


async def run_context(a, target, loads, dataset, serving):
    slo = load_yaml("spec.yaml")["slo"]
    run = run_record(
        "full-context ShareGPT",
        {
            "exact_lengths": True,
            "simultaneous_decode": True,
            "zero_preemptions": True,
            "device_health": True,
            "slo": slo,
        },
        {
            "contexts": a.contexts,
            "concurrency": loads,
            "output_tokens": a.output_tokens,
            "retrieval": a.retrieval,
            "wall_budget_s": a.wall_budget,
            "timeout_s_per_level": a.timeout,
            "stall_timeout_s": a.stall_timeout,
        },
    )
    out = Path(a.out or f"results/context_{a.target}.json")
    data = dict(
        bench="full-context",
        dataset="full-context ShareGPT",
        seed=a.seed,
        target=target,
        run=run,
        serving=serving,
        levels=[],
        measurements=[],
    )
    evidence = RunEvidence(out, data, sys.argv)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await sweep(a, target, loads, dataset, evidence, slo)
    except (
        asyncio.TimeoutError,
        asyncio.CancelledError,
        aiohttp.ClientError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        data["stop_reason"] = type(exc).__name__
        data["error"] = str(exc)
        return False
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        run["finished_at"] = time.time()
        evidence.checkpoint()


async def measure_level(endpoint, prompts, options, folder, end, server_limit, slo):
    inputs = len(prompts[0])
    before = await endpoint.metrics()
    journal_reader = await asyncio.to_thread(DeviceJournal, options.device_pci, options.kernel_log)
    if time.monotonic() >= end:
        raise asyncio.TimeoutError("level deadline reached before issuing requests")
    requests, samples, seconds, reason = await burst(
        endpoint,
        prompts,
        options.output_tokens,
        folder,
        options.stall_timeout,
        max(0, end - time.monotonic()),
    )
    after = await endpoint.metrics()
    health = (
        await endpoint.healthy(),
        await asyncio.to_thread(journal_reader.finish, folder / "kernel.log"),
    )
    level = summarize_burst(
        requests, inputs, options.output_tokens, seconds, reason, before, after, health, slo
    )
    level.update(
        server_limit=server_limit,
        input_fingerprint=fingerprint(prompts),
        last_scheduler_sample=samples,
    )
    level["active_at_last_sample"] = samples.get("num_requests_running")
    level["queued_at_last_sample"] = samples.get("num_requests_waiting")
    save_json(
        folder / "requests.json",
        [
            {
                "user": r.user,
                "state": r.state,
                "usage": r.usage,
                "issued_at_monotonic_s": r.start,
                "first_output_s": r.first,
                "last_output_s": r.last,
                "finished_at_s": r.end,
                "streamed_token_ids_count": r.streamed_tokens,
            }
            for r in requests
        ],
    )
    return level


def record_level(data, context, users, level):
    data["levels"].append(level)
    m = measurement(
        f"context {context} concurrency {users}",
        level["issued"],
        level["completed"],
        level["duration_s"],
        checks={"capacity": {"passed": True, "failed": False}.get(level["capacity_status"])},
        planned=users,
        stop_reason=level["stop_reason"],
    )
    m.update(
        joint_passes_per_min=60 * level["joint_passed"] / level["duration_s"],
        input_fingerprint=level["input_fingerprint"],
        output_tok_s=level["output_throughput"],
        p95_ttft_ms=level["ttft_ms"].get("p95"),
        p95_e2e_ms=level["e2el_ms"].get("p95"),
    )
    data["measurements"].append(m)


async def sweep(a, target, loads, dataset, evidence, slo):
    data = evidence.data
    root = evidence.root
    budget = SweepBudget(a.wall_budget)
    corpus = ShareGPT(dataset, a.seed)
    data["dataset_sha256"] = corpus.sha256
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10), connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        endpoint = Endpoint(session, target["base_url"], target["model"])
        models = await asyncio.wait_for(endpoint.json("/v1/models"), min(10, budget.remaining))
        save_json(root / "models.json", models)
        model = next((m for m in models["data"] if m["id"] == endpoint.model), None)
        if model is None:
            raise ValueError("configured model missing from /v1/models")
        for context, users in product(a.contexts, loads):
            if model.get("max_model_len") and context > model["max_model_len"]:
                raise ValueError(f"requested {context} exceeds server limit {model['max_model_len']}")
            work = users * context**2
            if not budget.allows(work, f"Context {context}, users {users}"):
                data["measurements"].append(
                    measurement(
                        f"context {context} concurrency {users}",
                        0,
                        0,
                        0,
                        planned=users,
                        checks={"capacity": None},
                        stop_reason="wall_budget",
                    )
                )
                data["stop_reason"] = "wall_budget"
                evidence.checkpoint()
                return False
            folder = root / f"ctx{context}-c{users}"
            folder.mkdir()
            inputs = context - a.output_tokens
            started = time.monotonic()
            end = started + min(a.timeout, budget.remaining)
            prompts = await asyncio.wait_for(
                corpus.prompts(users, inputs, endpoint.tokenize), end - time.monotonic()
            )
            level = await measure_level(endpoint, prompts, a, folder, end, model.get("max_model_len"), slo)
            record_level(data, context, users, level)
            evidence.checkpoint()
            if level["capacity_status"] != "passed":
                return False
            if a.retrieval:
                level["retrieval_status"] = await asyncio.wait_for(
                    retrieval(endpoint, prompts[0], a.seed, a.output_tokens, folder),
                    max(0, end - time.monotonic()),
                )
                evidence.checkpoint()
                if level["retrieval_status"] != "passed":
                    return False
            budget.completed(work, time.monotonic() - started)
    return True
