#!/usr/bin/env python3
"""A fake vLLM-style OpenAI server for testing the harness without a GPU.

Implements /v1/chat/completions (streaming, with usage and cached_tokens),
/v1/completions (streaming), /tokenize, /metrics and /v1/models. It simulates
a shared prefix cache and a token rate that slows as more requests run, so the
harness's throughput, latency and cache accounting can be checked end to end.
"""
import argparse
import asyncio
import hashlib
import json
import time
import re

from aiohttp import web

TPW = 1.3            # fake tokens per word
PREFILL_TOK_S = 40000
DECODE_TOK_S = 60    # per stream at concurrency 1
state = {"running": 0, "seen": set(), "q": 0, "h": 0, "prompt": 0, "gen": 0}


def ntok(text: str) -> int:
    return max(1, int(len(text.split()) * TPW))


def cached_prefix(messages):
    """Tokens of the longest message-prefix seen before (a crude prefix cache)."""
    cached, h = 0, hashlib.sha1()
    run = 0
    for m in messages:
        h.update(json.dumps(m, sort_keys=True).encode())
        key = h.hexdigest()
        run += ntok(m.get("content", ""))
        if key in state["seen"]:
            cached = run
        state["seen"].add(key)
    return cached


async def stream_tokens(resp, n_out, make_chunk):
    state["running"] += 1
    try:
        for i in range(n_out):
            rate = DECODE_TOK_S / (1 + 0.08 * (state["running"] - 1))
            await asyncio.sleep(1.0 / rate)
            await resp.write(("data: " + json.dumps(make_chunk(i)) + "\n\n").encode())
            state["gen"] += 1
    finally:
        state["running"] -= 1


async def chat(request):
    body = await request.json()
    msgs = body["messages"]
    prompt = sum(ntok(m.get("content", "")) for m in msgs)
    cached = cached_prefix(msgs)
    state["q"] += prompt
    state["h"] += cached
    state["prompt"] += prompt
    n_out = body.get("max_tokens", 16)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep((prompt - cached) / PREFILL_TOK_S + 0.01)
    await stream_tokens(resp, n_out, lambda i: {"choices": [{"index": 0, "delta": {"content": f" w{i}"}}]})
    usage = {"prompt_tokens": prompt, "completion_tokens": n_out, "total_tokens": prompt + n_out,
             "prompt_tokens_details": {"cached_tokens": cached}}
    await resp.write(("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def completions(request):
    body = await request.json()
    if not body.get("stream"):
        ids = body.get("prompt")
        text = "".join(chr(i) for i in ids) if isinstance(ids, list) else ""
        code = re.search(r"inventory code is ([0-9A-F]+)", text)
        return web.json_response({"choices": [{"text": code[1] if code else "two"}]})
    content = body.get("prompt", "")
    prompt = len(content) if isinstance(content, list) else ntok(content)
    state["prompt"] += prompt
    n_out = body.get("max_tokens", 16)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(prompt / PREFILL_TOK_S + 0.01)
    await stream_tokens(resp, n_out, lambda i: {"choices": [{"index": 0, "text": "<custom_token_5>", "token_ids": [5]}]})
    usage = {"prompt_tokens": prompt, "completion_tokens": n_out}
    await resp.write(("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def tokenize(request):
    body = await request.json()
    text = body.get("prompt", "")
    return web.json_response({"count": ntok(text), "tokens": list(map(ord, text)), "max_model_len": 32768})


async def metrics(request):
    lines = [
        f'vllm:prefix_cache_queries_total{{model_name="gemma31b-w4"}} {state["q"]}',
        f'vllm:prefix_cache_hits_total{{model_name="gemma31b-w4"}} {state["h"]}',
        f'vllm:num_preemptions_total{{model_name="gemma31b-w4"}} 0',
        f'vllm:prompt_tokens_total{{model_name="gemma31b-w4"}} {state["prompt"]}',
        f'vllm:generation_tokens_total{{model_name="gemma31b-w4"}} {state["gen"]}',
        f'vllm:kv_cache_usage_perc{{model_name="gemma31b-w4"}} {min(1.0, 0.05 * state["running"])}',
        f'vllm:num_requests_running{{model_name="gemma31b-w4"}} {state["running"]}',
        f'vllm:num_requests_waiting{{model_name="gemma31b-w4"}} 0',
    ]
    return web.Response(text="\n".join(lines) + "\n")


async def models(request):
    return web.json_response({"data": [{"id": "mock"}, {"id": "gemma31b-w4", "max_model_len": 32768}]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18001)
    a = ap.parse_args()
    app = web.Application()
    app.add_routes([web.post("/v1/chat/completions", chat), web.post("/v1/completions", completions),
                    web.post("/tokenize", tokenize), web.get("/health", models), web.get("/metrics", metrics), web.get("/v1/models", models)])
    web.run_app(app, port=a.port, print=None)


if __name__ == "__main__":
    main()
