"""Stream a synchronized burst and journal request and scheduler progress."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import time

import aiohttp

from llm_evidence import Progress, event, read_metrics


class Endpoint:
    def __init__(self, session, base, model):
        self.session = session
        self.base = base.rstrip("/").removesuffix("/v1")
        self.model = model

    async def json(self, route, body=None):
        async with self.session.request(
            "GET" if body is None else "POST", self.base + route, json=body
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def tokenize(self, text):
        data = await self.json(
            "/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False}
        )
        return data["tokens"]

    async def metrics(self):
        try:
            async with self.session.get(self.base + "/metrics", timeout=4) as response:
                response.raise_for_status()
                return read_metrics(await response.text(), self.model)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return {}

    async def healthy(self):
        try:
            async with self.session.get(self.base + "/health", timeout=4) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False


@dataclass
class Request:
    user: int
    state: str = "unissued"
    start: float = 0
    end: float = 0
    first: float | None = None
    last: float | None = None
    usage: dict = field(default_factory=dict)
    streamed_tokens: int = 0
    single_token_chunks: bool = True
    gaps_ms: list = field(default_factory=list)

    def exact(self, inputs, outputs):
        return (
            self.state == "completed"
            and self.first is not None
            and self.last is not None
            and self.usage.get("prompt_tokens") == inputs
            and self.usage.get("completion_tokens") == outputs
        )

    @property
    def ttft(self):
        return self.first - self.start

    def tpot(self, tokens):
        return 1000 * (self.last - self.first) / (tokens - 1)


async def stream(endpoint, request, ids, outputs, log, progress):
    request.start = time.monotonic()
    request.state = "active"
    event(log, user=request.user, state="issued", input_tokens=len(ids), planned_output=outputs)
    body = dict(
        model=endpoint.model,
        prompt=ids,
        max_tokens=outputs,
        ignore_eos=True,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        return_token_ids=True,
    )
    try:
        async with endpoint.session.post(endpoint.base + "/v1/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.content:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    request.state = "completed"
                    break
                message = json.loads(data)
                if message.get("error"):
                    raise ValueError(str(message["error"]))
                if message.get("usage"):
                    request.usage = message["usage"]
                for choice in message.get("choices", []):
                    count = len(choice.get("token_ids") or [])
                    if count or choice.get("text"):
                        request.streamed_tokens += count
                        request.single_token_chunks &= count == 1
                        now = time.monotonic()
                        if request.first is None:
                            request.first = now
                        if request.last is not None and count == 1:
                            request.gaps_ms.append(1000 * (now - request.last))
                        request.last = progress.last = now
                        event(
                            log,
                            user=request.user,
                            state="output",
                            elapsed_s=now - request.start,
                            token_ids_count=count,
                        )
            if request.state != "completed":
                raise ValueError("stream closed before [DONE]")
    except asyncio.CancelledError:
        request.state = "interrupted"
        raise
    except (aiohttp.ClientError, ValueError, asyncio.TimeoutError) as exc:
        request.state = "failed"
        event(log, user=request.user, error=str(exc))
    finally:
        request.end = time.monotonic()
        event(
            log,
            user=request.user,
            state=request.state,
            usage=request.usage,
            elapsed_s=request.end - request.start,
        )


async def monitor(endpoint, progress, stall, log, samples):
    while True:
        values = await endpoint.metrics()
        progress.observe(values)
        if values:
            samples.update(values)
        event(log, state="telemetry", metrics=values)
        if stall and time.monotonic() - progress.last >= stall:
            return "stalled"
        await asyncio.sleep(1)


async def burst(endpoint, prompts, outputs, folder, stall, timeout):
    requests = [Request(i) for i in range(len(prompts))]
    samples = {}
    progress = Progress()
    start = time.monotonic()
    reason = "completed"
    with (folder / "events.jsonl").open("x") as log:
        jobs = asyncio.gather(
            *(stream(endpoint, r, ids, outputs, log, progress) for r, ids in zip(requests, prompts))
        )
        watch = asyncio.create_task(monitor(endpoint, progress, stall, log, samples))
        try:
            done, _ = await asyncio.wait((jobs, watch), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if jobs in done:
                await jobs
            else:
                reason = watch.result() if watch in done else "wall_timeout"
        except asyncio.CancelledError:
            reason = "interrupted"
        finally:
            jobs.cancel()
            watch.cancel()
            await asyncio.gather(jobs, watch, return_exceptions=True)
    return requests, samples, time.monotonic() - start, reason
