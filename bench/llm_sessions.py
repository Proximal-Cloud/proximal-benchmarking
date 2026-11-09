#!/usr/bin/env python3
"""Socratic tutoring-session benchmark for the tutor LLMs in config/llm_targets.yaml.

Why sessions and not random prompts: the spec assumes ~91% of the tutor LLM's
input (2M of 2.2M tokens/day) is served from prefix cache. That only happens if
every student shares one long system prompt and each turn re-sends that
student's growing history. Random prompts would show almost no cache hits and
overstate cost several times. So each simulated student here:

  * sends [shared system prompt] + [own history] + [new message + RAG chunk]
  * appends the reply to its history
  * when the prompt would pass `compact_at_tokens`, replaces history with an
    (uncached) summary, as a real tutor app would

N students run concurrently (closed loop: next turn starts when the reply ends,
plus optional think time). The run sweeps N and reports, per level:
throughput, TTFT / TPOT / end-to-end latency percentiles, the measured cache
hit rate, KV-cache pressure (preemptions) and GPU utilisation/power.

Usage (from the repo root):
  python bench/llm_sessions.py --target gemma31b-w4 --concurrency 2,4,8,16 --gpus 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GpuSampler, load_yaml, summarize, write_result  # noqa: E402

# Ordinary classroom vocabulary, so the tokenizer sees realistic English text.
WORDS = """
the a an of to in and is are was were be been for on with as by at from that this
these those it its which who what when where why how can could should would will
may might must not no yes all any some each every more most less many much few
student teacher class lesson chapter page question answer example exercise problem
solution step steps method reason idea concept rule law theory fact result value
number numbers fraction decimal ratio percent equation variable function graph
angle triangle circle square area volume length width height speed time distance
force energy mass weight motion light sound heat electric current circuit magnet
cell plant animal water air soil rock earth sun moon planet star climate weather
history river city king empire trade war peace freedom constitution government
right duty citizen village market price cost profit loss interest bank money
grammar noun verb adjective sentence paragraph essay poem story character plot
reading writing listening speaking language word words meaning letter sound
think explain describe compare contrast define calculate measure observe predict
check verify prove show find solve draw write list name identify classify
first second third next then finally because therefore however although while
if unless until since before after during between among above below inside
simple important different similar common correct wrong true false possible
large small long short high low fast slow hot cold new old early late
experiment observation hypothesis conclusion data table chart diagram model map
practice homework test exam revision notes textbook board school friend family
""".split()


class TextFactory:
    """Makes filler text of roughly N tokens, unique per tag."""

    def __init__(self, tokens_per_word: float, corpus_words: list[str] | None = None):
        self.tpw = tokens_per_word
        self.corpus = corpus_words

    def text(self, rng: random.Random, n_tokens: int, tag: str) -> str:
        n_words = max(1, int(n_tokens / self.tpw))
        if self.corpus and len(self.corpus) > n_words:
            start = rng.randrange(0, len(self.corpus) - n_words)
            body = self.corpus[start:start + n_words]
        else:
            body = [rng.choice(WORDS) for _ in range(n_words)]
        # A unique tag at the start guarantees no accidental prefix sharing.
        return f"[{tag}] " + " ".join(body)


class Session:
    """One simulated student's conversation state."""

    def __init__(self, sid: str, prof: dict, tf: TextFactory, system_text: str,
                 system_tokens: int, rng: random.Random, preseed: bool = True):
        self.sid, self.p, self.tf, self.rng = sid, prof, tf, rng
        self.system = {"role": "system", "content": system_text}
        self.system_tokens = system_tokens
        self.history: list[dict] = []
        self.hist_tokens = 0
        self.turn = 0
        self.keep = prof.get("keep_history", True)
        if preseed and self.keep:
            # Start each student at a random point in its compaction cycle so the
            # measured window sees a steady-state mix of short and long histories.
            per_turn = prof["turn_new_input_tokens"] + prof["turn_output_tokens"]
            room = prof["compact_at_tokens"] - system_tokens - prof["turn_new_input_tokens"]
            k = rng.randrange(0, max(1, room // per_turn))
            for i in range(k):
                self.history.append({"role": "user", "content": tf.text(rng, prof["turn_new_input_tokens"], f"{sid}-pre{i}u")})
                self.history.append({"role": "assistant", "content": tf.text(rng, prof["turn_output_tokens"], f"{sid}-pre{i}a")})
            self.hist_tokens = k * per_turn

    def next_messages(self) -> tuple[list[dict], bool]:
        p = self.p
        self.turn += 1
        compacted = False
        if not self.keep:
            self.history, self.hist_tokens = [], 0
        elif self.system_tokens + self.hist_tokens + p["turn_new_input_tokens"] > p["compact_at_tokens"]:
            summary = self.tf.text(self.rng, p["summary_tokens"], f"{self.sid}-sum{self.turn}")
            self.history = [{"role": "user", "content": "Summary of our session so far: " + summary},
                            {"role": "assistant", "content": "Understood. Let's continue."}]
            self.hist_tokens = p["summary_tokens"] + 10
            compacted = True
        self._pending_user = {"role": "user",
                              "content": self.tf.text(self.rng, p["turn_new_input_tokens"], f"{self.sid}-t{self.turn}")}
        return [self.system, *self.history, self._pending_user], compacted

    def record_reply(self, reply_text: str, prompt_tokens: int | None, out_tokens: int | None):
        if not self.keep:
            return
        self.history.append(self._pending_user)
        self.history.append({"role": "assistant", "content": reply_text or "(reply)"})
        if prompt_tokens is not None and out_tokens is not None:
            self.hist_tokens = max(0, prompt_tokens - self.system_tokens) + out_tokens
        else:
            self.hist_tokens += self.p["turn_new_input_tokens"] + self.p["turn_output_tokens"]


# ---------------------------------------------------------------- HTTP layer

def root_url(base_url: str) -> str:
    return base_url[:-3] if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/")


async def calibrate_tokens_per_word(http, target, headers) -> float:
    """Ask vLLM's /tokenize how many tokens our filler words cost."""
    sample = " ".join(random.Random(0).choice(WORDS) for _ in range(2000))
    try:
        async with http.post(root_url(target["base_url"]).rstrip("/") + "/tokenize",
                             json={"model": target["model"], "prompt": sample},
                             headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status == 200:
                data = await r.json()
                count = data.get("count") or len(data.get("tokens", []))
                if count:
                    return count / 2000
    except Exception as e:  # remote APIs have no /tokenize
        print(f"[calibrate] /tokenize unavailable ({type(e).__name__}); assuming 1.33 tokens/word")
    return 1.33


async def scrape_metrics(http, target) -> dict:
    """Read vLLM's Prometheus /metrics and sum each metric across labels."""
    if not target.get("local"):
        return {}
    try:
        async with http.get(root_url(target["base_url"]).rstrip("/") + "/metrics",
                            timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                return {}
            text = await r.text()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name_labels, value = line.rsplit(" ", 1)
            name = name_labels.split("{", 1)[0]
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


def pick(metrics: dict, *needles: str):
    """First metric whose name contains all needles (names differ across vLLM versions)."""
    for k, v in metrics.items():
        if all(n in k for n in needles) and not k.endswith(("_created", "_bucket")):
            return v
    return None


async def one_turn(http, target, headers, messages, out_tokens, force_len, extra_body):
    body = {
        "model": target["model"], "messages": messages, "max_tokens": out_tokens,
        "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.7,
    }
    if force_len:  # vLLM-only knobs: generate exactly out_tokens tokens
        body["ignore_eos"] = True
        body["min_tokens"] = out_tokens
    body.update(extra_body or {})
    rec = {"t_start": time.time(), "ok": False}
    t0 = time.perf_counter()
    ttft, pieces, usage, err = None, [], None, None
    try:
        async with http.post(target["base_url"].rstrip("/") + "/chat/completions", json=body,
                             headers=headers, timeout=aiohttp.ClientTimeout(total=1800)) as r:
            if r.status != 200:
                err = f"HTTP {r.status}: {(await r.text())[:300]}"
            else:
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for ch in obj.get("choices") or []:
                        d = ch.get("delta") or {}
                        c = d.get("content") or ""
                        rc = d.get("reasoning_content") or d.get("reasoning") or ""
                        if (c or rc) and ttft is None:
                            ttft = time.perf_counter() - t0
                        if c:
                            pieces.append(c)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    e2e = time.perf_counter() - t0
    rec.update({"t_end": rec["t_start"] + e2e, "e2e_s": e2e, "ttft_s": ttft, "error": err})
    if usage:
        details = usage.get("prompt_tokens_details") or {}
        rec.update({
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": details.get("cached_tokens"),
        })
    if err is None and ttft is not None:
        rec["ok"] = True
        n = rec.get("completion_tokens") or 0
        rec["tpot_ms"] = (e2e - ttft) / (n - 1) * 1000 if n > 1 else None
    rec["reply"] = "".join(pieces)
    return rec


# ---------------------------------------------------------------- one level

def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


async def run_level(target, prof, concurrency, warmup_s, duration_s, think_s, tf,
                    system_text, system_tokens, force_len, extra_body, gpu, seed, grace_s,
                    warmup_turns=2):
    """Run one load level and measure it in steady state.

    The measured window starts only after every student has finished
    `warmup_turns` turns (capped at `warmup_s` seconds). Before that, each
    student's pre-seeded history is pushed into the prefix cache with a
    1-token request. Without both, the window is dominated by first turns
    whose history was never cached, which inflates TTFT and cost at high load.
    """
    headers = {"Content-Type": "application/json"}
    key_env = target.get("api_key_env")
    if key_env and os.environ.get(key_env):
        headers["Authorization"] = f"Bearer {os.environ[key_env]}"
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, read_bufsize=2 ** 20) as http:
        rng = random.Random(seed)
        sessions = [Session(f"c{concurrency}s{i}", prof, tf, system_text, system_tokens,
                            random.Random(rng.random()), preseed=True) for i in range(concurrency)]
        records: list[dict] = []
        turns_done = [0] * concurrency
        stop = asyncio.Event()
        st = {"t_warm": None, "t_stop": None, "warm_turns_reached": False}

        # 1) prime: put each student's starting history into the prefix cache
        async def prime(s: Session):
            if s.history:
                await one_turn(http, target, headers, [s.system, *s.history, {"role": "user", "content": "ok"}],
                               1, False, extra_body)
        if prof.get("keep_history", True):
            await asyncio.gather(*[prime(s) for s in sessions])

        async def worker(i: int, s: Session):
            while not stop.is_set():
                msgs, compacted = s.next_messages()
                rec = await one_turn(http, target, headers, msgs, prof["turn_output_tokens"], force_len, extra_body)
                rec["compacted"] = compacted
                s.record_reply(rec.pop("reply", ""), rec.get("prompt_tokens"), rec.get("completion_tokens"))
                records.append(rec)
                turns_done[i] += 1
                if think_s > 0 and not stop.is_set():
                    await asyncio.sleep(random.expovariate(1.0 / think_s))

        gauges: list[dict] = []

        async def gauge_loop():
            while not stop.is_set():
                if st["t_warm"] is not None:
                    gauges.append(await scrape_metrics(http, target))
                await asyncio.sleep(2)

        tasks = [asyncio.create_task(worker(i, s)) for i, s in enumerate(sessions)]
        gtask = asyncio.create_task(gauge_loop())
        # 2) warm-up: wait until every student has done `warmup_turns` turns
        t_begin = time.time()
        while min(turns_done) < warmup_turns and time.time() - t_begin < warmup_s:
            await asyncio.sleep(0.5)
        st["warm_turns_reached"] = min(turns_done) >= warmup_turns
        t_warm = st["t_warm"] = time.time()
        m0 = await scrape_metrics(http, target)
        # 3) measure
        await asyncio.sleep(duration_s)
        t_stop = time.time()
        m1 = await scrape_metrics(http, target)
        stop.set()
        done, pending = await asyncio.wait(tasks, timeout=grace_s)
        for t in pending:
            t.cancel()
        gtask.cancel()
        warmup_used = t_warm - t_begin

    # ---- aggregate the measurement window [t_warm, t_stop]
    W = t_stop - t_warm
    ok = [r for r in records if r["ok"]]
    in_win = [r for r in ok if r["t_start"] >= t_warm and r["t_end"] <= t_stop + grace_s]
    out_tok = prompt_tok = uncached_tok = turns = 0.0
    for r in ok:
        dec0 = r["t_start"] + (r["ttft_s"] or 0)
        frac_dec = overlap(dec0, r["t_end"], t_warm, t_stop) / max(1e-9, r["t_end"] - dec0)
        frac_pre = overlap(r["t_start"], dec0, t_warm, t_stop) / max(1e-9, dec0 - r["t_start"])
        out_tok += (r.get("completion_tokens") or 0) * frac_dec
        prompt_tok += (r.get("prompt_tokens") or 0) * frac_pre
        uncached_tok += ((r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0)) * frac_pre
        turns += overlap(r["t_start"], r["t_end"], t_warm, t_stop) / max(1e-9, r["t_end"] - r["t_start"])

    sum_prompt = sum(r.get("prompt_tokens") or 0 for r in in_win)
    sum_cached = sum(r.get("cached_tokens") or 0 for r in in_win)
    have_cached = any(r.get("cached_tokens") is not None for r in in_win)

    q_d = (pick(m1, "prefix_cache_queries") or 0) - (pick(m0, "prefix_cache_queries") or 0)
    h_d = (pick(m1, "prefix_cache_hits") or 0) - (pick(m0, "prefix_cache_hits") or 0)
    pre_d = (pick(m1, "num_preemptions") or 0) - (pick(m0, "num_preemptions") or 0)
    kv_vals = [v for g in gauges for v in [pick(g, "kv_cache_usage") or pick(g, "cache_usage_perc")] if v is not None]
    waiting = [pick(g, "num_requests_waiting") for g in gauges if pick(g, "num_requests_waiting") is not None]

    level = {
        "concurrency": concurrency,
        "window_s": W,
        "warmup_s_used": warmup_used,
        "warmup_turns_reached": st["warm_turns_reached"],
        "turns_per_student_in_window": len(in_win) / concurrency,
        "turns_completed_in_window": len(in_win),
        "turns_per_s": turns / W,
        "output_tok_per_s": out_tok / W,
        "prompt_tok_per_s": prompt_tok / W,
        "uncached_prompt_tok_per_s": uncached_tok / W if have_cached else None,
        "per_stream_output_tok_per_s": summarize([(r["completion_tokens"] or 0) / max(1e-9, r["e2e_s"] - r["ttft_s"]) for r in in_win if r.get("completion_tokens")]),
        "ttft_s": summarize([r["ttft_s"] for r in in_win]),
        "tpot_ms": summarize([r.get("tpot_ms") for r in in_win]),
        "e2e_s": summarize([r["e2e_s"] for r in in_win]),
        "mix_per_turn": {
            "prompt_tokens": summarize([r.get("prompt_tokens") for r in in_win]).get("mean"),
            "cached_tokens": summarize([r.get("cached_tokens") for r in in_win]).get("mean") if have_cached else None,
            "completion_tokens": summarize([r.get("completion_tokens") for r in in_win]).get("mean"),
            "compaction_share": (sum(1 for r in in_win if r.get("compacted")) / len(in_win)) if in_win else None,
        },
        "cache_hit_rate_usage": (sum_cached / sum_prompt) if (have_cached and sum_prompt) else None,
        "cache_hit_rate_metrics": (h_d / q_d) if q_d else None,
        "preemptions": pre_d if m1 else None,
        "kv_cache_usage_max": max(kv_vals) if kv_vals else None,
        "requests_waiting_max": max(waiting) if waiting else None,
        "errors": sum(1 for r in records if not r["ok"]),
        "error_samples": list({r["error"] for r in records if r["error"]})[:3],
        "gpu": gpu.window(t_warm, t_stop) if gpu else None,
    }
    return level


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


async def main_async(a):
    targets = load_yaml("llm_targets.yaml")
    spec = load_yaml("spec.yaml")
    target = targets["models"][a.target]
    prof = dict(targets["profiles"][a.profile])
    for k in ("system_prompt_tokens", "turn_new_input_tokens", "turn_output_tokens", "compact_at_tokens"):
        v = getattr(a, k)
        if v is not None:
            prof[k] = v
    if a.base_url:
        target["base_url"] = a.base_url
    if a.gpu_count is not None:
        target["gpus"] = a.gpu_count
    force_len = target.get("local", False) if a.force_len is None else a.force_len
    extra_body = json.loads(a.extra_body) if a.extra_body else {}

    headers = {"Content-Type": "application/json"}
    if target.get("api_key_env") and os.environ.get(target["api_key_env"]):
        headers["Authorization"] = f"Bearer {os.environ[target['api_key_env']]}"
    async with aiohttp.ClientSession() as http:
        tpw = await calibrate_tokens_per_word(http, target, headers)
    print(f"[calibrate] {tpw:.3f} tokens per filler word")

    corpus = None
    if a.corpus:
        corpus = Path(a.corpus).read_text(encoding="utf-8", errors="ignore").split()
    tf = TextFactory(tpw, corpus)
    # One system prompt, identical for every student and every level.
    system_text = ("You are a patient Socratic tutor. Never hand out final answers; ask guiding questions. "
                   + tf.text(random.Random(12345), prof["system_prompt_tokens"], "SYSTEM-PROMPT-v1"))

    gpu = GpuSampler(a.gpus.split(",") if a.gpus else None).start()
    levels = []
    slo = spec["slo"]
    try:
        for c in [int(x) for x in a.concurrency.split(",")]:
            print(f"\n=== {a.profile}:{a.target}  concurrency={c}  warmup={a.warmup}s  window={a.duration}s ===")
            lv = await run_level(target, prof, c, a.warmup, a.duration, a.think_time, tf, system_text,
                                 prof["system_prompt_tokens"], force_len, extra_body, gpu, a.seed + c, a.grace,
                                 warmup_turns=a.warmup_turns)
            levels.append(lv)
            print(f"  warm-up {lv['warmup_s_used']:.0f}s ({'ok' if lv['warmup_turns_reached'] else 'CAPPED before every student finished its warm-up turns'}), "
                  f"{lv['turns_per_student_in_window']:.1f} measured turns per student")
            if lv["turns_per_student_in_window"] < 2:
                print("  WARNING: under 2 measured turns per student; raise --duration for a stable number.")
            print(f"  out tok/s {fmt(lv['output_tok_per_s'],0)} | turns/s {fmt(lv['turns_per_s'],3)} | "
                  f"TTFT p95 {fmt(lv['ttft_s'].get('p95'))}s | TPOT p95 {fmt(lv['tpot_ms'].get('p95'),1)}ms | "
                  f"cache hit {fmt(lv['cache_hit_rate_usage'] or lv['cache_hit_rate_metrics'],3)} | "
                  f"preempt {fmt(lv['preemptions'])} | errors {lv['errors']}")
            t95, p95 = lv["ttft_s"].get("p95"), lv["tpot_ms"].get("p95")
            if a.stop_on_slo and t95 and p95 and (t95 > 3 * slo["llm_ttft_p95_s"] or p95 > 3 * slo["llm_tpot_p95_ms"]):
                print("  latency is >3x the SLO; stopping the sweep (higher load only gets worse).")
                break
            if lv["errors"] and lv["turns_completed_in_window"] == 0:
                print("  every request failed; stopping. First errors:", lv["error_samples"])
                break
    finally:
        gpu.stop()

    out = a.out or f"results/{a.profile}_{a.target}.json"
    write_result(out, {
        "bench": "llm_sessions", "profile": a.profile, "target_name": a.target,
        "target": {k: v for k, v in target.items() if k != "api_key_env"},
        "profile": prof, "tokens_per_word": tpw, "force_output_length": force_len,
        "think_time_s": a.think_time, "levels": levels,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="socratic", help="session shape under profiles: in config/llm_targets.yaml")
    ap.add_argument("--target", required=True, help="model name under models: in config/llm_targets.yaml")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32", help="comma list of simultaneous students")
    ap.add_argument("--duration", type=float, default=300, help="measured seconds per level")
    ap.add_argument("--warmup", type=float, default=1200, help="max unmeasured seconds per level (safety cap)")
    ap.add_argument("--warmup-turns", type=int, default=2, help="turns every student completes before measuring")
    ap.add_argument("--grace", type=float, default=180, help="seconds to let in-flight turns finish")
    ap.add_argument("--think-time", type=float, default=0.0, help="mean seconds a student pauses between turns")
    ap.add_argument("--gpus", default=None, help="GPU indices to sample, e.g. 0 or 1,2")
    ap.add_argument("--base-url", default=None, help="override the target's base_url")
    ap.add_argument("--gpu-count", type=int, default=None, help="override how many GPUs the deployment uses (for cost)")
    ap.add_argument("--corpus", default=None, help="text file (e.g. textbook chapters) to sample RAG text from")
    ap.add_argument("--extra-body", default=None, help='JSON merged into each request, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    ap.add_argument("--force-len", dest="force_len", action="store_true", default=None, help="force exact output length (vLLM only)")
    ap.add_argument("--no-force-len", dest="force_len", action="store_false")
    ap.add_argument("--stop-on-slo", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    for k in ("system_prompt_tokens", "turn_new_input_tokens", "turn_output_tokens", "compact_at_tokens"):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=int, default=None, help="override profile value")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
