#!/usr/bin/env python3
"""Text-to-speech benchmark for Bodhan Indic-Speak (spec line 2: 10 min/day).

Two modes:

  --mode hf    End-to-end with the model repo's own TTS class (transformers +
               SNAC + Vocos). One request at a time. Gives real audio, the
               real-time factor of the full pipeline, peak GPU memory, and the
               WAV files the ASR benchmark replays. Run in the TTS venv.

  --mode vllm  The speech LM (a Llama-3.2-3B that emits SNAC audio tokens)
               served by vLLM, driven with N concurrent requests. This is how
               you would serve it at scale. Audio tokens are converted to
               seconds of speech (SNAC 24 kHz = 7 tokens per 11.72 Hz frame =
               ~82 tokens/s of audio). The small Vocos decoder is not run in
               this mode; its cost shows up in --mode hf. Run from any env.

Per-stream speed matters for TTS: if a stream generates audio tokens slower
than playback, the student hears gaps. The operating point is the highest
concurrency where p95 stream speed stays >= slo.tts_realtime_margin x real time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GpuSampler, load_yaml, summarize, write_result, resolve_model_dir  # noqa: E402

SNAC_TOKENS_PER_AUDIO_S = 7 * 24000 / 2048  # 82.03

# Tutoring-style sample lines (Hindi, English, code-mixed, Kannada). Replace with
# your own via --texts (TSV: lang<TAB>speaker<TAB>text) for production numbers.
SAMPLES = [
    ("hi", "Amit", "प्रकाश की चाल लगभग तीन लाख किलोमीटर प्रति सेकंड होती है।"),
    ("hi", "Kavya", "आज हम भिन्न को जोड़ना सीखेंगे। पहले दोनों भिन्नों के हर देखो।"),
    ("hi", "Amit", "अगर किसी त्रिभुज के दो कोण साठ डिग्री के हैं, तो तीसरा कोण कितना होगा?"),
    ("hi", "Kavya", "Photosynthesis में पौधे sunlight, पानी और carbon dioxide से glucose बनाते हैं।"),
    ("hi", "Amit", "पानी सौ डिग्री सेल्सियस पर उबलता है। क्या पहाड़ों पर यह तापमान बदलता है?"),
    ("hi", "Kavya", "अगर दो पेन बीस रुपये के हैं, तो पाँच पेन कितने रुपये के होंगे? पहले एक पेन का दाम निकालो।"),
    ("en", "Kavya", "Good try! Before we check the answer, can you tell me what the question is really asking?"),
    ("en", "Amit", "Let's break this problem into smaller steps. What do we already know about the speed of the train?"),
    ("en", "Kavya", "Newton's second law says force equals mass times acceleration. What happens to the force if the mass doubles?"),
    ("en", "Amit", "That's close. Look at the units again: are we measuring distance in metres or in kilometres?"),
    ("en", "Kavya", "Great work. Now try the next question on your own, and tell me which formula you chose and why."),
    ("kn", "Deepika", "ನೀರು ನೂರು ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್‌ನಲ್ಲಿ ಕುದಿಯುತ್ತದೆ."),
]


def load_texts(path: str | None):
    if not path:
        return SAMPLES
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            lang, spk, text = line.split("\t", 2)
            rows.append((lang, spk, text))
    return rows


# ------------------------------------------------------------------ hf mode

def run_hf(a, texts):
    import numpy as np  # noqa: F401
    import soundfile as sf
    import torch

    model_dir = resolve_model_dir(a.model_dir)
    sys.path.insert(0, model_dir)
    from inference import TTS  # ships inside the model repo

    t_load = time.perf_counter()
    tts = TTS(model_dir)
    load_s = time.perf_counter() - t_load
    sr = getattr(tts, "sample_rate", 24000)
    wav_dir = Path(a.save_wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    # warm-up (CUDA kernels, caches)
    tts(texts[0][2], speaker=texts[0][1], seed=0)
    torch.cuda.reset_peak_memory_stats()

    gpu = GpuSampler([a.gpu] if a.gpu is not None else None).start()
    rows, manifest = [], []
    t0 = time.time()
    for rep in range(a.repeats):
        for i, (lang, spk, text) in enumerate(texts):
            torch.cuda.synchronize()
            s = time.perf_counter()
            wav = tts(text, speaker=spk, seed=1000 + i + rep)
            torch.cuda.synchronize()
            wall = time.perf_counter() - s
            dur = len(wav) / sr
            rows.append({"lang": lang, "speaker": spk, "chars": len(text), "wall_s": wall,
                         "audio_s": dur, "rtf": wall / dur if dur else None})
            if rep == 0:
                p = wav_dir / f"tts_{i:03d}_{lang}.wav"
                sf.write(p, wav, sr)
                manifest.append(f"{p}\t{lang}\t{text}")
            print(f"  [{lang}] {len(text):3d} chars -> {dur:5.2f}s audio in {wall:5.2f}s  (RTF {wall / dur:.2f})")
    t1 = time.time()
    gpu.stop()
    (wav_dir / "manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    total_audio = sum(r["audio_s"] for r in rows)
    total_wall = sum(r["wall_s"] for r in rows)
    return {
        "mode": "hf", "model_dir": model_dir, "load_s": load_s, "items": len(rows),
        "audio_s_total": total_audio, "wall_s_total": total_wall,
        "rtfx_single_stream": total_audio / total_wall,       # seconds of audio per GPU-second
        "rtf": summarize([r["rtf"] for r in rows]),
        "latency_s": summarize([r["wall_s"] for r in rows]),
        "chars_per_audio_s": sum(r["chars"] for r in rows) / total_audio,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "gpu": gpu.window(t0, t1), "rows": rows,
        "wav_manifest": str(wav_dir / "manifest.tsv"),
    }


# ---------------------------------------------------------------- vllm mode

def build_prompt(speaker: str, text: str) -> str:
    # Prompt contract from the Indic-Speak model card.
    return (f"<|start_of_human|><|begin_of_text|><|speaker>{speaker}<speaker|>\n{text}"
            f"<|eot_id|><|end_of_human|><|start_of_ai|><|start_of_speech|>")


async def one_tts(http, base_url, model, speaker, text, stops, max_tokens):
    import aiohttp

    body = {"model": model, "prompt": build_prompt(speaker, text), "max_tokens": max_tokens,
            "temperature": 0.6, "top_p": 0.9, "top_k": 50, "stream": True,
            "stream_options": {"include_usage": True},
            "add_special_tokens": False, "skip_special_tokens": False, "stop": stops}
    t0 = time.perf_counter()
    first, chunks, usage, err = None, 0, None, None
    t_28 = None
    try:
        async with http.post(base_url.rstrip("/") + "/completions", json=body,
                             timeout=aiohttp.ClientTimeout(total=600)) as r:
            if r.status != 200:
                err = f"HTTP {r.status}: {(await r.text())[:200]}"
            else:
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for ch in obj.get("choices") or []:
                        if ch.get("text"):
                            chunks += 1
                            now = time.perf_counter() - t0
                            first = first if first is not None else now
                            if chunks >= 28 and t_28 is None:
                                t_28 = now
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    e2e = time.perf_counter() - t0
    n = (usage or {}).get("completion_tokens") or chunks
    return {"ok": err is None and n > 0, "error": err, "e2e_s": e2e, "ttft_s": first,
            "first_4_frames_s": t_28, "tokens": n, "hit_max_tokens": n >= max_tokens, "chars": len(text)}


async def run_vllm_level(a, texts, conc, duration, warmup):
    import aiohttp

    recs = []
    t_begin = time.time()
    t_warm, t_stop = t_begin + warmup, t_begin + warmup + duration
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as http:
        async def worker(w):
            i = w
            while time.time() < t_stop:
                lang, spk, text = texts[i % len(texts)]
                i += conc
                r = await one_tts(http, a.base_url, a.model, spk, text, a.stop, a.max_tokens)
                r["t_end"] = time.time()
                r["t_start"] = r["t_end"] - r["e2e_s"]
                recs.append(r)
        await asyncio.gather(*[worker(w) for w in range(conc)])
    ok = [r for r in recs if r["ok"]]
    win = [r for r in ok if r["t_start"] >= t_warm]
    W = t_stop - t_warm
    tok_in_window = 0.0
    for r in ok:
        dec0 = r["t_start"] + (r["ttft_s"] or 0)
        span = max(1e-9, r["t_end"] - dec0)
        tok_in_window += r["tokens"] * max(0.0, min(r["t_end"], t_stop) - max(dec0, t_warm)) / span
    stream_audio_x = [r["tokens"] / max(1e-9, r["e2e_s"] - (r["ttft_s"] or 0)) / a.tokens_per_audio_s for r in win]
    return {
        "concurrency": conc, "window_s": W, "requests_in_window": len(win),
        "audio_tokens_per_s": tok_in_window / W,
        "audio_s_per_s": tok_in_window / W / a.tokens_per_audio_s,   # RTFx of the whole GPU
        "stream_speed_x_realtime": summarize(stream_audio_x),         # >1 = faster than playback
        "ttft_s": summarize([r["ttft_s"] for r in win]),
        "first_4_frames_s": summarize([r["first_4_frames_s"] for r in win]),
        "e2e_s": summarize([r["e2e_s"] for r in win]),
        "audio_s_per_request": summarize([r["tokens"] / a.tokens_per_audio_s for r in win]),
        "share_hit_max_tokens": (sum(r["hit_max_tokens"] for r in win) / len(win)) if win else None,
        "errors": sum(1 for r in recs if not r["ok"]),
        "error_samples": list({r["error"] for r in recs if r["error"]})[:3],
    }


def run_vllm(a, texts):
    margin = load_yaml("spec.yaml")["slo"]["tts_realtime_margin"]
    gpu = GpuSampler([a.gpu] if a.gpu is not None else None).start()
    levels = []
    try:
        for c in [int(x) for x in a.concurrency.split(",")]:
            t0 = time.time()
            lv = asyncio.run(run_vllm_level(a, texts, c, a.duration, a.warmup))
            lv["gpu"] = gpu.window(t0 + a.warmup, time.time())
            levels.append(lv)
            p5 = lv["stream_speed_x_realtime"].get("p50")
            print(f"  conc {c:3d}: {lv['audio_s_per_s']:.1f} s of audio per s | stream speed p50 "
                  f"{p5 if p5 is None else round(p5, 2)}x real time | TTFA(4 frames) p95 "
                  f"{lv['first_4_frames_s'].get('p95')} | hit max_tokens {lv['share_hit_max_tokens']} | errors {lv['errors']}")
            if lv["share_hit_max_tokens"] and lv["share_hit_max_tokens"] > 0.5:
                print("  WARNING: most requests ran to max_tokens. The stop tokens are probably wrong;"
                      " check token_contract.md in the model repo and pass --stop.")
            if p5 is not None and p5 < margin * 0.5:
                print("  streams are far slower than real time; stopping the sweep.")
                break
    finally:
        gpu.stop()
    return {"mode": "vllm", "tokens_per_audio_s": a.tokens_per_audio_s, "levels": levels,
            "realtime_margin": margin}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["hf", "vllm"], required=True)
    ap.add_argument("--model-dir", default="bodhan-ai/indic-speak", help="local snapshot or hub id (hf mode)")
    ap.add_argument("--texts", default=None, help="TSV of lang<TAB>speaker<TAB>text")
    ap.add_argument("--repeats", type=int, default=3, help="hf mode: passes over the texts")
    ap.add_argument("--save-wav-dir", default=os.path.join(os.environ.get("INPUTS", "/data/tokenfactory/bench-inputs"), "tts_wavs"), help="hf mode: where WAVs go (ASR input)")
    ap.add_argument("--base-url", default="http://localhost:8004/v1", help="vllm mode endpoint")
    ap.add_argument("--model", default="indic-speak", help="vllm served model name")
    ap.add_argument("--concurrency", default="1,4,8,16,32,64")
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--warmup", type=float, default=20)
    ap.add_argument("--max-tokens", type=int, default=2520)
    ap.add_argument("--stop", nargs="*", default=["<|end_of_speech|>", "<|end_of_ai|>"])
    ap.add_argument("--tokens-per-audio-s", type=float, default=SNAC_TOKENS_PER_AUDIO_S)
    ap.add_argument("--gpu", type=int, default=None, help="GPU index to sample")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    texts = load_texts(a.texts)
    res = run_hf(a, texts) if a.mode == "hf" else run_vllm(a, texts)
    write_result(a.out or f"results/tts_{a.mode}.json", {"bench": "tts", **res})


if __name__ == "__main__":
    main()
