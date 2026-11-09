#!/usr/bin/env python3
"""Turn benchmark results into cost per student-day and per student-month.

For each spec line it picks the operating point (the highest measured load that
still meets the latency SLOs in config/spec.yaml), converts the measured
throughput into GPU-seconds per student-day, and prices those GPU-seconds:

    $/student-day = GPU-seconds / 3600 x $/GPU-hour / utilization

Lines: 1. ASR, 2. TTS, 3. OCR, then one line per tutor LLM in
config/llm_targets.yaml (4. Gemma 4 31B, 5. ...), each costed on the Socratic
session benchmark. The public benchmark (ShareGPT via vllm bench serve) is shown
beside it for comparison. The total adds ASR + TTS + OCR + the cheapest LLM.

The $/GPU-hour comes from the machine type (spec.yaml `machines`) the results
were produced on; override with --machine or --usd-per-gpu-hour.

  python bench/cost_report.py --results results/cdac-20261105-1300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_yaml, machine_price  # noqa: E402
from html_report import render  # noqa: E402


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first(R: Path, *names):
    """The first of several file names that exists in R (new names first, then older ones)."""
    for n in names:
        d = load(R / n)
        if d:
            return d
    return None


def session_point(res, slo):
    good, best = [], None
    for lv in res["levels"]:
        t95, p95 = lv["ttft_s"].get("p95"), lv["tpot_ms"].get("p95")
        if t95 is None or p95 is None or not lv["output_tok_per_s"]:
            continue
        err_rate = lv["errors"] / max(1, lv["errors"] + lv["turns_completed_in_window"])
        if t95 <= slo["llm_ttft_p95_s"] and p95 <= slo["llm_tpot_p95_ms"] and err_rate < 0.01:
            good.append(lv)
        if best is None or t95 < best["ttft_s"]["p95"]:
            best = lv
    if good:
        return max(good, key=lambda x: x["output_tok_per_s"]), True
    return best, False


def public_point(res, slo):
    """Highest output throughput that meets the latency targets (else the lowest TTFT level)."""
    good, best = [], None
    for lv in res["levels"]:
        t95, p95 = lv["ttft_ms"].get("p95"), lv["tpot_ms"].get("p95")
        if t95 is None or p95 is None or not lv.get("output_throughput"):
            continue
        fail = (lv.get("failed") or 0) / max(1, (lv.get("failed") or 0) + (lv.get("completed") or 0))
        if t95 / 1000 <= slo["llm_ttft_p95_s"] and p95 <= slo["llm_tpot_p95_ms"] and fail < 0.01:
            good.append(lv)
        if best is None or t95 < best["ttft_ms"]["p95"]:
            best = lv
    if good:
        return max(good, key=lambda x: x["output_throughput"]), True
    return best, False


def chk(name, measured, target, unit, kind="max"):
    """One latency check. kind="max": measured must be <= target; "min": >= target."""
    met = None if measured is None else (measured <= target if kind == "max" else measured >= target)
    return {"name": name, "measured": measured, "target": target, "unit": unit, "kind": kind, "met": met}


def llm_price(api, target):
    """Hosted price entry for a model (its api_price key)."""
    table = api.get("llm") or {}
    key = (target or {}).get("api_price")
    return (key, table[key]) if key in table else (None, None)


def fmt_check(c):
    """'target ≤ 3.0 s, measured 5.41 s (1.8× the target)'."""
    if c["measured"] is None:
        return f"{c['name']}: target {'≤' if c['kind'] == 'max' else '≥'} {c['target']:g} {c['unit']}, not measured"
    m, t = c["measured"], c["target"]
    if c["kind"] == "max":
        verdict = (f"{m / t:.1f}× the target" if m > t else f"within, {m / t:.0%} of the target")
        return f"{c['name']}: target ≤ {t:g} {c['unit']}, measured {m:.2f} {c['unit']} ({verdict})"
    verdict = (f"{1 - m / t:.0%} below the target" if m < t else "within")
    return f"{c['name']}: target ≥ {t:g}{c['unit']}, measured {m:.2f}{c['unit']} ({verdict})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="folder with one run's result files")
    ap.add_argument("--out", default=None, help="markdown path (default <results>/report.md)")
    ap.add_argument("--machine", default=None, help="price as this machine type from spec.yaml (cdac, ryzen, ...)")
    ap.add_argument("--usd-per-gpu-hour", type=float, default=None, help="price at exactly this $/GPU-hour")
    a = ap.parse_args()
    R = Path(a.results)
    spec = load_yaml("spec.yaml")
    targets = load_yaml("llm_targets.yaml")
    models = targets.get("models") or {}
    day, slo, gcfg, bill = spec["student_day"], spec["slo"], spec["gpu"], spec["billing"]
    dpm, fx, util = bill["days_per_month"], bill["inr_per_usd"], gcfg["utilization"]
    api = spec["api_reference"]
    mref = spec.get("market_reference", {})

    # Which machine produced these results, and its $/GPU-hour.
    machine = {}
    for p in sorted(R.glob("*.json")):
        d = load(p)
        if isinstance(d, dict) and d.get("machine"):
            machine = d["machine"]
            if machine.get("gpus"):
                break
    pricing = machine_price(spec, machine, a.machine, a.usd_per_gpu_hour)
    price = pricing["usd_per_gpu_hour"]
    usd_per_gpu_s = price / 3600
    print(f"[price] ${price:.2f} per GPU-hour: {pricing['label']} ({pricing['source']}; {pricing['why_machine']})")

    rows, notes, units, glance, public = [], [], [], [], []
    raw = {"sessions": []}

    def unit(line, model, unit_name, throughput, gpu_s_per_unit, market=None, note=""):
        units.append({"line": line, "model": model, "unit": unit_name, "throughput": throughput,
                      "gpu_s_per_unit": gpu_s_per_unit,
                      "usd_per_unit": None if gpu_s_per_unit is None else gpu_s_per_unit * usd_per_gpu_s,
                      "market": market, "note": note})

    def add(line, model, measured, throughput, gpu_s, api_usd_day=None, slo_ok=True, kind="speech", extra=None):
        c_day = None if gpu_s is None else gpu_s * usd_per_gpu_s
        rows.append({"line": line, "kind": kind, "model": model, "measured": measured, "throughput": throughput,
                     "gpu_s_per_student_day": gpu_s, "usd_day_compute": c_day,
                     "usd_day_at_util": None if c_day is None else c_day / util,
                     "usd_month_at_util": None if c_day is None else c_day / util * dpm,
                     "api_usd_day": api_usd_day, "api_usd_month": None if api_usd_day is None else api_usd_day * dpm,
                     "slo_met": slo_ok, **(extra or {})})

    # ---- 1. ASR
    asr = load(R / "asr.json")
    asr_api = day["asr_minutes"] * api["bodhan_hosted"]["asr_inr_per_audio_minute"] / fx
    if asr and asr.get("throughput"):
        best = max(asr["throughput"], key=lambda x: x["rtfx"])
        ok = (asr["latency_batch1_s"].get("p95") or 99) <= slo["asr_latency_p95_s"]
        unit("1. ASR", "Indic-Transcribe-core", "1 minute of audio", f"RTFx {best['rtfx']:.0f} (batch {best['batch_size']})",
             60 / best["rtfx"], mref.get("asr_usd_per_min"))
        add("1. ASR (10 min)", "Indic-Transcribe-core", f"batch {best['batch_size']}, {asr['backend']}",
            f"RTFx {best['rtfx']:.0f}", day["asr_minutes"] * 60 / best["rtfx"], asr_api, ok,
            extra={"latency_p95_s": asr["latency_batch1_s"].get("p95"), "wer": asr.get("wer"),
                   "checks": [chk(f"p95 latency, ~{asr.get('utterance_s', 20):.0f} s utterance",
                                  asr["latency_batch1_s"].get("p95"), slo["asr_latency_p95_s"], "s")]})
    else:
        add("1. ASR (10 min)", "Indic-Transcribe-core", "not run yet", "-", None, asr_api, None)
        unit("1. ASR", "Indic-Transcribe-core", "1 minute of audio", "not run yet", None, mref.get("asr_usd_per_min"))

    # ---- 2. TTS
    tts_v, tts_h = load(R / "tts_vllm.json"), load(R / "tts_hf.json")
    cps = (tts_h or {}).get("chars_per_audio_s") or api["tts_chars_per_audio_second"]
    tts_api = day["tts_minutes"] * 60 * cps / 10000 * api["bodhan_hosted"]["tts_inr_per_10k_chars"] / fx
    if tts_v and tts_v.get("levels"):
        margin = tts_v.get("realtime_margin", slo["tts_realtime_margin"])
        good = [l for l in tts_v["levels"] if (l["stream_speed_x_realtime"].get("p5") or 0) >= margin and not l["errors"]]
        pick = max(good or tts_v["levels"], key=lambda l: l["audio_s_per_s"])
        unit("2. TTS", "Indic-Speak (vLLM)", "1 minute of audio", f"{pick['audio_s_per_s']:.0f} audio-s per GPU-s ({pick['concurrency']} streams)",
             60 / pick["audio_s_per_s"], None, "Vocos decoder not included")
        unit("2. TTS", "Indic-Speak (vLLM)", "1M characters of text", f"at {cps:.1f} characters per audio second",
             1e6 / cps / pick["audio_s_per_s"], mref.get("tts_usd_per_m_chars"))
        add("2. TTS (10 min)", "Indic-Speak (LM on vLLM)", f"{pick['concurrency']} streams",
            f"{pick['audio_s_per_s']:.0f} s audio / GPU-s", day["tts_minutes"] * 60 / pick["audio_s_per_s"], tts_api, bool(good),
            extra={"stream_speed_p5_x": pick["stream_speed_x_realtime"].get("p5"),
                   "share_hit_max_tokens": pick.get("share_hit_max_tokens"),
                   "checks": [chk("slowest 5% of streams vs playback speed", pick["stream_speed_x_realtime"].get("p5"),
                                  margin, "× real time", "min")]})
        notes.append("TTS (vLLM mode) excludes the Vocos decoder; compare with the single-stream run to size it.")
        if (pick.get("share_hit_max_tokens") or 0) > 0.5:
            notes.append("TTS WARNING: most requests hit max_tokens (wrong stop tokens?), so audio length and cost are unreliable.")
    elif tts_h:
        unit("2. TTS", "Indic-Speak (one stream)", "1 minute of audio", f"RTFx {tts_h['rtfx_single_stream']:.1f}, one stream",
             60 / tts_h["rtfx_single_stream"], None)
        unit("2. TTS", "Indic-Speak (one stream)", "1M characters of text", f"at {cps:.1f} characters per audio second",
             1e6 / cps / tts_h["rtfx_single_stream"], mref.get("tts_usd_per_m_chars"))
        add("2. TTS (10 min)", "Indic-Speak (transformers)", "single stream, end to end",
            f"RTFx {tts_h['rtfx_single_stream']:.1f}", day["tts_minutes"] * 60 / tts_h["rtfx_single_stream"], tts_api)
        notes.append("TTS is priced from the single-stream run, an upper bound on cost; the batched vLLM run didn't produce results.")
    else:
        add("2. TTS (10 min)", "Indic-Speak", "not run yet", "-", None, tts_api, None)
        unit("2. TTS", "Indic-Speak", "1M characters of text", "not run yet", None, mref.get("tts_usd_per_m_chars"))

    # ---- 3. OCR
    ocr = load(R / "ocr.json")
    ocr_api = day["ocr_pages"] * api["bodhan_hosted"]["ocr_inr_per_image"] / fx
    if ocr and ocr.get("levels"):
        good = [l for l in ocr["levels"] if (l["latency_s"].get("p95") or 99) <= slo["ocr_page_latency_p95_s"] and not l["errors"]]
        pick = max(good or ocr["levels"], key=lambda l: l["pages_per_s"])
        unit("3. OCR", "IndicOCR", "1,000 pages", f"{pick['pages_per_s']:.2f} pages/s ({pick['workers']} model copies)",
             1000 / pick["pages_per_s"], mref.get("ocr_usd_per_1k_pages"))
        add("3. OCR (10 pages)", "IndicOCR", f"{pick['workers']} model cop{'y' if pick['workers'] == 1 else 'ies'}",
            f"{pick['pages_per_s']:.2f} pages/s", day["ocr_pages"] / pick["pages_per_s"], ocr_api, bool(good),
            extra={"latency_p95_s": pick["latency_s"].get("p95"),
                   "checks": [chk("p95 page latency", pick["latency_s"].get("p95"), slo["ocr_page_latency_p95_s"], "s")]})
        if ocr.get("n_inputs", 99) < 50:
            notes.append(f"OCR ran on only {ocr.get('n_inputs')} distinct pages; add more to {'$INPUTS/pages'} for a steadier number.")
    else:
        add("3. OCR (10 pages)", "IndicOCR", "not run yet", "-", None, ocr_api, None)
        unit("3. OCR", "IndicOCR", "1,000 pages", "not run yet", None, mref.get("ocr_usd_per_1k_pages"))

    # ---- 4+. one line per tutor LLM, costed on the Socratic sessions
    llm_day = day["llm"]
    out_day = llm_day["output_tokens"]
    for i, (tname, tcfg) in enumerate(models.items()):
        n = 4 + i
        label = tcfg.get("name", tname)
        line = f"{n}. {label}"
        pname, pr = llm_price(api, tcfg)
        api_day = None
        if pr:
            api_day = (llm_day["cached_input_tokens"] * pr.get("usd_per_m_cached", pr["usd_per_m_input"])
                       + llm_day["uncached_input_tokens"] * pr["usd_per_m_input"] + out_day * pr["usd_per_m_output"]) / 1e6
        res = first(R, f"socratic_{tname}.json", f"workhorse_{tname}.json")
        lv, ok = session_point(res, slo) if res and res.get("levels") else (None, None)
        if lv is None:
            add(line, label, "not run yet", "-", None, api_day, None, kind="llm", extra={"target": tname})
            unit(line, label, "1M output tokens, all-in", "not run yet", None, None)
        else:
            raw["sessions"].append((res, lv["concurrency"], label))
            gpus = res["target"].get("gpus", 1)
            gpu_s = out_day / lv["output_tok_per_s"] * gpus
            mix = lv["mix_per_turn"]
            turns = out_day / (mix["completion_tokens"] or 1)
            cached_day = (mix["cached_tokens"] or 0) * turns if mix["cached_tokens"] is not None else None
            uncached_day = (mix["prompt_tokens"] - (mix["cached_tokens"] or 0)) * turns if mix["cached_tokens"] is not None else None
            plural = "student" if lv["concurrency"] == 1 else "students"
            who = f"{label} ({lv['concurrency']} {plural})"
            unit(line, who, "1M output tokens, all-in", f"{lv['output_tok_per_s']:.0f} output tok/s with the measured input mix",
                 1e6 / lv["output_tok_per_s"] * gpus, None, "includes reading the prompts that came with them")
            # Approximate split: each turn's time before its first token is prompt reading,
            # the rest is writing. Cached tokens cost memory, not compute.
            ttft_m, e2e_m = lv["ttft_s"].get("mean"), lv["e2e_s"].get("mean")
            if ttft_m and e2e_m and mix["cached_tokens"] is not None:
                share = ttft_m / e2e_m
                gpu_s_turn = gpus * mix["completion_tokens"] / lv["output_tok_per_s"]
                unc = max(1.0, mix["prompt_tokens"] - mix["cached_tokens"])
                unit(line, who, "1M cached input tokens", "kept in GPU memory", 0.0,
                     mref.get("llm_usd_per_m_cached"), "≈0 compute; their cost shows up as fewer students fitting in memory")
                unit(line, who, "1M new input tokens", f"{share:.0%} of each turn is prompt reading (approx.)",
                     share * gpu_s_turn / unc * 1e6, mref.get("llm_usd_per_m_input"), "approximate split")
                unit(line, who, "1M output tokens", f"{1 - share:.0%} of each turn is writing (approx.)",
                     (1 - share) * gpu_s_turn / mix["completion_tokens"] * 1e6, mref.get("llm_usd_per_m_output"), "approximate split")
            add(line, label, f"Socratic, {lv['concurrency']} {plural} x {gpus} GPU", f"{lv['output_tok_per_s']:.0f} out tok/s",
                gpu_s, api_day, ok, kind="llm",
                extra={"target": tname, "ttft_p95_s": lv["ttft_s"].get("p95"), "tpot_p95_ms": lv["tpot_ms"].get("p95"),
                       "checks": [chk("wait for first token, p95", lv["ttft_s"].get("p95"), slo["llm_ttft_p95_s"], "s"),
                                  chk("time per output token, p95", lv["tpot_ms"].get("p95"), slo["llm_tpot_p95_ms"], "ms")],
                       "cache_hit_rate": lv.get("cache_hit_rate_usage") or lv.get("cache_hit_rate_metrics"),
                       "measured_cached_tokens_per_day": cached_day, "measured_uncached_tokens_per_day": uncached_day,
                       "preemptions": lv.get("preemptions")})
            glance.append({"model": label, "workload": "Socratic sessions (our spec)", "op": f"{lv['concurrency']} {plural}",
                           "out_tps": lv["output_tok_per_s"], "mean_in": mix["prompt_tokens"], "mean_out": mix["completion_tokens"],
                           "gs_per_m_out": 1e6 / lv["output_tok_per_s"] * gpus,
                           "api_per_m_out": None if api_day is None else api_day / (out_day / 1e6), "slo_ok": ok})
            if cached_day is not None:
                notes.append(f"{label}: measured mix per student-day = {cached_day / 1e6:.2f}M cached + "
                             f"{uncached_day / 1e3:.0f}K uncached input (spec: {llm_day['cached_input_tokens'] / 1e6:.1f}M + "
                             f"{llm_day['uncached_input_tokens'] / 1e3:.0f}K); cache hit rate {(lv.get('cache_hit_rate_usage') or 0):.1%}.")
            if not ok:
                notes.append(f"{label}: no Socratic load level met the latency targets; priced at its lowest-latency level.")

        # Public benchmark for the same model (comparison only; not in the totals)
        pubs = sorted(R.glob(f"public_{tname}_*.json")) or sorted(R.glob(f"standard_{tname}_*.json"))
        for p in pubs:
            pres = load(p)
            if not pres or not pres.get("levels"):
                continue
            plv, pok = public_point(pres, slo)
            gpus = pres["target"].get("gpus", 1)
            for l in pres["levels"]:   # GPU-seconds used vs the hosted bill for the same tokens
                l["gpu_s"] = (l.get("duration_s") or 0) * gpus
                l["api_usd"] = None if not pr else ((l.get("total_input_tokens") or 0) * pr["usd_per_m_input"]
                                                    + (l.get("total_output_tokens") or 0) * pr["usd_per_m_output"]) / 1e6
            public.append({"res": pres, "label": label, "op": plv["concurrency"] if plv else None, "ok": pok, "api_name": pname})
            if plv and plv.get("total_output_tokens"):
                k = max(1, plv.get("completed") or 1)
                per_m = 1e6 / plv["total_output_tokens"]
                glance.append({"model": label, "workload": "Public benchmark, " + pres.get("dataset_desc", pres["dataset"]).split(":")[0],
                               "op": f"{plv['concurrency']} requests at once", "out_tps": plv["output_throughput"],
                               "mean_in": (plv.get("total_input_tokens") or 0) / k, "mean_out": plv["total_output_tokens"] / k,
                               "gs_per_m_out": plv["gpu_s"] * per_m,
                               "api_per_m_out": None if plv["api_usd"] is None else plv["api_usd"] * per_m, "slo_ok": pok})
                # Unit costs from the public run, same rows as the Socratic ones (comparison only, not in totals)
                ds_name = pres.get("dataset_desc", pres["dataset"]).split(":")[0]
                who = f"{label} ({ds_name}, {plv['concurrency']} requests at once)"
                tag = "public benchmark, for comparison; not in the totals"
                mi, mo = (plv.get("total_input_tokens") or 0) / k, plv["total_output_tokens"] / k
                unit(line, who, "1M output tokens, all-in",
                     f"{plv['output_throughput']:.0f} output tok/s, ~{mi:,.0f} in / {mo:,.0f} out tokens per request",
                     plv["gpu_s"] * per_m, None, tag + "; includes reading the prompts")
                ttft_m, e2e_m = plv["ttft_ms"].get("mean"), plv["e2el_ms"].get("mean")
                if ttft_m and e2e_m and plv.get("total_input_tokens"):
                    share = min(1.0, ttft_m / e2e_m)
                    unit(line, who, "1M input tokens", f"{share:.0%} of each request is prompt reading (approx.)",
                         share * plv["gpu_s"] / plv["total_input_tokens"] * 1e6, mref.get("llm_usd_per_m_input"),
                         tag + "; approximate split")
                    unit(line, who, "1M output tokens", f"{1 - share:.0%} of each request is writing (approx.)",
                         (1 - share) * plv["gpu_s"] * per_m, mref.get("llm_usd_per_m_output"), tag + "; approximate split")
            if not pok:
                notes.append(f"{label}, public benchmark ({pres['dataset']}): no level met the latency targets; shown at its lowest-latency level.")

    # ---- totals: ASR + TTS + OCR + the cheapest LLM that ran
    speech = [r for r in rows if r["kind"] == "speech"]
    llms = [r for r in rows if r["kind"] == "llm" and r["usd_day_compute"] is not None]
    llm_pick = min(llms, key=lambda r: r["usd_day_compute"]) if llms else None
    chosen = [r for r in speech if r["usd_day_compute"] is not None] + ([llm_pick] if llm_pick else [])
    missing = [r["line"].split(" (")[0] for r in speech if r["usd_day_compute"] is None] + ([] if llm_pick else ["LLM"])
    pure_day = sum(r["usd_day_compute"] for r in chosen)
    total = pure_day / util * dpm
    if len(llms) > 1:
        notes.append(f"Totals use the cheapest LLM line ({llm_pick['model']}); the others are shown for comparison.")

    def f(x, d=2, pre="$"):
        return "-" if x is None else f"{pre}{x:,.{d}f}"

    lines = [
        f"# Cost per student: {pricing['label']}", "",
        f"${price:.2f} per GPU-hour ({pricing['source']}; machine chosen by {pricing['why_machine']}), utilization {util:.0%}, "
        f"{dpm} days/month, INR/USD {fx}. Budget target ${bill['target_usd_per_student_month']:.2f}/student-month.", "",
        "| Line | Model | Operating point | Throughput | GPU-s / student-day | $/day (pure compute) | $/month at utilization | Hosted API $/month | Latency target vs measured |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        checks = "; ".join(fmt_check(c) for c in r.get("checks") or []) or "-"
        lines.append(f"| {r['line']} | {r['model']} | {r['measured']} | {r['throughput']} | "
                     f"{f(r['gpu_s_per_student_day'], 1, '')} | {f(r['usd_day_compute'], 4)} | "
                     f"{f(r['usd_month_at_util'])} | {f(r.get('api_usd_month'))} | {checks} |")
    miss = f" (missing: {', '.join(missing)})" if missing else ""
    with_llm = f" with {llm_pick['model']}" if llm_pick else ""
    lines += ["", f"**Total{with_llm}: {f(pure_day * dpm)} per student-month pure compute, "
                  f"{f(total)} at {util:.0%} utilization{miss}**", ""]
    if glance:
        lines += ["## LLM: Socratic sessions vs public benchmark", "",
                  "| Model | Workload | Operating point | Output tok/s | Mean tokens in / out | $ per 1M output (all-in) | Hosted API, same tokens | Ratio |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g in glance:
            usd_m = g["gs_per_m_out"] * usd_per_gpu_s
            ratio = "-" if not g["api_per_m_out"] else f"{usd_m / g['api_per_m_out']:.1f}x"
            lines.append(f"| {g['model']} | {g['workload']} | {g['op']}{'' if g['slo_ok'] else ' (missed latency target)'} | "
                         f"{g['out_tps']:,.0f} | {g['mean_in']:,.0f} / {g['mean_out']:,.0f} | {f(usd_m)} | {f(g['api_per_m_out'])} | {ratio} |")
        lines.append("")
    lines += [f"- Not costed yet: {n}" for n in spec.get("not_costed_yet", [])]
    lines += [""] + [f"- {n}" for n in notes]
    raw.update({"asr": asr, "tts_vllm": tts_v, "tts_hf": tts_h, "ocr": ocr})
    page = render({"spec": spec, "rows": rows, "notes": notes, "missing": missing, "raw": raw, "machine": machine,
                   "units": units, "chosen": chosen, "llm_pick": llm_pick, "pricing": pricing, "glance": glance,
                   "public": public})
    html_path = R / "report.html"
    html_path.write_text(page, encoding="utf-8")
    out = Path(a.out or R / "report.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (R / "report.json").write_text(json.dumps({"pricing": pricing, "lines": rows, "units": units, "llm_comparison": glance,
                                               "total_usd_month_pure_compute": pure_day * dpm, "total_usd_month_at_util": total,
                                               "missing": missing, "notes": notes, "spec": spec}, indent=2, default=str),
                                   encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[saved] {html_path} (open in a browser), {out} and {R / 'report.json'}")


if __name__ == "__main__":
    main()
