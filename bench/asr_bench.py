#!/usr/bin/env python3
"""Speech-to-text benchmark for Bodhan Indic-Transcribe (spec line 1: 10 min/day).

Measures, on one GPU:
  * batch-1 latency per utterance (what a student waits for after speaking)
  * throughput at several batch sizes, as RTFx = seconds of audio transcribed
    per second of GPU time (the number the cost report uses)
  * peak GPU memory, mean utilisation, power
  * WER / CER when reference transcripts are available

Input: either a manifest TSV (path<TAB>lang<TAB>reference) such as the one the
TTS benchmark writes, or a folder of WAVs plus --lang. Short clips are joined
into ~20 s utterances (the model's sweet spot; it accepts up to 30 s per pass).

Backends:
  nemo  The repo's NeMo checkpoint; batches clips (fastest). Needs nemo_toolkit[asr].
  hf    The repo's own Python class; one clip at a time (no NeMo needed).

Run in the ASR venv:
  python bench/asr_bench.py --model-dir bodhan-ai/indic-transcribe-core \
      --manifest /data/tokenfactory/bench-inputs/tts_wavs/manifest.tsv --gpu 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GpuSampler, edit_distance, summarize, write_result, resolve_model_dir  # noqa: E402


def read_inputs(a):
    items = []
    if a.manifest:
        base = Path(a.manifest).resolve().parent
        for line in Path(a.manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                parts = line.split("\t")
                path = parts[0] if Path(parts[0]).is_absolute() else str(base / parts[0])  # relative = next to manifest
                items.append((path, parts[1] if len(parts) > 1 else a.lang, parts[2] if len(parts) > 2 else None))
    else:
        refs = {}
        ref_file = Path(a.audio_dir) / "refs.tsv"
        if ref_file.exists():
            for line in ref_file.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    name, txt = line.split("\t", 1)
                    refs[name] = txt
        for p in sorted(Path(a.audio_dir).glob("*.wav")):
            items.append((str(p), a.lang, refs.get(p.name)))
    if not items:
        sys.exit("no audio found: pass --manifest or --audio-dir with .wav files")
    return items


def make_utterances(items, out_dir: Path, target_s: float, max_s: float = 29.0):
    """Join same-language clips (16 kHz mono) into ~target_s utterances."""
    import numpy as np
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    by_lang: dict[str, list] = {}
    for path, lang, ref in items:
        by_lang.setdefault(lang, []).append((path, ref))
    utts = []
    for lang, clips in by_lang.items():
        buf, refs, n = [], [], 0
        for i, (path, ref) in enumerate(clips + clips):  # loop twice so short sets still fill
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                idx = np.linspace(0, len(wav) - 1, int(len(wav) * 16000 / sr))
                wav = np.interp(idx, np.arange(len(wav)), wav).astype("float32")
            wav = wav[: int(max_s * 16000)]
            cur = sum(len(b) for b in buf) / 16000
            if buf and cur + len(wav) / 16000 > max_s:
                utts.append(_flush(out_dir, lang, n, buf, refs)); n += 1; buf, refs = [], []
            buf += [wav, np.zeros(int(0.4 * 16000), dtype="float32")]
            refs.append(ref)
            if sum(len(b) for b in buf) / 16000 >= target_s:
                utts.append(_flush(out_dir, lang, n, buf, refs)); n += 1; buf, refs = [], []
        if buf:
            utts.append(_flush(out_dir, lang, n, buf, refs))
    return utts


def _flush(out_dir, lang, n, buf, refs):
    import numpy as np
    import soundfile as sf

    wav = np.concatenate(buf)
    p = out_dir / f"utt_{lang}_{n:03d}.wav"
    sf.write(p, wav, 16000)
    ref = " ".join(r for r in refs if r) if all(refs) else None
    return {"path": str(p), "lang": lang, "ref": ref, "audio_s": len(wav) / 16000}


def norm_words(s: str) -> list[str]:
    s = "".join(ch for ch in unicodedata.normalize("NFC", s) if not unicodedata.category(ch).startswith("P"))
    return s.lower().split()


class Backend:
    def __init__(self, kind: str, model_dir: str):
        model_dir = resolve_model_dir(model_dir)
        self.kind, self.model_dir = kind, model_dir
        if kind == "nemo":
            sys.path.insert(0, f"{model_dir}/nemo")
            from load_nemo import load_nemo_model  # ships inside the model repo
            self.model = load_nemo_model(model_dir)
            self.model.eval()
        else:
            sys.path.insert(0, model_dir)
            from indic_transcribe import IndicTranscribe  # ships inside the model repo
            self.model = IndicTranscribe.from_pretrained(model_dir)

    def transcribe(self, paths: list[str], lang: str, batch_size: int) -> list[str]:
        if self.kind == "nemo":
            kw = dict(source_lang=lang, target_lang=lang, pnc="yes", batch_size=batch_size)
            try:
                hyps = self.model.transcribe(paths, verbose=False, **kw)
            except TypeError:  # older NeMo: no `verbose`
                hyps = self.model.transcribe(paths, **kw)
            if isinstance(hyps, tuple):
                hyps = hyps[0]
            return [getattr(h, "text", h) for h in hyps]
        return [self.model(p, lang=lang) for p in paths]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="bodhan-ai/indic-transcribe-core")
    ap.add_argument("--backend", choices=["nemo", "hf"], default="nemo")
    ap.add_argument("--manifest", default=None, help="TSV path<TAB>lang<TAB>reference")
    ap.add_argument("--audio-dir", default=None, help="folder of .wav (optional refs.tsv: name<TAB>text)")
    ap.add_argument("--lang", default="hi", help="language for --audio-dir inputs")
    ap.add_argument("--utt-seconds", type=float, default=20.0, help="join clips into utterances of this length")
    ap.add_argument("--work-dir", default=os.path.join(os.environ.get("INPUTS", "/data/tokenfactory/bench-inputs"), "asr_utts"))
    ap.add_argument("--latency-n", type=int, default=30, help="utterances timed one at a time")
    ap.add_argument("--batch-sizes", default="1,8,16,32,64")
    ap.add_argument("--throughput-audio-min", type=float, default=30.0, help="minutes of audio per batch-size run")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default="results/asr.json")
    a = ap.parse_args()

    import torch

    utts = make_utterances(read_inputs(a), Path(a.work_dir), a.utt_seconds)
    print(f"[inputs] {len(utts)} utterances, {sum(u['audio_s'] for u in utts) / 60:.1f} min of audio")

    t = time.perf_counter()
    be = Backend(a.backend, a.model_dir)
    load_s = time.perf_counter() - t
    be.transcribe([utts[0]["path"]], utts[0]["lang"], 1)  # warm-up
    torch.cuda.reset_peak_memory_stats()
    gpu = GpuSampler([a.gpu] if a.gpu is not None else None).start()

    # 1) batch-1 latency + quality
    lat, errs_w, errs_c, ref_w, ref_c = [], 0, 0, 0, 0
    for u in (utts * (a.latency_n // len(utts) + 1))[: a.latency_n]:
        torch.cuda.synchronize(); s = time.perf_counter()
        hyp = be.transcribe([u["path"]], u["lang"], 1)[0]
        torch.cuda.synchronize(); lat.append(time.perf_counter() - s)
        if u["ref"]:
            rw, hw = norm_words(u["ref"]), norm_words(hyp)
            errs_w += edit_distance(rw, hw); ref_w += len(rw)
            rc, hc = list("".join(rw)), list("".join(hw))
            errs_c += edit_distance(rc, hc); ref_c += len(rc)
    print(f"[latency] batch-1 p50 {summarize(lat)['p50']:.3f}s p95 {summarize(lat)['p95']:.3f}s "
          f"for ~{a.utt_seconds:.0f}s utterances")

    # 2) throughput per batch size
    levels = []
    for bs in [int(x) for x in a.batch_sizes.split(",")]:
        if a.backend == "hf" and bs > 1:
            print(f"[throughput] hf backend has no batching; skipping batch {bs}")
            continue
        by_lang: dict[str, list] = {}
        for u in utts:
            by_lang.setdefault(u["lang"], []).append(u)
        reps = max(1, int(a.throughput_audio_min * 60 / sum(u["audio_s"] for u in utts)) + 1)
        audio = 0.0
        torch.cuda.synchronize(); t0w, s = time.time(), time.perf_counter()
        for lang, us in by_lang.items():
            paths = [u["path"] for u in us] * reps
            be.transcribe(paths, lang, bs)
            audio += sum(u["audio_s"] for u in us) * reps
        torch.cuda.synchronize(); wall = time.perf_counter() - s
        lv = {"batch_size": bs, "audio_s": audio, "wall_s": wall, "rtfx": audio / wall,
              "gpu": gpu.window(t0w, time.time())}
        levels.append(lv)
        print(f"[throughput] batch {bs:3d}: RTFx {lv['rtfx']:.0f} ({audio / 60:.1f} min audio in {wall:.1f}s)")
    gpu.stop()

    write_result(a.out, {
        "bench": "asr", "backend": a.backend, "model_dir": be.model_dir, "load_s": load_s,
        "utterance_s": a.utt_seconds, "latency_batch1_s": summarize(lat),
        "wer": errs_w / ref_w if ref_w else None, "cer": errs_c / ref_c if ref_c else None,
        "throughput": levels, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    })


if __name__ == "__main__":
    main()
