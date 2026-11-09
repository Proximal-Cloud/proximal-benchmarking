#!/usr/bin/env python3
"""OCR benchmark for Bodhan IndicOCR (spec line 3: 10 pages/day).

Runs the full two-stage parser (layout detection + block recognition) on a
folder of page images and reports pages/second, per-page latency, blocks and
characters per page, and GPU memory/utilisation.

Throughput is measured by running N worker processes on the same GPU, each
with its own copy of the model (~2 GB each), because the stock parser handles
one page at a time. Bodhan's own H100 figure (~6 pages/s) used a vLLM-served
recognizer; if the repo exposes that backend, pass its options with
--parser-kwargs and compare.

Use real inputs: 20-50 phone photos of homework sheets (including handwriting)
and textbook pages, in the languages your students use. Synthetic pages from
prep_inputs.py are for a smoke test only.

Run in the OCR venv:
  python bench/ocr_bench.py --pages-dir /data/tokenfactory/bench-inputs/pages --workers 1,2,4,8 --gpu 2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GpuSampler, summarize, write_result, resolve_model_dir  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def resolve_repo(repo: str) -> str:
    return resolve_model_dir(repo)


def _worker(repo, parser_kwargs, pages, q, ready, go):
    sys.path.insert(0, repo)
    from indic_ocr import IndicOCR  # ships inside the model repo

    parser = IndicOCR.from_pretrained(repo, **parser_kwargs)
    parser.parse(pages[0])  # warm-up
    ready.wait()
    go.wait()
    for p in pages:
        s = time.perf_counter()
        try:
            page = parser.parse(p)
            blocks = page.get("blocks", [])
            q.put({"ok": True, "latency_s": time.perf_counter() - s, "blocks": len(blocks),
                   "chars": len(page.get("markdown", "")), "t_end": time.time()})
        except Exception as e:  # keep going; report the failure
            q.put({"ok": False, "error": f"{type(e).__name__}: {e}", "latency_s": time.perf_counter() - s,
                   "t_end": time.time()})


def run_level(repo, parser_kwargs, pages, workers, pages_per_worker):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ready = ctx.Barrier(workers + 1)
    go = ctx.Event()
    procs = []
    for w in range(workers):
        mine = [pages[(w * pages_per_worker + i) % len(pages)] for i in range(pages_per_worker)]
        p = ctx.Process(target=_worker, args=(repo, parser_kwargs, mine, q, ready, go))
        p.start()
        procs.append(p)
    ready.wait()  # all models loaded and warmed up
    t0 = time.time()
    go.set()
    recs = [q.get() for _ in range(workers * pages_per_worker)]
    t1 = time.time()
    for p in procs:
        p.join()
    return recs, t0, t1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="bodhan-ai/indic-ocr")
    ap.add_argument("--pages-dir", required=True)
    ap.add_argument("--workers", default="1,2,4,8", help="parallel model copies on the GPU")
    ap.add_argument("--pages-per-worker", type=int, default=25)
    ap.add_argument("--parser-kwargs", default="{}", help='JSON passed to IndicOCR.from_pretrained, e.g. \'{"table_format":"markdown"}\'')
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default="results/ocr.json")
    a = ap.parse_args()

    pages = sorted(str(p) for p in Path(a.pages_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if not pages:
        sys.exit(f"no images in {a.pages_dir}")
    repo = resolve_repo(a.repo)
    kwargs = json.loads(a.parser_kwargs)
    print(f"[inputs] {len(pages)} page images")

    gpu = GpuSampler([a.gpu] if a.gpu is not None else None).start()
    levels = []
    try:
        for w in [int(x) for x in a.workers.split(",")]:
            recs, t0, t1 = run_level(repo, kwargs, pages, w, a.pages_per_worker)
            ok = [r for r in recs if r["ok"]]
            lv = {"workers": w, "pages": len(ok), "wall_s": t1 - t0,
                  "pages_per_s": len(ok) / (t1 - t0),
                  "latency_s": summarize([r["latency_s"] for r in ok]),
                  "blocks_per_page": summarize([r["blocks"] for r in ok]).get("mean"),
                  "chars_per_page": summarize([r["chars"] for r in ok]).get("mean"),
                  "errors": len(recs) - len(ok),
                  "error_samples": list({r.get("error") for r in recs if not r["ok"]})[:3],
                  "gpu": gpu.window(t0, t1)}
            levels.append(lv)
            print(f"[workers {w}] {lv['pages_per_s']:.2f} pages/s | latency p50 {lv['latency_s'].get('p50', 0):.2f}s "
                  f"p95 {lv['latency_s'].get('p95', 0):.2f}s | errors {lv['errors']}")
    finally:
        gpu.stop()
    write_result(a.out, {"bench": "ocr", "repo": repo, "parser_kwargs": kwargs, "n_inputs": len(pages), "levels": levels})


if __name__ == "__main__":
    main()
