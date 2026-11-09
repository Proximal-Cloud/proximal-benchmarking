#!/usr/bin/env python3
"""Build the baseline ASR and OCR inputs from public datasets.

Run it on the benchmark machine itself, or on a laptop if the machine's internet
is slow, then rsync the output folder (~0.5-1 GB) across.

  ASR (writes audio/manifest.tsv with path, language, reference transcript):
    * ReadNet (Pratham): children aged 5-16 reading aloud, Hindi + Marathi   CC BY-NC 4.0
    * Google FLEURS: adults reading sentences, Hindi + Kannada               CC BY 4.0
  OCR (writes page images to pages/):
    * DrawEduMath: handwritten maths answers by US school students           CC BY-NC 4.0
    * OmniDocBench: textbook, exam-paper and handwritten-note pages          research use only
    * Wikisource: proofread scanned book pages in Hindi + Kannada            scans mostly public domain

Licences: internal benchmarking only. Don't redistribute these files, train
on them or ship them in a product. SOURCES.md in the output lists them.

Usage (needs huggingface_hub, numpy and soundfile, e.g. the tts env):
  python bench/prep_public_inputs.py --out $INPUTS --cache $TF_HOME/dataset-cache
  python bench/prep_public_inputs.py --inspect        # print each source's columns, if a step fails
  # from a laptop instead:
  rsync -a --partial --progress ~/tf-transfer/bench-inputs/ <user>@<server>:<INPUTS folder>/

Each source is independent: if one fails, the others still finish, and the
summary at the end says what to fix. Re-running skips files already present.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import tarfile
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

UA = "TokenFactoryBench/1.0 (benchmark input prep; contact via huggingface account)"
CSV_LIMIT = sys.maxsize
csv.field_size_limit(min(CSV_LIMIT, 2 ** 31 - 1))


# ------------------------------------------------------------------ helpers

def log(msg):
    print(msg, flush=True)


def http_get(url: str, timeout=60, tries=3) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({last})")


def hf_file(repo: str, filename: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", cache_dir=str(cache)))


def to_wav16k(data: bytes | Path, dst: Path) -> float:
    """Decode any soundfile-readable audio, write 16 kHz mono WAV, return seconds."""
    import numpy as np
    import soundfile as sf

    src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else str(data)
    wav, sr = sf.read(src, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(wav) - 1, int(len(wav) * 16000 / sr))
        wav = np.interp(idx, np.arange(len(wav)), wav).astype("float32")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), wav, 16000)
    return len(wav) / 16000


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t or "")).strip()


def has_script(text: str, lo: int, hi: int) -> bool:
    return any(lo <= ord(c) <= hi for c in text)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    raw = path.read_bytes().decode("utf-8-sig", "replace")
    try:
        dialect = csv.Sniffer().sniff(raw[:5000], delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(raw), dialect=dialect))
    return (list(rows[0].keys()) if rows else []), rows


def guess_url_col(cols, rows, exts):
    """Column whose values look like links to files with one of `exts`."""
    best, score = None, 0
    for c in cols:
        vals = [str(r.get(c) or "") for r in rows[:200]]
        n = sum(1 for v in vals if v.startswith("http") or v.lower().endswith(exts))
        if n > score:
            best, score = c, n
    return best if score else None


def guess_text_col(cols, rows, lo=0x0900, hi=0x097F, exclude=()):
    """Column with the most Devanagari (by default) text."""
    best, score = None, 0
    for c in cols:
        if c in exclude:
            continue
        n = sum(len(re.findall(f"[{chr(lo)}-{chr(hi)}]", str(r.get(c) or ""))) for r in rows[:200])
        if n > score:
            best, score = c, n
    return best


# ------------------------------------------------------------------ ASR sources

def readnet(a, out_audio: Path, cache: Path, manifest: list):
    """Pratham ReadNet: children reading. CSVs list each recording's link + text."""
    files = {"hi": "HI_Paras_Stories.csv", "mr": "MR_Paras_Stories.csv"}
    for lang, fname in files.items():
        path = hf_file("PrathamOrgAI/ReadNet", fname, cache)
        cols, rows = read_csv(path)
        acol = a.readnet_audio_col or guess_url_col(cols, rows, (".wav", ".mp3", ".flac", ".m4a", ".ogg"))
        tcol = a.readnet_text_col or guess_text_col(cols, rows, exclude=(acol,))
        if not acol or not tcol:
            raise RuntimeError(f"{fname}: could not tell which columns hold the audio link and text. "
                               f"Columns: {cols}. Re-run with --inspect, then pass "
                               f"--readnet-audio-col/--readnet-text-col.")
        log(f"  ReadNet {lang}: {len(rows)} rows; audio column '{acol}', text column '{tcol}'")
        rng = random.Random(1)
        rng.shuffle(rows)
        got = 0
        for r in rows:
            if got >= a.readnet_per_lang:
                break
            url, text = str(r.get(acol) or "").strip(), clean(str(r.get(tcol) or ""))
            if not url or not text:
                continue
            if not url.startswith("http"):  # a path inside the repo
                try:
                    data = hf_file("PrathamOrgAI/ReadNet", url.lstrip("/"), cache).read_bytes()
                except Exception:  # noqa: BLE001
                    continue
            else:
                try:
                    data = http_get(url)
                except Exception as e:  # noqa: BLE001
                    log(f"    skip (download failed): {e}")
                    continue
            dst = out_audio / f"readnet_{lang}_{got:03d}.wav"
            try:
                secs = to_wav16k(data, dst)
            except Exception as e:  # noqa: BLE001
                log(f"    skip (could not decode audio): {e}")
                continue
            if secs < 1.0 or secs > 30.0:
                dst.unlink(missing_ok=True)
                continue
            manifest.append((dst.name, lang, text, "ReadNet"))
            got += 1
        log(f"  ReadNet {lang}: kept {got} clips")


def fleurs(a, out_audio: Path, cache: Path, manifest: list):
    """Google FLEURS test split: data/<lang>/test.tsv + data/<lang>/audio/test.tar.gz."""
    for code, lang in (("hi_in", "hi"), ("kn_in", "kn")):
        tsv = hf_file("google/fleurs", f"data/{code}/test.tsv", cache)
        refs = {}
        for line in tsv.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                refs[parts[1]] = clean(parts[2])   # file_name -> raw transcription
        log(f"  FLEURS {code}: {len(refs)} test sentences; downloading the audio archive (a few hundred MB)")
        tar_path = hf_file("google/fleurs", f"data/{code}/audio/test.tar.gz", cache)
        wanted = list(refs)
        random.Random(2).shuffle(wanted)
        wanted = set(wanted[: a.fleurs_per_lang])
        got = 0
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                name = Path(m.name).name
                if not m.isfile() or name not in wanted:
                    continue
                dst = out_audio / f"fleurs_{lang}_{got:03d}.wav"
                to_wav16k(tf.extractfile(m).read(), dst)
                manifest.append((dst.name, lang, refs[name], "FLEURS"))
                got += 1
        log(f"  FLEURS {code}: kept {got} clips")


# ------------------------------------------------------------------ OCR sources

def drawedumath(a, out_pages: Path, cache: Path, sources: list):
    path = hf_file("Heffernan-WPI-Lab/DrawEduMath", "Data/DrawEduMath_QA.csv", cache)
    cols, rows = read_csv(path)
    ucol = a.drawedumath_image_col or guess_url_col(cols, rows, (".png", ".jpg", ".jpeg", ".webp"))
    if not ucol:
        raise RuntimeError(f"DrawEduMath: no image-link column found. Columns: {cols}. "
                           f"Re-run with --inspect, then pass --drawedumath-image-col.")
    urls = []
    for r in rows:
        u = str(r.get(ucol) or "").strip()
        if u.startswith("http") and u not in urls:
            urls.append(u)
    random.Random(3).shuffle(urls)
    log(f"  DrawEduMath: {len(urls)} distinct student images in column '{ucol}'")
    got = 0
    for u in urls:
        if got >= a.drawedumath_n:
            break
        ext = Path(urllib.parse.urlparse(u).path).suffix.lower() or ".png"
        dst = out_pages / f"drawedumath_{got:03d}{ext}"
        if not dst.exists():
            try:
                dst.write_bytes(http_get(u))
            except Exception as e:  # noqa: BLE001
                log(f"    skip: {e}")
                continue
        got += 1
    sources.append(("DrawEduMath", got))
    log(f"  DrawEduMath: kept {got} images")


def omnidocbench(a, out_pages: Path, cache: Path, sources: list):
    meta = json.loads(hf_file("opendatalab/OmniDocBench", "OmniDocBench.json", cache).read_text(encoding="utf-8"))
    want = ("textbook", "book", "exam", "note")
    picks = []
    for p in meta:
        info = p.get("page_info", p)
        attr = info.get("page_attribute", {}) if isinstance(info, dict) else {}
        src = str(attr.get("data_source", "")).lower()
        langv = str(attr.get("language", "")).lower()
        img = info.get("image_path") if isinstance(info, dict) else None
        if img and any(w in src for w in want) and ("english" in langv or not langv):
            picks.append((src, img))
    if not picks:
        raise RuntimeError("OmniDocBench: no textbook/exam/note pages matched; run --inspect to see the fields.")
    random.Random(4).shuffle(picks)
    by_src: dict[str, int] = {}
    for src, _ in picks:
        by_src[src] = by_src.get(src, 0) + 1
    log(f"  OmniDocBench: matching pages by type {by_src}")
    got = 0
    for src, img in picks:
        if got >= a.omnidoc_n:
            break
        try:
            path = hf_file("opendatalab/OmniDocBench", f"images/{img}", cache)
        except Exception as e:  # noqa: BLE001
            log(f"    skip {img}: {e}")
            continue
        dst = out_pages / f"omnidoc_{re.sub(r'[^a-z]+', '', src)}_{got:03d}{Path(img).suffix}"
        dst.write_bytes(path.read_bytes())
        got += 1
    sources.append(("OmniDocBench", got))
    log(f"  OmniDocBench: kept {got} pages")


def wikisource(a, out_pages: Path, refs_dir: Path, sources: list):
    """Random proofread pages from Hindi and Kannada Wikisource, with their text."""
    for lang in ("hi", "kn"):
        api = f"https://{lang}.wikisource.org/w/api.php"
        info = json.loads(http_get(f"{api}?action=query&meta=proofreadinfo&format=json"))
        ns = info["query"]["proofreadnamespaces"]["page"]["id"]
        got, tries = 0, 0
        while got < a.wikisource_per_lang and tries < 15:
            tries += 1
            q = (f"{api}?action=query&format=json&generator=random&grnnamespace={ns}&grnlimit=20"
                 f"&prop=proofread|imageforpage|revisions&rvprop=content&rvslots=main"
                 f"&prppifpprop=filename|fullsize|responsiveimages&prppifpsize=1600")
            pages = json.loads(http_get(q)).get("query", {}).get("pages", {})
            for p in pages.values():
                if got >= a.wikisource_per_lang:
                    break
                quality = (p.get("proofread") or {}).get("quality", 0)
                ifp = p.get("imageforpage") or {}
                url = (ifp.get("responsiveimages") or {}).get("2") or (ifp.get("responsiveimages") or {}).get("1.5") \
                    or ifp.get("fullsize")
                text = ""
                revs = p.get("revisions") or []
                if revs:
                    text = (revs[0].get("slots", {}).get("main", {}) or {}).get("*", "") or revs[0].get("*", "")
                lo, hi = (0x0900, 0x097F) if lang == "hi" else (0x0C80, 0x0CFF)
                if quality < 3 or not url or not has_script(text, lo, hi):
                    continue
                if url.startswith("//"):
                    url = "https:" + url
                ext = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
                dst = out_pages / f"wikisource_{lang}_{got:03d}{ext}"
                try:
                    dst.write_bytes(http_get(url))
                except Exception as e:  # noqa: BLE001
                    log(f"    skip: {e}")
                    continue
                body = re.sub(r"<noinclude>.*?</noinclude>", "", text, flags=re.S)
                (refs_dir / f"{dst.stem}.txt").write_text(
                    f"source: https://{lang}.wikisource.org/wiki/{urllib.parse.quote(p.get('title', ''))}\n\n{body}",
                    encoding="utf-8")
                got += 1
            time.sleep(1)  # be polite to Wikimedia
        sources.append((f"Wikisource ({lang})", got))
        log(f"  Wikisource {lang}: kept {got} proofread pages")


# ------------------------------------------------------------------ inspect

def inspect(cache: Path):
    def show_csv(repo, fname):
        try:
            cols, rows = read_csv(hf_file(repo, fname, cache))
            log(f"\n== {repo}/{fname}: {len(rows)} rows\ncolumns: {cols}")
            for r in rows[:2]:
                log("  " + json.dumps({k: str(v)[:120] for k, v in r.items()}, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            log(f"\n== {repo}/{fname}: FAILED {e}")
    show_csv("PrathamOrgAI/ReadNet", "HI_Paras_Stories.csv")
    show_csv("Heffernan-WPI-Lab/DrawEduMath", "Data/DrawEduMath_QA.csv")
    try:
        tsv = hf_file("google/fleurs", "data/hi_in/test.tsv", cache).read_text(encoding="utf-8").splitlines()
        log(f"\n== google/fleurs data/hi_in/test.tsv: {len(tsv)} lines\n  {tsv[0][:300]}")
    except Exception as e:  # noqa: BLE001
        log(f"\n== FLEURS: FAILED {e}")
    try:
        meta = json.loads(hf_file("opendatalab/OmniDocBench", "OmniDocBench.json", cache).read_text(encoding="utf-8"))
        first = meta[0]
        log(f"\n== OmniDocBench.json: {len(meta)} pages; first page_info:\n  "
            + json.dumps(first.get("page_info", {k: first[k] for k in list(first)[:5]}), ensure_ascii=False)[:600])
    except Exception as e:  # noqa: BLE001
        log(f"\n== OmniDocBench: FAILED {e}")


# ------------------------------------------------------------------ main

SOURCES_MD = """# Benchmark inputs: sources and licences

Internal benchmarking only. Do not redistribute, train on, or ship these files.

| Folder | Source | Licence |
| --- | --- | --- |
| audio/readnet_* | ReadNet, Pratham (huggingface.co/datasets/PrathamOrgAI/ReadNet): children aged 5-16 reading aloud | CC BY-NC 4.0 |
| audio/fleurs_* | Google FLEURS (huggingface.co/datasets/google/fleurs), test split | CC BY 4.0 |
| pages/drawedumath_* | DrawEduMath (huggingface.co/datasets/Heffernan-WPI-Lab/DrawEduMath) | CC BY-NC 4.0 |
| pages/omnidoc_* | OmniDocBench (huggingface.co/datasets/opendatalab/OmniDocBench) | research use only, not commercial |
| pages/wikisource_* | Hindi / Kannada Wikisource scans; proofread text in pages/refs/ | scans mostly public domain; text CC BY-SA |
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path.home() / "tf-transfer" / "bench-inputs"))
    ap.add_argument("--cache", default=str(Path.home() / "tf-transfer" / "dataset-cache"),
                    help="download cache, kept OUT of the hf/ folder you rsync for models")
    ap.add_argument("--inspect", action="store_true", help="print each source's columns and stop")
    ap.add_argument("--only", nargs="*", choices=["readnet", "fleurs", "drawedumath", "omnidocbench", "wikisource"])
    ap.add_argument("--readnet-per-lang", type=int, default=100)
    ap.add_argument("--fleurs-per-lang", type=int, default=50)
    ap.add_argument("--drawedumath-n", type=int, default=30)
    ap.add_argument("--omnidoc-n", type=int, default=20)
    ap.add_argument("--wikisource-per-lang", type=int, default=10)
    ap.add_argument("--readnet-audio-col"), ap.add_argument("--readnet-text-col")
    ap.add_argument("--drawedumath-image-col")
    a = ap.parse_args()

    cache = Path(a.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    if a.inspect:
        inspect(cache)
        return
    out = Path(a.out).expanduser()
    out_audio, out_pages, refs_dir = out / "audio", out / "pages", out / "pages" / "refs"
    for d in (out_audio, out_pages, refs_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest: list = []
    sources: list = []
    status = {}
    steps = [("readnet", lambda: readnet(a, out_audio, cache, manifest)),
             ("fleurs", lambda: fleurs(a, out_audio, cache, manifest)),
             ("drawedumath", lambda: drawedumath(a, out_pages, cache, sources)),
             ("omnidocbench", lambda: omnidocbench(a, out_pages, cache, sources)),
             ("wikisource", lambda: wikisource(a, out_pages, refs_dir, sources))]
    for name, fn in steps:
        if a.only and name not in a.only:
            continue
        log(f"\n== {name}")
        try:
            fn()
            status[name] = "ok"
        except Exception as e:  # noqa: BLE001
            status[name] = f"FAILED: {e}"
            log(f"  !! {name} failed: {e}")

    # manifest paths are relative to audio/, so they work on cn14 unchanged
    if manifest:
        old = {}
        mpath = out_audio / "manifest.tsv"
        if mpath.exists() and a.only:  # keep rows from sources not re-run
            for line in mpath.read_text(encoding="utf-8").splitlines():
                p = line.split("\t")
                old[p[0]] = line
        for f, lang, text, _ in manifest:
            old[f] = f"{f}\t{lang}\t{text}"
        mpath.write_text("\n".join(old.values()) + "\n", encoding="utf-8")
    (out / "SOURCES.md").write_text(SOURCES_MD, encoding="utf-8")

    n_audio = len(list(out_audio.glob("*.wav")))
    n_pages = len([p for p in out_pages.iterdir() if p.is_file()])
    log("\n== summary")
    for k, v in status.items():
        log(f"  {k:13s} {v}")
    log(f"  {n_audio} audio clips in {out_audio} (manifest.tsv), {n_pages} page images in {out_pages}")
    log(f"\nIf you built this on another computer, copy it to the benchmark machine's $INPUTS folder:\n"
        f"  rsync -a --partial --progress {out}/ <user>@<server>:<INPUTS folder>/")
    if any(v != "ok" for v in status.values()):
        log("\nSome sources failed. Run with --inspect and send the output, or re-run just those with --only <name>.")


if __name__ == "__main__":
    main()
