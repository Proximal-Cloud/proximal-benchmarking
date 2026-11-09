"""Shared helpers: config loading, percentiles, GPU telemetry, result files.

Kept dependency-light (stdlib + PyYAML) so every benchmark venv can import it.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def resolve_model_dir(repo_or_path: str) -> str:
    """Local folder for a model: a path as given, else its newest snapshot in the
    Hugging Face cache, else download it. Reading the cache folder directly works
    offline even when parts of the repo (e.g. an unused nemo/ folder) were skipped."""
    p = Path(repo_or_path).expanduser()
    if p.exists():
        return str(p)
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snaps = sorted((hf_home / "hub" / ("models--" + repo_or_path.replace("/", "--")) / "snapshots").glob("*/"),
                   key=lambda d: d.stat().st_mtime)
    if snaps:
        return str(snaps[-1])
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_or_path)


def load_yaml(name: str) -> dict:
    import yaml  # imported lazily so --help works without PyYAML

    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def pct(values, q: float):
    """Percentile with linear interpolation; None for empty input."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def summarize(values) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "p5": pct(vals, 5),
        "p50": pct(vals, 50),
        "p90": pct(vals, 90),
        "p95": pct(vals, 95),
        "p99": pct(vals, 99),
        "max": max(vals),
    }


class GpuSampler:
    """Samples nvidia-smi once a second in a background thread.

    Records utilisation, memory and power for the given GPU indices so each
    benchmark can report mean utilisation, peak memory and energy used.
    Silently does nothing on machines without nvidia-smi (e.g. the mock test).
    """

    FIELDS = "index,utilization.gpu,memory.used,memory.total,power.draw"

    def __init__(self, gpu_indices=None, interval_s: float = 1.0):
        self.gpu_indices = None if gpu_indices is None else {int(i) for i in gpu_indices}
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = None
        self.available = shutil.which("nvidia-smi") is not None

    def _read(self):
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        t = time.time()
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                idx = int(parts[0])
                if self.gpu_indices is not None and idx not in self.gpu_indices:
                    continue
                self.samples.append({
                    "t": t, "gpu": idx, "util": float(parts[1]),
                    "mem_mib": float(parts[2]), "mem_total_mib": float(parts[3]),
                    "power_w": float(parts[4]) if parts[4] not in ("[N/A]", "N/A") else None,
                })
            except ValueError:
                continue

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._read()
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self):
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self

    def window(self, t0: float, t1: float) -> dict:
        """Aggregate samples between t0 and t1 (epoch seconds)."""
        s = [x for x in self.samples if t0 <= x["t"] <= t1]
        if not s:
            return {"available": self.available, "n": 0}
        gpus = sorted({x["gpu"] for x in s})
        per_gpu = {}
        total_energy_j = 0.0
        for g in gpus:
            gs = [x for x in s if x["gpu"] == g]
            powers = [x["power_w"] for x in gs if x["power_w"] is not None]
            mean_p = sum(powers) / len(powers) if powers else None
            if mean_p is not None:
                total_energy_j += mean_p * (t1 - t0)
            per_gpu[str(g)] = {
                "util_mean": sum(x["util"] for x in gs) / len(gs),
                "mem_peak_mib": max(x["mem_mib"] for x in gs),
                "mem_total_mib": gs[0]["mem_total_mib"],
                "power_mean_w": mean_p,
            }
        return {"available": True, "n": len(s), "per_gpu": per_gpu,
                "energy_j": total_energy_j, "seconds": t1 - t0}


def machine_info() -> dict:
    info = {"host": platform.node(), "python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "machine_type": os.environ.get("TF_MACHINE") or None}
    if shutil.which("nvidia-smi"):
        try:
            info["gpus"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        except Exception:
            pass
    return info


def gpu_names(machine: dict) -> list[str]:
    """GPU model names from machine_info()["gpus"] lines like '0, NVIDIA A100-PCIE-40GB, 40960 MiB, 595.71.05'."""
    names = []
    for line in (machine or {}).get("gpus", []):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1] not in names:
            names.append(parts[1])
    return names


def machine_price(spec: dict, machine: dict, machine_override: str | None = None,
                  price_override: float | None = None) -> dict:
    """$ per GPU-hour for the machine the results came from, by machine TYPE.

    Machine: --machine on the command line, else MACHINE saved with the results
    (machine_info's "machine_type"), else the first entry of spec `machines` whose
    `hosts` globs match the hostname, else `default_machine`.
    Price: --usd-per-gpu-hour, else that machine's usd_per_gpu_hour."""
    import fnmatch

    machines = spec.get("machines") or {}
    host = (machine or {}).get("host", "")
    key, why = None, ""
    if machine_override:
        key, why = machine_override, "--machine on the command line"
    elif (machine or {}).get("machine_type"):
        key, why = machine["machine_type"], "MACHINE set when the benchmark ran"
    else:
        for k, m in machines.items():
            if any(fnmatch.fnmatch(host.lower(), g.lower()) for g in (m.get("hosts") or [])):
                key, why = k, f"hostname {host} matches machines.{k}.hosts"
                break
    if key is None:
        key, why = spec.get("default_machine"), "default_machine (MACHINE wasn't set and the hostname didn't match)"
    if key not in machines:
        raise SystemExit(f"machine {key!r} is not in spec.yaml machines: (have: {', '.join(machines)})")
    m = machines[key]
    out = {"machine": key, "label": m.get("label", key), "usd_per_gpu_hour": float(m["usd_per_gpu_hour"]),
           "source": f"machines.{key} in spec.yaml", "why_machine": why,
           "gpus": gpu_names(machine), "all": {k: {"label": v.get("label", k), "usd_per_gpu_hour": float(v["usd_per_gpu_hour"])}
                                               for k, v in machines.items()}}
    if price_override is not None:
        out["usd_per_gpu_hour"], out["source"] = float(price_override), "--usd-per-gpu-hour on the command line"
    return out


def write_result(path: str | os.PathLike, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"machine": machine_info(), **payload}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[saved] {p}")
    return p


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance over token lists (for WER/CER without extra deps)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]
