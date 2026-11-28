"""Evidence shared by the short-request and full-context clients."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from common import machine_info


def save_json(path, data):
    """Publish complete checkpoints, including when a reader is polling the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False)
    try:
        with temp as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temp.name, path)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def event(file, **fields):
    file.write(json.dumps({"time": time.time(), **fields}) + "\n")
    file.flush()


def verdict(checks):
    if any(check is False for check in checks):
        return "failed"
    if not checks or any(check is None for check in checks):
        return "insufficient evidence"
    return "passed"


def decode_peak(intervals):
    """Peak positive overlap of client-observed first to last token intervals."""
    events = []
    for first, last in intervals:
        if last > first:
            events.extend(((first, 1), (last, -1)))
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return peak


def joint_latency(samples, planned, slo):
    """Use all planned requests as the denominator, including failures."""
    if samples is None:
        return {"joint_passed": None, "joint_fraction": None, "latency_status": "insufficient evidence"}
    passed = sum(ttft <= slo["llm_ttft_p95_s"] and tpot <= slo["llm_tpot_p95_ms"] for ttft, tpot in samples)
    fraction = passed / planned
    return {
        "joint_passed": passed,
        "joint_fraction": fraction,
        "latency_status": verdict([fraction >= slo["llm_min_joint_pass_fraction"]]),
    }


def public_outcomes(raw, users, planned, slo):
    """Older vLLM versions can omit some or all detailed request arrays."""
    done = raw.get("completed")
    complete = done == planned and raw.get("failed", 0) == 0
    fields = ("ttfts", "latencies", "output_lens")
    detailed = all(isinstance(raw.get(k), list) for k in fields)
    samples = None
    peak = None
    if (
        detailed
        and done is not None
        and len(raw["ttfts"]) >= done
        and len({len(raw[k]) for k in fields}) == 1
    ):
        intervals = []
        samples = []
        starts = raw.get("start_times") or []
        errors = raw.get("errors") or []
        has_starts = len(starts) == len(raw["ttfts"]) and all(isinstance(t, (int, float)) for t in starts)
        for i, (ttft, duration, tokens) in enumerate(zip(*(raw[k] for k in fields))):
            if i < len(errors) and errors[i]:
                continue
            if None in (ttft, duration, tokens):
                # A successful request without timings cannot establish either outcome.
                samples = None
                break
            if tokens < 2 or duration < ttft:
                continue
            samples.append((ttft, 1000 * (duration - ttft) / (tokens - 1)))
            if has_starts:
                intervals.append((starts[i] + ttft, starts[i] + duration))
        if samples is not None and has_starts:
            peak = decode_peak(intervals)
    return {
        "completion_status": verdict([complete]),
        "verified_decode_concurrency": peak,
        # The external CLI lacks full-length and device-health checks.
        "capacity_status": verdict([complete, None if peak is None else peak >= users, None]),
        **joint_latency(samples, planned, slo),
    }


COUNTERS = (
    "prompt_tokens_total",
    "generation_tokens_total",
    "num_preemptions_total",
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
    "iteration_tokens_total_sum",
    "request_success_total",
)


def read_metrics(text, model):
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r"vllm:(\w+)(?:\{(.*)\})?\s+([\d.eE+-]+)", line)
        if not match or match[1] not in COUNTERS:
            continue
        label = re.search(r'(?:^|,)model_name="([^"]*)"', match[2] or "")
        if label and label[1] != model:
            continue
        name, value = match[1], float(match[3])
        if name == "request_success_total":
            reason = re.search(r'(?:^|,)finished_reason="([^"]+)"', match[2] or "")
            if reason:
                name += "_" + reason[1]
        if name == "kv_cache_usage_perc":
            values[name] = max(values.get(name, 0), value)
        else:
            values[name] = values.get(name, 0) + value
    return values


class SweepBudget:
    """Estimate the next level from the most recently completed level's work and wall time."""

    def __init__(self, seconds, start=None):
        started = time.monotonic() if start is None else start
        self.deadline = started + seconds if seconds is not None else float("inf")
        self.seconds_per_unit = None

    @property
    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def allows(self, work, label):
        estimate = self.seconds_per_unit * work if self.seconds_per_unit is not None else None
        description = f"{estimate:.1f}s" if estimate is not None else "unavailable before the first level"
        print(f"{label}: estimated duration {description}", flush=True)
        return self.remaining > 0 and (estimate is None or estimate <= self.remaining)

    def completed(self, work, elapsed):
        self.seconds_per_unit = elapsed / work


class Progress:
    def __init__(self):
        self.last = time.monotonic()
        self.counters = {}

    def observe(self, metrics):
        counters = {
            k: metrics[k]
            for k in (
                "prompt_tokens_total",
                "generation_tokens_total",
                "iteration_tokens_total_sum",
                "kv_cache_usage_perc",
            )
            if k in metrics
        }
        if counters and counters != self.counters:
            self.last = time.monotonic()
            self.counters = counters


def write_tables(output, rows):
    """Both formats render the same records. Empty fields mean unavailable."""
    output = Path(output)
    if output.suffix not in ("", ".json"):
        output = Path(str(output) + ".json")
    fields = list(rows[0]) if rows else ["Outcome"]
    with output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def line(values):
        cells = [str(v).replace("|", r"\|").replace("\n", " ") if v is not None else "—" for v in values]
        return "| " + " | ".join(cells) + " |"

    lines = [line(fields), line(["---"] * len(fields))]
    lines.extend(line(row.values()) for row in rows)
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")


def report_rows(data):
    rows = []
    levels = list(data["levels"])
    recorded = {(level.get("context"), level["concurrency"]) for level in levels}
    for measure in data.get("measurements", []):
        match = re.fullmatch(r"(?:context (\d+) )?concurrency (\d+)", measure["name"])
        if not match:
            continue
        context = int(match[1]) if match[1] else None
        users = int(match[2])
        if (context, users) not in recorded:
            levels.append(
                {
                    "concurrency": users,
                    "context": context,
                    "completed": measure.get("completed"),
                    "num_prompts": measure.get("planned"),
                    "stop_reason": measure["stop_reason"],
                    "active_at_last_sample": measure.get("active_at_last_sample"),
                    "queued_at_last_sample": measure.get("queued_at_last_sample"),
                    "completion_status": measure["status"],
                }
            )
    for level in levels:
        rows.append(
            {
                "Model": data["target"]["model"],
                "Workload": data["dataset"],
                "Seed": data.get("seed"),
                "Trial": data.get("run", {}).get("id"),
                "Sample": "single burst" if data["bench"] == "full-context" else "single trial",
                "Server limit": level.get("server_limit"),
                "Target tokens/user": level.get("context"),
                "Users": level["concurrency"],
                "Peak decode overlap": level.get("verified_decode_concurrency"),
                "Completed": level["completed"],
                "Planned": level["num_prompts"],
                "Mean input tokens": level["total_input_tokens"] / level["completed"]
                if level["completed"] and level.get("total_input_tokens") is not None
                else None,
                "Mean output tokens": level["total_output_tokens"] / level["completed"]
                if level["completed"] and level.get("total_output_tokens") is not None
                else None,
                "Failed": level.get("failed"),
                "Interrupted": level.get("interrupted"),
                "Last server active": level.get("active_at_last_sample"),
                "Last server queued": level.get("queued_at_last_sample"),
                "Stop reason": level.get("stop_reason"),
                "Preemptions": level.get("preemptions"),
                "Simultaneous decode s": level.get("simultaneous_decode_s"),
                "Completion": level.get("completion_status"),
                "Capacity": level.get("capacity_status"),
                "Joint latency": level.get("latency_status"),
                "Joint pass fraction": level.get("joint_fraction"),
                "Output tok/s": level.get("output_throughput"),
                "p95 TTFT ms": level.get("ttft_ms", {}).get("p95"),
                "p95 TPOT ms": level.get("tpot_ms", {}).get("p95"),
                "p95 ITL ms": level.get("itl_ms", {}).get("p95"),
                "p95 E2E ms": level.get("e2el_ms", {}).get("p95"),
                "Retrieval": level.get("retrieval_status"),
                **{
                    k: data.get("serving", {}).get(k)
                    for k in ("weights", "activations", "kv_cache", "device_ids")
                },
            }
        )
    return rows


class RunEvidence:
    """Keep each invocation independently recoverable and publish a current report."""

    def __init__(self, output, data, argv):
        self.output = Path(output)
        self.data = data
        self.root = self.output.parent / (self.output.stem + "-" + data["run"]["id"])
        self.root.mkdir(parents=True)
        self.data["artifacts"] = str(self.root)
        self.data["machine"] = machine_info()
        save_json(self.root / "command.json", argv)
        self.checkpoint()

    def checkpoint(self):
        self.data["run"]["updated_at"] = time.time()
        save_json(self.root / "result.json", self.data)
        save_json(self.output, self.data)
        write_tables(self.output, report_rows(self.data))


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
