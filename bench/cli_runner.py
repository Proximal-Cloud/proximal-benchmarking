"""Supervise the external vLLM load generator and preserve its artifacts."""

import json
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

from llm_evidence import Progress, event, read_metrics, save_json


def metrics_snapshot(base, model, timeout):
    try:
        with urlopen(base + "/metrics", timeout=timeout) as response:
            return read_metrics(response.read().decode(), model)
    except (URLError, OSError):
        return {}


def model_info(base, model):
    try:
        with urlopen(base + "/v1/models", timeout=2) as response:
            models = json.load(response).get("data", [])
        return next((info for info in models if info.get("id") == model), {})
    except (URLError, OSError, ValueError):
        return {}


def run_cli(command, folder, base, model, timeout, stall):
    save_json(folder / "command.json", command)
    deadline = time.monotonic() + timeout
    progress = Progress()
    before = metrics_snapshot(base, model, min(2, timeout))
    latest = before
    reason = "interrupted"
    with (folder / "client.log").open("w") as log, (folder / "events.jsonl").open("w") as events:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "timeout"
                    break
                try:
                    code = process.wait(timeout=min(1, remaining))
                    reason = "completed" if code == 0 else "runner_failure"
                    break
                except subprocess.TimeoutExpired:
                    pass
                metrics = metrics_snapshot(base, model, max(0.01, min(2, deadline - time.monotonic())))
                if metrics:
                    latest = metrics
                    event(events, metrics=metrics)
                    progress.observe(metrics)
                if stall and time.monotonic() - progress.last >= stall:
                    reason = "stalled"
                    break
        except KeyboardInterrupt:
            reason = "interrupted"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            event(events, stop_reason=reason, returncode=process.returncode)
            save_json(folder / "scheduler.json", {"before": before, "last_observed": latest})
    return reason
