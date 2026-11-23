#!/usr/bin/env bash
# Validate the OpenAI endpoints used by the Socratic and ShareGPT clients.
#   scripts/smoke_llm.sh gemma31b-w4 8001
set -euo pipefail
python3 - "${1:?model name}" "${2:?port}" <<'PY'
import json
import sys
from urllib.request import Request, urlopen

model, port = sys.argv[1:]
base = f"http://127.0.0.1:{port}"


def request(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    return urlopen(Request(base + path, data, {"Content-Type": "application/json"}), timeout=180)


with request("/v1/models") as response:
    assert model in {entry["id"] for entry in json.load(response)["data"]}, model
with request("/v1/completions", {"model": model, "prompt": "One plus one is", "max_tokens": 8}) as response:
    assert json.load(response)["choices"], "Missing completion choices"

prompt = "You are a tutor. Ask one guiding question. " + "Explain each step with a short example. " * 1500
body = {
    "model": model, "max_tokens": 16, "ignore_eos": True,
    "stream": True, "stream_options": {"include_usage": True},
    "messages": [{"role": "system", "content": prompt},
                 {"role": "user", "content": "How do I find the area of a triangle?"}],
}
for attempt in (1, 2):
    usage = None
    with request("/v1/chat/completions", body) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                usage = json.loads(line[6:]).get("usage") or usage
    assert usage and usage["prompt_tokens"] > 0 and usage["completion_tokens"] == 16, usage
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"Streaming call {attempt} | prompt {usage['prompt_tokens']} | output 16 | "
          f"cached {cached if cached is not None else 'unavailable'}")

try:
    with request("/metrics") as response:
        metrics = response.read().decode().splitlines()
except OSError as exc:
    print(f"Cache metrics unavailable ({exc})")
else:
    prefixes = ("vllm:prefix_cache", "vllm:kv_cache_usage", "vllm:num_preemptions")
    print("\n".join(line for line in metrics if line.startswith(prefixes)) or "Cache metrics unavailable")
PY
