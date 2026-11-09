#!/usr/bin/env bash
# Quick check that a served LLM answers and reports cached tokens.
#   scripts/smoke_llm.sh gemma31b-w4 8001
set -euo pipefail
name=${1:?name}; port=${2:?port}
body='{"model":"'"$name"'","max_tokens":80,"messages":[
  {"role":"system","content":"You are a Socratic tutor for class 8. Ask one guiding question; never give the final answer."},
  {"role":"user","content":"How do I find the area of a triangle with base 6 cm and height 4 cm?"}]}'
echo "== first call"
curl -s "http://localhost:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"]); print("usage:", r["usage"])'
echo "== same prompt again: prompt_tokens_details.cached_tokens should now be > 0"
curl -s "http://localhost:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" \
  | python3 -c 'import json,sys; print("usage:", json.load(sys.stdin)["usage"])'
echo "== prefix-cache counters from /metrics"
curl -s "http://localhost:$port/metrics" | grep -E "^vllm:(prefix_cache|kv_cache_usage|num_preemptions)" | head
