#!/usr/bin/env bash
# Runs the LLM + TTS-vLLM benchmarks and the cost report against a mock server.
# No GPU needed. Takes ~3 minutes.   bash tests/test_end_to_end.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(mktemp -d)
PORT=$((18100 + RANDOM % 800))
python3 tests/mock_openai_server.py --port $PORT & MOCK=$!
trap 'kill $MOCK 2>/dev/null; [ -n "${KEEP:-}" ] && cp "$OUT"/report.* "$KEEP"/; rm -rf "$OUT"' EXIT
sleep 1.5
export TF_MACHINE=cdac
python3 bench/llm_sessions.py --target gemma31b-w4 --base-url http://localhost:$PORT/v1 \
  --concurrency 2,8 --duration 25 --warmup-turns 2 --grace 60 --turn-output-tokens 80 \
  --out "$OUT/socratic_gemma31b-w4.json"
python3 bench/tts_bench.py --mode vllm --base-url http://localhost:$PORT/v1 --concurrency 1,4 \
  --duration 10 --warmup 2 --max-tokens 100 --out "$OUT/tts_vllm.json"
mkdir -p "$OUT/tok" && printf '[{"conversations":[]}]' > "$OUT/sharegpt.json"
python3 bench/standard_bench.py --target gemma31b-w4 --dataset sharegpt --dataset-path "$OUT/sharegpt.json" \
  --base-url http://localhost:$PORT/v1 --tokenizer "$OUT/tok" --vllm "python3 tests/fake_vllm.py" \
  --out "$OUT/public_gemma31b-w4_sharegpt.json"
cp tests/fixtures/*.json "$OUT/"   # ASR / OCR / one-stream TTS results from the Oct 29 cn14 run
python3 bench/cost_report.py --results "$OUT" > "$OUT/report.txt"
grep -q "| 4. Gemma 4 31B | Gemma 4 31B | Socratic" "$OUT/report.md"
grep -q "Public benchmark, ShareGPT" "$OUT/report.md"
grep -q '\$0.20 per GPU-hour (machines.cdac' "$OUT/report.md"
if grep -qi "elite\|workhorse" "$OUT/report.html"; then echo "FAIL: report still mentions elite/workhorse"; exit 1; fi
grep -q 'id="machine"' "$OUT/report.html"
grep -q "public benchmark (sharegpt)" "$OUT/report.html"
python3 bench/cost_report.py --results "$OUT" --machine ryzen > "$OUT/report2.txt"
grep -q '\$0.50 per GPU-hour' "$OUT/report2.txt"
python3 bench/cost_report.py --results "$OUT" > /dev/null   # leave report.html at the configured price
python3 - "$OUT/socratic_gemma31b-w4.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for lv in d["levels"]:
    assert lv["errors"] == 0, lv["error_samples"]
    assert lv["output_tok_per_s"] > 0 and lv["ttft_s"]["n"] > 0
    assert 0 < lv["cache_hit_rate_usage"] < 1
    assert abs(lv["cache_hit_rate_usage"] - lv["cache_hit_rate_metrics"]) < 0.05
print("socratic checks passed; cache hit", round(d["levels"][-1]["cache_hit_rate_usage"], 3))
PY
echo "END-TO-END TEST PASSED"
