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
bash scripts/smoke_llm.sh mock "$PORT"
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
    m = lv["measurement"]
    assert m["status"] == "qualified"
    assert len(lv["requests"]) == m["issued"] == m["completed"] == lv["turns_completed_in_window"]
    assert m["joint_passed"] > 0 and all(r["outcome"] == "completed" for r in lv["requests"])
    assert lv["errors"] == 0, lv["error_samples"]
    assert lv["output_tok_per_s"] > 0 and lv["ttft_s"]["n"] > 0
    assert 0 < lv["cache_hit_rate_usage"] < 1
    assert abs(lv["cache_hit_rate_usage"] - lv["cache_hit_rate_metrics"]) < 0.05
print("socratic checks passed; cache hit", round(d["levels"][-1]["cache_hit_rate_usage"], 3))
PY
mkdir "$OUT/vllm" "$OUT/llm-scaler"
for engine in vllm llm-scaler; do
  printf '%s\n' "$engine" > "$OUT/$engine/llm_engine"
  printf '1\n' > "$OUT/$engine/quick"
  cp "$OUT"/socratic_gemma31b-w4.json "$OUT/$engine/"
done
cp "$OUT"/public_gemma31b-w4_sharegpt.json "$OUT/vllm/"
python3 bench/compare_llm.py "$OUT/vllm" "$OUT/llm-scaler" > "$OUT/compare.md"
grep -q '| vllm | vllm | ShareGPT | 2 |' "$OUT/compare.md"
grep -q '| llm-scaler | llm-scaler | ShareGPT | 8 | Missing |' "$OUT/compare.md"
grep -q 'GPU-seconds per student-day' "$OUT/compare.md"
printf '4\n' > "$OUT/vllm/socratic_concurrency"
printf '16\n' > "$OUT/vllm/public_concurrency"
python3 bench/compare_llm.py "$OUT/vllm" > "$OUT/compare-custom.md"
grep -q '| vllm | vllm | Socratic | 4 | Missing |' "$OUT/compare-custom.md"
grep -q '| vllm | vllm | ShareGPT | 16 |' "$OUT/compare-custom.md"
if grep -q '| vllm | vllm | ShareGPT | 2 |' "$OUT/compare-custom.md"; then exit 1; fi
if FAKE_PARTIAL=1 python3 bench/standard_bench.py --target gemma31b-w4 --dataset random \
    --tokenizer "$OUT/tok" --vllm "python3 tests/fake_vllm.py" --concurrency 2 --num-prompts 10 \
    --out "$OUT/partial.json"; then
  echo "FAIL: partial completions returned success"
  exit 1
fi
python3 - "$OUT" <<'PYTEST'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
m = json.loads((root / "partial.json").read_text())["measurements"][0]
assert (m["status"], m["issued"], m["completed"]) == ("failed", 10, 1), m
report = json.loads((root / "report.json").read_text())
assert report["evidence"] and "Run evidence" in (root / "report.html").read_text()
assert "OCR" in " ".join(report["missing"])
sys.path.insert(0, "bench")
from paired_stats import median_interval
assert median_interval([1] * 5) is None
assert median_interval(list(range(6))) == (0, 5)
from cost_report import session_point
socratic = json.loads((root / "socratic_gemma31b-w4.json").read_text())
for level in socratic["levels"]:
    level["measurement"]["status"] = "failed"
assert session_point(socratic, report["spec"]["slo"]) == (None, False)
PYTEST
echo "END-TO-END TEST PASSED"
