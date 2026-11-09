#!/usr/bin/env bash
# Run every benchmark, then build the cost report. ~2.5-3 hours on cn14.
# Run inside tmux so a dropped SSH doesn't kill it:
#   tmux new -s bench
#   MACHINE=cdac scripts/run_all.sh 2>&1 | tee results/run_all.log
#
# MACHINE   machine type from config/spec.yaml `machines:` (cdac, ryzen, ...). Sets the
#           $/GPU-hour in the report. If unset, the report matches the hostname.
# MODELS    LLMs to benchmark, names from config/llm_targets.yaml (default: all listed there)
# What runs, in order (each one's results go to results/<machine>-<date-time>/):
#   LLM on GPU 0:   Socratic sessions at 2,4,8,16 students, then the public benchmark
#                   (ShareGPT via vllm bench serve) at 2,4,8,16 requests at once
#   Bodhan on the last GPU: TTS (one stream), ASR, OCR, then TTS batched on vLLM
# Skip parts: SKIP_LLM=1 SKIP_SOCRATIC=1 SKIP_PUBLIC=1 SKIP_BODHAN=1 SKIP_TTS_VLLM=1
# Quick dry run (~30 min, short levels):  QUICK=1 MACHINE=cdac scripts/run_all.sh
set -uo pipefail   # no -e: one failed benchmark should not kill a 3-hour run
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_HUB_OFFLINE=1            # everything is pre-downloaded; never wait on the network
export TF_MACHINE=${MACHINE:-${TF_MACHINE:-}}   # saved with every result file
PY="$VENVS/bench/bin/python"
[ -x "$PY" ] || PY="$VENVS/vllm/bin/python"   # the vLLM env has everything the clients need
STAMP=$(date +%Y%m%d-%H%M)
RES="results/${TF_MACHINE:-$(hostname -s)}-$STAMP"
mkdir -p "$RES"
echo "== run $STAMP on ${TF_MACHINE:-hostname $(hostname -s)}; results in $RES; client python: $PY"
[ -n "$TF_MACHINE" ] || echo "   (MACHINE not set: the report will pick the price by hostname; set MACHINE=cdac to be explicit)"

NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true); NGPU=${NGPU:-1}; [ "$NGPU" -ge 1 ] || NGPU=1
LLM_GPU=${LLM_GPU:-0}
BODHAN_GPU=${BODHAN_GPU:-$((NGPU - 1))}    # cn14: GPU 2
have_model() { ls -d "$HF_HOME/hub/models--${1//\//--}"/snapshots/*/ >/dev/null 2>&1; }
cfg() { "$PY" -c "import sys; sys.path.insert(0, 'bench'); from common import load_yaml; c = load_yaml('llm_targets.yaml'); $1"; }

if [ "${QUICK:-0}" = "1" ]; then
  SOC_ARGS="--concurrency 2,8 --duration 120 --warmup-turns 1"; PUB_ARGS="--concurrency 2,8 --num-prompts 40"
  TTS_ARGS="--concurrency 1,8,32 --duration 40 --warmup 10"; OCR_ARGS="--workers 1,4 --pages-per-worker 5"
else
  SOC_ARGS="--concurrency 2,4,8,16 --duration 300 --warmup-turns 2 --no-stop-on-slo"; PUB_ARGS=""   # public levels: llm_targets.yaml
  TTS_ARGS="--concurrency 1,4,8,16,32,64 --duration 120 --warmup 20"; OCR_ARGS=""
fi

# ---------------------------------------------------------------- 4+. LLMs on GPU 0
if [ "${SKIP_LLM:-0}" != "1" ]; then
  for name in ${MODELS:-$(cfg "print(' '.join(c['models']))")}; do
    repo=$(cfg "print(c['models']['$name']['repo'])") || { echo "!! $name is not in config/llm_targets.yaml"; continue; }
    if ! have_model "$repo"; then echo "!! skipping $name: $repo is not in $HF_HOME (run scripts/01_download_models.sh)"; continue; fi
    echo "== LLM $name on GPU $LLM_GPU"
    if GPU_IDS=$LLM_GPU MAX_SEQS=64 scripts/serve_llm.sh "$name"; then
      if [ "${SKIP_SOCRATIC:-0}" != "1" ]; then
        echo "   Socratic sessions (~45 min; log: $RES/socratic_$name.log)"
        "$PY" bench/llm_sessions.py --target "$name" --gpus "$LLM_GPU" $SOC_ARGS \
          --out "$RES/socratic_$name.json" > "$RES/socratic_$name.log" 2>&1
        tail -n 6 "$RES/socratic_$name.log"
      fi
      if [ "${SKIP_PUBLIC:-0}" != "1" ]; then
        if [ -s "$INPUTS/datasets/ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
          echo "   public benchmark, ShareGPT (~20 min; log: $RES/public_${name}_sharegpt.log)"
          "$PY" bench/standard_bench.py --target "$name" --dataset sharegpt --gpus "$LLM_GPU" $PUB_ARGS \
            --out "$RES/public_${name}_sharegpt.json" > "$RES/public_${name}_sharegpt.log" 2>&1
          tail -n 4 "$RES/public_${name}_sharegpt.log"
        else
          echo "!! skipping the public benchmark: ShareGPT isn't in $INPUTS/datasets (run scripts/01_download_models.sh)"
        fi
      fi
    else
      echo "!! $name did not start; see $TF_HOME/logs/tf-$name.log"
    fi
    scripts/serve_llm.sh stop "$name"
  done
fi

# ---------------------------------------------------------------- 1-3. Bodhan on the last GPU
if [ "${SKIP_BODHAN:-0}" != "1" ]; then
  echo "== Bodhan models on GPU $BODHAN_GPU"
  export CUDA_VISIBLE_DEVICES=$BODHAN_GPU
  # --gpu only tells the telemetry sampler which GPU to watch (nvidia-smi numbering).
  # TTS first: it writes the WAVs the ASR benchmark replays when there are no recordings.
  echo "   TTS, one stream end to end"
  "$VENVS/tts/bin/python" bench/tts_bench.py --mode hf --gpu "$BODHAN_GPU" --save-wav-dir "$INPUTS/tts_wavs" \
    --out "$RES/tts_hf.json" > "$RES/tts_hf.log" 2>&1; tail -n 3 "$RES/tts_hf.log"

  if [ -s "$INPUTS/audio/manifest.tsv" ]; then ASR_IN="--manifest $INPUTS/audio/manifest.tsv"   # public recordings
  else ASR_IN="--manifest $INPUTS/tts_wavs/manifest.tsv"; fi
  echo "   ASR"
  { "$VENVS/asr/bin/python" bench/asr_bench.py $ASR_IN --work-dir "$INPUTS/asr_utts" --gpu "$BODHAN_GPU" --out "$RES/asr.json" \
    || "$VENVS/asr/bin/python" bench/asr_bench.py $ASR_IN --work-dir "$INPUTS/asr_utts" --gpu "$BODHAN_GPU" --backend hf --out "$RES/asr.json"; } \
    > "$RES/asr.log" 2>&1; tail -n 3 "$RES/asr.log"

  if [ -x "$VENVS/ocr/bin/python" ]; then
    if ! ls "$INPUTS/pages"/* >/dev/null 2>&1; then
      echo "!! no real pages in $INPUTS/pages; using placeholder pages (numbers will be optimistic)"
      "$VENVS/ocr/bin/python" bench/prep_inputs.py --pages-dir "$INPUTS/pages" --n 12
    fi
    echo "   OCR ($(ls "$INPUTS/pages" | wc -l) pages)"
    "$VENVS/ocr/bin/python" bench/ocr_bench.py --pages-dir "$INPUTS/pages" --gpu "$BODHAN_GPU" $OCR_ARGS \
      --out "$RES/ocr.json" > "$RES/ocr.log" 2>&1; tail -n 5 "$RES/ocr.log"
  else
    echo "!! skipping OCR: no $VENVS/ocr environment yet"
  fi
  unset CUDA_VISIBLE_DEVICES
fi

if [ "${SKIP_TTS_VLLM:-0}" != "1" ]; then
  echo "== TTS batched on vLLM, GPU $BODHAN_GPU"
  if GPU_IDS=$BODHAN_GPU scripts/serve_llm.sh indic-speak; then
    "$PY" bench/tts_bench.py --mode vllm --gpu "$BODHAN_GPU" $TTS_ARGS --out "$RES/tts_vllm.json" \
      > "$RES/tts_vllm.log" 2>&1; tail -n 4 "$RES/tts_vllm.log"
  else
    echo "!! Indic-Speak did not start under vLLM; see $TF_HOME/logs/tf-indic-speak.log (the single-stream TTS result still counts)"
  fi
  scripts/serve_llm.sh stop indic-speak
fi

# ---------------------------------------------------------------- report
scripts/serve_llm.sh stop all
"$PY" bench/cost_report.py --results "$RES" > "$RES/report.log" 2>&1 || { cat "$RES/report.log"; exit 1; }
head -n 3 "$RES/report.log"
cp "$RES/report.html" "results/report_$(basename "$RES").html"
echo
echo "Report: $(pwd)/results/report_$(basename "$RES").html"
echo "All raw results and logs: $(pwd)/$RES"
