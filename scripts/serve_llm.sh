#!/usr/bin/env bash
# Start (or stop) one vLLM server and wait until it answers. Runs from the
# $VENVS/vllm Python env (no sudo needed), or in Docker if 00_prep_machine.sh was
# run with USE_DOCKER=1. The choice is in $TF_HOME/.runtime; override with RUNTIME=docker|venv.
#
#   scripts/serve_llm.sh gemma31b-w4        # GPU 0, port 8001
#   scripts/serve_llm.sh gemma26b-a4b-fp8   # GPU 1, port 8002
#   scripts/serve_llm.sh gemma31b-tp2       # GPUs 1+2, port 8003 (stop 26b and Bodhan first)
#   scripts/serve_llm.sh indic-speak        # GPU 2, port 8004 (TTS speech LM)
#   scripts/serve_llm.sh stop <name>|all
#
# Variants (env vars):
#   KV_FP8=1    store the KV cache in 8-bit: ~2x more students fit in memory (test quality!)
#   BF16=1      use unquantized Google weights (26b and tp2 only), if FP8 fails to load on A100
#   MAX_SEQS=N  cap on concurrent sequences (default 64; run_all.sh uses 256 for the short-request benchmark)
#   GPU_IDS=0   put the model on other GPU(s) than its default, e.g. on a 1-GPU machine
#   PORT_OVERRIDE=9001   serve on another port
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

RUNTIME=${RUNTIME:-$(cat "$TF_HOME/.runtime" 2>/dev/null || echo venv)}
LOGDIR="$TF_HOME/logs"; mkdir -p "$LOGDIR"

stop_one() {  # $1 = tf-<name>
  if [ "$RUNTIME" = "docker" ]; then docker rm -f "$1" >/dev/null 2>&1 || true
  elif [ -f "$LOGDIR/$1.pid" ]; then
    pid=$(cat "$LOGDIR/$1.pid"); kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    rm -f "$LOGDIR/$1.pid"
  fi
}

name=${1:?usage: serve_llm.sh <name> | stop <name|all>}
if [ "$name" = "stop" ]; then
  which=${2:-all}
  if [ "$which" = "all" ]; then
    if [ "$RUNTIME" = "docker" ]; then docker ps -q --filter name=tf- | xargs -r docker rm -f
    else for f in "$LOGDIR"/tf-*.pid; do [ -e "$f" ] && stop_one "$(basename "$f" .pid)"; done; fi
  else stop_one "tf-$which"; fi
  exit 0
fi

MAXLEN=40960
EXTRA=()
case "$name" in
  gemma31b-w4)
    GPUS=0; PORT=8001; CPUS=$CPUS_GPU0; MEM=0.92
    MODEL=google/gemma-4-31B-it-qat-w4a16-ct ;;
  gemma26b-a4b-fp8)
    GPUS=1; PORT=8002; CPUS=$CPUS_GPU12; MEM=0.92
    MODEL=RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic
    if [ "${BF16:-0}" = "1" ]; then GPUS=1,2; MODEL=google/gemma-4-26B-A4B-it; EXTRA+=(--tensor-parallel-size 2); fi ;;
  gemma31b-tp2)
    GPUS=1,2; PORT=8003; CPUS=$CPUS_GPU12; MEM=0.92
    MODEL=RedHatAI/gemma-4-31B-it-FP8-block; EXTRA+=(--tensor-parallel-size 2)
    if [ "${BF16:-0}" = "1" ]; then MODEL=google/gemma-4-31B-it; MEM=0.95; fi ;;
  indic-speak)
    GPUS=2; PORT=8004; CPUS=$CPUS_GPU12; MEM=0.85; MAXLEN=4096
    MODEL=bodhan-ai/indic-speak ;;
  *) echo "unknown model name: $name"; exit 1 ;;
esac

# Overrides for other machines.
if [ -n "${GPU_IDS:-}" ]; then
  GPUS=$GPU_IDS
  case "$GPUS" in 0|0,*) CPUS=$CPUS_GPU0 ;; *) CPUS=$CPUS_GPU12 ;; esac
fi
PORT=${PORT_OVERRIDE:-$PORT}
# CPU pinning uses cn14's core layout; skip it where those cores don't exist.
PIN=(taskset -c "$CPUS")
if [ -z "$CPUS" ] || ! taskset -c "$CPUS" true 2>/dev/null; then PIN=(); CPUSET=(); else CPUSET=(--cpuset-cpus "$CPUS"); fi

if [ "$name" != "indic-speak" ]; then
  EXTRA+=(--language-model-only)             # text only: skip the vision tower, more room for KV cache
fi
[ "${KV_FP8:-0}" = "1" ] && EXTRA+=(--kv-cache-dtype fp8)

ARGS=(--served-model-name "$name"
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$MEM"
  --max-num-seqs "${MAX_SEQS:-64}"
  --enable-prefix-caching --enable-prompt-tokens-details
  "${EXTRA[@]}")

stop_one "tf-$name"
echo "== starting tf-$name ($RUNTIME): $MODEL on GPU(s) $GPUS, port $PORT"
if [ "$RUNTIME" = "docker" ]; then
  docker run -d --name "tf-$name" \
    --gpus "\"device=$GPUS\"" --ipc=host --shm-size 16g \
    "${CPUSET[@]}" -p "$PORT:8000" \
    -v "$HF_HOME:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$VLLM_IMAGE" --model "$MODEL" "${ARGS[@]}" >/dev/null
  alive() { docker ps -q --filter "name=tf-$name" | grep -q .; }
  logs()  { docker logs --tail 60 "tf-$name"; }
else
  [ -x "$VENVS/vllm/bin/vllm" ] || { echo "!! $VENVS/vllm missing: run scripts/00_prep_machine.sh"; exit 1; }
  CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 setsid nohup "${PIN[@]}" \
    "$VENVS/vllm/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT" "${ARGS[@]}" \
    > "$LOGDIR/tf-$name.log" 2>&1 < /dev/null &
  echo $! > "$LOGDIR/tf-$name.pid"
  alive() { kill -0 "$(cat "$LOGDIR/tf-$name.pid")" 2>/dev/null; }
  logs()  { tail -n 60 "$LOGDIR/tf-$name.log"; }
  echo "   log: $LOGDIR/tf-$name.log"
fi

echo -n "== waiting for http://localhost:$PORT (model load takes 2-10 min) "
for i in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then echo " ready"; exit 0; fi
  if ! alive; then echo; echo "!! server exited. Last log lines:"; logs; exit 1; fi
  echo -n "."; sleep 5
done
echo; echo "!! not ready after 15 min. Last log lines:"; logs; exit 1
