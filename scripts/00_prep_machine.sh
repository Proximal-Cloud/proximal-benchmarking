#!/usr/bin/env bash
# Step 1: check the machine, lay out folders and install the Python environments.
# Safe to re-run. No sudo needed. Takes 10-20 min (vLLM + PyTorch are ~10 GB).
#   $VENVS/vllm   vLLM 0.23 (serves the LLM and TTS, and runs `vllm bench serve`)
#   $VENVS/bench  the benchmark clients (small)
# vLLM runs straight from the vllm env. To run it in Docker instead, set USE_DOCKER=1
# (needs your account in the docker group and nvidia-container-toolkit).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "== GPUs"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

echo "== Disk (weights go to $DATA_ROOT; needs ~150 GB free)"
mkdir -p "$DATA_ROOT" 2>/dev/null || { echo "!! can't create $DATA_ROOT; export DATA_ROOT=<a folder you own> and re-run"; exit 1; }
df -h "$DATA_ROOT" | sed 1d

mkdir -p "$TF_HOME" "$HF_HOME" "$VENVS" "$INPUTS"/{pages,audio,datasets} "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_HOME"
echo "== Folders ready under $TF_HOME and $HF_HOME"

echo "== uv (fast Python env manager)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}

echo "== vLLM env ($VENVS/vllm)"
uv venv -q --python 3.12 "$VENVS/vllm" 2>/dev/null || true
# transformers 5.12.1: newer releases fail to load Gemma 4 in vLLM 0.23 (head_dim error)
VIRTUAL_ENV="$VENVS/vllm" uv pip install "vllm==0.23.0" "transformers==5.12.1"
"$VENVS/vllm/bin/vllm" --version

echo "== Benchmark client env ($VENVS/bench)"
uv venv -q --python 3.12 "$VENVS/bench" 2>/dev/null || true
VIRTUAL_ENV="$VENVS/bench" uv pip install -q aiohttp pyyaml numpy pillow huggingface_hub

runtime=venv
if [ "${USE_DOCKER:-0}" = "1" ]; then
  echo "== Can Docker containers use the GPUs?"
  if docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi -L; then runtime=docker
  else echo "!! Docker can't use the GPUs for this account; using the vllm env instead"; fi
fi
echo "$runtime" > "$TF_HOME/.runtime"
echo "Done. vLLM runtime: $runtime. Next: scripts/01_download_models.sh"
