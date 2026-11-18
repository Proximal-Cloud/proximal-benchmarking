#!/usr/bin/env bash
# Step 1: check the machine, lay out folders and install the Python environments.
# Safe to re-run. No sudo needed. Takes 10-20 min (vLLM + PyTorch are ~10 GB).
#   $VENVS/vllm   vLLM 0.23 (serves the LLM and TTS, and runs `vllm bench serve`)
#   $VENVS/bench  the benchmark clients (small)
# vLLM runs straight from the vllm env. To run it in Docker instead, set USE_DOCKER=1
# (needs your account in the docker group; NVIDIA also needs nvidia-container-toolkit).
# Intel XPU (xpu-smi, for example a B70) installs the XPU wheel and skips NVIDIA packages.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "== GPUs ($GPU_BACKEND)"
# xpu-smi does not accept nvidia-smi --query-gpu flags. That is a different tool.
if [ "$GPU_BACKEND" = "xpu" ]; then
  # 1=id, 2=name, 8=driver, 16=memory. See: xpu-smi discovery -h
  xpu-smi discovery --dump 1,2,8,16 || xpu-smi discovery
elif [ "$GPU_BACKEND" = "cuda" ]; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
  echo "!! No supported GPU management tool found (xpu-smi or nvidia-smi)"
  exit 1
fi

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

# Remove CUDA wheels left by an earlier vllm install. Those wheels keep the
# `nvidia` module importable after torch itself has been replaced by the XPU build.
drop_nvidia_packages() {
  [ -x "$VENVS/vllm/bin/python" ] || return 0
  local names
  names=$("$VENVS/vllm/bin/python" - <<'PY'
import importlib.metadata as md
print(" ".join(d.metadata["Name"] for d in md.distributions()
               if (d.metadata.get("Name") or "").lower().startswith("nvidia")))
PY
)
  if [ -n "$names" ]; then
    echo "== removing leftover CUDA packages: $names"
    # Package names have no spaces.
    VIRTUAL_ENV="$VENVS/vllm" uv pip uninstall $names || true
  fi
}

echo "== vLLM env ($VENVS/vllm)"
uv venv -q --python 3.12 "$VENVS/vllm" 2>/dev/null || true
if [ "$GPU_BACKEND" = "xpu" ]; then
  # A previous CUDA install in this env pulls nvidia-* and cannot run on XPU.
  if [ -x "$VENVS/vllm/bin/python" ] && ! "$VENVS/vllm/bin/python" -c 'import sys,torch; sys.exit(0 if getattr(torch.version, "xpu", None) else 1)' 2>/dev/null; then
    echo "== removing CUDA torch from $VENVS/vllm"
    VIRTUAL_ENV="$VENVS/vllm" uv pip uninstall vllm torch torchvision torchaudio triton || true
  fi
  drop_nvidia_packages
  # No XPU build of vLLM 0.23. The indexes below keep torch/triton off the CUDA wheels.
  echo "== Intel XPU: installing $VLLM_XPU_SPEC (NVIDIA packages are not installed)"
  VIRTUAL_ENV="$VENVS/vllm" uv pip install "$VLLM_XPU_SPEC" \
    --extra-index-url "$XPU_VLLM_INDEX" \
    --extra-index-url "$XPU_TORCH_INDEX" \
    --extra-index-url "$XPU_TRITON_INDEX" \
    --extra-index-url "$XPU_KERNELS_INDEX" \
    --index-strategy unsafe-best-match
  drop_nvidia_packages
  "$VENVS/vllm/bin/python" - <<'PY' || { echo "!! vLLM env does not have the Intel XPU torch build"; exit 1; }
import importlib.metadata as md
import importlib.util
import sys
import torch
print("torch", torch.__version__, "xpu", getattr(torch.version, "xpu", None))
xpu = bool(getattr(torch.version, "xpu", None)) or "+xpu" in torch.__version__
left = [d.metadata["Name"] for d in md.distributions()
        if (d.metadata.get("Name") or "").lower().startswith("nvidia")]
spec = importlib.util.find_spec("nvidia")
if left or spec is not None:
    print("!! CUDA packages are still installed:", ", ".join(left) or getattr(spec, "origin", "nvidia"))
if not xpu or left or spec is not None:
    sys.exit(1)
PY
else
  # transformers 5.12.1: newer releases fail to load Gemma 4 in vLLM 0.23 (head_dim error)
  VIRTUAL_ENV="$VENVS/vllm" uv pip install "vllm==0.23.0" "transformers==5.12.1"
fi
"$VENVS/vllm/bin/vllm" --version

echo "== Benchmark client env ($VENVS/bench)"
uv venv -q --python 3.12 "$VENVS/bench" 2>/dev/null || true
VIRTUAL_ENV="$VENVS/bench" uv pip install -q aiohttp pyyaml numpy pillow huggingface_hub

runtime=venv
if [ "${USE_DOCKER:-0}" = "1" ]; then
  echo "== Can Docker containers use the GPUs?"
  if [ "$GPU_BACKEND" = "xpu" ]; then
    if docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
        --entrypoint python "$VLLM_IMAGE" -c "import torch; assert torch.xpu.device_count() > 0; print(torch.xpu.device_count())"; then
      runtime=docker
    else echo "!! Docker can't use the XPUs for this account; using the vllm env instead"; fi
  elif docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi -L; then runtime=docker
  else echo "!! Docker can't use the GPUs for this account; using the vllm env instead"; fi
fi
echo "$runtime" > "$TF_HOME/.runtime"
echo "Done. vLLM runtime: $runtime. Next: scripts/01_download_models.sh"
