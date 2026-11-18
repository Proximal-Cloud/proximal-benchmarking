# Source this in every shell:  source scripts/env.sh
# Everything big (Python envs, model weights, caches, benchmark inputs) goes under
# DATA_ROOT, which should be a disk with ~150 GB free. Default /data; on another
# machine set it first, e.g.  export DATA_ROOT=$HOME/bench-data  (before sourcing).

export DATA_ROOT=${DATA_ROOT:-/data}
export TF_HOME=${TF_HOME:-$DATA_ROOT/tokenfactory}     # venvs, inputs, logs
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf}               # Hugging Face cache (weights)
export HF_XET_HIGH_PERFORMANCE=1          # faster Hugging Face downloads
export UV_CACHE_DIR=${UV_CACHE_DIR:-$DATA_ROOT/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}
export TORCH_HOME=${TORCH_HOME:-$DATA_ROOT/torch-cache}
export VENVS=${VENVS:-$TF_HOME/venvs}
export INPUTS=${INPUTS:-$TF_HOME/bench-inputs}

# GPU backend. Intel XPU machines have xpu-smi, not nvidia-smi.
# Set GPU_BACKEND=xpu or GPU_BACKEND=cuda before sourcing to override detection.
if [ -z "${GPU_BACKEND:-}" ]; then
  if command -v xpu-smi >/dev/null 2>&1; then GPU_BACKEND=xpu
  elif command -v nvidia-smi >/dev/null 2>&1; then GPU_BACKEND=cuda
  else GPU_BACKEND=none
  fi
fi
export GPU_BACKEND

# vLLM container image.
# NVIDIA: v0.23.0 is the version the GLM-5.2 / Gemma 4 recipes use.
# With driver 595 (CUDA 13.2) the default image is fine; if it complains about
# CUDA, try the -cu129 tag.
# Intel: there is no XPU build of v0.23. Use the stable XPU image and wheel.
if [ "$GPU_BACKEND" = "xpu" ]; then
  export GPU_VISIBLE_ENV=ZE_AFFINITY_MASK
  export VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai-xpu:latest}
  export VLLM_XPU_SPEC=${VLLM_XPU_SPEC:-vllm==0.34.0+xpu}
  export XPU_VLLM_INDEX=${XPU_VLLM_INDEX:-https://wheels.vllm.ai/0.34.0/xpu}
  export XPU_TORCH_INDEX=${XPU_TORCH_INDEX:-https://download.pytorch.org/whl/xpu}
  export XPU_TRITON_INDEX=${XPU_TRITON_INDEX:-https://wheels.vllm.ai/xpu}
  export XPU_KERNELS_INDEX=${XPU_KERNELS_INDEX:-https://vllm-project.github.io/vllm-xpu-kernels/whl}
else
  export GPU_VISIBLE_ENV=CUDA_VISIBLE_DEVICES
  export VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.23.0}
fi

# CPU cores next to each GPU, so each model server stays on its own NUMA node.
# Set for C-DAC cn14 (from `nvidia-smi topo -m`: GPU0 -> NUMA 0, GPUs 1-2 -> NUMA 1).
# Elsewhere pinning is off unless you set these yourself before sourcing.
case "$(hostname -s)" in
  cn14*) export CPUS_GPU0=${CPUS_GPU0:-"0-31,64-95"} CPUS_GPU12=${CPUS_GPU12:-"32-63,96-127"} ;;
  *)     export CPUS_GPU0=${CPUS_GPU0:-} CPUS_GPU12=${CPUS_GPU12:-} ;;
esac

# FlashInfer's sampler compiles CUDA code on first use and needs nvcc, which cn14
# doesn't have. Use vLLM's built-in PyTorch sampler instead (same output).
# Intel XPU does not use FlashInfer; do not set this there.
if [ "$GPU_BACKEND" != "xpu" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

# Print how many GPUs the management tool can see. Echoes 0 if it cannot tell.
gpu_count() {
  local n=""
  if [ "$GPU_BACKEND" = "xpu" ] && command -v xpu-smi >/dev/null 2>&1; then
    n=$(xpu-smi discovery -j 2>/dev/null | python3 -c 'import json,sys
raw=sys.stdin.read()
try:
    d=json.loads(raw)
except Exception:
    raise SystemExit
devs=d.get("device_list") or d.get("deviceList") or []
print(len(devs))' || true)
    if [ -z "$n" ] || [ "$n" = "0" ]; then
      n=$(xpu-smi discovery 2>/dev/null | grep -c 'Device Name:' || true)
    fi
  elif command -v nvidia-smi >/dev/null 2>&1; then
    n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  fi
  echo "${n:-0}"
}

# Hugging Face login is stored by huggingface_hub under $HF_HOME, never in this repo.

# Triton compiles a small helper and links -lze_loader.
# The driver package often installs only libze_loader.so.1, which the linker will not use.
setup_xpu_ze_loader() {
  [ "${GPU_BACKEND:-}" = "xpu" ] || return 0
  local p real dir linkdir="$TF_HOME/xpu-link"
  for p in \
    /usr/lib/x86_64-linux-gnu/libze_loader.so \
    /usr/lib64/libze_loader.so \
    /usr/lib/libze_loader.so \
    /usr/local/lib/libze_loader.so \
    /usr/local/lib64/libze_loader.so
  do
    [ -e "$p" ] && return 0
  done
  real=""
  if command -v ldconfig >/dev/null 2>&1; then
    real=$(ldconfig -p 2>/dev/null | awk '/libze_loader\.so/ { print $NF; exit }')
  fi
  if [ -z "$real" ]; then
    for p in \
      /usr/lib/x86_64-linux-gnu/libze_loader.so.1 \
      /usr/lib64/libze_loader.so.1 \
      /lib/x86_64-linux-gnu/libze_loader.so.1 \
      /usr/local/lib/libze_loader.so.1 \
      /opt/intel/oneapi/lib/libze_loader.so \
      /opt/intel/oneapi/lib/libze_loader.so.1 \
      /opt/intel/oneapi/compiler/latest/lib/libze_loader.so \
      /opt/intel/oneapi/level-zero/latest/lib/libze_loader.so
    do
      if [ -e "$p" ]; then real=$p; break; fi
    done
  fi
  if [ -z "$real" ]; then
    echo "!! libze_loader not found. Triton cannot compile XPU kernels." >&2
    echo "   Install Level Zero, then re-run: sudo apt-get install -y libze1 libze-dev" >&2
    return 0
  fi
  dir=$(dirname "$real")
  if [ ! -e "$dir/libze_loader.so" ]; then
    mkdir -p "$linkdir"
    ln -sfn "$real" "$linkdir/libze_loader.so"
    dir=$linkdir
    echo "== XPU: Triton will link libze_loader via $linkdir" >&2
  fi
  case ":${LIBRARY_PATH:-}:" in
    *":$dir:"*) ;;
    *) export LIBRARY_PATH="$dir${LIBRARY_PATH:+:$LIBRARY_PATH}" ;;
  esac
}
setup_xpu_ze_loader

true  # sourcing must always succeed, or scripts using `set -e` exit silently
