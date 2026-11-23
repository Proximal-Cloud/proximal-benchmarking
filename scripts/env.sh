# Source this in every shell:  source scripts/env.sh
# Everything big (Python envs, model weights, caches, benchmark inputs) goes under
# DATA_ROOT, which should be a disk with ~150 GB free. Default /data; on another
# machine set it first, e.g.  export DATA_ROOT=$HOME/bench-data  (before sourcing).

# Repo root, so the helpers below work from any working directory.
BENCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

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

# GPU list for one model. A 32GB B70 cannot hold Gemma 4 31B (4-bit) and a 40K
# context on one card (about 3 GiB of KV cache left; 40K needs about 7 GiB).
# Two cards fit the spec. Set TP=1 to keep a single card.
model_gpus() { # $1 model name, $2 requested id or list
  local name=$1 requested=${2:-0} n next
  if [ "$GPU_BACKEND" = "xpu" ] && [ "$name" = "gemma31b-w4" ] && [ "${TP:-auto}" != "1" ]; then
    case "$requested" in
      *,*) printf '%s\n' "$requested" ;;
      *)
        n=$(gpu_count)
        next=$((requested + 1))
        if [ "${n:-0}" -gt "$next" ]; then printf '%s\n' "$requested,$next"
        else printf '%s\n' "$requested"; fi
        ;;
    esac
  else
    printf '%s\n' "$requested"
  fi
}

# Print how many GPUs the management tool can see. Echoes 0 if it cannot tell.
gpu_count() {
  python3 "$BENCH_ROOT/bench/common.py" gpu-count 2>/dev/null || echo 0
}

# True when the Hugging Face cache holds a weight file for this repo.
# A snapshot with only the tokenizer and config is left by an interrupted download.
have_model() { # $1 = repo, e.g. google/gemma-4-31B-it-qat-w4a16-ct
  local dir="$HF_HOME/hub/models--${1//\//--}/snapshots" f size
  [ -d "$dir" ] || return 1
  while IFS= read -r -d '' f; do
    [ -f "$f" ] || continue
    size=$(stat -Lc%s "$f" 2>/dev/null || stat -Lf%z "$f" 2>/dev/null || echo 0)
    [ "$size" -gt 1000000 ] && return 0
  done < <(find "$dir" \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.gguf' \) -print0)
  return 1
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

# Triton runs `ocloc` to turn SPIR-V into an XPU binary. It is a separate package
# from the Level Zero loader, and it must be on PATH.
setup_xpu_ocloc() {
  [ "${GPU_BACKEND:-}" = "xpu" ] || return 0
  command -v ocloc >/dev/null 2>&1 && return 0
  local p
  for p in \
    /usr/bin/ocloc \
    /usr/local/bin/ocloc \
    /opt/intel/oneapi/ocloc/latest/bin/ocloc \
    /opt/intel/oneapi/bin/ocloc \
    /usr/lib/x86_64-linux-gnu/intel-opencl/ocloc
  do
    if [ -x "$p" ]; then
      export PATH="$(dirname "$p"):${PATH}"
      echo "== XPU: using ocloc at $p" >&2
      return 0
    fi
  done
  echo "!! ocloc not found. Triton cannot compile XPU kernels." >&2
  echo "   Install the package that matches this machine's compute runtime:" >&2
  echo "     sudo apt-get install -y intel-ocloc" >&2
  return 0
}
setup_xpu_ocloc

true  # sourcing must always succeed, or scripts using `set -e` exit silently
