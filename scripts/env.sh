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

# vLLM container image. v0.23.0 is the version the GLM-5.2 / Gemma 4 recipes use.
# With driver 595 (CUDA 13.2) the default image is fine; if it complains about
# CUDA, try the -cu129 tag.
export VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.23.0}

# CPU cores next to each GPU, so each model server stays on its own NUMA node.
# Set for C-DAC cn14 (from `nvidia-smi topo -m`: GPU0 -> NUMA 0, GPUs 1-2 -> NUMA 1).
# Elsewhere pinning is off unless you set these yourself before sourcing.
case "$(hostname -s)" in
  cn14*) export CPUS_GPU0=${CPUS_GPU0:-"0-31,64-95"} CPUS_GPU12=${CPUS_GPU12:-"32-63,96-127"} ;;
  *)     export CPUS_GPU0=${CPUS_GPU0:-} CPUS_GPU12=${CPUS_GPU12:-} ;;
esac

# FlashInfer's sampler compiles CUDA code on first use and needs nvcc, which cn14
# doesn't have. Use vLLM's built-in PyTorch sampler instead (same output).
export VLLM_USE_FLASHINFER_SAMPLER=0

# Hugging Face login is stored by huggingface_hub under $HF_HOME, never in this repo.
true  # sourcing must always succeed, or scripts using `set -e` exit silently
