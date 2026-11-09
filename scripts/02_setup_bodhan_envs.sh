#!/usr/bin/env bash
# Step 3: Python environments for the three Bodhan models (ASR, TTS, OCR).
# Run after 01_download_models.sh. Safe to re-run.
#
# Default (what runs on C-DAC cn14): the Bodhan models share the vllm env, plus
# three small packages. $VENVS/asr, tts and ocr become links to it.
#
# SEPARATE_ENVS=1 instead builds one env per model (needed only if a model's
# dependencies clash with vLLM's): asr = torch + NeMo, tts = torch + transformers + snac,
# ocr = whatever the OCR repo's own install.sh picks.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export PATH="$HOME/.local/bin:$PATH"
COMMON="pyyaml numpy soundfile huggingface_hub"

if [ "${SEPARATE_ENVS:-0}" != "1" ]; then
  [ -x "$VENVS/vllm/bin/python" ] || { echo "!! no $VENVS/vllm yet: run scripts/00_prep_machine.sh first"; exit 1; }
  VIRTUAL_ENV="$VENVS/vllm" uv pip install snac soundfile accelerate pyyaml
  for e in asr tts ocr; do
    if [ -L "$VENVS/$e" ] || [ ! -e "$VENVS/$e" ]; then ln -sfn "$VENVS/vllm" "$VENVS/$e"; echo "== $VENVS/$e -> vllm env"
    else echo "== $VENVS/$e already exists as its own env; leaving it"; fi
  done
  echo "Done. Next: scripts/smoke_bodhan.sh"
  exit 0
fi

echo "== ASR env"
uv venv -q --python 3.11 "$VENVS/asr" 2>/dev/null || true
VIRTUAL_ENV="$VENVS/asr" uv pip install --torch-backend=auto torch torchaudio
VIRTUAL_ENV="$VENVS/asr" uv pip install transformers sentencepiece $COMMON
if [ "${SKIP_NEMO:-0}" != "1" ]; then
  VIRTUAL_ENV="$VENVS/asr" uv pip install "nemo_toolkit[asr]" \
    || echo "!! NeMo failed to install; the ASR benchmark falls back to the hf backend (no batching)"
fi

echo "== TTS env"
uv venv -q --python 3.11 "$VENVS/tts" 2>/dev/null || true
VIRTUAL_ENV="$VENVS/tts" uv pip install --torch-backend=auto torch
VIRTUAL_ENV="$VENVS/tts" uv pip install "transformers>=5" accelerate snac $COMMON

echo "== OCR env (runs the repo's own installer)"
uv venv -q --python 3.11 "$VENVS/ocr" 2>/dev/null || true
OCR_DIR=$("$VENVS/bench/bin/python" -c "import sys; sys.path.insert(0, 'bench'); from common import resolve_model_dir; print(resolve_model_dir('bodhan-ai/indic-ocr'))")
( source "$VENVS/ocr/bin/activate" && cd "$OCR_DIR" && ./install.sh )
VIRTUAL_ENV="$VENVS/ocr" uv pip install pillow $COMMON

echo "Done. Next: scripts/smoke_bodhan.sh"
