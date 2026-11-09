#!/usr/bin/env bash
# Step 1: make sure the benchmark's models and data are on this machine. Safe to re-run.
#   Models (Gemma 4 31B, Bodhan x3): checked in $HF_HOME; anything already there is skipped.
#   ShareGPT (~650 MB, public): downloaded with curl, no Hugging Face login needed.
# Needs no `hf` command and no pip install. A login is only needed if a gated model
# (Gemma, Bodhan) is missing; the script says so and shows how.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
unset HF_HUB_OFFLINE
PY="$VENVS/bench/bin/python"; [ -x "$PY" ] || PY="$VENVS/vllm/bin/python"

have_model() { ls -d "$HF_HOME/hub/models--${1//\//--}"/snapshots/*/ >/dev/null 2>&1; }

# ---- models
repos=$("$PY" -c "import sys; sys.path.insert(0, 'bench'); from common import load_yaml; print(' '.join(m['repo'] for m in load_yaml('llm_targets.yaml')['models'].values()))")
repos="$repos bodhan-ai/indic-transcribe-core bodhan-ai/indic-speak hubertsiuzdak/snac_24khz bodhan-ai/indic-ocr"
missing=()
for r in $repos; do
  if have_model "$r"; then echo "== $r: already here"; else echo "== $r: MISSING"; missing+=("$r"); fi
done
for r in "${missing[@]}"; do
  echo "== downloading $r"
  "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$r" \
    || { echo "!! $r failed. If it is gated (Gemma, Bodhan): accept its terms on huggingface.co, then log in with"
         echo "     $PY -c 'from huggingface_hub import login; login()'"
         echo "   paste a read token from huggingface.co/settings/tokens, and re-run this script."; }
done

# ---- ShareGPT for the public benchmark (public file: plain HTTPS, no token)
SG="$INPUTS/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
mkdir -p "$INPUTS/datasets"
if [ -s "$SG" ]; then
  echo "== ShareGPT: already here"
else
  echo "== ShareGPT (~650 MB)"
  # -C - resumes a partial download if you re-run after a dropped connection
  if curl -fL --retry 5 -C - -o "$SG.part" "$URL"; then mv "$SG.part" "$SG"
  else echo "!! ShareGPT download failed; re-run this script to resume"; fi
fi
[ -s "$SG" ] && "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); print(f'== ShareGPT OK: {len(d):,} conversations')" "$SG" \
  || echo "!! ShareGPT file is incomplete or not valid JSON; delete $SG and re-run"
echo "== Done."
