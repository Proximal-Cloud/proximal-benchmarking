#!/usr/bin/env bash
# Step 3b: one tiny run of each Bodhan model (on the last GPU) to prove the installs work.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=${BODHAN_GPU:-$(( $(nvidia-smi -L | grep -c '^GPU') - 1 ))}
mkdir -p "$INPUTS/smoke"

echo "== TTS: one Hindi sentence -> $INPUTS/smoke/hello.wav"
"$VENVS/tts/bin/python" - <<EOF
import sys, time
sys.path.insert(0, "bench"); from common import resolve_model_dir
d = resolve_model_dir("bodhan-ai/indic-speak"); sys.path.insert(0, d)
from inference import TTS
tts = TTS(d)
t = time.time(); wav = tts("नमस्ते! आज हम त्रिभुज का क्षेत्रफल निकालना सीखेंगे।", speaker="Kavya")
print(f"  {len(wav)/24000:.1f}s of audio in {time.time()-t:.1f}s")
tts.save("$INPUTS/smoke/hello.wav", wav)
EOF

echo "== ASR: transcribe that WAV"
"$VENVS/asr/bin/python" - <<EOF
import sys
sys.path.insert(0, "bench"); from common import resolve_model_dir
d = resolve_model_dir("bodhan-ai/indic-transcribe-core"); sys.path.insert(0, d)
from indic_transcribe import IndicTranscribe
asr = IndicTranscribe.from_pretrained(d)
print("  ", asr("$INPUTS/smoke/hello.wav", lang="hi"))
EOF

echo "== OCR: one placeholder page"
"$VENVS/ocr/bin/python" bench/prep_inputs.py --pages-dir "$INPUTS/smoke/pages" --n 1
"$VENVS/ocr/bin/python" - <<EOF
import sys
sys.path.insert(0, "bench"); from common import resolve_model_dir
d = resolve_model_dir("bodhan-ai/indic-ocr"); sys.path.insert(0, d)
from indic_ocr import IndicOCR
page = IndicOCR.from_pretrained(d).parse("$INPUTS/smoke/pages/synthetic_000.png")
print(page["markdown"][:400])
EOF
echo "All three Bodhan models ran."
