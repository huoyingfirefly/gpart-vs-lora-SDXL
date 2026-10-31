#!/usr/bin/env bash
# Scale sweep for the character LoRA.
#
# Needed because the 2026-10-30 stack test showed aima_noobai_v1 at scale 1.0
# barely moves the model (mean |delta| vs base 0.033-0.078, against 0.177-0.223
# for the GPart character adapter and 0.118-0.203 for the style LoRA alone).
# Its metadata says 530 steps, r16, lr 1e-4 - a much smaller run than the 1200
# steps GPart got, so scale 1.0 was never going to be a fair pairing.
#
# Finds the scale at which the LoRA actually expresses the character, so the
# stacked comparison can be re-run at a matched strength.
set -u
cd /d/Gpart || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT=outputs/stacktest_2026-10-30
COMMON="--steps 28 --cfg 5 --sched euler --spacing leading --seeds 31337"
LORA_CHAR=D:/AI/models/loras/aima_noobai_v1.safetensors

run() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "$@" \
    --prompts-file "$OUT/prompts.jsonl" $COMMON --grid \
    --output "$OUT/$tag" --tag "$tag" || echo "FAILED $tag"
}

run lo_char15 --lora "$LORA_CHAR" --lora-scale 1.5
run lo_char20 --lora "$LORA_CHAR" --lora-scale 2.0
run lo_char25 --lora "$LORA_CHAR" --lora-scale 2.5

echo "==== SWEEP DONE  $(date +%H:%M:%S) ===="
