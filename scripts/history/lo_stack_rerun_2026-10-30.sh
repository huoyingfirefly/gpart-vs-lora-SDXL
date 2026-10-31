#!/usr/bin/env bash
# Re-run the LoRA stack at character scales that actually express the character.
#
# The first lo_stack paired LoRA style 1.5 with LoRA aima 1.0. The zoom comparison
# (review/zoom_p3.png) shows aima at 1.0 barely registers, at 1.5 the X hair
# ornament appears, at 2.0 the pink gradient and pink eyes appear. So 1.0 was
# below the threshold where the character survives on top of a 1.5 style LoRA.
set -u
cd /d/Gpart || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT=outputs/stacktest_2026-10-30
COMMON="--steps 28 --cfg 5 --sched euler --spacing leading --seeds 31337"
LORA_STYLE=D:/AI/models/loras/reserve1999_style_v3.safetensors
LORA_CHAR=D:/AI/models/loras/aima_noobai_v1.safetensors

run() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "$@" \
    --prompts-file "$OUT/prompts.jsonl" $COMMON --grid \
    --output "$OUT/$tag" --tag "$tag" || echo "FAILED $tag"
}

run lo_stack_c15 --lora "$LORA_STYLE" --lora "$LORA_CHAR" --lora-scale 1.5 1.5
run lo_stack_c20 --lora "$LORA_STYLE" --lora "$LORA_CHAR" --lora-scale 1.5 2.0

echo "==== RERUN DONE  $(date +%H:%M:%S) ===="
