#!/usr/bin/env bash
# Fairness fix: the attn-style rows were sampled from a HALF-TRAINED checkpoint.
#
# longrun_2026-10-29/d50m trained for 1200 steps. The final test sampled
# step_00600 from it, while every other GPart row used its final checkpoint.
# Theta was still climbing at 600 (70.83 -> 94.23 at 1200, +33%), so that row is
# not a final model and the comparison was unfair to it.
#
# This re-samples configs 04 / 08 / 10 from step_01200. Everything else is
# byte-identical to final_test_2026-10-30.sh (same prompts, seeds, sampler,
# negative, resolution), so the checkpoint is the only changed variable.
#
# Matched alpha is rescaled for the new theta: own-coverage RMS at step 1200 is
# 94.230 / sqrt(545.6M) = 0.004034, so alpha = 0.000115 / 0.004034 = 0.0285
# (the LoRA's attn2-only RMS, same convention config 10 used). The character
# adapter keeps 0.031.
#
# 18 images, ~11 min.

set -u
cd "$(dirname "$0")/.."

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT="outputs/final_2026-10-30"
ATTN=outputs/longrun_2026-10-29/d50m/step_01200
CHAR=outputs/charrun_2026-10-30/aima_d23m/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 310977364113511 12345 67890 --size 832 1216
        --prompts-file "$OUT/prompts.jsonl" --negative "$NEG" --merge)

run() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
      --output "$OUT/$tag" --tag "$tag" "$@"
}

run gp_style_attn_f --adapter-dir "$ATTN" --alpha 1.0
run gp_stack_attn_f --adapter-dir "$ATTN" --adapter-dir "$CHAR" --alpha 1.0 1.0
run gp_stack_attn_fm --adapter-dir "$ATTN" --adapter-dir "$CHAR" --alpha 0.0285 0.031

echo "==== ATTN FINAL RERUN DONE  $(date +%H:%M:%S) ===="
