#!/usr/bin/env bash
# Style + character stacking test: GPart vs LoRA.
#
# Both adapters are applied by adding their deltas to the UNet weights, so the two
# rows are built the same way and the only difference is where the deltas come from.
# Configs chosen so the matrix answers the actual question:
#   base / style only / char only -> the two single-adapter references
#   gp_stack / lo_stack           -> the stacked pair under each method
#   gp_char12                     -> whether 1200-step theta beats 600 (norm already
#                                    reached 75, past the 70.8 that was visually best
#                                    on the style run)
#
# Prompts: 2 carry both triggers, 1 style only, 1 character only, so leakage of
# either adapter alone is visible in the grid.
set -u
cd /d/Gpart || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT=outputs/stacktest_2026-10-30
PROMPTS=$OUT/prompts.jsonl
# euler + leading: euler_ancestral silently wrecks this model's outputs
COMMON="--steps 28 --cfg 5 --sched euler --spacing leading --seeds 31337"
STYLE=outputs/style_alllinear_2026-10-30/r1999_d23m
CHAR=outputs/charrun_2026-10-30/aima_d23m
LORA_STYLE=D:/AI/models/loras/reserve1999_style_v3.safetensors
LORA_CHAR=D:/AI/models/loras/aima_noobai_v1.safetensors

mkdir -p "$OUT"

# one process at a time: each needs ~11 GB of system commit, and two at once got
# silently OOM-killed by the OS on 2026-10-29
run() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "$@" \
    --prompts-file "$PROMPTS" $COMMON --grid \
    --output "$OUT/$tag" --tag "$tag" || echo "FAILED $tag"
}

run base
run gp_style600 --adapter-dir "$STYLE/step_00600" --merge
run gp_char600  --adapter-dir "$CHAR/step_00600" --merge
# scales are positional: one value per --lora / --adapter-dir, both from the same flag
run gp_stack    --adapter-dir "$STYLE/step_00600" --adapter-dir "$CHAR/step_00600" \
                --alpha 1.0 1.0 --merge
# strength-matched variant: on the v2b prompt set GPart alpha=1.0 warmed the image
# 2.7x more than LoRA@1.5, so alpha~0.4 is the comparable point. Without this row the
# stack comparison would repeat the strength confound that was retracted once already.
run gp_stack_a04 --adapter-dir "$STYLE/step_00600" --adapter-dir "$CHAR/step_00600" \
                --alpha 0.4 0.4 --merge
run lo_style15  --lora "$LORA_STYLE" --lora-scale 1.5
run lo_char10   --lora "$LORA_CHAR" --lora-scale 1.0
run lo_stack    --lora "$LORA_STYLE" --lora "$LORA_CHAR" --lora-scale 1.5 1.0
run gp_char12   --adapter-dir "$CHAR/adapter" --merge

echo "==== STACK TEST DONE  $(date +%H:%M:%S) ===="
