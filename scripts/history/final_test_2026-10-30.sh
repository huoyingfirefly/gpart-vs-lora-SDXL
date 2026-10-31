#!/usr/bin/env bash
# Final controlled comparison: 11 adapters under one identical recipe.
#
# Recipe is ComfyUI's, not our old one: euler_ancestral + trailing spacing
# (~= ComfyUI's "normal" scheduler) + 30 steps. The three-way sampler check in
# outputs/final_2026-10-30/_samplercheck/compare3.png showed that our old
# euler/leading recipe starts at sigma 11.48 while ComfyUI starts at 14.61, so
# the two were never the same pipeline.
#
# Prompt set has TWO lines, both present in every run so the adapters are the
# only variable:
#   p0 "clean"  - content + quality tags only, no character-identity tags and no
#                 style adjectives; adapters are the only source of identity/style
#   p1 "full"   - the identity/description tags written out, like a real ComfyUI
#                 prompt; shows how much of the look comes from text alone
#
# Everything is alpha/scale 1.0 except the two *_m rows, which rescale GPart so
# its per-weight RMS equals the LoRA's at 1.0 (see docs in the report).
set -u
cd /d/Gpart || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT=outputs/final_2026-10-30
COMMON="--steps 30 --cfg 5 --sched euler_a --spacing trailing --seeds 310977364113511 12345 67890"
NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

L_STYLE=D:/AI/models/loras/reserve1999_style_v3.safetensors
L_CHAR=D:/AI/models/loras/aima_noobai_v1.safetensors
GP_STYLE_ALL=outputs/style_alllinear_2026-10-30/r1999_d23m/step_00600
GP_STYLE_ATTN=outputs/longrun_2026-10-29/d50m/step_00600
GP_CHAR=outputs/charrun_2026-10-30/aima_d23m/adapter

run() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "$@" \
    --prompts-file "$OUT/prompts.jsonl" --negative "$NEG" $COMMON \
    --output "$OUT/$tag" --tag "$tag" || echo "FAILED $tag"
}

# --- no adapter: the reference every row is judged against -------------------
run base

# --- singles ---------------------------------------------------------------
run lo_char        --lora "$L_CHAR" --lora-scale 1.0
run lo_style       --lora "$L_STYLE" --lora-scale 1.0
run gp_style_all   --adapter-dir "$GP_STYLE_ALL" --merge
run gp_style_attn  --adapter-dir "$GP_STYLE_ATTN" --merge
run gp_char        --adapter-dir "$GP_CHAR" --merge

# --- stacks at nominal 1.0 --------------------------------------------------
run lo_stack       --lora "$L_STYLE" --lora "$L_CHAR" --lora-scale 1.0 1.0
run gp_stack_all   --adapter-dir "$GP_STYLE_ALL" --adapter-dir "$GP_CHAR" \
                   --alpha 1.0 1.0 --merge
run gp_stack_attn  --adapter-dir "$GP_STYLE_ATTN" --adapter-dir "$GP_CHAR" \
                   --alpha 1.0 1.0 --merge

# --- displacement-matched GPart: alpha so per-weight RMS == LoRA at 1.0 ------
# style all-linear: 0.0012084 -> 0.000138   => 0.114
# style attn-only : 0.0030327 -> 0.000115   => 0.038
# character       : 0.0033554 -> 0.000103   => 0.031
run gp_stack_all_m  --adapter-dir "$GP_STYLE_ALL" --adapter-dir "$GP_CHAR" \
                    --alpha 0.114 0.031 --merge
run gp_stack_attn_m --adapter-dir "$GP_STYLE_ATTN" --adapter-dir "$GP_CHAR" \
                    --alpha 0.038 0.031 --merge

echo "==== FINAL DONE  $(date +%H:%M:%S) ===="
