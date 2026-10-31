#!/usr/bin/env bash
# GPart character strength sweep - the cell that was never sampled.
#
# LoRA got 3-4 strength points on the character task; GPart got exactly one
# (alpha 1.0). Comparing "LoRA's optimum" against "GPart's default" is not a
# fair test, and REPORT.md 7 says so without ever fixing it.
#
# Direction: gp_char @1.0 has per-weight RMS 0.003351, while the matched LoRA
# peaks at 0.001472 and gets WORSE as displacement grows (0.70 -> 0.57 -> 0.47
# at scale 1.0 / 1.5 / 2.0). So the interesting points are BELOW 1.0.
#
#   0.44  exact displacement match with lo_char_matched @1.0
#   0.65  between the match point and the default
#   0.30  well under - tests whether "less is more" holds for GPart too
#
# Usage: sweep_gp_char_2026-11-01.sh [tag ...]     (default: all three)
# Costs ~4 min per alpha. Do NOT run while a sampler is active (each needs
# ~11 GB commit); wait until the training job on the GPU has exited.

set -u
cd "$(dirname "$0")/.."

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT="outputs/final_2026-10-30"
GP_CHAR=outputs/charrun_2026-10-30/aima_d23m/adapter
GP_ALL=outputs/charrun_2026-11-01/aima_alllinear_d23m/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"
COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 310977364113511 12345 67890 --size 832 1216
        --prompts-file "$OUT/prompts.jsonl" --negative "$NEG")

run() {
  local tag="$1" dir="$2" alpha="$3"
  if [ ! -d "$dir" ]; then echo "?? $tag: no adapter at $dir"; return 1; fi
  echo "==== $tag  alpha=$alpha  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
      --adapter-dir "$dir" --alpha "$alpha" \
      --output "$OUT/$tag" --tag "$tag"
}

# attn2-only GPart (gp_char), the config that scored 0.43
run gp_char_s44 "$GP_CHAR" 0.44
run gp_char_s65 "$GP_CHAR" 0.65
run gp_char_s30 "$GP_CHAR" 0.30

# all-linear GPart (job A) - same sweep once it finishes, so the sweep is not
# confounded with the layer-coverage question.
if [ -d "$GP_ALL" ]; then
  run gp_char_al_s44 "$GP_ALL" 0.44
  run gp_char_al_s65 "$GP_ALL" 0.65
fi

echo "==== SWEEP DONE  $(date +%H:%M:%S) ===="
