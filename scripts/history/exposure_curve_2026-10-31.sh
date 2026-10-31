#!/usr/bin/env bash
# Exposure-matched character comparison: kill the "1200 vs 530 steps is unfair" objection.
#
# The raw step counts are not comparable because the effective batch differs
# (LoRA 1x4, GPart 1x2). Converted to image exposures they are nearly equal:
#   LoRA   530 steps x 4 = 2120 samples, 21 images -> ~101 views/image
#   GPart 1200 steps x 2 = 2400 samples, 21 images -> ~114 views/image
#
# Both trainers kept intermediate checkpoints, so the two ladders can be aligned
# on views/image instead of on steps:
#
#   views/img   GPart (aima_d23m)      LoRA (aima_noobai_v1-00000N)
#   ~55-60      step_00600  (1140/21)  000006  (epoch 6, 60)
#   ~72-80      step_00800  (1520/21)  000008  (epoch 8, 80)
#   ~90-95      step_01000  (1900/21)  000009  (epoch 9, 90)
#   ~114/101    step_01200 (already)   final   (already, the 1.0 row)
#
# 6 configs x 3 seeds = 18 images, ~10 min. GPU is busy with the strength sweep
# until that finishes - run this only afterwards.
#
# Note what this does NOT equalise: learning rate (GPart 5e-4 vs LoRA 1e-4).
# Matching exposures will show how fast each one picks up the design, not how
# far it can go.

set -u
cd "$(dirname "$0")/.."

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT="outputs/final_2026-10-30"
GP=outputs/charrun_2026-10-30/aima_d23m
L=D:/AI/models/loras

P0="r1999style, aima, 1girl, solo, full body, standing, from front, looking at viewer, long hair, black coat, white dress, outdoors, city street, evening, crowd in background, masterpiece, best quality, very aesthetic, absurdres, highly detailed"

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 310977364113511 12345 67890 --size 832 1216
        --prompt "$P0" --negative "$NEG")

run() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
      --output "$OUT/$tag" --tag "$tag" "$@"
}

for STEP in 600 800 1000; do
  run "gp_c${STEP}" --adapter-dir "$GP/step_0${STEP}" --alpha 1.0
done

for EP in 6 8 9; do
  run "lo_c${EP}" --lora "$L/aima_noobai_v1-00000${EP}.safetensors" --lora-scale 1.0
done

echo "==== EXPOSURE CURVE DONE  $(date +%H:%M:%S) ===="
