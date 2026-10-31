#!/usr/bin/env bash
# Does a stronger LoRA deliver the full character design, or only more hair colour?
#
# At scale 1.0 the LoRA rows reached a tag score of -0.07 (below the base model)
# while GPart reached +0.30. The open question is whether that gap is a property
# of the LoRA's dW or just its operating point: 1.0 is ~1/30 of GPart's per-weight
# displacement. This sweeps the LoRA up to 3.0 on the same clean prompt.
#
# p0 (clean) only: p1 describes the character in text and drowns the adapter.
# 3 configs x 3 scales x 3 seeds = 27 images, ~22 min at ~40 s/image.

set -u
cd "$(dirname "$0")/.."

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT="outputs/final_2026-10-30"
L_STYLE=D:/AI/models/loras/reserve1999_style_v3.safetensors
L_CHAR=D:/AI/models/loras/aima_noobai_v1.safetensors

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

for S in 1.5 2.0 3.0; do
  # --lora-scale is positional: it must come after every --lora it applies to
  run "lo_char_s${S/"."/}"  --lora "$L_CHAR"  --lora-scale "$S"
done

for S in 1.5 2.0 3.0; do
  run "lo_style_s${S/"."/}" --lora "$L_STYLE" --lora-scale "$S"
done

for S in 1.5 2.0 3.0; do
  run "lo_stack_s${S/"."/}" --lora "$L_STYLE" --lora "$L_CHAR" --lora-scale "$S" "$S"
done

echo "==== SWEEP DONE  $(date +%H:%M:%S) ===="
