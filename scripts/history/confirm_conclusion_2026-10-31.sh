#!/usr/bin/env bash
# Confirmation runs for the "matched LoRA beats GPart on the character" result.
#
# Three ways the 0.63 vs 0.40 gap could be an artifact, and one run per way:
#   seeds   n=3 -> n=6. Adds seeds 111111/222222/333333 to the existing dirs, so
#           tag_eval sees 6 images per config without any renaming.
#   p2      a second clean prompt in a different scene (indoor, upper body) to
#           check the gap is not specific to one composition.
#   scale   the matched LoRA at 1.5 / 2.0, to see whether 1.0 is already its
#           best operating point or whether it is simply under-driven.
#
# Usage: confirm_conclusion_2026-10-31.sh seeds|p2|scale [tag ...]

set -u
cd "$(dirname "$0")/.."

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT="outputs/final_2026-10-30"
M=outputs/charrun_2026-10-31/aima_lora_r8_matched
NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"
BASE=(--steps 30 --cfg 5 --sched euler_a --spacing trailing --size 832 1216)

adapter_for() {  # tag -> extra args
  case "$1" in
    base)               echo "" ;;
    lo_char)            echo "--lora D:/AI/models/loras/aima_noobai_v1.safetensors --lora-scale 1.0" ;;
    lo_char_matched)    echo "--adapter-dir $M/adapter --alpha 1.0" ;;
    lo_char_matched_s15) echo "--adapter-dir $M/adapter_s15 --alpha 1.0" ;;
    lo_char_matched_s20) echo "--adapter-dir $M/adapter_s20 --alpha 1.0" ;;
    gp_char)            echo "--adapter-dir outputs/charrun_2026-10-30/aima_d23m/adapter --alpha 1.0" ;;
    *) echo "?? $1"; return 1 ;;
  esac
}

case "${1:-}" in
  seeds)
    shift
    for tag in "$@"; do
      extra=$(adapter_for "$tag") || exit 1
      echo "==== seeds $tag  $(date +%H:%M:%S) ===="
      PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${BASE[@]}" \
        --seeds 111111 222222 333333 \
        --prompts-file "$OUT/prompts.jsonl" --negative "$NEG" \
        $extra --output "$OUT/$tag" --tag "$tag"
    done
    ;;
  p2)
    shift
    for tag in "$@"; do
      extra=$(adapter_for "$tag") || exit 1
      echo "==== p2 $tag  $(date +%H:%M:%S) ===="
      PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${BASE[@]}" \
        --seeds 310977364113511 12345 67890 \
        --prompt "$(cat "$OUT/prompts_p2.txt")" --negative "$NEG" \
        $extra --output "$OUT/${tag}_p2" --tag "${tag}_p2"
    done
    ;;
  scale)
    shift
    for tag in "$@"; do
      extra=$(adapter_for "$tag") || exit 1
      echo "==== scale $tag  $(date +%H:%M:%S) ===="
      PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${BASE[@]}" \
        --seeds 310977364113511 12345 67890 \
        --prompts-file "$OUT/prompts.jsonl" --negative "$NEG" \
        $extra --output "$OUT/$tag" --tag "$tag"
    done
    ;;
  *)
    echo "usage: $0 seeds|p2|scale [tag ...]"; exit 2 ;;
esac

echo "==== CHUNK DONE  $(date +%H:%M:%S) ===="
