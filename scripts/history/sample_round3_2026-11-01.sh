#!/usr/bin/env bash
# Round 3 sampling. Run AFTER job A and/or job B finish; each config is
# independent, so any subset can be sampled as its training output appears.
#
#   gp_char_alllinear        job A alone            -> answers "was it coverage?"
#   lo_style_matched         job B alone            -> style domain, matched recipe
#   gp_stack_alllinear_char  style attn_f + job A
#   lo_stack_matched         job B + lo_char_matched
#
# Same recipe as every earlier round: p0/p1 from prompts.jsonl, 3 seeds,
# euler_a + trailing, 30 steps, cfg 5, 832x1216.
#
# Usage: sample_round3_2026-11-01.sh <tag> [tag ...]

set -u
cd "$(dirname "$0")/.."

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT="outputs/final_2026-10-30"
A=outputs/charrun_2026-11-01/aima_alllinear_d23m/adapter
B=outputs/stylerun_2026-11-01/r1999_lora_r8_matched/adapter
STYLE_ATTN=outputs/longrun_2026-10-29/d50m/step_01200
CHAR_MATCHED=outputs/charrun_2026-10-31/aima_lora_r8_matched/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"
COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 310977364113511 12345 67890 --size 832 1216
        --prompts-file "$OUT/prompts.jsonl" --negative "$NEG")

run() {
  local tag="$1"; shift
  case "$tag" in
    gp_char_alllinear)        set -- --adapter-dir "$A" --alpha 1.0 ;;
    lo_style_matched)         set -- --adapter-dir "$B" --alpha 1.0 ;;
    gp_stack_alllinear_char)  set -- --adapter-dir "$STYLE_ATTN" --adapter-dir "$A" \
                                   --alpha 1.0 1.0 --merge ;;
    lo_stack_matched)         set -- --adapter-dir "$B" --adapter-dir "$CHAR_MATCHED" \
                                   --alpha 1.0 1.0 --merge ;;
    *) echo "?? $tag"; return 1 ;;
  esac
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" "$@" \
      --output "$OUT/$tag" --tag "$tag"
}

for t in "$@"; do run "$t" || exit 1; done
echo "==== ROUND3 SAMPLE DONE  $(date +%H:%M:%S) ===="
