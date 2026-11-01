#!/usr/bin/env bash
# 留出集验证：混族叠加到底成不成立。
#
# 和 stylechar_2026-11-01.sh 的区别：
#   - 两个没见过的 prompt（近景 / 大远景，故意跨取景）
#   - 三个没用过的 seed
#   - 只跑五个关键臂
#
# α=0.25 是在 s0 上调出来的，这里就是用来检验它在未见数据上还成不成立。
#
# 用法: stylechar_holdout_2026-11-01.sh all | <tag> ...
set -u
cd "$(dirname "$0")/.." || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
BASE=outputs/stylechar_2026-11-01
OUT=$BASE/holdout

LO_STYLE=$BASE/../stylerun_2026-11-01/r1999_lora_r8_matched/adapter
GP_STYLE=$BASE/../longrun_2026-10-29/d50m/step_01200
LO_CHAR=$BASE/../charrun_2026-10-31/aima_lora_r8_matched/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 424242 777777 999983 --size 832 1216
        --prompts-file "$BASE/prompts_holdout.jsonl" --negative "$NEG")

run() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" "$@" \
      --output "$OUT/$tag" --tag "$tag"
}

one() {
  case "$1" in
    base)        run base ;;
    lo_s10c10)   run lo_s10c10 --adapter-dir "$LO_STYLE" --adapter-dir "$LO_CHAR" \
                                --alpha 1.0 1.0 --merge ;;
    gp_s10c13)   run gp_s10c13 --adapter-dir "$GP_STYLE" --adapter-dir "$BASE/../charrun_2026-10-30/aima_d23m/adapter" \
                                --alpha 1.0 1.3 --merge ;;
    hy_gp025_lo10) run hy_gp025_lo10 --adapter-dir "$GP_STYLE" --adapter-dir "$LO_CHAR" \
                                --alpha 0.250 1.0 --merge ;;
    hy_gp047_lo10) run hy_gp047_lo10 --adapter-dir "$GP_STYLE" --adapter-dir "$LO_CHAR" \
                                --alpha 0.469 1.0 --merge ;;
    *) echo "?? $1"; return 1 ;;
  esac
}

if [ "${1:-}" = all ]; then
  set -- base lo_s10c10 gp_s10c13 hy_gp025_lo10 hy_gp047_lo10
fi
for t in "$@"; do one "$t" || exit 1; done
echo "==== HOLDOUT DONE  $(date +%H:%M:%S) ===="
