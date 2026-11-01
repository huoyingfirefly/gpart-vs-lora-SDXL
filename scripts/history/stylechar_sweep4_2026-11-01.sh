#!/usr/bin/env bash
# α 定标：4 个留出 prompt × 3 个 seed × 7 个臂 = 每臂 12 张。
#
# 目的不是再比一次谁赢，是把混族的画风强度扫密一点（0.15/0.25/0.35/0.469），
# 让"最优 α 在哪"这个之前两个口径打架的问题有足够的分辨率。
#
# 全部重跑（不复用 holdout/），保证 7 个臂在完全一致的输入上比较。
# 采样是确定性的，所以 h0/h1 会产出和之前逐像素相同的图。
#
# 用法: stylechar_sweep4_2026-11-01.sh all | <tag> ...
set -u
cd "$(dirname "$0")/.." || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
BASE=outputs/stylechar_2026-11-01
OUT=$BASE/sweep4

LO_STYLE=$BASE/../stylerun_2026-11-01/r1999_lora_r8_matched/adapter
GP_STYLE=$BASE/../longrun_2026-10-29/d50m/step_01200
LO_CHAR=$BASE/../charrun_2026-10-31/aima_lora_r8_matched/adapter
GP_CHAR=$BASE/../charrun_2026-10-30/aima_d23m/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 424242 777777 999983 --size 832 1216
        --prompts-file "$BASE/prompts_holdout4.jsonl" --negative "$NEG")

run() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" "$@" \
      --output "$OUT/$tag" --tag "$tag"
}

hy() {  # hy <tag> <gpart style alpha>
  run "$1" --adapter-dir "$GP_STYLE" --adapter-dir "$LO_CHAR" \
           --alpha "$2" 1.0 --merge
}

one() {
  case "$1" in
    base)      run base ;;
    lo_s10c10) run lo_s10c10 --adapter-dir "$LO_STYLE" --adapter-dir "$LO_CHAR" \
                              --alpha 1.0 1.0 --merge ;;
    # 纯 LoRA 叠也扫同一条画风轴，否则拿它没调过的点去比是错的
    lo_s025c10) run lo_s025c10 --adapter-dir "$LO_STYLE" --adapter-dir "$LO_CHAR" \
                              --alpha 0.250 1.0 --merge ;;
    lo_s035c10) run lo_s035c10 --adapter-dir "$LO_STYLE" --adapter-dir "$LO_CHAR" \
                              --alpha 0.350 1.0 --merge ;;
    lo_s047c10) run lo_s047c10 --adapter-dir "$LO_STYLE" --adapter-dir "$LO_CHAR" \
                              --alpha 0.469 1.0 --merge ;;
    gp_s10c13) run gp_s10c13 --adapter-dir "$GP_STYLE" --adapter-dir "$GP_CHAR" \
                              --alpha 1.0 1.3 --merge ;;
    # 纯 GPart 叠自己的画风扫描：之前它一直固定在 1.0，和扫过的混族比不公平
    gp_s025c13) run gp_s025c13 --adapter-dir "$GP_STYLE" --adapter-dir "$GP_CHAR" \
                              --alpha 0.250 1.3 --merge ;;
    gp_s035c13) run gp_s035c13 --adapter-dir "$GP_STYLE" --adapter-dir "$GP_CHAR" \
                              --alpha 0.350 1.3 --merge ;;
    gp_s047c13) run gp_s047c13 --adapter-dir "$GP_STYLE" --adapter-dir "$GP_CHAR" \
                              --alpha 0.469 1.3 --merge ;;
    hy015)     hy hy015 0.150 ;;
    hy025)     hy hy025 0.250 ;;
    hy035)     hy hy035 0.350 ;;
    hy047)     hy hy047 0.469 ;;
    *) echo "?? $1"; return 1 ;;
  esac
}

if [ "${1:-}" = all ]; then
  set -- base lo_s10c10 gp_s10c13 hy015 hy025 hy035 hy047
fi
for t in "$@"; do one "$t" || exit 1; done
echo "==== SWEEP4 DONE  $(date +%H:%M:%S) ===="
