#!/usr/bin/env bash
# 补两个缺口。
#
# A. 单挂画风（lo_s047 / gp_s047）
#    之前"单挂画风"那一层用的是 s0 那套 prompt，而 kohya 单挂和整个横评用的是 holdout4。
#    跨 prompt 集排表等于没好。这两臂补到同一边，才能公平地排
#    底模 / 自训r8单挂 / GPart单挂 / kohya单挂。
#
# B. GPart 全参画风接进横评
#    style_alllinear_2026-10-30/r1999_d23m 是 attn1+attn2+FFN 全覆盖版本（700 模块），
#    但只训了 600 步，‖θ‖=56.48，而 attn2-only 的 d50m 是 1200 步、‖θ‖=94.23。
#    位移对齐系数 = 94.23 / 56.48 = 1.668。
#      横评冠军的画风分量是 d50m@0.469 = 44.19  →  全参版 α = 0.469 × 1.668 = 0.782
#    跑 0.782 / 1.0 / 1.668 三档，角色固定 1.3，只变画风 adapter。
#    如果全参版在等位移下不比 attn2 版差，说明"覆盖率不是瓶颈"在画风上也成立；
#    如果明显差，说明 10-30 报告 §6.3 的结论可以外推到画风。
#
# 用法: stylechar_round2_2026-11-02.sh all | solo | alllinear
set -u
cd "$(dirname "$0")/.." || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
BASE=outputs/stylechar_2026-11-01
OUT_SOLO="$BASE/单挂画风_4留出prompt×3seed"
OUT_SWEEP="$BASE/横评_4留出prompt×3seed"

LO_STYLE=$BASE/../stylerun_2026-11-01/r1999_lora_r8_matched/adapter
GP_STYLE=$BASE/../longrun_2026-10-29/d50m/step_01200
GP_STYLE_ALL=$BASE/../style_alllinear_2026-10-30/r1999_d23m/step_00600
GP_CHAR=$BASE/../charrun_2026-10-30/aima_d23m/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 424242 777777 999983 --size 832 1216
        --prompts-file "$BASE/prompts_holdout4.jsonl" --negative "$NEG")

run() {
  local out="$1" tag="$2"; shift 2
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" "$@" \
    --output "$out/$tag" --tag "$tag" || exit 1
}

do_solo() {
  run "$OUT_SOLO" "LoRA单挂-画风自训r8@0.47+无角色" \
      --adapter-dir "$LO_STYLE" --alpha 0.469
  run "$OUT_SOLO" "GPart单挂-画风attn2版@0.47+无角色" \
      --adapter-dir "$GP_STYLE" --alpha 0.469
}

do_alllinear() {
  local a="$1" label="$2"
  run "$OUT_SWEEP" "GPart叠-画风全参版@${label}+角色GPart@1.3" \
      --adapter-dir "$GP_STYLE_ALL" --adapter-dir "$GP_CHAR" --alpha "$a" 1.3 --merge
}

case "${1:-}" in
  all)       do_solo; do_alllinear 0.782 0.78; do_alllinear 1.0 1.0; do_alllinear 1.668 1.67 ;;
  solo)      do_solo ;;
  alllinear) do_alllinear 0.782 0.78; do_alllinear 1.0 1.0; do_alllinear 1.668 1.67 ;;
  *) echo "?? $1"; exit 1 ;;
esac
echo "==== ROUND2 DONE  $(date +%H:%M:%S) ===="
