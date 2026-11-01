#!/usr/bin/env bash
# 拆开"角色 adapter 本身"和"画风 adapter 的干扰"。
#
# 矛盾：同一个 LoRA 角色 adapter，2026-10-30 单独用是 0.70，今天叠着 GPart 画风
# 只有 0.50；同一个 GPart 角色 adapter，10-30 是 0.43，今天是 0.63（还是 α=1.3）。
#
# 两天之间只改了 prompt 集（p0 全身街景 → holdout4 四种景别）和"是否叠画风 adapter"。
# 这组把角色 adapter 单独挂到 holdout4 上跑，于是：
#   昨天值 vs 这组值   → 差在 prompt 集
#   这组值 vs 今天叠画风的值 → 差在画风 adapter 的干扰
#
# 用法: stylechar_charsolo_2026-11-01.sh all
set -u
cd "$(dirname "$0")/.." || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
BASE=outputs/stylechar_2026-11-01
OUT="$BASE/角色单挂_4留出prompt×3seed"

LO_CHAR=$BASE/../charrun_2026-10-31/aima_lora_r8_matched/adapter
GP_CHAR=$BASE/../charrun_2026-10-30/aima_d23m/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 424242 777777 999983 --size 832 1216
        --prompts-file "$BASE/prompts_holdout4.jsonl" --negative "$NEG")

one() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" "$@" \
    --output "$OUT/$tag" --tag "$tag" || exit 1
}

case "${1:-}" in
  all)
    one "角色单挂-LoRA@1.0+无画风"  --adapter-dir "$LO_CHAR" --alpha 1.0
    one "角色单挂-GPart@1.0+无画风" --adapter-dir "$GP_CHAR" --alpha 1.0
    one "角色单挂-GPart@1.3+无画风" --adapter-dir "$GP_CHAR" --alpha 1.3
    ;;
  *) echo "?? $1"; exit 1 ;;
esac
echo "==== CHARSOLO DONE  $(date +%H:%M:%S) ===="
