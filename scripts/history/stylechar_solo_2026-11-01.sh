#!/usr/bin/env bash
# 补齐"单挂画风 adapter"这一层。
#
# 坑在哪：之前比"单挂画风"用的是 s0 那套 prompt（prompts.jsonl），而 sweep4 / kohya
# 用的是 holdout4。跨 prompt 集排表等于又比了一次不同条件。kohya 那三档单挂
# （koh_s05/10/15）是在 holdout4 上跑的，所以 LoRA 和 GPart 的单挂也得在同一边补。
#
# 用法: stylechar_solo_2026-11-01.sh all | <tag> ...
set -u
cd "$(dirname "$0")/.." || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
BASE=outputs/stylechar_2026-11-01
OUT=$BASE/solo

LO_STYLE=$BASE/../stylerun_2026-11-01/r1999_lora_r8_matched/adapter
GP_STYLE=$BASE/../longrun_2026-10-29/d50m/step_01200

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 424242 777777 999983 --size 832 1216
        --prompts-file "$BASE/prompts_holdout4.jsonl" --negative "$NEG")

one() {
  case "$1" in
    lo_s047) PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
               --adapter-dir "$LO_STYLE" --alpha 0.469 \
               --output "$OUT/lo_s047" --tag lo_s047 ;;
    gp_s047) PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
               --adapter-dir "$GP_STYLE" --alpha 0.469 \
               --output "$OUT/gp_s047" --tag gp_s047 ;;
    # 拿 kohya 单挂最强的那档做参照，顺便确认它和 kohya/ 目录里那张逐像素一致
    koh_s15) PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
               --lora "D:/AI/models/loras/reserve1999_style_v3.safetensors" --lora-scale 1.5 \
               --output "$OUT/koh_s15" --tag koh_s15 ;;
    *) echo "?? $1"; return 1 ;;
  esac
}

if [ "${1:-}" = all ]; then
  set -- lo_s047 gp_s047 koh_s15
fi
for t in "$@"; do
  echo "==== $t  $(date +%H:%M:%S) ===="
  one "$t" || exit 1
done
echo "==== SOLO DONE  $(date +%H:%M:%S) ===="
