#!/usr/bin/env bash
# 用社区 kohya LoRA 当"主流方法"基线的横评。
#
# 为什么之前那场对比不算数：基线是自训的 diffusers r8（rank 8、只有 transformer_blocks、
# 6 类 target、1200 步、||dW||=44.2）。真正的 mianstream 是 reserve1999_style_v3.safetensors：
# 同一份 6_reserve1999 数据集、同一个 r1999style 触发词、同一个底模，kohya sd-scripts 训的
# rank 32 / 全 UNet 722 层 / 1310 步 / lr 1e-4，||dW||=6.50。
#
# 设置和 sweep4 完全一致（同 prompt、同 seed、同采样），这样新臂能直接接进那张表里比。
#
# 用法: stylechar_kohya_2026-11-01.sh all | <tag> ...
set -u
cd "$(dirname "$0")/.." || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
BASE=outputs/stylechar_2026-11-01
OUT=$BASE/kohya
KOH="D:/AI/models/loras/reserve1999_style_v3.safetensors"

LO_CHAR=$BASE/../charrun_2026-10-31/aima_lora_r8_matched/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 424242 777777 999983 --size 832 1216
        --prompts-file "$BASE/prompts_holdout4.jsonl" --negative "$NEG")

one() {
  case "$1" in
    # 单挂：先摸清 kohya LoRA 自己的强度轴落在哪
    koh_s05)  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
                --lora "$KOH" --lora-scale 0.5 --tag koh_s05 --output "$OUT/koh_s05" ;;
    koh_s10)  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
                --lora "$KOH" --lora-scale 1.0 --tag koh_s10 --output "$OUT/koh_s10" ;;
    koh_s15)  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
                --lora "$KOH" --lora-scale 1.5 --tag koh_s15 --output "$OUT/koh_s15" ;;
    # 叠加：论文里"混族"真正要打的那个基线
    koh_s10c10) PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
                --lora "$KOH" --lora-scale 1.0 \
                --adapter-dir "$LO_CHAR" --alpha 1.0 --merge \
                --tag koh_s10c10 --output "$OUT/koh_s10c10" ;;
    koh_s15c10) PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
                --lora "$KOH" --lora-scale 1.5 \
                --adapter-dir "$LO_CHAR" --alpha 1.0 --merge \
                --tag koh_s15c10 --output "$OUT/koh_s15c10" ;;
    koh_s20c10) PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
                --lora "$KOH" --lora-scale 2.0 \
                --adapter-dir "$LO_CHAR" --alpha 1.0 --merge \
                --tag koh_s20c10 --output "$OUT/koh_s20c10" ;;
    # kohya 画风 + GPart 角色：另一个方向的混族，补一个角落
    koh_s10c13) PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
                --lora "$KOH" --lora-scale 1.0 \
                --adapter-dir "$BASE/../charrun_2026-10-30/aima_d23m/adapter" \
                --alpha 1.3 --merge \
                --tag koh_s10c13 --output "$OUT/koh_s10c13" ;;
    *) echo "?? $1"; return 1 ;;
  esac
}

if [ "${1:-}" = all ]; then
  set -- koh_s05 koh_s10 koh_s15 koh_s10c10 koh_s15c10 koh_s20c10 koh_s10c13
fi
for t in "$@"; do
  echo "==== $t  $(date +%H:%M:%S) ===="
  one "$t" || exit 1
done
echo "==== KOHYA SWEEP DONE  $(date +%H:%M:%S) ===="
