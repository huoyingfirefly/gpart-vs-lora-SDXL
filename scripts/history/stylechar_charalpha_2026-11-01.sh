#!/usr/bin/env bash
# 补 GPart 角色 adapter 的 α 对照。
#
# 问题：横评里冠军臂是「GPart 画风 0.47 + GPart 角色 1.3」，而所有 LoRA 臂的角色是 1.0。
# 名义 α 不可比：角色这一对 ‖ΔW‖ 是 LoRA 68.8 / GPart 78.4（α=1.0），
# GPart 开到 1.3 就是 101.9，比对手大 48%。0.63 的身份分很可能是剂量买来的。
#
# 2026-10-30 的等位移量对比给的是 LoRA 0.70 > GPart 0.43，和今天这张表矛盾，
# 差的就是这一步没对齐。
#
# 除角色 α 外，其余（画风 adapter、画风 α、prompt、seed、采样参数）与横评完全一致，
# 结果直接写进同一个目录，可以和已有臂逐像素对比。
#
# 用法: stylechar_charalpha_2026-11-01.sh all | <tag> ...
set -u
cd "$(dirname "$0")/.." || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
BASE=outputs/stylechar_2026-11-01
OUT="$BASE/横评_4留出prompt×3seed"

GP_STYLE=$BASE/../longrun_2026-10-29/d50m/step_01200
GP_CHAR=$BASE/../charrun_2026-10-30/aima_d23m/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 424242 777777 999983 --size 832 1216
        --prompts-file "$BASE/prompts_holdout4.jsonl" --negative "$NEG")

one() {
  local alpha="$1" tag="$2"
  echo "==== $tag (角色 α=$alpha)  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" \
    --adapter-dir "$GP_STYLE" --adapter-dir "$GP_CHAR" \
    --alpha 0.469 "$alpha" --merge \
    --output "$OUT/$tag" --tag "$tag" || exit 1
}

case "${1:-}" in
  all)
    one 1.0 "GPart叠-画风attn2版@0.47+角色GPart@1.0"
    # 位移对齐 LoRA 角色 68.8：78.4 * 0.878 = 68.8
    one 0.88 "GPart叠-画风attn2版@0.47+角色GPart@0.88"
    one 1.6 "GPart叠-画风attn2版@0.47+角色GPart@1.6"
    ;;
  a10) one 1.0 "GPart叠-画风attn2版@0.47+角色GPart@1.0" ;;
  a088) one 0.88 "GPart叠-画风attn2版@0.47+角色GPart@0.88" ;;
  a16) one 1.6 "GPart叠-画风attn2版@0.47+角色GPart@1.6" ;;
  *) echo "?? $1"; exit 1 ;;
esac
echo "==== CHARALPHA DONE  $(date +%H:%M:%S) ===="
