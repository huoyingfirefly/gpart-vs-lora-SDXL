#!/usr/bin/env bash
# 画风 + 角色：LoRA vs GPart，两边各自取最优强度。
#
# 和之前几组的区别：
#   1. 两边都用**配方对齐**的适配器（lr 5e-4, 1200 步），不是 kohya 的 v1
#   2. 强度可以分别调 —— sample.py 的 --alpha 现在同时缩放 GPart theta 和 LoRA lora_B
#   3. prompt 走修好的分块编码器，长 prompt 不再丢画质标签
#
# 主流做法是"画风 0.5–0.7、角色 0.8–1.0"，所以每组都扫三档画风强度；
# GPart 的角色侧用 1.3（已知工作点，见 REPORT §6.9）。
#
# 用法: stylechar_2026-11-01.sh all | <tag> ...
set -u
cd "$(dirname "$0")/.." || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT=outputs/stylechar_2026-11-01

# 画风：配方对齐的 r8 LoRA / attn d=50M GPart
LO_STYLE=outputs/stylerun_2026-11-01/r1999_lora_r8_matched/adapter
GP_STYLE=outputs/longrun_2026-10-29/d50m/step_01200
# 角色：同一套配方
LO_CHAR=outputs/charrun_2026-10-31/aima_lora_r8_matched/adapter
GP_CHAR=outputs/charrun_2026-10-30/aima_d23m/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

COMMON=(--steps 30 --cfg 5 --sched euler_a --spacing trailing
        --seeds 310977364113511 12345 67890 --size 832 1216
        --prompts-file "$OUT/prompts.jsonl" --negative "$NEG")

run() {
  local tag="$1"; shift
  echo "==== $tag  $(date +%H:%M:%S) ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "${COMMON[@]}" "$@" \
      --output "$OUT/$tag" --tag "$tag"
}

one() {
  case "$1" in
    base)      run base ;;

    # --- singles: 定位每条曲线的形状 -------------------------------------
    lo_s10)    run lo_s10 --adapter-dir "$LO_STYLE" --alpha 1.0 ;;
    gp_s10)    run gp_s10 --adapter-dir "$GP_STYLE" --alpha 1.0 ;;
    lo_c10)    run lo_c10 --adapter-dir "$LO_CHAR"  --alpha 1.0 ;;
    gp_c13)    run gp_c13 --adapter-dir "$GP_CHAR"  --alpha 1.3 ;;

    # --- 等位移量对照 ---------------------------------------------------
    # 名义 α 不可比：画风 LoRA ||dW||=44.2，画风 GPart ||dW||=94.2。
    # 下面两臂各自把对方的口径拉平，用来看"GPart 画风更准"是剂量还是结构。
    lo_s213)   run lo_s213 --adapter-dir "$LO_STYLE" --alpha 2.130 ;;
    gp_s047)   run gp_s047 --adapter-dir "$GP_STYLE" --alpha 0.469 ;;
    # 按 UNet 输出扰动对齐时 GPart 的等效 α（见 scripts/measure_effect_2026-11-01.py）
    gp_s067)   run gp_s067 --adapter-dir "$GP_STYLE" --alpha 0.670 ;;
    lo_s213c10) run lo_s213c10 --adapter-dir "$LO_STYLE" --adapter-dir "$LO_CHAR" \
                               --alpha 2.130 1.0 --merge ;;
    gp_s047c13) run gp_s047c13 --adapter-dir "$GP_STYLE" --adapter-dir "$GP_CHAR" \
                               --alpha 0.469 1.3 --merge ;;

    # --- 混族叠加：画风用 GPart，角色用 LoRA -----------------------------
    # 依据：身份绑定 LoRA 强，全局调色 GPart 强（等效果 α≈0.67，见 effect_dose.json）。
    # 两个 delta 都是加性的，--merge 顺序无关；这里两种顺序都跑，顺带验证交换律。
    hy_gp067_lo10) run hy_gp067_lo10 --adapter-dir "$GP_STYLE" --adapter-dir "$LO_CHAR" \
                                      --alpha 0.670 1.0 --merge ;;
    hy_gp047_lo10) run hy_gp047_lo10 --adapter-dir "$GP_STYLE" --adapter-dir "$LO_CHAR" \
                                      --alpha 0.469 1.0 --merge ;;
    # 角色 adapter 自己抬饱和、画风 adapter 压饱和，两者在 α≈0.25 处穿过参考值
    hy_gp025_lo10) run hy_gp025_lo10 --adapter-dir "$GP_STYLE" --adapter-dir "$LO_CHAR" \
                                      --alpha 0.250 1.0 --merge ;;
    hy_rev_lo10gp067) run hy_rev_lo10gp067 --adapter-dir "$LO_CHAR" --adapter-dir "$GP_STYLE" \
                                      --alpha 1.0 0.670 --merge ;;

    # --- LoRA 叠加：角色固定 1.0，扫画风 --------------------------------
    lo_s10c10) run lo_s10c10 --adapter-dir "$LO_STYLE" --adapter-dir "$LO_CHAR" \
                              --alpha 1.0 1.0 --merge ;;
    lo_s07c10) run lo_s07c10 --adapter-dir "$LO_STYLE" --adapter-dir "$LO_CHAR" \
                              --alpha 0.7 1.0 --merge ;;
    lo_s05c10) run lo_s05c10 --adapter-dir "$LO_STYLE" --adapter-dir "$LO_CHAR" \
                              --alpha 0.5 1.0 --merge ;;

    # --- GPart 叠加：角色固定 1.3（已知工作点），扫画风 ------------------
    gp_s10c13) run gp_s10c13 --adapter-dir "$GP_STYLE" --adapter-dir "$GP_CHAR" \
                              --alpha 1.0 1.3 --merge ;;
    gp_s07c13) run gp_s07c13 --adapter-dir "$GP_STYLE" --adapter-dir "$GP_CHAR" \
                              --alpha 0.7 1.3 --merge ;;
    gp_s05c13) run gp_s05c13 --adapter-dir "$GP_STYLE" --adapter-dir "$GP_CHAR" \
                              --alpha 0.5 1.3 --merge ;;
    *) echo "?? $1"; return 1 ;;
  esac
}

if [ "${1:-}" = all ]; then
  set -- base lo_s10 gp_s10 lo_c10 gp_c13 lo_s10c10 lo_s07c10 lo_s05c10 \
         gp_s10c13 gp_s07c13 gp_s05c13
fi
for t in "$@"; do one "$t" || exit 1; done
echo "==== STYLECHAR DONE  $(date +%H:%M:%S) ===="
