#!/usr/bin/env bash
# Round 3 evaluation. Prints the three tables that decide REPORT.md 6.8's
# open question. Nothing here samples or trains.
#
#   part A  character fidelity: did GPart all-linear close the 0.43 -> 0.70 gap?
#   part B  style domain: color/texture of the matched style LoRA vs GPart
#   part C  character leakage: does the style adapter drag the character in?
#
# Usage: eval_round3_2026-11-01.sh [a|b|c|all]

set -u
cd "$(dirname "$0")/.."
ENV_PY="C:/Users/asus/anaconda3/envs/ai_env/python.exe"
D=outputs/final_2026-10-30
what="${1:-all}"

if [ "$what" = all ] || [ "$what" = a ]; then
  echo "########## A. 角色还原（配方对齐后，层覆盖这一项是否解释差距）"
  "$ENV_PY" scripts/tag_eval.py --md --pat '*_00_*' \
    --dirs base=$D/base \
           lo_char_matched=$D/lo_char_matched \
           gp_char=$D/gp_char \
           gp_char_alllinear=$D/gp_char_alllinear 2>&1 | tail -6
  echo
  echo "  参考：base 是地板；只看 score 的增幅，不要看绝对值。"
fi

if [ "$what" = all ] || [ "$what" = b ]; then
  echo "########## B. 画风（色彩/纹理统计，画风侧没有 tagger 指标）"
  "$ENV_PY" - <<'EOF'
import numpy as np
from pathlib import Path
from PIL import Image
B = Path("outputs/final_2026-10-30"); seeds = [310977364113511, 12345, 67890]
def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
def stats(a):
    mx, mn = a.max(2), a.min(2)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean(); v = a.max(2).mean()
    g = np.abs(np.diff(a.mean(2), axis=0)).mean() + np.abs(np.diff(a.mean(2), axis=1)).mean()
    return s, v, g
ref = {(pi, s): load(B / "base" / f"base_{pi:02d}_s{s}_832x1216.png") for pi in (0, 1) for s in seeds}
print(f"{'config':26s} {'dRGB':>6s} {'S':>6s} {'V':>6s} {'edge':>6s}")
for tag in ["base", "lo_style", "lo_style_matched", "gp_style_all", "gp_style_attn_f"]:
    rows = []
    for pi in (0, 1):
        for s in seeds:
            p = B / tag / f"{tag}_{pi:02d}_s{s}_832x1216.png"
            if not p.exists(): continue
            a = load(p); rows.append([np.abs(a - ref[(pi, s)]).mean()] + list(stats(a)))
    if rows:
        m = np.mean(rows, 0)
        print(f"{tag:26s} " + " ".join(f"{x:6.3f}" for x in m))
EOF
fi

if [ "$what" = all ] || [ "$what" = c ]; then
  echo "########## C. 画风适配器有没有把角色带进来（应当接近 base）"
  "$ENV_PY" scripts/tag_eval.py --md --pat '*_00_*' \
    --dirs base=$D/base \
           lo_style=$D/lo_style \
           lo_style_matched=$D/lo_style_matched \
           gp_style_attn_f=$D/gp_style_attn_f 2>&1 | tail -6
fi
