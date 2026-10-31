"""Build the horizontal-comparison sheets for the final 2026-10-30 run.

Layout, all under outputs/final_2026-10-30/review/:
  横向_p0_clean.png   rows = 11 configs, cols = 3 seeds   (clean prompt)
  横向_p1_full.png    same, full-description prompt
  按配置/<NN>_<tag>.png  each config alone: 2 prompts x 3 seeds
  legend.txt

Every cell is labelled so the sheets can be read without cross-referencing.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2] / "outputs" / "final_2026-10-30"
OUT = BASE / "review"

# 04/08/10 use the *_f reruns: the first pass sampled step_00600 from a run that
# continued to step_01200, so those three rows were half-trained while every other
# row was a final checkpoint. See REPORT.md 4.2.
CONFIGS = [
    ("base", "无 adapter（基准）", "—"),
    ("lo_char", "LoRA 角色", "scale 1.0"),
    ("lo_style", "LoRA 画风 v3", "scale 1.0"),
    ("gp_style_all", "GPart 画风（全参）", "alpha 1.0"),
    ("gp_style_attn_f", "GPart 画风（只 attn2，终版）", "alpha 1.0"),
    ("gp_char", "GPart 角色", "alpha 1.0"),
    ("lo_stack", "LoRA 画风+角色", "1.0 / 1.0"),
    ("gp_stack_all", "GPart 全参 画风+角色", "alpha 1.0 / 1.0"),
    ("gp_stack_attn_f", "GPart 非全参 画风+角色", "alpha 1.0 / 1.0"),
    ("gp_stack_all_m", "GPart 全参 叠加（位移对齐）", "alpha 0.114 / 0.031"),
    ("gp_stack_attn_fm", "GPart 非全参 叠加（位移对齐）", "alpha 0.0285 / 0.031"),
]

PROMPTS = [(0, "clean", "干净 prompt"), (1, "full", "完整描述 prompt")]
SEEDS = [310977364113511, 12345, 67890]


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                 "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def frame(tag: str, pi: int, seed: int) -> Path | None:
    p = BASE / tag / f"{tag}_{pi:02d}_s{seed}_832x1216.png"
    return p if p.exists() else None


def sheet_for_prompt(pi: int, slug: str, title: str,
                     subset: list[int] | None = None, suffix: str = "") -> None:
    idxs = list(range(len(CONFIGS))) if subset is None else subset
    rows = [(i, *CONFIGS[i]) for i in idxs if any(frame(CONFIGS[i][0], pi, s) for s in SEEDS)]
    if not rows:
        print(f"p{pi}: nothing yet")
        return

    cell_w = 300
    sample = Image.open(frame(rows[0][1], pi, SEEDS[0]))
    cell_h = round(cell_w * sample.height / sample.width)
    f_cell, f_hdr, f_lab = font(20), font(26), font(19)
    lab_w = 300
    head_h = 70

    sheet = Image.new(
        "RGB",
        (lab_w + len(SEEDS) * cell_w, head_h + len(rows) * (cell_h + 46)),
        "white",
    )
    d = ImageDraw.Draw(sheet)
    d.text((12, 10), title, fill="black", font=f_hdr)
    for j, seed in enumerate(SEEDS):
        d.text((lab_w + j * cell_w + 8, head_h - 26), f"seed {seed}", fill="black", font=f_cell)

    for r, (idx, tag, desc, params) in enumerate(rows):
        y = head_h + r * (cell_h + 46)
        d.text((12, y + 8), f"{idx:02d}  {tag}", fill="black", font=f_cell)
        d.text((12, y + 34), desc, fill="dimgray", font=f_lab)
        d.text((12, y + 58), params, fill="gray", font=f_lab)
        for j, seed in enumerate(SEEDS):
            src = frame(tag, pi, seed)
            if src is None:
                continue
            im = Image.open(src).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
            sheet.paste(im, (lab_w + j * cell_w, y))
        d.line([(0, y + cell_h + 40), (sheet.width, y + cell_h + 40)], fill="#dddddd")

    path = OUT / f"横向_p{pi}_{slug}{suffix}.png"
    sheet.save(path)
    print(f"saved {path.name}  {sheet.width}x{sheet.height}  ({len(rows)} configs)")


def sheet_for_config(idx: int, tag: str, desc: str, params: str) -> None:
    cells = [(pi, s) for pi, _, _ in PROMPTS for s in SEEDS if frame(tag, pi, s)]
    if not cells:
        return
    cell_w = 340
    sample = Image.open(frame(tag, cells[0][0], cells[0][1]))
    cell_h = round(cell_w * sample.height / sample.width)
    f_cell, f_lab = font(21), font(19)
    lab_w = 190
    head_h = 68

    sheet = Image.new("RGB", (lab_w + len(SEEDS) * cell_w, head_h + len(PROMPTS) * (cell_h + 30)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 6), f"{idx:02d} {tag}  —  {desc}  ({params})", fill="black", font=f_cell)
    for j, seed in enumerate(SEEDS):
        d.text((lab_w + j * cell_w + 8, head_h - 26), f"seed {seed}", fill="black", font=f_lab)

    for r, (pi, slug, pname) in enumerate(PROMPTS):
        y = head_h + r * (cell_h + 30)
        d.text((10, y + cell_h // 2 - 10), f"p{pi}\n{pname}", fill="black", font=f_lab)
        for j, seed in enumerate(SEEDS):
            src = frame(tag, pi, seed)
            if src is not None:
                im = Image.open(src).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
                sheet.paste(im, (lab_w + j * cell_w, y))

    path = OUT / "按配置" / f"{idx:02d}_{tag}.png"
    sheet.save(path)
    print(f"saved 按配置/{path.name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "按配置").mkdir(exist_ok=True)
    for idx, (tag, desc, params) in enumerate(CONFIGS):
        sheet_for_config(idx, tag, desc, params)
    for pi, slug, title in PROMPTS:
        # full sheet for on-screen review
        sheet_for_prompt(pi, slug, title)
        # 3 rows per chunk: the full sheet is ~5400 px tall, which no A4 page can
        # hold without slicing an image in half
        n = len(CONFIGS)
        for k in range(0, n, 3):
            part = list(range(k, min(k + 3, n)))
            label = f"{title}  ({part[0] + 1:02d}–{part[-1] + 1:02d} / {n})"
            sheet_for_prompt(pi, slug, label, part, f"_part{k // 3 + 1}")

    lines = [
        "最终对比 2026-10-30  NoobAI-XL v1.1",
        "配方: euler_ancestral + trailing spacing + 30 steps + cfg 5 + 832x1216",
        "     (= ComfyUI 的 euler_ancestral + normal, 见 _samplercheck/compare3.png)",
        "seed: 310977364113511 / 12345 / 67890",
        "",
        "配置:",
    ]
    for i, (tag, desc, params) in enumerate(CONFIGS):
        lines.append(f"  {i:02d}  {tag:16s} {desc}   {params}")
    (OUT / "legend.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
