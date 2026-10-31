"""Contact sheet for the style+character stacking comparison (p0).

    python -I scripts/compose_stack_sheet.py

Output: outputs/final_2026-10-30/横向_叠加_p0.png

Rows: base / kohya pair / matched-recipe pairs. Character score is the p0
tag_eval mean (n=3 seeds).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "outputs" / "final_2026-10-30"
SEEDS = ["310977364113511", "12345", "67890"]

ROWS = [
    ("base", "无适配器（基准）", "char 0.08"),
    ("lo_stack", "LoRA 画风+角色（kohya 那对）", "char 0.00"),
    ("lo_stack_matched", "LoRA 画风+角色（配方对齐）", "char 0.57"),
    ("gp_stack_attn_f", "GPart attn2 画风 + attn2 角色", "char 0.40"),
    ("gp_stack_alllinear_char", "GPart attn2 画风 + 全参角色", "char 0.27"),
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def find(tag: str, seed: str) -> Path | None:
    p = B / tag / f"{tag}_00_s{seed}_832x1216.png"
    return p if p.exists() else None


def main() -> None:
    cell_w = 300
    base = find("base", SEEDS[0])
    assert base is not None
    im0 = Image.open(base)
    cell_h = round(cell_w * im0.height / im0.width)
    im0.close()

    lab_w, head_h, pad = 290, 84, 40
    f_title, f_row, f_small = font(32), font(23), font(19)

    w = lab_w + len(SEEDS) * cell_w
    h = head_h + len(ROWS) * (cell_h + pad)
    sheet = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(sheet)

    d.text((12, 10), "画风 + 角色 叠加 · p0 干净提示词（行末为角色 tag score）",
           fill="black", font=f_title)
    for j, s in enumerate(SEEDS):
        d.text((lab_w + j * cell_w + 6, head_h - 24), f"seed {s}", fill="black", font=f_small)

    for r, (tag, label, score) in enumerate(ROWS):
        y = head_h + r * (cell_h + pad)
        d.text((10, y + 6), label, fill="black", font=f_row)
        d.text((10, y + 36), score, fill="#555555", font=f_small)
        for j, seed in enumerate(SEEDS):
            src = find(tag, seed)
            if src is None:
                continue
            im = Image.open(src).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
            sheet.paste(im, (lab_w + j * cell_w, y))
            d.rectangle([lab_w + j * cell_w, y, lab_w + (j + 1) * cell_w - 1, y + cell_h - 1],
                        outline="#999999")
        d.line([(0, y + cell_h + pad - 10), (w, y + cell_h + pad - 10)], fill="#dddddd")

    out = B / "横向_叠加_p0.png"
    sheet.save(out)
    print(f"saved {out.name}  {sheet.width}x{sheet.height}")


if __name__ == "__main__":
    main()
