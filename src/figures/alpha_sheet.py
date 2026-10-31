"""Contact sheet for the all-linear GPart alpha sweep (character, p0).

    python -I scripts/compose_alpha_sheet.py

Output: outputs/final_2026-10-30/横向_全参角色_alpha扫描.png

Rows are ordered by tag_eval score so the peak is visible in the middle.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "outputs" / "final_2026-10-30"
SEEDS = ["310977364113511", "12345", "67890"]

ROWS = [
    ("base", "无适配器（基准）", "base 0.08"),
    ("lo_char_matched", "LoRA 角色（配方对齐）", "0.70"),
    ("gp_char", "GPart attn2 · α1.0", "0.43"),
    ("gp_char_alllinear", "GPart 全参 · α1.0", "0.40"),
    ("gp_char_alllinear_a13", "GPart 全参 · α1.3", "0.55"),
    ("gp_char_alllinear_a15", "GPart 全参 · α1.5", "0.57"),
    ("gp_char_alllinear_a18", "GPart 全参 · α1.8", "0.40"),
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
    p0 = find("base", SEEDS[0])
    assert p0 is not None
    im0 = Image.open(p0)
    cell_h = round(cell_w * im0.height / im0.width)
    im0.close()

    # Split into chunks so each sheet stays readable when scaled into an A4 page.
    chunks = [ROWS[:4], ROWS[4:]]
    suffixes = ["", "_下半"]
    for rows, suffix in zip(chunks, suffixes):
        build(rows, cell_w, cell_h, suffix)


def build(rows, cell_w: int, cell_h: int, suffix: str) -> None:
    lab_w, head_h, pad = 290, 84, 40
    f_title, f_row, f_small = font(32), font(24), font(19)

    w = lab_w + len(SEEDS) * cell_w
    h = head_h + len(rows) * (cell_h + pad)
    sheet = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(sheet)

    d.text((12, 10), "全参 GPart 角色 · α 扫描（p0，行末为 tag score）", fill="black", font=f_title)
    for j, s in enumerate(SEEDS):
        d.text((lab_w + j * cell_w + 6, head_h - 24), f"seed {s}", fill="black", font=f_small)

    for r, (tag, label, score) in enumerate(rows):
        y = head_h + r * (cell_h + pad)
        d.text((10, y + 6), label, fill="black", font=f_row)
        d.text((10, y + 36), f"score {score}", fill="#555555", font=f_small)
        for j, seed in enumerate(SEEDS):
            src = find(tag, seed)
            if src is None:
                continue
            im = Image.open(src).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
            sheet.paste(im, (lab_w + j * cell_w, y))
            d.rectangle([lab_w + j * cell_w, y, lab_w + (j + 1) * cell_w - 1, y + cell_h - 1],
                        outline="#999999")
        d.line([(0, y + cell_h + pad - 10), (w, y + cell_h + pad - 10)], fill="#dddddd")

    out = B / f"横向_全参角色_alpha扫描{suffix}.png"
    sheet.save(out)
    print(f"saved {out.name}  {sheet.width}x{sheet.height}")


if __name__ == "__main__":
    main()
