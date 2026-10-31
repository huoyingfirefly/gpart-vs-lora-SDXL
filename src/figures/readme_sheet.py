"""Build the two showcase sheets linked from the README.

    python -I src/figures/readme_sheet.py

Each sheet is base model vs GPart at its best setting, three seeds, no LoRA
rows -- the README uses these to show what the adapter produces, not to carry
the comparison (the tables do that).

The individual sample images live in `outputs/`, which is not redistributed,
so the rows are cropped back out of the committed contact sheets in
`results/figures/`. Sources must keep their current row order; the row indices
below are what pins that down.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "figures"
OUT = SRC / "readme"

SEEDS = ["310977364113511", "12345", "67890"]

# source sheet -> (first row y, row height, row pitch, label column width)
GEOMETRY = {
    "alpha": (84, 438, 478, 290),
    "style": (86, 497, 539, 300),
}

# (source, row index) pairs, in draw order
SHEETS = {
    "showcase_character": {
        "title": "角色 · 基模 vs GPart（全参 α1.5，tag score 0.57 vs base 0.08）",
        "rows": [("alpha", 0, "horizontal_alpha_sweep_top.png"),   # base
                 ("alpha", 1, "horizontal_alpha_sweep_bottom.png")],  # GPart α1.5
    },
    "showcase_style": {
        "title": "画风 · 基模 vs GPart（attn2，终版）",
        "rows": [("style", 0, "horizontal_full_prompt_style.png"),  # base
                 ("style", 3, "horizontal_full_prompt_style.png")],  # GPart attn2
    },
}

WEB_WIDTH = 1000


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def crop_row(name: str, index: int, source: str) -> Image.Image:
    y0, row_h, pitch, _ = GEOMETRY[name]
    sheet = Image.open(SRC / source).convert("RGB")
    top = y0 + index * pitch
    return sheet.crop((0, top, sheet.width, top + row_h))


def build(key: str, spec: dict) -> None:
    rows = [crop_row(name, idx, src) for name, idx, src in spec["rows"]]
    width = rows[0].width
    _, row_h, pitch, lab_w = GEOMETRY[spec["rows"][0][0]]
    assert all(r.width == width for r in rows), "rows must share a width"

    header = pitch - row_h + 40     # same gap the source sheets use, plus title room
    height = header + row_h * len(rows) + (pitch - row_h) * (len(rows) - 1) + 40

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 14), spec["title"], fill=(0, 0, 0), font=font(32))

    cell_w = (width - lab_w) / len(SEEDS)
    for i, seed in enumerate(SEEDS):
        f = font(19)
        tw = draw.textlength(seed, font=f)
        x = lab_w + cell_w * i + (cell_w - tw) / 2
        draw.text((x, header - 26), f"seed {seed}", fill=(90, 90, 90), font=f)

    y = header
    for row in rows:
        sheet.paste(row, (0, y))
        y += row_h + (pitch - row_h)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{key}.jpg"
    sheet.resize((WEB_WIDTH, round(height * WEB_WIDTH / width)), Image.LANCZOS).save(
        out, quality=92, optimize=True, subsampling=0
    )
    print(f"{key}: {width}x{height} -> {WEB_WIDTH}px, {out.stat().st_size/1024:.0f} KB")


def main() -> None:
    for key, spec in SHEETS.items():
        build(key, spec)


if __name__ == "__main__":
    main()
