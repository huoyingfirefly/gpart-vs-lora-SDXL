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
    "stack": (84, 438, 478, 290),
    "style": (86, 497, 539, 300),
}

# (source, row index, source sheet, optional label lines) in draw order.
# A label is [(text, size, grey)] and overwrites the label column -- used where
# the source sheet's own label is clipped by its 290 px column.
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
    "showcase_stacking": {
        "title": "画风 + 角色叠加 · 基模 vs GPart（画风 attn2 + 角色 attn2）",
        "rows": [
            ("stack", 0, "horizontal_stacking_p0.png"),  # base
            ("stack", 3, "horizontal_stacking_p0.png",   # GPart attn2 pair
             [("GPart 叠加", 24, 0), ("画风 attn2 +", 19, 0),
              ("角色 attn2", 19, 0), ("char 0.40", 19, 90)]),
        ],
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
    rows = [crop_row(row[0], row[1], row[2]) for row in spec["rows"]]
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
    for spec_row, row in zip(spec["rows"], rows):
        sheet.paste(row, (0, y))
        if len(spec_row) > 3:          # redraw the label column
            draw.rectangle((0, y, lab_w, y + row_h), fill="white")
            ty = y + 34
            for text, size, grey in spec_row[3]:
                draw.text((14, ty), text, fill=(grey, grey, grey), font=font(size))
                ty += size + 12
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
