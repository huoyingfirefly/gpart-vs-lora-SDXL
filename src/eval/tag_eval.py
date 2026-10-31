"""Objective evaluation of generated images with a local WD14 (booru) tagger.

Turns "looks like the character" into a number: for each adapter configuration
we measure how often the tagger actually finds the target character's tags
(hair colour, beret, X hair ornament, ...) across seeds.

  # what does the tagger see in these images?
  python scripts/tag_eval.py --probe outputs/final_2026-10-30/base/*.png

  # hit rate of the aima design tags, per configuration
  python scripts/tag_eval.py --dirs base=outputs/final_2026-10-30/base \
                                    gp_char=outputs/final_2026-10-30/gp_char

Runs on CPU; needs onnxruntime (present in the `ai_env` conda env, not in the
sampling venv). Model input is NHWC float32, 0-255, BGR, 448x448.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# Override with --model, or WD14_MODEL=/path/to/model.onnx.
# selected_tags.csv must sit next to the ONNX file.
DEFAULT_MODEL = Path(os.environ.get(
    "WD14_MODEL",
    "models/wd14/SmilingWolf_wd-v1-4-moat-tagger-v2/model.onnx",
))
SIZE = 448

# the aima design, as booru tags (underscores are how the tagger spells them)
CHAR_TAGS = ["white_hair", "pink_hair", "gradient_hair", "red_eyes", "beret",
             "hair_flower", "hair_ornament", "choker", "black_dress", "black_ribbon"]


def load_names(tags_csv: Path) -> tuple[list[str], np.ndarray]:
    names, cats = [], []
    with open(tags_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            cats.append(int(row["category"]))
    return names, np.array(cats)


def prepare(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", im.size, (255, 255, 255, 255))
    canvas.alpha_composite(im)
    im = canvas.convert("RGB")
    side = max(im.size)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    square = square.resize((SIZE, SIZE), Image.BICUBIC)
    arr = np.asarray(square)[:, :, ::-1]  # RGB -> BGR
    return arr.astype(np.float32)[None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="+", help="image files: print top tags")
    ap.add_argument("--dirs", nargs="+", help="tag=directory pairs")
    ap.add_argument("--tags", default=",".join(CHAR_TAGS))
    ap.add_argument("--pat", default="*.png", help="glob within each dir, e.g. '*_00_*'")
    ap.add_argument("--md", action="store_true", help="emit a markdown table for the report")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="WD14 tagger ONNX; selected_tags.csv must be beside it")
    args = ap.parse_args()

    model = Path(args.model)
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    names, cats = load_names(model.parent / "selected_tags.csv")
    general = cats == 0

    def predict(path: Path) -> np.ndarray:
        return sess.run(None, {sess.get_inputs()[0].name: prepare(path)})[0][0]

    if args.probe:
        for p in args.probe:
            probs = predict(Path(p))
            order = np.argsort(-probs)[: args.top]
            seen = ", ".join(f"{names[i]}:{probs[i]:.2f}" for i in order)
            print(f"{Path(p).name}\n  {seen}\n")
        return

    wanted = [t.strip() for t in args.tags.split(",") if t.strip()]
    idx = {t: names.index(t) for t in wanted if t in names}
    missing = [t for t in wanted if t not in idx]
    if missing:
        print(f"!! tagger has no tag(s): {missing}")

    cols = list(idx)
    if args.md:
        print("| config | n | " + " | ".join(cols) + " | score |")
        print("|---" * (len(cols) + 2) + "|")
    else:
        print(f"{'config':22s} {'n':>3s} " + " ".join(f"{t[:12]:>12s}" for t in cols))

    for spec in args.dirs:
        tag, _, d = spec.partition("=")
        paths = sorted(Path(d).glob(args.pat))
        if not paths:
            print(f"{tag:22s}   0 (no images in {d})")
            continue
        probs = np.stack([predict(p) for p in paths])
        sub = probs[:, list(idx.values())]
        hits = (sub >= args.thr).mean(0)
        mean = sub.mean(0)
        score = hits.mean()
        if args.md:
            print(f"| {tag} | {len(paths)} | "
                  + " | ".join(f"{h:.2f} / {m:.2f}" for h, m in zip(hits, mean))
                  + f" | **{score:.2f}** |")
        else:
            print(f"{tag:22s} {len(paths):3d} " + " ".join(f"{h:5.2f}/{m:4.2f}" for h, m in zip(hits, mean))
                  + f"   score={score:.2f}")


if __name__ == "__main__":
    main()
