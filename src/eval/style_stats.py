"""Compare generated image sets against a style reference set with cheap CPU stats.

No CLIP image tower is available offline, so this uses colour/texture statistics:
hue-agnostic palette spread, colourfulness, edge density and luminance contrast.
Lower distance-to-reference is better; the base model's numbers are the control.

Usage:
    python src/style_stats.py --ref D:/AI/lora/datasets_reserve1999_clean/6_reserve1999 \
        --dirs outputs/style_r1999_base outputs/style_r1999_gpart
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

KEYS = ("sat_mean", "sat_std", "val_mean", "val_std", "colorfulness", "edge_density")


def features(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(2), rgb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    val = mx

    # Hasler-Süsstrunk colourfulness
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(
        rg.mean() ** 2 + yb.mean() ** 2
    )

    gray = rgb.mean(2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()

    return np.array(
        [sat.mean(), sat.std(), val.mean(), val.std(), colorful, gx + gy],
        dtype=np.float64,
    )


def collect(paths: list[Path]) -> np.ndarray:
    feats = [features(p) for p in paths]
    return np.stack(feats)


def summarise(label: str, feats: np.ndarray) -> str:
    mean, std = feats.mean(0), feats.std(0)
    cells = "  ".join(f"{k}={m:.3f}±{s:.3f}" for k, m, s in zip(KEYS, mean, std))
    return f"{label:24s} n={len(feats):3d}  {cells}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="directory of style reference images")
    ap.add_argument("--dirs", nargs="+", required=True, help="directories of generated images")
    args = ap.parse_args()

    ref_paths = sorted(p for p in Path(args.ref).iterdir() if p.suffix.lower() in {".jpg", ".png"})
    ref = collect(ref_paths)
    ref_mean, ref_std = ref.mean(0), ref.std(0)
    ref_std[ref_std < 1e-6] = 1.0
    print(summarise("REFERENCE", ref))

    for d in args.dirs:
        paths = sorted(p for p in Path(d).iterdir() if p.suffix.lower() == ".png")
        if not paths:
            continue
        feats = collect(paths)
        z = np.abs((feats.mean(0) - ref_mean) / ref_std).mean()
        print(summarise(Path(d).name, feats) + f"  |z|={z:.2f}")


if __name__ == "__main__":
    main()
