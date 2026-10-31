"""Build a training cache where one latent is paired with several text conditions.

The trigger alone must carry identity, so every image also appears with its
appearance tags stripped: "img_01v0" = full caption, "v1" = trigger + scene tags,
"v2" = trigger only.

Usage:
    python src/prep_caption_variants.py \
        --src D:/AI/lora/datasets_aima/10_aima \
        --dst D:/AI/lora/datasets_aima/10_aima_var
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "GPart"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import LATENT_RE  # noqa: E402
from sample import Encoder  # noqa: E402

TRIGGER = "aima"

# appearance / outfit / body tags: everything the model should have to get from TRIGGER
IDENTITY = {
    "white hair", "pink hair", "gradient hair", "short hair", "red eyes",
    "black beret", "hair flower", "x hair ornament", "black choker",
    "shirt", "white shirt", "collared shirt", "ribbon", "belt", "white flower",
    "dress", "bow", "sailor collar", "thigh strap", "butterfly", "hairclip",
    "shorts", "black ribbon", "bowtie", "pink flower", "jacket", "shoes",
    "hair ribbon", "sleeves past wrists", "white bow", "buckle", "jewelry",
    "puffy sleeves", "black footwear", "puffy long sleeves", "gloves", "frills",
    "white gloves", "medium hair", "open jacket", "white ribbon", "black dress",
    "hair between eyes", "thighs", "ahoge", "chain", "floral print",
    "black sailor collar", "school uniform", "animal ears", "dog ears",
    "dog tail", "tail", "skin fang", "fang", "skeleton",
}


def build_variants(caption: str, trigger: str = TRIGGER,
                   identity: set[str] = IDENTITY) -> list[str]:
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    rest = [
        t for t in tags
        if t.lower() not in identity and t.lower() not in {trigger, "1girl", "solo"}
    ]
    full = ", ".join(tags)
    scene = ", ".join([trigger, "1girl", "solo", *rest])
    bare = f"{trigger}, 1girl, solo"
    return [full, scene, bare]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="kohya dataset dir with cached *_sdxl.npz + *_te_outputs.npz")
    ap.add_argument("--dst", required=True,
                    help="output dir; cached latents are copied, not re-encoded")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aux", default=str(ROOT / "data" / "noobai_v11_aux"),
                    help="SDXL aux dir: text_encoder/, text_encoder_2/, vae/")
    ap.add_argument("--trigger", default=TRIGGER,
                    help="trigger token; default keeps the aima run reproducible")
    ap.add_argument("--identity", default=None,
                    help="comma-separated appearance tags to strip from the scene/bare "
                         "variants; default is the aima set")
    args = ap.parse_args()

    trigger = args.trigger
    identity = (set(IDENTITY) if args.identity is None
                else {t.strip().lower() for t in args.identity.split(",") if t.strip()})

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    aux = Path(args.aux)
    # without this check, from_pretrained() treats the missing path as a hub repo
    # id and raises "Repo id must use alphanumeric chars..."
    if not (aux / "text_encoder").is_dir():
        raise SystemExit(
            f"no text encoder at {aux / 'text_encoder'}. "
            f"Pass --aux, or see quickstart/02-models.md.")
    enc = Encoder(aux, args.device)
    written = 0
    for latent_path in sorted(src.glob("*_sdxl.npz")):
        match = LATENT_RE.match(latent_path.name)
        if match is None:
            continue
        sample_id, w, h = match.group("id"), match.group("w"), match.group("h")
        caption = (src / f"{sample_id}.txt").read_text(encoding="utf-8").strip()
        variants = build_variants(caption, trigger, identity)

        # keep the original scale/target/crop metadata as-is
        for k, text in enumerate(variants):
            vid = f"{sample_id}v{k}"
            link_or_copy(latent_path, dst / f"{vid}_{w}x{h}_sdxl.npz")
            (dst / f"{vid}.txt").write_text(text, encoding="utf-8")
            embeds, pooled = enc.encode(text)
            embeds, pooled = embeds[0].cpu(), pooled[0].cpu()
            np.savez(
                dst / f"{vid}_te_outputs.npz",
                hidden_state1=embeds[:, :768].numpy(),
                hidden_state2=embeds[:, 768:].numpy(),
                pool2=pooled.numpy(),
            )
            written += 1
            print(f"{vid}: {text[:90]}", flush=True)
    print(f"wrote {written} samples to {dst}", flush=True)


if __name__ == "__main__":
    main()
