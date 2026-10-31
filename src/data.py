"""Dataset over kohya's cached SDXL latents and text-encoder outputs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

VAE_SCALE_FACTOR = 0.13025
LATENT_RE = re.compile(r"^(?P<id>.+)_(?P<w>\d+)x(?P<h>\d+)_sdxl\.npz$")


class CachedSDXLDataset(Dataset):
    def __init__(self, data_dir: str | Path, max_resolution: int | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.items: list[dict] = []
        for latent_path in sorted(self.data_dir.glob("*_sdxl.npz")):
            match = LATENT_RE.match(latent_path.name)
            if match is None:
                continue
            sample_id = match.group("id")
            te_path = self.data_dir / f"{sample_id}_te_outputs.npz"
            if not te_path.exists():
                continue
            with np.load(latent_path) as z:
                lat_key = next(k for k in z.files if k.startswith("latents_"))
                latents = z[lat_key]
                orig = z[next(k for k in z.files if k.startswith("original_size_"))]
                crop = z[next(k for k in z.files if k.startswith("crop_ltrb_"))]
            if max_resolution is not None and max(latents.shape[-2:]) * 8 > max_resolution:
                continue
            self.items.append(
                {
                    "id": sample_id,
                    "latent_path": latent_path,
                    "te_path": te_path,
                    "latent_key": lat_key,
                    "orig": orig,
                    "crop": crop,
                    "latent_shape": latents.shape,
                }
            )

        if not self.items:
            raise ValueError(f"no cached samples found in {self.data_dir}")

        tall = [it for it in self.items if it["latent_shape"][-2] >= it["latent_shape"][-1]]
        self.resolutions = sorted({(it["latent_shape"][-2] * 8, it["latent_shape"][-1] * 8) for it in self.items})
        print(
            f"dataset: {len(self.items)} samples, {len(self.resolutions)} resolutions, "
            f"max={max(max(r) for r in self.resolutions)}px, portrait-ish={len(tall)}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        with np.load(item["latent_path"]) as z:
            latents = torch.from_numpy(z[item["latent_key"]]).float() * VAE_SCALE_FACTOR
        with np.load(item["te_path"]) as z:
            hidden1 = torch.from_numpy(z["hidden_state1"]).float()
            hidden2 = torch.from_numpy(z["hidden_state2"]).float()
            pooled = torch.from_numpy(z["pool2"]).float()

        prompt_embeds = torch.cat([hidden1, hidden2], dim=-1)  # (77, 2048)
        h, w = latents.shape[-2:]
        target_h, target_w = h * 8, w * 8
        orig_w, orig_h = float(item["orig"][0]), float(item["orig"][1])
        crop_left, crop_top = float(item["crop"][0]), float(item["crop"][1])
        time_ids = torch.tensor(
            [orig_h, orig_w, crop_top, crop_left, target_h, target_w], dtype=torch.float32
        )

        caption_path = item["latent_path"].with_name(f"{item['id']}.txt")
        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""

        return {
            "id": item["id"],
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "pooled": pooled,
            "time_ids": time_ids,
            "caption": caption,
        }
