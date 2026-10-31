# `data/` — nothing here is committed

Model weights, latent caches and source images are all excluded by `.gitignore`.
This file documents what the code expects so you can rebuild it.

## 1. Base UNet

`src/train.py` loads a diffusers-format `UNet2DConditionModel` directory:

```
data/noobai_v11_unet/
├── config.json
└── diffusion_pytorch_model.safetensors
```

If your base is a single-file checkpoint, convert it:

```bash
python -m src.prep.convert_unet --checkpoint /path/to/base.safetensors \
    --config data/sd_xl_base.yaml --output data/noobai_v11_unet
```

`data/sd_xl_base.yaml` is the stock SDXL UNet config. Use whatever matches your
checkpoint's architecture.

## 2. Latent / text-embedding cache

The training loop reads kohya sd-scripts' SDXL cache format directly, so the
easiest path is to let kohya produce it. Drop your images into a kohya dataset
folder, then run any sd-scripts SDXL trainer for a single step with caching
enabled (or use `sdxl_train_util.cache_batch` in a small script of your own).
The result is, per image:

```
10_my_character/
├── 0001_0.png                     source image (not read at train time)
├── 0001.txt                       caption
├── 0001_sdxl.npz                  latents_*, original_size_*, crop_ltrb_*
└── 0001_te_outputs.npz            hidden_state1, hidden_state2, pool2
```

`src/data.py` globs `*_sdxl.npz`, pairs each with `<id>_te_outputs.npz` and
`<id>.txt`, and skips any pair whose resolution exceeds `--max-res`.

Scaling: the cached latents are multiplied by `0.13025` (`VAE_SCALE_FACTOR`) at
load time, so **store raw VAE means, not pre-scaled latents**.

## 3. Caption variants

The protocol trains on three caption variants per image (full / trigger + scene /
trigger only) so the trigger token has to carry identity:

```bash
python -m src.prep.caption_variants \
    --src data/10_my_character \
    --dst data/10_my_character_var \
    --trigger mychar \
    --identity "white hair,red eyes,beret"
```

This copies the cached `.npz` alongside the new caption files — it does not
re-encode images, so it takes seconds.

## 4. What you need to supply yourself

| | |
|---|---|
| base model | NoobAI-XL v1.1 or any SDXL-family UNet |
| WD14 tagger | `SmilingWolf/wd-v1-4-moat-tagger-v2`, ONNX; pass `--model` |
| source images | **not** redistributed here — see `docs/06-release-checklist.md` |
