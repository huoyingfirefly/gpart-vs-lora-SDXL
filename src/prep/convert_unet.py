"""Convert an original SDXL checkpoint into a diffusers UNet folder.

Only the UNet is extracted, so no HuggingFace tokenizer/config download is needed.
The converted tensors are written straight to safetensors (most are zero-copy
views into the source file), which keeps host RAM usage low.

Usage:
    D:\\AI\\lora\\venv\\Scripts\\python.exe D:\\Gpart\\src\\convert_ckpt.py \\
        --checkpoint D:\\AI\\models\\checkpoints\\NoobAI-XL-v1.1.safetensors \\
        --config D:\\Gpart\\data\\sd_xl_base.yaml \\
        --output D:\\Gpart\\data\\noobai_v11_unet
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml
from accelerate import init_empty_weights
from diffusers import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
    create_unet_diffusers_config,
)
from safetensors import safe_open
from safetensors.torch import save_file


def rss_gib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(args.config, "r", encoding="utf-8") as f:
        original_config = yaml.safe_load(f)

    unet_config = create_unet_diffusers_config(original_config, image_size=args.image_size)
    unet_config["upcast_attention"] = False
    print("unet_config:", json.dumps(unet_config, indent=1))
    print(f"[{time.time()-t0:6.1f}s] rss={rss_gib():.2f} GiB", flush=True)

    # Load only the UNet tensors; the checkpoint also contains VAE and text encoders.
    sd: dict[str, torch.Tensor] = {}
    with safe_open(args.checkpoint, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith("model.diffusion_model.")]
        print(f"unet tensors: {len(keys)}")
        for i, key in enumerate(keys):
            sd[key] = f.get_tensor(key)
            if (i + 1) % 400 == 0:
                print(f"  loaded {i+1}/{len(keys)} rss={rss_gib():.2f} GiB", flush=True)
    print(f"[{time.time()-t0:6.1f}s] loaded rss={rss_gib():.2f} GiB", flush=True)

    converted = convert_ldm_unet_checkpoint(
        sd, unet_config, path=args.checkpoint, extract_ema=False
    )
    del sd
    gc.collect()
    print(
        f"[{time.time()-t0:6.1f}s] converted {len(converted)} tensors "
        f"rss={rss_gib():.2f} GiB",
        flush=True,
    )

    # Validate the key set against a meta-device model (no allocation).
    with init_empty_weights():
        meta = UNet2DConditionModel(**unet_config)
    expected = set(dict(meta.named_parameters()).keys()) | set(dict(meta.named_buffers()).keys())
    missing = expected - set(converted)
    unexpected = set(converted) - expected
    if missing or unexpected:
        raise RuntimeError(
            f"key mismatch: missing={sorted(missing)[:10]} unexpected={sorted(unexpected)[:10]}"
        )
    meta.save_config(str(out))
    del meta
    gc.collect()

    tensors = {}
    for name, tensor in converted.items():
        tensor = tensor.to(dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensors[name] = tensor
    del converted
    gc.collect()
    print(f"[{time.time()-t0:6.1f}s] prepared rss={rss_gib():.2f} GiB", flush=True)

    save_file(tensors, str(out / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    info = {
        "source_checkpoint": args.checkpoint,
        "source_config": args.config,
        "image_size": args.image_size,
        "dtype": args.dtype,
        "num_tensors": len(tensors),
        "seconds": round(time.time() - t0, 1),
        "unet_config": unet_config,
    }
    (out / "conversion_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"saved to {out}  [{time.time()-t0:6.1f}s] rss={rss_gib():.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
