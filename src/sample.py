"""Sample the NoobAI UNet (base, GPart or LoRA adapter) with the converted VAE/TEs.

Example:
    python src/sample.py --adapter-dir D:/Gpart/outputs/smoke_gpart/adapter \
        --prompt "r1999style, 1girl, solo, short hair, outdoors" \
        --size 832 1216 --steps 25 --cfg 5 --seed 1 --output D:/Gpart/outputs/samples
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "GPart"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from peft import PeftModel  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer  # noqa: E402

CLIP_SKIP = 2  # sd-scripts caches TE1 hidden_states[11] and TE2 hidden_states[-2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--unet", default=str(ROOT / "data" / "noobai_v11_unet"))
    p.add_argument("--aux", default=str(ROOT / "data" / "noobai_v11_aux"))
    # repeatable: pass --adapter-dir twice to stack two GPart adapters. Stacking
    # requires --merge, otherwise the second PeftModel would wrap the first.
    p.add_argument("--adapter-dir", action="append", default=None)
    p.add_argument("--merge", action="store_true", help="merge adapter weights into the UNet")
    # GPart has no built-in strength knob; scaling theta is exactly one.
    # one value applies to every adapter, or give one per --adapter-dir
    p.add_argument("--alpha", nargs="+", type=float, default=[1.0],
                   help="scale GPart theta before merging")
    p.add_argument("--lora", action="append", default=None,
                   help="kohya-format SDXL LoRA safetensors to merge; repeatable")
    p.add_argument("--lora-scale", nargs="+", type=float, default=[1.0],
                   help="extra multiplier on the LoRA delta")
    p.add_argument("--prompt", action="append", default=None)
    p.add_argument("--prompts-file", default=None, help="JSONL with {\"prompt\": ..., \"size\": [w, h]?}")
    p.add_argument("--negative", default="low quality, worst quality, blurry, watermark, text")
    p.add_argument("--size", nargs=2, type=int, default=[832, 1216])
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=5.0)
    # euler_a washes this model out badly (see outputs/matrix_strip.png); euler is
    # the default so a plain run reproduces NoobAI's actual look
    p.add_argument("--sched", choices=["euler_a", "euler", "dpmpp"], default="euler")
    # SDXL's official scheduler config uses "leading"; "trailing" shifts every sigma
    p.add_argument("--spacing", choices=["leading", "trailing"], default="leading")
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--tag", default="sample")
    p.add_argument("--grid", action="store_true", help="also write a contact sheet")
    return p.parse_args()


def load_prompts(args) -> list[dict]:
    items: list[dict] = []
    if args.prompt:
        for text in args.prompt:
            items.append({"prompt": text, "size": list(args.size)})
    if args.prompts_file:
        for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                obj = {"prompt": obj}
            obj.setdefault("size", list(args.size))
            items.append(obj)
    if not items:
        raise ValueError("provide --prompt or --prompts-file")
    return items


class Encoder:
    def __init__(self, aux: Path, device: str) -> None:
        self.te1 = CLIPTextModel.from_pretrained(aux / "text_encoder", local_files_only=True).to(device)
        self.te2 = CLIPTextModelWithProjection.from_pretrained(
            aux / "text_encoder_2", local_files_only=True
        ).to(device)
        self.tok1 = CLIPTokenizer.from_pretrained(aux / "text_encoder", local_files_only=True)
        self.tok2 = CLIPTokenizer.from_pretrained(aux / "text_encoder_2", local_files_only=True)
        self.device = device

    @staticmethod
    def _chunk_ids(tok, text: str) -> list[list[int]]:
        """Split text into 75-token bodies, one 77-token window each.

        CLIPTokenizer(max_length=77, truncation=True) silently drops everything
        past 77 tokens. ComfyUI and A1111 do not: they encode consecutive
        77-token windows and concatenate the hidden states along the sequence
        axis, so a long prompt stays long. Truncating made this sampler
        disagree with ComfyUI exactly when the prompt is detailed enough to
        matter -- which is the case every character test cares about.

        A prompt that already fits in one window produces embeddings identical
        to the old truncating path, so short-prompt results are unaffected.
        """
        ids = tok(text, truncation=False, padding=False).input_ids
        body = ids[1:-1]  # BOS/EOS are re-added per window
        if not body:
            return [[tok.bos_token_id, tok.eos_token_id]]
        return [[tok.bos_token_id, *body[i:i + 75], tok.eos_token_id]
                for i in range(0, len(body), 75)]

    @staticmethod
    def _pad(tok, ids: list[int]) -> torch.Tensor:
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        return torch.tensor([ids + [pad] * (77 - len(ids))])

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        chunks1 = self._chunk_ids(self.tok1, text)
        chunks2 = self._chunk_ids(self.tok2, text)
        if len(chunks1) != len(chunks2):
            raise ValueError(f"tokenizers disagree on chunk count for {text!r}")
        hs1, hs2, proj = [], [], None
        with torch.no_grad():
            for c1, c2 in zip(chunks1, chunks2):
                ids1 = self._pad(self.tok1, c1).to(self.device)
                ids2 = self._pad(self.tok2, c2).to(self.device)
                o1 = self.te1(ids1, output_hidden_states=True, return_dict=True)
                o2 = self.te2(ids2, output_hidden_states=True, return_dict=True)
                hs1.append(o1.hidden_states[11])
                hs2.append(o2.hidden_states[-2])
                # sd-scripts pools at the EOS token itself, not at argmax(input_ids).
                # With several windows this is the EOS of the last one.
                eos_idx = int((ids2[0] == self.tok2.eos_token_id).nonzero()[0])
                proj = o2.last_hidden_state[0, eos_idx] @ self.te2.text_projection.weight.T
            h1 = torch.cat(hs1, dim=1)
            h2 = torch.cat(hs2, dim=1)
        return torch.cat([h1, h2], dim=-1).float(), proj.unsqueeze(0).float()

    def to_cpu(self) -> None:
        self.te1.to("cpu")
        self.te2.to("cpu")
        torch.cuda.empty_cache()


class CachedEmbeds:
    def __init__(self, data_dir: str | Path) -> None:
        self.by_caption: dict[str, Path] = {}
        for npz in sorted(Path(data_dir).glob("*_te_outputs.npz")):
            txt = npz.with_name(npz.name.replace("_te_outputs.npz", ".txt"))
            if txt.exists():
                self.by_caption[txt.read_text(encoding="utf-8").strip()] = npz
        print(f"cached captions: {len(self.by_caption)}")

    def get(self, text: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        path = self.by_caption.get(text.strip())
        if path is None:
            return None
        with np.load(path) as z:
            h1 = torch.from_numpy(z["hidden_state1"]).float().unsqueeze(0)
            h2 = torch.from_numpy(z["hidden_state2"]).float().unsqueeze(0)
            pooled = torch.from_numpy(z["pool2"]).float().unsqueeze(0)
        return torch.cat([h1, h2], dim=-1), pooled


def _kohya_sdxl_name(name: str) -> str | None:
    """Map a kohya/ComfyUI SDXL LoRA module name to a diffusers UNet module path."""
    if name.startswith("middle_block_1_"):
        prefix, suffix = "mid_block.attentions.0.", name[len("middle_block_1_"):]
    else:
        m = re.match(r"(input_blocks|output_blocks)_(\d+)_1_(.+)$", name)
        if not m:
            return None
        kind, idx, suffix = m.group(1), int(m.group(2)), m.group(3)
        if kind == "input_blocks":
            block = 1 if idx in (4, 5) else 2
            layer = idx - (4 if block == 1 else 7)
            prefix = f"down_blocks.{block}.attentions.{layer}."
        else:
            block = 0 if idx < 3 else 1
            layer = idx if block == 0 else idx - 3
            prefix = f"up_blocks.{block}.attentions.{layer}."

    m = re.match(r"transformer_blocks_(\d+)_(attn[12])_(to_[qkv]|to_out_0)$", suffix)
    if m:
        tb, attn, tail = m.groups()
        tail = {"to_out_0": "to_out.0"}.get(tail, tail)
        return f"{prefix}transformer_blocks.{tb}.{attn}.{tail}"
    m = re.match(r"transformer_blocks_(\d+)_(ff_net_0_proj|ff_net_2)$", suffix)
    if m:
        tb, tail = m.groups()
        tail = {"ff_net_0_proj": "ff.net.0.proj", "ff_net_2": "ff.net.2"}[tail]
        return f"{prefix}transformer_blocks.{tb}.{tail}"
    if suffix in ("proj_in", "proj_out"):
        return prefix + suffix
    return None


def merge_kohya_lora(unet: UNet2DConditionModel, path: str, scale: float = 1.0) -> None:
    """Fold a kohya-format SDXL LoRA (lora_unet_* keys) into the UNet weights.

    Hand-rolled because diffusers' kohya converter mixes SD1.5 block indexing
    into SDXL names (down_blocks.4 does not exist), and the vendored GPart peft
    fork rejects diffusers' adapter injection anyway.
    """
    from safetensors.torch import load_file

    state = load_file(path)
    merged, missing = 0, []
    for key in [k for k in state if k.endswith(".lora_down.weight")]:
        base = key[: -len(".lora_down.weight")]
        module_name = _kohya_sdxl_name(base.removeprefix("lora_unet_"))
        if module_name is None:
            missing.append(base)
            continue
        down = state[key].float()
        up = state[base + ".lora_up.weight"].float()
        module = unet.get_submodule(module_name)
        rank = down.shape[0]
        alpha = float(state[base + ".alpha"]) if base + ".alpha" in state else rank
        delta = up.reshape(up.shape[0], -1) @ down.reshape(rank, -1)
        delta = (delta * (alpha / rank) * scale).reshape(module.weight.shape)
        module.weight.data += delta.to(device=module.weight.device, dtype=module.weight.dtype)
        merged += 1
    print(f"merged {merged} LoRA module(s) from {Path(path).name}")
    if missing:
        raise RuntimeError(f"{len(missing)} unmapped LoRA keys, e.g. {missing[:3]}")


def _expand(vals: list[float], n: int, name: str) -> list[float]:
    """One value applies to every item; otherwise there must be one value per item."""
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"{name}: got {len(vals)} value(s) for {n} item(s)")
    return vals


def main() -> None:
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.adapter_dir and len(args.adapter_dir) > 1 and not args.merge:
        raise ValueError("stacking several GPart adapters requires --merge")

    prompts = load_prompts(args)
    unet = UNet2DConditionModel.from_pretrained(
        args.unet, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device
    )
    unet.eval()
    if args.lora:
        # deltas are additive in weight space, so merging in sequence is stacking
        for path, scale in zip(args.lora, _expand(args.lora_scale, len(args.lora), "--lora-scale")):
            merge_kohya_lora(unet, path, scale)
    if args.adapter_dir:
        for adapter_dir, alpha in zip(
            args.adapter_dir, _expand(args.alpha, len(args.adapter_dir), "--alpha")
        ):
            # GPart theta reloads onto CPU; without this every denoise step round-trips
            # the whole adapter over PCIe (~7x slower).
            unet = PeftModel.from_pretrained(unet, adapter_dir).to(device).eval()
            if alpha != 1.0:
                scaled = 0
                for name, param in unet.named_parameters():
                    if "gpart_theta_d" in name:
                        param.data.mul_(alpha)
                        scaled += 1
                print(f"scaled {scaled} gpart theta tensor(s) by {alpha}")
            if args.merge:
                # delta_w does not read the base weight, so merging one adapter and
                # then loading the next gives exactly W + dW1 + dW2
                unet = unet.merge_and_unload().eval()
                print(f"adapter {Path(adapter_dir).name} merged into UNet")

    vae = AutoencoderKL.from_pretrained(args.aux + "/vae", torch_dtype=torch.float32)
    vae.eval()

    cached = CachedEmbeds(args.cache_dir) if args.cache_dir else None
    enc = Encoder(Path(args.aux), device)

    def condition(text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if cached is not None:
            hit = cached.get(text)
            if hit is not None:
                return hit
        return enc.encode(text)

    neg_embeds, neg_pool = condition(args.negative)
    neg_embeds = neg_embeds.to(device, dtype)
    neg_pool = neg_pool.to(device, dtype)
    conds = []
    for item in prompts:
        embeds, pool = condition(item["prompt"])
        conds.append((embeds.to(device, dtype), pool.to(device, dtype)))
    enc.to_cpu()
    print(f"encoded {len(prompts)} prompts")

    sched_kwargs = dict(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", prediction_type="epsilon",
        timestep_spacing=args.spacing, steps_offset=1,
    )
    if args.sched == "euler_a":
        scheduler = EulerAncestralDiscreteScheduler(**sched_kwargs)
    elif args.sched == "euler":
        scheduler = EulerDiscreteScheduler(**sched_kwargs)
    else:
        scheduler = DPMSolverMultistepScheduler(
            **sched_kwargs, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )
    scheduler.set_timesteps(args.steps)

    results: list[tuple[str, torch.Tensor, dict]] = []
    for pi, (item, (pos_embeds, pos_pool)) in enumerate(zip(prompts, conds)):
        w, h = (int(v) for v in item["size"])
        time_ids = torch.tensor([[h, w, 0, 0, h, w]], device=device, dtype=dtype)
        pos_cond = {"text_embeds": pos_pool, "time_ids": time_ids}
        neg_cond = {"text_embeds": neg_pool, "time_ids": time_ids}
        for seed in args.seeds:
            # step_index lives on the scheduler, so reset it for every image
            scheduler.set_timesteps(args.steps)
            generator = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn((1, 4, h // 8, w // 8), generator=generator, device=device, dtype=dtype)
            latents = latents * scheduler.init_noise_sigma
            for t in scheduler.timesteps:
                inp = scheduler.scale_model_input(latents, t)
                with torch.no_grad():
                    noise_uncond = unet(inp, t, encoder_hidden_states=neg_embeds,
                                        added_cond_kwargs=neg_cond).sample
                    noise_text = unet(inp, t, encoder_hidden_states=pos_embeds,
                                      added_cond_kwargs=pos_cond).sample
                noise = noise_uncond + args.cfg * (noise_text - noise_uncond)
                latents = scheduler.step(noise, t, latents,
                                         generator=generator).prev_sample
            name = f"{args.tag}_{pi:02d}_s{seed}"
            results.append((name, latents.to("cpu"), {"prompt": item["prompt"], "w": w, "h": h, "seed": seed}))
            print(f"sampled {name} {w}x{h}")

    unet.to("cpu")
    del unet
    torch.cuda.empty_cache()
    vae.to(device)

    saved = []
    for name, latents, meta in results:
        with torch.no_grad():
            image = vae.decode(latents.to(device).float() / vae.config.scaling_factor).sample
        arr = (image / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        arr = (arr * 255).round().astype(np.uint8)
        path = out / f"{name}_{meta['w']}x{meta['h']}.png"
        Image.fromarray(arr).save(path)
        saved.append((path, arr, meta))
        print(f"saved {path}")

    if args.grid and saved:
        cell_w = min(arr.shape[1] for _, arr, _ in saved) // 2
        cell_h = min(arr.shape[0] for _, arr, _ in saved) // 2
        cols = min(4, len(saved))
        rows = (len(saved) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        for i, (_, arr, _) in enumerate(saved):
            img = Image.fromarray(arr).resize((cell_w, cell_h), Image.LANCZOS)
            sheet.paste(img, ((i % cols) * cell_w, (i // cols) * cell_h))
        sheet_path = out / f"{args.tag}_grid.png"
        sheet.save(sheet_path)
        print(f"saved {sheet_path}")


if __name__ == "__main__":
    main()
