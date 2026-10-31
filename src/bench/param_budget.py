"""Report GPart/LoRA parameter budgets for the converted SDXL UNet on meta device."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import UNet2DConditionModel

ROOT = Path(__file__).resolve().parents[2]
UNET_DIR = ROOT / "data" / "noobai_v11_unet"

TARGET_SETS = {
    "attn": ["to_q", "to_k", "to_v", "to_out.0"],
    "attn+ff": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
    "attn+ff+proj": [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
        "proj_in",
        "proj_out",
    ],
}


def matches(name: str, suffixes: list[str]) -> bool:
    return any(name == s or name.endswith("." + s) for s in suffixes)


def main() -> None:
    config = UNet2DConditionModel.load_config(str(UNET_DIR))
    with init_empty_weights():
        unet = UNet2DConditionModel.from_config(config)

    linears = [(n, m) for n, m in unet.named_modules() if isinstance(m, torch.nn.Linear)]
    total_all = sum(p.numel() for p in unet.parameters())
    print(f"unet total params: {total_all/1e6:.1f}M")
    print(f"linear modules:    {len(linears)}")
    print(f"transformer blocks present: {sum('transformer_blocks' in n for n, _ in linears)}")
    print()

    for label, suffixes in TARGET_SETS.items():
        chosen = [(n, m) for n, m in linears if matches(n, suffixes)]
        n_params = sum(m.weight.numel() for _, m in chosen)
        lora16 = sum(16 * (m.in_features + m.out_features) for _, m in chosen)
        print(f"[{label}] modules={len(chosen)}  adapted_weights={n_params/1e6:.1f}M  "
              f"lora_r16_params={lora16/1e6:.2f}M")
        for d in (131_072, 524_288, 1_000_000, 5_000_000, lora16):
            print(f"    d={d:>9,}  avg group={n_params/d:8.1f}  "
                  f"subspace={100*d/n_params:6.2f}%  theta_fp32={4*d/2**20:7.1f} MiB")
        print()

    names = [n for n, _ in linears if "attn2" in n][:4]
    print("sample names:", json.dumps(names, indent=1))
    print("block ids:", json.dumps(sorted({
        ".".join(n.split(".")[:4]) for n, _ in linears if "transformer_blocks" in n
    })[:8], indent=1))


if __name__ == "__main__":
    main()
