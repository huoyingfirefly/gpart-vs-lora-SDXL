"""Verify a saved adapter against the real NoobAI UNet: isometry, delta, merge.

Usage:
    python src/check_adapter.py --adapter D:/Gpart/outputs/smoke_gpart/adapter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "GPart"))

import torch  # noqa: E402
from diffusers import UNet2DConditionModel  # noqa: E402
from peft import PeftModel  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unet", default=str(ROOT / "data" / "noobai_v11_unet"))
    p.add_argument("--adapter", required=True)
    p.add_argument("--size", type=int, default=512)
    args = p.parse_args()

    torch.manual_seed(0)
    unet = UNet2DConditionModel.from_pretrained(
        args.unet, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda"
    )
    model = PeftModel.from_pretrained(unet, args.adapter).to("cuda").eval()

    theta = model.base_model.gpart_theta_d.default.detach().float()
    print(f"theta: n={theta.numel():,} norm={theta.norm():.4f}")

    deltas = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "get_delta_weight"):
                try:
                    deltas[name] = module.get_delta_weight("default")
                except Exception as exc:  # adapters with extra args (e.g. dropout)
                    print(f"  skip {name}: {exc}")
    print(f"adapted modules with delta: {len(deltas)}")
    nz = sum(1 for d in deltas.values() if d.float().abs().max() > 0)
    print(f"non-zero deltas: {nz}")
    norms = {n: float(d.float().norm()) for n, d in deltas.items()}
    sample = list(norms.items())[:5]
    for n, v in sample:
        print(f"  {n}: |dW|={v:.5f}")

    lat = args.size // 8
    torch.manual_seed(1)
    x = torch.randn(1, 4, lat, lat, device="cuda", dtype=torch.bfloat16)
    t = torch.tensor([500], device="cuda")
    pe = torch.randn(1, 77, 2048, device="cuda", dtype=torch.bfloat16)
    pooled = torch.randn(1, 1280, device="cuda", dtype=torch.bfloat16)
    tids = torch.tensor([[args.size, args.size, 0, 0, args.size, args.size]], device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        before = model(x, t, encoder_hidden_states=pe,
                       added_cond_kwargs={"text_embeds": pooled, "time_ids": tids}).sample.clone()

    merged = model.merge_and_unload()
    with torch.no_grad():
        after = merged(x, t, encoder_hidden_states=pe,
                       added_cond_kwargs={"text_embeds": pooled, "time_ids": tids}).sample
    diff = (after.float() - before.float()).abs()
    print(f"output |before-after| max={diff.max():.3e} mean={diff.mean():.3e}")
    print(f"base weight changed modules: "
          f"{sum(1 for n, m in merged.named_modules() if isinstance(m, torch.nn.Linear) and 'to_q' in n)} linear to_q modules")


if __name__ == "__main__":
    main()
