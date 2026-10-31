"""Measure GPart forward/backward cost as a function of d on the real UNet."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "GPart"))

import torch  # noqa: E402
from diffusers import UNet2DConditionModel  # noqa: E402
from peft import get_peft_model  # noqa: E402
from peft.tuners.gpart import GPartConfig  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[1_000_000])
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    ds = args.d
    size = args.size
    x = torch.randn(1, 4, size // 8, size // 8, device="cuda", dtype=torch.bfloat16)
    t = torch.tensor([500], device="cuda")
    pe = torch.randn(1, 77, 2048, device="cuda", dtype=torch.bfloat16)
    pooled = torch.randn(1, 1280, device="cuda", dtype=torch.bfloat16)
    tids = torch.tensor([[size, size, 0, 0, size, size]], device="cuda", dtype=torch.bfloat16)

    for d in ds:
        base = UNet2DConditionModel.from_pretrained(
            str(ROOT / "data" / "noobai_v11_unet"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cuda",
        )
        base.enable_gradient_checkpointing()
        cfg = GPartConfig(
            d=d,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            assignment_backend="stateless",
            proj_seed=42,
            task_type=None,
        )
        model = get_peft_model(base, cfg)
        model.train()
        model = model.to("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # warmup
        out = model(x, t, encoder_hidden_states=pe,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": tids}).sample
        out.float().square().mean().backward()
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(2):
            out = model(x, t, encoder_hidden_states=pe,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": tids}).sample
        torch.cuda.synchronize()
        t_fwd = (time.time() - t0) / 2

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(2):
            out = model(x, t, encoder_hidden_states=pe,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": tids}).sample
            out.float().square().mean().backward()
        torch.cuda.synchronize()
        t_bwd = (time.time() - t0) / 2
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"d={d:>9,} fwd={t_fwd:6.2f}s fwd+bwd={t_bwd:6.2f}s peak={peak:5.2f}GiB", flush=True)

        del model, base, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
