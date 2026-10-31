"""Time each phase of a real training micro-batch to locate slowdowns."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "GPart"))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from diffusers import DDPMScheduler, UNet2DConditionModel  # noqa: E402
from peft import get_peft_model  # noqa: E402
from peft.tuners.gpart import GPartConfig  # noqa: E402

from data import CachedSDXLDataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="D:/AI/lora/datasets_aima/10_aima")
    p.add_argument("--d", type=int, default=23_224_320)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--optimizer-step", action="store_true")
    p.add_argument("--config", choices=["train", "simple"], default="train",
                   help="'simple' matches bench_d.py's default GPartConfig")
    p.add_argument("--dummy", action="store_true",
                   help="feed random tensors instead of dataset samples")
    p.add_argument("--dummy-size", type=int, default=None,
                   help="square latent size for --dummy (e.g. 152 -> 152x152)")
    p.add_argument("--bench-mode", action="store_true",
                   help="replicate bench_d.py setup: model.to(cuda), empty_cache, warmup+zero_grad")
    p.add_argument("--to-cuda", action="store_true", help="call model.to('cuda') after wrapping")
    p.add_argument("--empty-cache", action="store_true")
    p.add_argument("--warmup", action="store_true")
    args = p.parse_args()

    ds = CachedSDXLDataset(args.data, max_resolution=1536)
    unet = UNet2DConditionModel.from_pretrained(
        str(ROOT / "data" / "noobai_v11_unet"), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="cuda")
    unet.enable_gradient_checkpointing()
    if args.config == "simple":
        cfg = GPartConfig(d=args.d, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                          assignment_backend="stateless", proj_seed=42, task_type=None)
    else:
        cfg = GPartConfig(
            d=args.d,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            gpart_dropout=0.0,
            bias="none",
            proj_seed=42,
            projection_type="partition",
            partition_scope="global",
            grouping_strategy="random",
            assignment_backend="stateless",
            isometric=True,
            layers_pattern=None,
            task_type=None,
        )
    model = get_peft_model(unet, cfg)
    model.train()
    if args.bench_mode or args.to_cuda:
        model = model.to("cuda")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"theta dtype={trainable[0].dtype} device={trainable[0].device} "
          f"base dtype={next(model.parameters()).dtype}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    sched = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", steps_offset=1,
                          prediction_type="epsilon", clip_sample=False)
    alphas = sched.alphas_cumprod.cuda()

    if args.bench_mode or args.empty_cache:
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args.bench_mode or args.warmup:
        # bench_d.py style warmup: one fwd+bwd at 1216px, then drop the grads
        x0 = torch.randn(1, 4, 152, 152, device="cuda", dtype=torch.bfloat16)
        pe0 = torch.randn(1, 77, 2048, device="cuda", dtype=torch.bfloat16)
        pool0 = torch.randn(1, 1280, device="cuda", dtype=torch.bfloat16)
        tids0 = torch.tensor([[1216, 1216, 0, 0, 1216, 1216]], device="cuda", dtype=torch.bfloat16)
        out0 = model(x0, torch.tensor([500], device="cuda"), encoder_hidden_states=pe0,
                     added_cond_kwargs={"text_embeds": pool0, "time_ids": tids0}).sample
        out0.float().square().mean().backward()
        model.zero_grad(set_to_none=True)
        del x0, pe0, pool0, tids0, out0
        torch.cuda.synchronize()

    for i in range(args.iters):
        t0 = time.time()
        if args.dummy:
            s = args.dummy_size
            lat = (torch.randn(1, 4, s, s, device="cuda") if s
                   else torch.randn(1, 4, 112, 144, device="cuda"))
            pe = torch.randn(1, 77, 2048, device="cuda", dtype=torch.bfloat16)
            pool = torch.randn(1, 1280, device="cuda", dtype=torch.bfloat16)
            tids = torch.tensor([[896, 1152, 0, 0, 896, 1152]], device="cuda", dtype=torch.bfloat16)
        else:
            idx = i % len(ds)
            item = ds[idx]
            lat = item["latents"].unsqueeze(0).cuda().float()
            pe = item["prompt_embeds"].unsqueeze(0).cuda().bfloat16()
            pool = item["pooled"].unsqueeze(0).cuda().bfloat16()
            tids = item["time_ids"].unsqueeze(0).cuda().bfloat16()
        torch.cuda.synchronize()
        t1 = time.time()

        noise = torch.randn_like(lat)
        t = torch.randint(0, 1000, (1,), device="cuda")
        noisy = sched.add_noise(lat, noise, t).bfloat16()
        pred = model(noisy, t, encoder_hidden_states=pe,
                     added_cond_kwargs={"text_embeds": pool, "time_ids": tids}).sample
        if i == 0:
            print(f"noisy dtype={noisy.dtype} pred dtype={pred.dtype}", flush=True)
        torch.cuda.synchronize()
        t2 = time.time()

        loss = (pred.float() - noise).pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.time()

        if args.optimizer_step:
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        t4 = time.time()
        shape = tuple(lat.shape[-2:])
        print(f"iter {i} latent={shape} data={t1-t0:5.2f}s fwd={t2-t1:5.2f}s "
              f"bwd={t3-t2:5.2f}s opt={t4-t3:5.2f}s total={t4-t0:5.2f}s "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
              f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
