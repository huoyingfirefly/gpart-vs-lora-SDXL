"""Minimal SDXL fine-tuning loop for GPart vs LoRA on kohya-cached data.

Example:
    D:\\AI\\lora\\venv\\Scripts\\python.exe D:\\Gpart\\src\\train_sdxl.py \\
        --adapter gpart --d 1000000 --steps 50 --max-res 768 \\
        --output D:\\Gpart\\outputs\\smoke_gpart
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "GPart"))

import torch  # noqa: E402
from diffusers import DDPMScheduler, UNet2DConditionModel  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from peft.tuners.gpart import GPartConfig  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from transformers import get_cosine_schedule_with_warmup  # noqa: E402

try:  # works both as `python -m src.train` and as `python src/train.py`
    from .data import CachedSDXLDataset  # noqa: E402
except ImportError:
    from data import CachedSDXLDataset  # noqa: E402


# top-level YAML keys that map straight onto argparse dests
_CFG_TOP = {
    "unet", "data", "adapter", "d", "rank", "alpha", "gpart_alpha", "targets",
    "scope", "layers_pattern", "backend", "partition_mode", "output",
}


def load_config(path: Path, parser: argparse.ArgumentParser) -> dict:
    """Read a configs/*.yaml into a dict of argparse defaults.

    Only the top-level keys in _CFG_TOP and the `train:` block are read.  The
    `sample:` block records how the row was scored (alpha / seeds / sampler) and
    belongs to src/sample.py, not to training.
    """
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = {k: v for k, v in cfg.items() if k in _CFG_TOP}
    merged.update(cfg.get("train") or {})
    dests = {action.dest for action in parser._actions}
    unknown = sorted(set(merged) - dests)
    if unknown:
        raise SystemExit(f"{path}: unknown config key(s): {', '.join(unknown)}")
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="configs/*.yaml; CLI flags override the file")
    p.add_argument("--unet", default=str(ROOT / "data" / "noobai_v11_unet"))
    p.add_argument("--data", default=None,
                   help="kohya-style dataset dir (images + .npz latents + captions)")
    p.add_argument("--adapter", choices=["gpart", "lora"], default="gpart")
    p.add_argument("--d", type=int, default=1_000_000)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--gpart-alpha", type=float, default=1.0,
                   help="global scale on GPart theta (Delta W = alpha * P theta); "
                        "1.0 reproduces the paper's parameterization")
    p.add_argument(
        "--targets",
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
    )
    p.add_argument("--scope", choices=["global", "transformer_block"], default="global")
    p.add_argument("--layers-pattern", default=None)
    p.add_argument("--backend", choices=["stateless", "materialized"], default="stateless")
    p.add_argument("--partition-mode", choices=["splitmix64", "int32"], default="splitmix64",
                   help="group-id mixer for the implicit partition. int32 is ~4x faster to "
                        "compute but changes theta's basis, so checkpoints are NOT "
                        "interchangeable between modes. Only valid with --backend stateless")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=20)
    # cosine-to-zero makes long runs plateau by construction; constant keeps LR alive
    p.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--max-res", type=int, default=1536)
    p.add_argument("--noise-offset", type=float, default=0.05)
    p.add_argument("--min-snr-gamma", type=float, default=5.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--theta-norm-max", type=float, default=0.0,
                   help="hard cap on the GPart update norm ||dW|| = gpart_alpha*||theta||; "
                        "theta is rescaled back onto that sphere after every step so that "
                        "'how long' and 'how far from the base model' are separate knobs. "
                        "0 disables the cap (the paper's unconstrained setting)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout", type=int, default=8)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="stop gracefully and checkpoint after this many seconds")
    p.add_argument("--output", default=None)

    # two-phase parse so that --config only supplies defaults: an explicit CLI
    # flag always beats the file.
    pre, _ = p.parse_known_args()
    if pre.config:
        p.set_defaults(**load_config(Path(pre.config), p))
    args = p.parse_args()

    for required in ("data", "output"):
        if not getattr(args, required):
            p.error(f"--{required} is required (on the command line or in --config)")
    if args.backend == "materialized" and args.partition_mode != "splitmix64":
        p.error("--partition-mode only applies to the stateless backend")
    return args


def build_adapter(model: UNet2DConditionModel, args: argparse.Namespace):
    if args.adapter == "gpart":
        if args.scope == "transformer_block" and not args.layers_pattern:
            raise ValueError("--layers-pattern is required with --scope transformer_block")
        cfg = GPartConfig(
            d=args.d,
            target_modules=list(args.targets),
            gpart_dropout=0.0,
            bias="none",
            proj_seed=args.seed,
            projection_type="partition",
            partition_scope=args.scope,
            grouping_strategy="random",
            assignment_backend=args.backend,
            partition_mode=args.partition_mode,
            isometric=True,
            alpha=args.gpart_alpha,
            layers_pattern=args.layers_pattern,
            task_type=None,
        )
    else:
        cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=list(args.targets),
            lora_dropout=0.0,
            bias="none",
            task_type=None,
        )
    return get_peft_model(model, cfg)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = "cuda"
    dataset = CachedSDXLDataset(args.data, max_resolution=args.max_res)
    n = len(dataset)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_idx = perm[: args.holdout]
    train_idx = perm[args.holdout :]
    train_ds = Subset(dataset, train_idx)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    print(f"split: train={len(train_idx)} val={len(val_idx)}")

    unet = UNet2DConditionModel.from_pretrained(
        args.unet,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    unet.enable_gradient_checkpointing()
    ckpt_dir = out / "checkpoint"
    resuming = args.resume and (ckpt_dir / "state.pt").exists()
    if resuming:
        model = PeftModel.from_pretrained(unet, ckpt_dir / "adapter", is_trainable=True)
        print(f"resumed adapter from {ckpt_dir / 'adapter'}")
    else:
        model = build_adapter(unet, args)
    # peft builds the GPart theta parameters on CPU; leaving them there makes every
    # step bounce ~d floats across PCIe twice (measured 6.7x slower at d=23.2M).
    model = model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    # GPart keeps its whole adapter in one theta vector, so ||theta|| is exactly ||dW||
    # (times alpha); the cap below needs a handle on that single tensor
    theta_param = next(
        (p for n, p in model.named_parameters() if "gpart_theta_d" in n), None
    )
    theta_cap = (
        args.theta_norm_max / max(args.gpart_alpha, 1e-8)
        if args.theta_norm_max > 0 and theta_param is not None
        else 0.0
    )
    if theta_cap:
        print(f"theta norm capped at {theta_cap:.3f} (||dW|| <= {args.theta_norm_max})",
              flush=True)
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable device: {trainable[0].device}")
    n_total = sum(p.numel() for p in model.parameters())
    print(f"adapter={args.adapter} trainable={n_train/1e6:.2f}M ({100*n_train/n_total:.3f}%)")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    total_steps = max(args.steps, 1)
    if args.lr_schedule == "constant":
        def lr_lambda(step: int) -> float:
            return min((step + 1) / max(args.warmup, 1), 1.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup, num_training_steps=total_steps
        )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        prediction_type="epsilon",
        clip_sample=False,
    )
    alphas = noise_scheduler.alphas_cumprod.to(device, dtype=torch.float32)

    start_step = 0
    if resuming:
        state = torch.load(ckpt_dir / "state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        print(f"resumed at step {start_step}")

    val_items = []
    for idx in val_idx[: args.val_batches]:
        item = dataset[idx]
        g = torch.Generator().manual_seed(10_000 + idx)
        latents = item["latents"].unsqueeze(0)
        noise = torch.randn(latents.shape, generator=g)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), generator=g)
        val_items.append(
            (
                latents.to(device, dtype=torch.float32),
                item["prompt_embeds"].unsqueeze(0).to(device, dtype=torch.bfloat16),
                item["pooled"].unsqueeze(0).to(device, dtype=torch.bfloat16),
                item["time_ids"].unsqueeze(0).to(device, dtype=torch.bfloat16),
                noise.to(device),
                t.to(device),
            )
        )

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total = 0.0
        for latents, prompt, pooled, tids, noise, t in val_items:
            noisy = noise_scheduler.add_noise(latents, noise, t).to(torch.bfloat16)
            pred = model(
                noisy,
                t,
                encoder_hidden_states=prompt,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": tids},
            ).sample
            mse = (pred.float() - noise).pow(2).mean(dim=(1, 2, 3))
            alpha = alphas[t]
            snr = alpha / (1.0 - alpha)
            weight = torch.minimum(snr, torch.full_like(snr, args.min_snr_gamma)) / snr
            total += float((mse * weight).mean())
        model.train()
        return total / len(val_items)

    log_path = out / "train.jsonl"
    if val_items:
        initial_val = validate()
        print(f"initial val loss {initial_val:.6f}", flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": start_step, "val": initial_val}) + "\n")
    started = time.time()
    micro = 0
    step = start_step
    running_loss = 0.0
    running_n = 0

    def save_checkpoint(current_step: int) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir / "adapter")
        torch.save(
            {
                "step": current_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt_dir / "state.pt",
        )
        print(f"saved checkpoint at step {current_step}", flush=True)

    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    for batch in infinite(loader):
        latents = batch["latents"].to(device, dtype=torch.float32)
        prompt = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
        pooled = batch["pooled"].to(device, dtype=torch.bfloat16)
        tids = batch["time_ids"].to(device, dtype=torch.bfloat16)

        noise = torch.randn_like(latents)
        if args.noise_offset > 0:
            noise = noise + args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=device
            )
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
        noisy = noise_scheduler.add_noise(latents, noise, t).to(torch.bfloat16)

        pred = model(
            noisy,
            t,
            encoder_hidden_states=prompt,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": tids},
        ).sample

        mse = (pred.float() - noise.float()).pow(2).mean(dim=(1, 2, 3))
        if args.min_snr_gamma > 0:
            alpha = alphas[t]
            snr = alpha / (1.0 - alpha)
            weight = torch.minimum(snr, torch.full_like(snr, args.min_snr_gamma)) / snr
            loss = (mse * weight).mean()
        else:
            loss = mse.mean()

        (loss / args.grad_accum).backward()
        running_loss += loss.item()
        running_n += 1
        micro += 1
        if micro < args.grad_accum:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        if theta_cap:
            # project back onto the sphere: AdamW keeps pushing theta's norm up even
            # when the loss is flat, and that drift is what saturates the outputs
            with torch.no_grad():
                norm = torch.linalg.vector_norm(theta_param)
                if norm > theta_cap:
                    theta_param.mul_(theta_cap / norm)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        step += 1

        # cheap per-step heartbeat so scripts/watch_progress.py can draw a smooth
        # bar; train.jsonl only updates every --log-every steps
        (out / "status.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "loss": round(loss.item(), 6),
                    "sec": round(time.time() - started, 1),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
            ),
            encoding="utf-8",
        )

        if step % args.log_every == 0 or step == 1:
            if args.adapter == "gpart":
                theta_norm = next(p for p in trainable).detach().float().norm().item()
            else:
                theta_norm = float("nan")
            val_loss = None
            if args.val_batches and args.val_every and step % args.val_every == 0:
                val_loss = validate()
            record = {
                "step": step,
                "loss": running_loss / running_n,
                "val": val_loss,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "theta_norm": theta_norm,
                "sec": round(time.time() - started, 1),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            running_loss = 0.0
            running_n = 0

        if args.save_every and step % args.save_every == 0:
            model.save_pretrained(out / f"step_{step:05d}")
            # also overwrite checkpoint/ so an interrupted run can --resume from here
            # instead of falling back to the last completed step
            save_checkpoint(step)
            print(f"saved step_{step:05d}", flush=True)

        if step >= args.steps:
            break
        if args.max_seconds and time.time() - started > args.max_seconds:
            print(f"time budget reached at step {step}", flush=True)
            break

    save_checkpoint(step)
    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    print(f"saved adapter to {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
