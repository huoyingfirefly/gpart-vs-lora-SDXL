#!/usr/bin/env python
"""One-shot GPart training: dataset folder in, trained adapter out.

    python scripts/train_gpart.py \
        --data  D:/AI/lora/datasets_aima/10_aima \
        --name  aima --trigger aima \
        --identity "white hair,pink hair,gradient hair,red eyes,beret" \
        --preset attn2 --steps 1200

Does four things, in order, and stops at the first one that fails:

  1. checks the dataset is actually trainable (latents, captions, TE cache)
  2. expands each caption into 3 variants (full / trigger+scene / trigger only)
  3. trains GPart with the matched recipe from docs/01-protocol.md
  4. writes <output>/run_manifest.json and prints the sampling command

Nothing here is specific to the aima run: --data / --name / --trigger /
--identity are all required, deliberately. The one default that would be
dangerous to keep (caption_variants.py falling back to the aima identity set)
is refused rather than inherited.

Use --dry-run to see the exact commands without running them.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# two target presets; see configs/README.md for the module/weight counts
PRESETS = {
    # 280 modules, 545.6 M covered weights -- ~8.1 s/step, ~2.7 h for 1200 steps
    "attn2": ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"],
    # 700 modules, 2184.0 M covered weights -- ~20.5 s/step, multi-hour
    "alllinear": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
}

MATCHED_RECIPE = {
    "lr": "5e-4",
    "lr_schedule": "constant",
    "warmup": 50,
    "grad_accum": 2,
    "max_res": 1536,
    "noise_offset": 0.05,
    "min_snr_gamma": 5.0,
    "grad_clip": 1.0,
    "weight_decay": 0.0,
}


def die(msg: str) -> None:
    print(f"\n[FAIL] {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def audit_dataset(data: Path) -> dict:
    """Fail loudly on the three things that make a dataset silently untrainable."""
    if not data.is_dir():
        die(f"dataset directory not found: {data}")

    latents = sorted(data.glob("*_sdxl.npz"))
    if not latents:
        die(
            f"no *_sdxl.npz in {data}\n"
            "  This repo reads kohya sd-scripts' SDXL latent cache. Point --data at the\n"
            "  folder kohya wrote the cache into, or make the cache first (data/README.md)."
        )

    missing_te, missing_txt = [], []
    for lp in latents:
        sample_id = lp.name[: -len("_sdxl.npz")].rsplit("_", 1)[0]
        if not (data / f"{sample_id}_te_outputs.npz").exists():
            missing_te.append(sample_id)
        if not (data / f"{sample_id}.txt").exists():
            missing_txt.append(sample_id)

    if missing_te:
        die(
            f"{len(missing_te)} sample(s) have no *_te_outputs.npz "
            f"(first: {missing_te[0]})\n"
            "  Caption variants re-encode text, but training reads the TE cache. Regenerate\n"
            "  the cache with kohya, or rebuild it from the captions."
        )
    if missing_txt:
        die(f"{len(missing_txt)} sample(s) have no .txt caption (first: {missing_txt[0]})")

    if len(latents) < 5:
        print(f"[warn] only {len(latents)} samples -- under ~10 the trigger token "
              f"will not learn an identity", file=sys.stderr)

    return {"samples": len(latents), "latents_dir": str(data)}


def audit_models(unet: Path, aux: Path, need_aux: bool) -> None:
    if not (unet / "config.json").exists():
        die(
            f"no diffusers UNet at {unet}\n"
            "  Convert a single-file checkpoint with src/prep/convert_unet.py, "
            "or pass --unet."
        )
    if need_aux and not (aux / "text_encoder_2").exists():
        die(
            f"no SDXL text encoders at {aux}\n"
            "  --aux must contain text_encoder/, text_encoder_2/ and vae/. "
            "Needed to encode the caption variants; see data/README.md."
        )


def run(cmd: list[str], dry: bool, what: str) -> None:
    # shlex-quoted so the echo is copy-pasteable (the identity tag list contains spaces)
    print("  " + " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if dry:
        return
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        die(f"{what} exited with code {result.returncode}")
    print(f"  {what} ok in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a GPart adapter on a kohya-cached dataset, end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data", required=True, type=Path,
                   help="kohya dataset dir: images, .txt captions, *_sdxl.npz, *_te_outputs.npz")
    p.add_argument("--name", required=True,
                   help="run name; outputs go to <out-root>/<name>")
    p.add_argument("--trigger", required=True,
                   help="trigger token inserted into the training captions")
    p.add_argument("--identity", required=True,
                   help="comma-separated appearance tags stripped from the scene/bare "
                        "variants, so the trigger has to carry identity")
    p.add_argument("--preset", choices=sorted(PRESETS), default="attn2",
                   help="attn2: 280 modules ~8.1 s/step (default). "
                        "alllinear: 700 modules ~20.5 s/step, matches the LoRA coverage")
    p.add_argument("--steps", type=int, default=1200,
                   help="1200 = matched recipe. 800 is the cheapest budget that keeps "
                        "accessories; below that identity tags drop non-uniformly")
    p.add_argument("--d", type=int, default=23224320,
                   help="GPart group count; 23224320 is what every row in results/ uses")
    p.add_argument("--gpart-alpha", type=str, default="1.0",
                   help="training-time scale on theta (not the sampling strength)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-every", type=int, default=200,
                   help="0 disables validation (useful for a smoke run)")
    p.add_argument("--save-every", type=int, default=200,
                   help="0 keeps only the final adapter")
    p.add_argument("--extra", default="",
                   help="extra flags appended to the train command, shlex-split; "
                        "e.g. --extra '--theta-norm-max 70 --resume'")
    p.add_argument("--unet", type=Path, default=ROOT / "data" / "noobai_v11_unet")
    p.add_argument("--aux", type=Path, default=ROOT / "data" / "noobai_v11_aux")
    p.add_argument("--out-root", type=Path, default=ROOT / "runs")
    p.add_argument("--variant-dir", type=Path, default=None,
                   help="where the caption variants go (default: <data>_var)")
    p.add_argument("--device", default="cuda", help="device for the text encoder")
    p.add_argument("--reuse-variants", action="store_true",
                   help="skip variant generation if the directory already has samples")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running anything")
    args = p.parse_args()

    t0 = time.time()
    data = args.data.resolve()
    out = (args.out_root / args.name).resolve()
    variant_dir = (args.variant_dir or data.with_name(data.name + "_var")).resolve()

    if variant_dir == data:
        die("--variant-dir must differ from --data; the variants are a separate dataset")

    step("1/4 auditing the dataset")
    data_info = audit_dataset(data)
    print(f"  {data_info['samples']} cached samples in {data}", flush=True)

    has_variants = bool(list(variant_dir.glob("*_te_outputs.npz")))
    audit_models(args.unet, args.aux, need_aux=not (args.reuse_variants and has_variants))

    step("2/4 caption variants")
    if args.reuse_variants and has_variants:
        n = len(list(variant_dir.glob("*_te_outputs.npz")))
        print(f"  reusing {n} samples already in {variant_dir}")
    else:
        if variant_dir.exists() and any(variant_dir.iterdir()):
            if args.dry_run:
                print(f"  [dry-run] would clear {variant_dir}")
            else:
                print(f"  clearing stale variants in {variant_dir}")
                shutil.rmtree(variant_dir)
        run([
            sys.executable, "-m", "src.prep.caption_variants",
            "--src", str(data), "--dst", str(variant_dir),
            "--trigger", args.trigger,
            "--identity", args.identity,
            "--device", args.device,
            "--aux", str(args.aux),
        ], args.dry_run, "caption variants")

    step("3/4 training")
    train_cmd = [
        sys.executable, "-m", "src.train",
        "--adapter", "gpart",
        "--d", str(args.d),
        "--gpart-alpha", args.gpart_alpha,
        "--targets", *PRESETS[args.preset],
        "--scope", "transformer_block",
        "--layers-pattern", "transformer_blocks",
        "--backend", "stateless",
        "--partition-mode", "splitmix64",
        "--data", str(variant_dir),
        "--unet", str(args.unet),
        "--output", str(out),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--holdout", "4",
        "--val-every", str(args.val_every),
        "--val-batches", "4",
        "--log-every", "10",
        "--save-every", str(args.save_every),
    ]
    for key, value in MATCHED_RECIPE.items():
        train_cmd += [f"--{key.replace('_', '-')}", str(value)]
    if args.extra:
        train_cmd += shlex.split(args.extra)

    eta_h = {"attn2": 8.11, "alllinear": 20.5}[args.preset] * args.steps / 3600
    print(f"  preset={args.preset}  steps={args.steps}  "
          f"estimated wall clock ~{eta_h:.1f} h on an 8 GB card")
    run(train_cmd, args.dry_run, "training")

    step("4/4 manifest")
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name": args.name,
        "trigger": args.trigger,
        "identity": args.identity,
        "preset": args.preset,
        "targets": PRESETS[args.preset],
        "d": args.d,
        "gpart_alpha": args.gpart_alpha,
        "steps": args.steps,
        "seed": args.seed,
        "recipe": MATCHED_RECIPE,
        "data": str(data),
        "variants": str(variant_dir),
        "output": str(out),
        "unet": str(args.unet),
        "dataset": data_info,
        "python": sys.version.split()[0],
        "elapsed_s": round(time.time() - t0, 1),
    }
    try:
        import torch

        manifest["torch"] = torch.__version__
        manifest["cuda"] = torch.version.cuda
        manifest["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # torch is optional here; the manifest is still useful without it
        pass

    if args.dry_run:
        print("  [dry-run] " + json.dumps(manifest, indent=2))
    else:
        (out / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {out / 'run_manifest.json'}")

    adapter = out / "adapter"
    print(f"""
done: {time.time() - t0:.0f}s

  adapter   {adapter}
  log       {out / 'train.jsonl'}
  manifest  {out / 'run_manifest.json'}

sample it:

  python -m src.sample --adapter-dir "{adapter}" --alpha 1.0 \\
      --prompts-file configs/prompts/p0_p1.jsonl --seeds 1 2 3 \\
      --steps 30 --cfg 5 --size 832 1216 \\
      --sched euler_a --spacing trailing --output "{out / 'samples'}"

--alpha is the sampling strength, not --gpart-alpha: sweep it (1.0 / 1.3 / 1.5)
before concluding anything about this adapter. In the aima run the difference
between alpha 1.0 and 1.5 was 0.40 -> 0.57 on the same checkpoint.
""")


if __name__ == "__main__":
    main()
