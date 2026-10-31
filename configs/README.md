# `configs/`

One file per row of `results/tables/`. They record the exact settings, and
`src/train.py --config <file>` reads them (CLI flags still win).

## Schema

```yaml
name: <row id matching results/tables/character.csv>
adapter: gpart | lora
# --- gpart ---
d: 23224320
backend: stateless | materialized
partition_mode: splitmix64 | int32
# --- lora ---
rank: 8
alpha: 4                 # LoRA alpha; NOT the sampling strength
# --- both ---
targets: [...]           # bare names hit attn1+attn2; prefixed names hit one
scope: transformer_block
layers_pattern: transformer_blocks
data: <cache dir>
output: <run dir>
train: {steps, lr, lr_schedule, warmup, grad_accum, max_res, noise_offset,
        min_snr_gamma, grad_clip, seed, holdout, val_every, val_batches, save_every}
sample: {alpha, steps, cfg, size, sched, spacing, seeds, prompts}
```

## Two blocks, two consumers

- `train:` and the top-level keys are read by `src/train.py`.
- `sample:` is **not** read by the trainer. It records how the row was scored —
  the α used, the seeds, the sampler — and is what you pass to `src/sample.py`.
  This split exists because `alpha` means two different things (LoRA strength
  during training vs. adapter scale at sampling); keeping them in separate blocks
  makes that impossible to mix up.

## Set `sched` and `spacing` explicitly

`src/sample.py` defaults to `--sched euler --spacing leading`. Neither is what
produced these numbers. Every config here pins `sched: euler_a` and
`spacing: trailing`, which is what matches ComfyUI's "normal" schedule; the
`leading` default will not reproduce `results/`. See `docs/01-protocol.md` §1.

## Key values

| | |
|---|---|
| `targets: ["attn2.*"]` | 280 modules, 545.6 M weights |
| `targets: ["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"]` | 700 modules, 2184.0 M weights (bare names match both attn1 and attn2) |

The bare-name form is the one that matters: prefixes are what make the attn2-only
run *not* cover the matched LoRA's modules.

## Rows

| file | row | score |
|---|---|---|
| `gpart_char_attn2.yaml` | `gp_char` | 0.43 |
| `gpart_char_alllinear.yaml` | `gp_char_alllinear` (α sweep inside) | 0.40–0.57 |
| `lora_char_matched.yaml` | `lo_char_matched` | 0.70 |
| `lora_char_kohya.yaml` | `lo_char_kohya` — the confound, not the opponent | 0.05 |
| `gpart_style_attn.yaml` | `gp_style_attn` | edge 0.017 |
| `lora_style_matched.yaml` | `lo_style_matched` | edge 0.028 |

The kohya recipe in `lora_char_kohya.yaml` is approximated from the shipped
config (10 epochs, batch 4, cosine-with-restarts); its `steps: 160` is the
derived count, not a value from kohya's log. See `docs/01-protocol.md` §3.
