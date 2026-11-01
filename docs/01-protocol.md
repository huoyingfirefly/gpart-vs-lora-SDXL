# Protocol

The whole point of this repository is the protocol, so it is written down before
the numbers.

## 0. The failure mode this protocol is designed against

A PEFT comparison is only meaningful if the two adapters receive **the same
training signal**. It is easy to produce a 30× difference in weight displacement
without touching the algorithm:

| knob | effect |
|---|---|
| learning rate | 5× is enough to change a "dead" adapter into a strong one |
| number of optimizer steps × grad-accum × batch | exposure count |
| lr schedule (constant vs cosine-with-restarts) | effective step size over time |
| which modules are adapted | capacity and coverage radius |
| the strength you *sample* with | post-hoc tuning of one side only |

The shipped GPart and kohya-LoRA configs differ on four of these five. Any
headline produced from them compares recipes.

## 1. Fixed sampling protocol

All images in `results/` use one sampler configuration, chosen so it matches
ComfyUI's "normal" schedule (diffusers calls it `trailing` — this matters, the
default `leading` shifts every image):

```
scheduler   Euler ancestral
spacing     trailing
steps       30
CFG         5.0
size        832 x 1216
dtype       bf16
```

## 2. Training recipe (all matched runs)

```
optimizer        AdamW (weight_decay 0)
lr               5e-4, constant, warmup 50 steps
steps            1200  (800 for the shortened character runs, see §5)
grad accumulation 2  (effective batch 2)
noise offset     0.05
min-SNR gamma    5
grad clip        1.0
max resolution   1536 (bucketed)
seed             42
holdout          4 images, val every 200 steps
base model       NoobAI-XL v1.1 (SDXL), UNet only
```

Data: 21 source images per character, each expanded to **3 caption variants**
(full caption / trigger + scene tags / trigger only) so that the trigger token
has to carry identity. 63 samples, ~38 epochs at 1200 steps.

## 3. Adapter configurations

| row | adapter | targets | modules | covered weights | params |
|---|---|---|---|---|---|
| `gpart_attn2` | GPart, `d = 23,224,320` | `attn2.{to_q,to_k,to_v,to_out.0}` | 280 | 545.6 M | 23.2 M |
| `gpart_alllinear` | GPart, same `d` | + `ff.net.0.proj`, `ff.net.2` | 700 | 2184.0 M | 23.2 M |
| `lora_r8_matched` | LoRA `r=8, α=4` | same 700 | 700 | 2184.0 M | 20.9 M |
| `lora_kohya` | LoRA `r=16, α=8`, kohya | all linear | 700 | 2184.0 M | 42.7 M |

`gpart_alllinear` exists only to answer "how much did coverage explain?" — it
changes **only** `--targets` relative to `gpart_attn2`.

`lora_kohya` is the open-box reference: lr 1e-4, cosine-with-restarts, 10 epochs,
batch 1×4, 1024 buckets, AdamW8bit. It is included to quantify the confound,
**not** as the fair opponent.

## 4. Metrics

**Character identity** — WD14 booru tagger over 10 canonical tags:

```
white_hair  pink_hair  gradient_hair  red_eyes  beret
hair_flower hair_ornament choker black_dress black_ribbon
```

A tag counts as a hit at probability ≥ 0.35. `score` = mean hit rate across the
10 tags. `base` (no adapter) is the floor; read deltas, not absolute values.

Prompts: `p0` (clean scene, trigger only — the discriminating prompt) and `p1`
(full description — everything the adapter should supply is spelled out, so it
measures almost nothing about the adapter). Report on `p0`.

`p0` is also the most structurally demanding prompt in the set (full body,
standing, crowd in the background), which is why §`02` §5 adds a **four-framing
holdout set** — close-up, wide rooftop, medium/train, from-behind, 3 seeds each:

```
h0  upper body, close-up, looking at viewer, indoors, cafe, night, warm lamp light
h1  wide shot, full body, standing, from side, outdoors, rooftop, sunset
h2  medium shot, sitting, hugging knees, indoors, train interior, dusk light
h3  full body, walking, from behind, outdoors, rainy street, neon reflections, night
```

(each prefixed with `r1999style, aima, 1girl, solo,` and suffixed with the quality
tags; full text in `configs/prompts/holdout4.jsonl`)

`h0`–`h3` span the range where an identity adapter has little to do (`h0`) to
where it has a lot (`h3`, where two of the ten tags — `red_eyes`, `choker` — are
structurally invisible and the ceiling is 0.8 rather than 1.0). Any identity
number quoted without naming its prompt set is a number for whichever set the
author happened to use; the two agree on direction only for `p0`.

**Style / image change** — against the same seed with no adapter:

```
dRGB   mean |I - I_base| per channel
S      mean saturation
V      mean value
edge   mean |gradient| of the luminance channel
```

`edge` is the useful one: it separates "the image changed a lot" from "the image
kept its detail".

**Displacement** — `‖ΔW‖` = α‖θ‖ for GPart, `‖BA‖` for LoRA; per-weight RMS uses
the number of weights the adapter actually covers. The convention matters: an
attn2-only adapter normalized over the full model would look ~2× smaller. Both
conventions are reported where it matters.

## 5. Strength is always swept on both sides

Never compare one method's optimum to the other's default. Sweeps actually run:

| | swept |
|---|---|
| kohya LoRA | 1.0 / 1.5 / 2.0 / 3.0 |
| matched LoRA | 1.0 / 1.5 / 2.0 |
| GPart attn2 | 1.0 |
| GPart all-linear | 1.0 / 1.3 / 1.5 / 1.8 |

Two sweeps are missing and are stated as gaps: GPart attn2 above 1.0, and GPart
at strength 1.0 after the all-linear switch (done separately).

Step budget for the *shortened* runs (800 instead of 1200) is derived from the
step curve in `docs/02-results-character.md` §4, not chosen arbitrarily.

## 6. Seeds

Character rows: 6 seeds (`310977364113511, 12345, 67890, 111111, 222222, 333333`).
Style and stacking rows: 3 seeds × 2 prompts. All tables state `n`.

## 7. What is not controlled

- GPart's `d` (two points: 23.2 M, 50 M) and learning rate (one value).
- LoRA rank (r8 and r16 only, and they come from different recipes).
- Text encoder: never trained. The vendored GPart path trains the UNet only.
- Base-model variance across checkpoints of NoobAI-XL (v1.1 only).

Any conclusion that depends on these being swept is out of scope for this repo.
