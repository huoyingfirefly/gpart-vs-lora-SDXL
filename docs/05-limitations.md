# Limitations

Consolidated. Each item says what was not done and what conclusion would be
overreach if it were ignored.

## 1. Scope

| | tested | not tested |
|---|---|---|
| base model | NoobAI-XL v1.1 (SDXL) | any SD1.5/SD3/Flux model |
| tasks | 1 character, 1 style, 1 additive stack of those two | composition (multi-character), concept learning, pose, lighting, control |
| characters | `aima`; `juxueli` in progress as single-arm | — |
| encoders | UNet only | text encoder (the vendored GPart path cannot train it) |
| resolution | 832×1216, buckets ≤1536 | high-res, tiling |

One character and one style is enough to falsify "GPart's advantage comes from
its parameterization", because that claim is universal and one counter-example
kills it. It is **not** enough to establish "LoRA is generally better" — that is
a universal claim we did not test.

## 2. Hyperparameters not swept

| knob | points tested | gap |
|---|---|---|
| GPart `d` | 23.2 M, 50 M | two points, one character; no interaction with lr |
| GPart lr | 5e-4 only | the entire headline is about lr, and we never varied GPart's |
| GPart target modules | attn2, all-linear, `ff` only | partial coverage between the two, e.g. attn+ff without conv |
| LoRA rank | r8, r16 | r4, r32, r64 |
| LoRA α/rank ratio | 0.5 | fixed by both recipes |
| caption expansion | 3 variants | variant mix not ablated |
| α (strength) | see `01-protocol.md` §5 | GPart attn2 above 1.0 not swept; the attn2 vs all-linear α comparison is across different runs |

The **lr** row is the important one. We concluded the open-box result was an lr
artifact, and we demonstrated that by moving *LoRA* to GPart's lr. We never moved
GPart to a different lr. So strictly: "at lr 5e-4, matched everything, LoRA
wins". A GPart run that does better at 1e-3 is not excluded.

## 3. Metric limitations

- **Tag scores are not image quality.** The α sweep makes this concrete: the
  score rises to 0.57 while saturation, edge energy and composition degrade.
  Every score in this repo should be read next to the corresponding
  `dRGB`/`S`/`edge` row and the images.
- **No perceptual / feature-space metrics.** No CLIP, DINO, or LPIPS distance, no
  human preference study. The image statistics are cheap proxies.
- **The two style metrics disagree, at the same magnitude as the effects.**
  `|z|` (whole-image colour) and `SWD` (patch-level texture) rank the same arms
  differently — `03-results-style-stacking.md` §5. Neither is a ground truth, so
  a claim resting on one of them alone is a claim about that metric.
- **The style metrics are framing-sensitive.** Cropping 65% of an image moves
  `|z|` by 0.8–2.4, which is larger than the gap between most arms. The content
  gate (WD14 `1girl` ≥ 0.5) removes the worst failure — an empty image scoring as
  a perfect style match — but not framing sensitivity.
- **The identity metric is prompt-dependent, and the dependence is large.**
  Changing the prompt set moves it 0.22; stacking a style adapter moves it 0.09
  (`02-results-character.md` §5). It is a measurement of a configuration, not of
  a method.
- **Threshold sensitivity.** Tag hits use p ≥ 0.35 with no sensitivity analysis.
  A different threshold would move every absolute number; the base row anchors
  the comparison.
- **Tag sets are bespoke.** 10 tags per character, chosen post-hoc by caption
  frequency and filtered for base discrimination. This is a reasonable procedure
  but not a preregistered one.
- **Images were looked at.** The qualitative observations (colour cast at α 1.5,
  background smear at 1.8) are not quantified beyond the statistics. They are
  reported as observations, not measurements.

## 4. Implementation limitations

- **The 4.2× training cost is this repo's `stateless` backend.** See `04-cost-model.md`.
  A materialized-`P` implementation would be faster and more memory-hungry; we
  could not run it in 8 GB.
- **No text encoder training.** LoRA in kohya can train the TE; the vendored GPart
  path here cannot. This is an asymmetry in favor of LoRA that we did not remove —
  though note that in these runs the LoRA also never trained the TE (`train_unet_only`),
  so the comparison itself is symmetric.
- **Composition was tested only additively.** The structural claim that motivates
  GPart — exact linear composition in shared coordinates under non-linear merging
  operations — was not implemented. This is the largest untested gap for anyone
  evaluating GPart on its own terms.
- **The α sweep was scored on `p0` only**, `n = 6`, on a single character.
- **`p2` (held-out caption) exists for three rows only.**

## 5. Statistical

- Character rows: `n = 6` seeds, but one character. Seed variance is measured;
  across-character variance is not (`juxueli` pending).
- Style and stacking: `n = 3` seeds × 2 prompts = 6 images per cell. Small.
- The step curve uses `n = 3` at 400/600/800; differences between those three
  points are within noise and are not read as a curve (only the 800 vs 400
  accessory gap is robust, and it is large).
- **The four-framing holdout set is 12 images per arm** (4 prompts × 3 seeds).
  Paired bootstrap over those cells puts the 95% interval of a 0.05–0.13 identity
  difference across zero, and of a 0.13 difference barely outside it. Do not read
  a 0.05 gap between arms as a result at this `n`.
- **Confidence intervals are reported only where they were computed** — the
  identity comparisons in `02-results-character.md` §5. The tables from `p0`
  alone (`n = 6`) have none; treat ±0.05 as noise and do not over-read 0.55 vs
  0.57.
- **No per-arm image counts were increased for the style metrics.** 12 images is
  thin for a Wasserstein distance over patch clouds; the `SWD` ordering between
  adjacent arms is within its own sampling noise.

## 6. Reproducibility

- The exact base checkpoint (NoobAI-XL v1.1) is not redistributed here. Numbers
  depend on it.
- Training used specific latent caches; re-encoding the source images with a
  different VAE build could shift results slightly.
- The sampler must use `trailing` spacing. `leading` (the diffusers default for
  many schedulers) produces visibly different images at the same settings and
  will not reproduce these numbers.
- **Prompts longer than 77 CLIP tokens must be chunked, not truncated.** Until
  this was fixed, `src/sample.py` used `truncation=True` and silently dropped
  everything past token 77. ComfyUI and A1111 encode consecutive 77-token
  windows instead. Any result produced by this repo before the fix that used a
  detailed prompt is a result for a shorter prompt than the one printed. The
  chunking path now in `src/sample.py` returns identical embeddings for prompts
  that already fit in one window, so short-prompt numbers are unaffected.
- The community kohya style LoRA (`reserve1999_style_v3`, rank 32) is an external
  download and is not redistributed here. Its `‖ΔW‖` and module list are recorded
  in `results/tables/holdout_three_family.csv`; its identity could not be pinned
  to a specific training run beyond the metadata it ships with.

## 7. What a fair referee would say

> You tested one character, one style and one base model, varying only the recipe
> and the strength. You showed that the originally reported gap does not survive
> that control, which is a useful negative result. You did not show that low-rank
> adaptation is generally better, and you did not test the composition property
> that is GPart's actual stated advantage.

That is the correct reading, and `00-claims.md` is written to match it.
