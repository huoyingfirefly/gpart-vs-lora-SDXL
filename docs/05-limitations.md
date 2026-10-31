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
- No confidence intervals are reported for the tag scores. Treat ±0.05 as noise
  and do not over-read 0.55 vs 0.57.

## 6. Reproducibility

- The exact base checkpoint (NoobAI-XL v1.1) is not redistributed here. Numbers
  depend on it.
- Training used specific latent caches; re-encoding the source images with a
  different VAE build could shift results slightly.
- The sampler must use `trailing` spacing. `leading` (the diffusers default for
  many schedulers) produces visibly different images at the same settings and
  will not reproduce these numbers.

## 7. What a fair referee would say

> You tested one character, one style and one base model, varying only the recipe
> and the strength. You showed that the originally reported gap does not survive
> that control, which is a useful negative result. You did not show that low-rank
> adaptation is generally better, and you did not test the composition property
> that is GPart's actual stated advantage.

That is the correct reading, and `00-claims.md` is written to match it.
