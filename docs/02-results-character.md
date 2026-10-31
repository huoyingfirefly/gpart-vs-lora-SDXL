# Character task

One character (`aima`, 21 source images × 3 caption variants), NoobAI-XL v1.1,
UNet-only, prompt `p0` (clean scene, trigger token only), WD14 tag score over 10
tags at threshold 0.35. `n = 6` seeds for every row except where noted.

`p0` is the discriminating prompt. Under `p1` (the full description, with hair
colour, eye colour and every accessory spelled out) the base model scores almost
as well as any adapter, because there is nothing left for the adapter to supply.
Anything measured on `p1` is a statement about text following, not about the
adapter.

## 1. The open-box result, and why it isn't one

| config | recipe | modules | α | ‖ΔW‖ | per-weight RMS | score |
|---|---|---|---|---|---|---|
| base | — | — | — | 0 | 0 | 0.08 |
| LoRA (kohya, shipped) | lr 1e-4, cosine-restarts, 10 ep | all-linear r16 | 1.0 | 8.20 | 0.000103 | **0.05** |
| GPart (shipped) | lr 5e-4, constant, 1200 st | attn2 | 1.0 | 78.29 | 0.003355 | **0.43** |

Taken at face value this is a 33× displacement gap and a 0.43 vs 0.05 score gap,
i.e. "GPart learns the character, LoRA does nothing". It is not what it looks
like:

- The kohya adapter lands at **1/33 of GPart's per-weight displacement**. That is
  an under-trained ΔW, not a fully-trained one at strength 1.0.
- It scores **below base** (0.05 vs 0.08). An adapter whose output is
  indistinguishable from "no adapter" cannot be evidence about LoRA.
- Raising the kohya LoRA's strength recovers part of it (optimum at scale 2.0 →
  0.27 before saturation), which is the same finding from the other direction: the
  ΔW is simply too small.

The shipped GPart and the shipped LoRA differ in learning rate (5×), schedule,
step count, batch size and rank — four of the five confounds in
`01-protocol.md` §0.

## 2. Coverage is not the explanation

The obvious next hypothesis: GPart only touches `attn2` (280 modules, 545.6 M
weights) while LoRA touches all 700 linear modules (2184 M weights). Move GPart
up to the same coverage — same recipe, same `d`, only `--targets` changes:

| config | modules | covered | α | score | Δ |
|---|---|---|---|---|---|
| GPart attn2 | 280 | 545.6 M | 1.0 | 0.43 | — |
| GPart all-linear | 700 | 2184.0 M | 1.0 | **0.40** | −0.03 |

Coverage is falsified as an explanation: it slightly *hurts*. Per tag it is a
reshuffle, not a gain —

```
pink_hair  0.17 → 0.50     beret         0.67 → 0.17
red_eyes   0.67 → 1.00     black_ribbon  0.50 → 0.17
```

Hair and eye colour improve; the accessories regress. Same displacement budget
spread over 4× the weights.

## 3. Matched recipe, matched coverage, each at its own optimum

The matched LoRA is r8/α4, all-linear (same 700 modules), lr 5e-4 constant,
1200 steps, same latents, same caption variants. 20.9 M parameters vs GPart's
23.2 M — GPart has *more* capacity here.

| config | modules | α | ‖ΔW‖ | score (p0) | score (p2 held-out) |
|---|---|---|---|---|---|
| base | — | — | 0 | 0.08 | — |
| **LoRA matched** | all-linear r8 | 1.0 | 68.77 | **0.70** | **0.73** |
| GPart attn2 | 280 | 1.0 | 78.29 | 0.43 | 0.53 |
| GPart all-linear | 700 | 1.0 | — | 0.40 | — |
| GPart all-linear | 700 | 1.3 | — | 0.55 | — |
| **GPart all-linear (peak)** | 700 | 1.5 | — | **0.57** | — |
| GPart all-linear | 700 | 1.8 | — | 0.40 | — |

Both methods are at their own optimum. The matched LoRA saturates at 1.0 and
degrades at 1.5 and 2.0; GPart peaks at 1.3–1.5 and collapses by 1.8. Nobody in
this table is under-scaled.

### The α sweep is a real peak, and the peak costs image quality

Going from α 1.0 to 1.5 buys +0.17 score, and from the contact sheet it is clear
what is being bought: accessories appear, but clothing detail does not keep up;
by α 1.5 there is a global colour cast; at α 1.8 the background smears and the
composition distorts. The tag score rises while the images get worse — which is
exactly why the score alone is not a quality metric (`00-claims.md`).

### Displacement efficiency

To reach a *lower* score, GPart needs roughly 2× the matched LoRA's per-weight
RMS (0.0029 at α 1.3 vs LoRA's own displacement at 1.0). Same order of magnitude
of ‖ΔW‖, less character information per unit of it — the finding that shows up
again in the style and stacking tasks (§`03`).

## 4. Step budget: what under-training actually removes

Scoring intermediate checkpoints of the 1200-step run (`p0`, `n = 3`):

| steps | score | beret | hair_ornament | note |
|---|---|---|---|---|
| 400 | 0.30 | **0.00** | 0.33 | hair colour and eyes present, accessories gone |
| 600 | 0.23 | 0.00 | 0.67 | within noise of 400 |
| 800 | 0.33 | 0.67 | 0.67 | accessories appear |
| 1200 | 0.43 | 0.67 | 0.83 | reference (`n = 6`) |

Under-training is not uniform decay. The coarse, high-contrast identity cues
(hair colour, eye colour) arrive first; the accessories arrive last. At 400 steps
`white_hair` and `red_eyes` are already at 1.00 while `beret` is 0.00.

Practical consequence: **800 steps is the cheapest budget that preserves
accessories.** (The 400/600/800 points differ by less than their `n = 3` noise —
do not read the ordering between them as a curve.)

## 5. Images

| figure | shows |
|---|---|
| `results/figures/horizontal_full_prompt_character.png` | full prompt, 6 seeds: base / kohya LoRA / GPart attn2 / matched LoRA |
| `results/figures/horizontal_alpha_sweep_top.png` | `p0`, 3 seeds: base / matched LoRA / GPart attn2 (α 1.0) / GPart all-linear (α 1.0) |
| `results/figures/horizontal_alpha_sweep_bottom.png` | `p0`, 3 seeds: GPart all-linear at α 1.3 / 1.5 / 1.8 |

Compare the α 1.5 and 1.8 rows in the bottom sheet against the α 1.0 row in the
top sheet before accepting the 0.57 peak number.

## 6. Caveats specific to this task

- One character. A second character (`juxueli`, same acquisition pipeline) is in
  progress as a single-arm replication — it can check that GPart learns another
  character at all, but it will **not** carry a GPart-vs-LoRA conclusion.
- The tag set is 10 tags chosen by caption frequency and checked for base
  discrimination; a different tag set would move the absolute scores. The base
  row is the anchor.
- `p2` (held-out caption) was scored for the three main rows only.
