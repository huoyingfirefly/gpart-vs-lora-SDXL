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

## 5. The identity gap is prompt-dependent, and at equal displacement it is zero

Everything above is measured on `p0` alone. Repeating §3's comparison on four
different framings, 3 seeds each (12 images per arm, same adapters, same
`‖ΔW‖` at nominal α 1.0 — 68.8 vs 78.4):

| | framing | LoRA @1.0 | GPart @1.0 |
|---|---|---|---|
| h0 | close-up | 0.77 | 0.70 |
| h1 | wide rooftop | 0.47 | 0.40 |
| h2 | medium / train | 0.67 | 0.53 |
| h3 | from behind | 0.30 | 0.37 |
| | **mean** | **0.55** | **0.50** |

Paired bootstrap over the 12 (prompt, seed) cells, 4000 resamples:
**+0.050, 95% CI [−0.058, +0.175]** — not distinguishable from zero. Per
framing, every CI also crosses zero (`scripts/boot_identity_2026-11-02.py`).

So §3's 0.27 gap is `p0`. `p0` is `full body, standing, from front, … crowd in
background` — the most structurally demanding prompt in the set. LoRA covers
attn1 + FFN, which carries spatial structure; GPart covers attn2 only, which
modulates prompt injection. **The more a prompt constrains composition, the more
LoRA wins.** On a close-up with nothing to compose, the two tie.

### The strength ceiling is asymmetric

Both methods are exactly linear in α (`ΔW` scales with the LoRA scale, and
`ΔW = P·diag(s)·(αθ)` for GPart). The optimum is not:

| GPart char α | ‖ΔW‖ | identity (stacked on GPart style @0.469) |
|---|---|---|
| 0.88 | 68.8 | 0.42 |
| 1.0 | 78.4 | 0.54 |
| **1.3** | 101.9 | **0.63** |
| 1.6 | 125.4 | 0.56 |

LoRA peaks at 1.0 and degrades at 1.5 and 2.0; GPart peaks at 1.3–1.6. Since both
are linear in their scale parameter, that asymmetry is a property of the two
`ΔW` **directions** — LoRA's has components that turn destructive when amplified,
GPart's sparse attn2-only support has fewer. It means "each at its own optimum"
is not a neutral comparison:

- **equal displacement** (GPart @0.88 vs LoRA @1.0, ‖ΔW‖ 68.8 both): GPart 0.417
  vs LoRA 0.500, GPart − LoRA = −0.083, CI [−0.225, +0.058] — **no difference**.
- **each at its own optimum** (GPart @1.3, 1.8× the displacement): GPart 0.633
  vs LoRA 0.500, GPart − LoRA = +0.133, CI [+0.025, +0.233] — **GPart ahead**.

Which one is "right" depends on the question. For "which parameterization carries
more identity per unit of weight change", this is a tie. For "which adapter
should I ship at its best setting", GPart's ability to absorb more displacement
is a real asset.

`results/tables/holdout_three_family.csv` has the full three-family table these
rows come from (§`03` §5).

## 6. Images

| figure | shows |
|---|---|
| `results/figures/horizontal_full_prompt_character.png` | full prompt, 6 seeds: base / kohya LoRA / GPart attn2 / matched LoRA |
| `results/figures/horizontal_alpha_sweep_top.png` | `p0`, 3 seeds: base / matched LoRA / GPart attn2 (α 1.0) / GPart all-linear (α 1.0) |
| `results/figures/horizontal_alpha_sweep_bottom.png` | `p0`, 3 seeds: GPart all-linear at α 1.3 / 1.5 / 1.8 |

Compare the α 1.5 and 1.8 rows in the bottom sheet against the α 1.0 row in the
top sheet before accepting the 0.57 peak number.

## 7. Caveats specific to this task

- One character. A second character (`juxueli`, same acquisition pipeline) is in
  progress as a single-arm replication — it can check that GPart learns another
  character at all, but it will **not** carry a GPart-vs-LoRA conclusion.
- **The identity score is not a stable ranking.** It moves 0.22 when the prompt
  set changes and another 0.09 when a style adapter is stacked (§5). Any single
  number quoted from this doc is a number *for a configuration*, not for a method.
- The tag set is 10 tags chosen by caption frequency and checked for base
  discrimination; a different tag set would move the absolute scores. The base
  row is the anchor.
- `p2` (held-out caption) was scored for the three main rows only.
