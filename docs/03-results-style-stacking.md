# Style transfer and adapter stacking

Two more tasks, run because the character result alone can be explained away as
"LoRA happens to suit a single face". Both start from the same base model, the
same sampler, the same 3 seeds × 2 prompts, and report image statistics against
the no-adapter image of the same seed.

```
dRGB   mean |I − I_base| per channel   (how much the image changed)
S      mean saturation                 (base 0.297)
edge   mean |∇ luminance|              (base 0.028)
```

`dRGB` is the control variable. If two adapters changed the image by the same
amount and one of them kept its detail while the other did not, that difference
is about the adapter, not about the strength.

## 1. Style transfer

Data: one style (`reserve1999`, 76 images). Same two recipes as the character
task (matched lo 5e-4/1200, kohya 1e-4/cosine).

| config | dRGB | S | V | edge |
|---|---|---|---|---|
| base | 0.000 | 0.297 | 0.501 | 0.028 |
| LoRA style (kohya) | 0.131 | 0.200 | 0.488 | 0.027 |
| **LoRA style (matched)** | **0.168** | **0.240** | 0.435 | **0.028** |
| GPart style (all-linear) | 0.146 | 0.233 | 0.483 | 0.020 |
| **GPart style (attn2)** | **0.172** | **0.155** | 0.444 | **0.017** |

Two things fall out of this table.

**(a) "GPart shifts the style harder" is also a recipe artifact.** At equal
displacement — matched LoRA 0.168, GPart attn2 0.172 — the image change is the
same. The larger figure you get from the open-box run is the larger ΔW from the
larger learning rate, again.

**(b) At equal displacement, GPart spends it on a global wash.** Saturation drops
to 0.155 and edge energy to 0.017, a **39% loss of high-frequency detail relative
to base**. The matched LoRA, for the same 0.168 dRGB, ends at S 0.240 and edge
0.028 — edge energy identical to base, detail fully retained.

The obvious counter-explanation is that this particular style is flat and blurry,
and GPart simply reproduced it better. It is not: a kohya LoRA trained on the same
76 images scores edge 0.027. The detail loss is a property of the GPart run, not
of the style.

Note also that GPart's own all-linear run (edge 0.020) retains more than its
attn2 run (0.017) at *less* displacement (0.146 vs 0.172) — spreading the same
parameter budget over attention-only modules produces a more global, less
structured change.

## 2. Stacking a style adapter and a character adapter

Composition is where full-rank partitioning is supposed to matter: GPart's `ΔW =
P·diag(s)·θ` is linear in `θ`, so two adapters composed in the shared
partitioned coordinates add exactly, whereas two LoRAs composed in factor space
produce cross terms (`B₁A₁ + B₂A₂` is not the same as any single `BA`).

We tested the plain additive case — load both adapters, add their ΔW — because
that is what users do:

| config | character score | dRGB | S | edge |
|---|---|---|---|---|
| base | 0.08 | 0.000 | 0.297 | 0.028 |
| LoRA stacking (kohya pair) | **0.00** | — | — | — |
| **LoRA stacking (matched pair)** | **0.57** | 0.206 | 0.249 | **0.029** |
| GPart attn2 style + attn2 character | 0.40 | 0.206 | 0.161 | **0.021** |
| GPart attn2 style + all-linear character | 0.27 | 0.177 | 0.129 | 0.021 |

**(a) "LoRA stacking just averages the adapters away" was a kohya artifact.**
The open-box pair collapses to 0.00 — the character disappears entirely — because
each constituent is already at ~1/33 displacement. The matched pair scores 0.57
with accessories intact.

**(b) "GPart is the better composition base" does not hold either.** At identical
dRGB (0.206) the stacking result reproduces the single-adapter pattern exactly:
LoRA keeps S 0.249 / edge 0.029, GPart falls to S 0.161 / edge 0.021. Composition
neither rescues nor amplifies the difference.

Stacking was also run in **displacement-matched** variants (`gp_stack_attn_m`,
`gp_stack_attn_fm`, `gp_stack_all_m` in the working tree) so that the comparison
does not rest on dRGB being the right control variable. The ordering is the same
there. Their raw images and logs were not packaged into `results/` because only
the ordering is being claimed — if you want the per-row numbers, they are
regenerable from `scripts/history/stacktest_2026-10-30.sh`.

## 3. What stacking does *not* test

GPart's published claim is not that additive merging is easy — it is that
composition **in shared coordinates** stays exact under operations that are
non-linear in matrix space: TIES merging, orthogonalization against other
adapters, sign conflicts. In factor space those operations destroy or
misattribute LoRA's components.

None of that was implemented in this repo. The table above is plain `ΔW₁ + ΔW₂`
for both methods, which is the one composition operation where LoRA's cross terms
do not matter. So:

- ✅ We can say: plain additive stacking shows no GPart advantage **on the
  dRGB/saturation/edge family of metrics** (§1–§2). On a style-*match* metric it
  does, and that is §5 — read them together before quoting either.
- ❌ We cannot say: GPart is not a better composition base. That hypothesis is
  untested here, and it remains the most plausible place for GPart's structural
  advantage to show up.

See `00-claims.md`, "claims we do NOT make", row 4.

## 4. From "how much changed" to "did it move toward the target"

Every metric in §1–§3 is relative to the *same seed with no adapter*. That makes
them control variables, not scores: `dRGB` says an image changed, `S` and `edge`
say how the change is distributed, and none of them knows what the target style
looks like. A style adapter can produce a large, detail-preserving, entirely
wrong change and score well on all three.

Answering "did it move toward the style" needs a reference set — the style's own
source images — and a distance to it. §5 adds two such distances and re-runs the
stacking comparison with them, which is where the strongest result in this
document comes from.

## 5. Three families, a style-*match* metric, and the composability difference

§1–§2 use `dRGB`, `S` and `edge`, which measure *how much* changed and whether
detail survived. They do not measure whether the image moved **toward the
target style**. Adding that measurement, and swapping in a community-trained
LoRA as the mainstream baseline, changes part of the picture.

### The baseline, and why "our LoRA was too small" is not the explanation

| | modules | attn1 | attn2 | FFN | rank | ‖ΔW‖ |
|---|---|---|---|---|---|---|
| self-trained r8 (matched) | 700 | 280 | 280 | 140 | 8 | 44.2 |
| kohya `reserve1999_style_v3` | 722 | 280 | 280 | 140 | 32 | **6.50** |

Same dataset (`6_reserve1999`), same trigger, same base model, kohya sd-scripts,
1310 steps, lr 1e-4. Nearly identical module coverage (722 vs 700), and the
self-trained adapter carries **6.8× the displacement** — yet the kohya one is the
better style adapter. Coverage and magnitude are both falsified as explanations
for why the self-trained LoRA underperformed; only the direction of the update
differs.

### Style match on a four-framing holdout set

`|z|` = mean absolute z-score over six whole-image colour statistics against the
style's own reference set; `SWD` = sliced Wasserstein distance between 32×32
Oklab patch clouds. Both are gated on WD14 `1girl` ≥ 0.5 so an image that lost
its subject cannot score well by being empty. Lower is better. 12 images per arm.

| arm | identity | \|z\| | SWD |
|---|---|---|---|
| base, no adapter | 0.02 | 1.459 | 8.711 |
| **GPart style @0.469 + GPart char @1.3** | **0.63** | **0.408** | 3.713 |
| GPart style @0.469 + LoRA char @1.0 | 0.50 | 0.575 | **3.585** |
| kohya style @1.0 + GPart char @1.3 | 0.52 | 0.530 | 5.103 |
| kohya style @1.0 + LoRA char @1.0 | 0.57 | 0.652 | 4.666 |
| kohya style @1.5 + LoRA char @1.0 | 0.47 | 0.583 | 5.011 |
| kohya style @2.0 + LoRA char @1.0 | 0.47 | 0.726 | 5.559 |
| self-trained style r8 @0.469 + LoRA char @1.0 | 0.50 | 0.721 | 4.863 |
| *reference: kohya style @1.5 alone, no character* | — | 0.616 | **3.336** |

Two things to notice before reading a winner out of it:

1. **`|z|` and `SWD` disagree.** `|z|` ranks the pure-GPart stack first by a wide
   margin; `SWD` puts the hybrid first and GPart second. The gap between metric
   families is the same size as the gap between arms, so neither ranking is a
   result on its own — quote both or neither.
2. **The identity column is not a ranking** — at equal displacement the two
   character adapters are statistically indistinguishable (`02` §5).

### The composability difference

The row that does not depend on which style metric you prefer is the *reference*
row, and its comparison against the stacked rows:

```
kohya style alone            SWD 3.336   ← best style number in the whole table
kohya style + LoRA char      SWD 4.666 (α1.0) / 5.011 (α1.5) / 5.559 (α2.0)
kohya style + GPart char     SWD 5.103
GPart style + LoRA char      SWD 3.585
GPart style + GPart char     SWD 3.713
```

The kohya style adapter is the **best alone** and the **worst stacked**, and it
gets worse monotonically as its α rises. The GPart style adapter stays at
3.59–3.71 against every character adapter tried, including one from the other
family. (Its own solo SWD is not in this table — the solo-style arms were run on
a separate prompt set and are not comparable to these rows.)

**Hypothesis (untested):** kohya is dense rank-32 over attn1 + attn2 + FFN, and
the character LoRA occupies the same module set, so the two low-rank increments
compete for the same directions. GPart is attn2-only with a sparse 50 M support,
overlapping the character LoRA's support much less.

This hypothesis has not been tested against alternatives — for instance, that the
kohya adapter is simply saturated at the α we stacked it at, or that its training
data overlaps the character images. Until one of those is excluded, the honest
form is: **we found a concrete pair of adapters where the GPart one composes and
the LoRA one does not.** That is a fact about those two files, not yet about the
two parameterizations.

Full table: `results/tables/holdout_three_family.csv`.

## 6. Figures

| figure | shows |
|---|---|
| `results/figures/horizontal_full_prompt_style.png` | style rows at the full prompt |
| `results/figures/horizontal_stacking_p0.png` | base / kohya stack / matched stack / both GPart stacks |

The four-framing holdout arms have no contact sheet in `results/figures/` — they
were scored from the working tree. `scripts/history/stylechar_*.sh` regenerates
them; `results/tables/holdout_three_family.{csv,md}` holds every number.
