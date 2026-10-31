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

- ✅ We can say: plain additive stacking shows no GPart advantage on these tasks.
- ❌ We cannot say: GPart is not a better composition base. That hypothesis is
  untested here, and it remains the most plausible place for GPart's structural
  advantage to show up.

See `00-claims.md`, "claims we do NOT make", row 4.

## 4. Figures

| figure | shows |
|---|---|
| `results/figures/horizontal_full_prompt_style.png` | style rows at the full prompt |
| `results/figures/horizontal_stacking_p0.png` | base / kohya stack / matched stack / both GPart stacks |
