# What this repo does and does not claim

Written first, deliberately. Every public statement should be traceable to a row
here. If a sentence in the README or a paper draft is not supported by this list,
it is overreach.

The list is split in two because the two halves behave differently. **Robust**
claims survived every control we ran. **Conditional** claims are true on the
configurations we measured and flip when the configuration changes — they are
reported with their conditions attached, and quoting one without its condition
is a misquote.

## Robust claims

| # | Claim | Evidence |
|---|---|---|
| R1 | The open-box "GPart beats LoRA" result on this task is explained by the training recipe, not by the parameterization. The shipped LoRA ran at lr 1e-4 and reached 1/33 of GPart's per-weight displacement. | `02-results-character.md` §1 |
| R2 | Layer coverage is not the explanation, at least for the character task: an all-linear GPart at matched coverage scores 0.40 vs 0.43 for attn2-only. | `02-results-character.md` §2 |
| R3 | At equal image displacement, GPart produces lower saturation and lower edge energy than LoRA — in the style task, the stacking task, and the character task. | `03-results-style-stacking.md` §1 |
| R4 | Under-training removes identity tags non-uniformly: at 400 steps the hair colour and eye colour transfer but the accessories score 0.00; they appear by 800 steps. | `02-results-character.md` §4 |
| R5 | Magnitude is not the explanation either. A community kohya style LoRA (rank 32, 722 modules, lr 1e-4) has ‖ΔW‖ **6.8× smaller** than our self-trained r8 (6.50 vs 44.2) over essentially the same module set (722 vs 700), and is a *better* style adapter. So "the self-trained LoRA lost because it was narrow or under-trained" is false: direction matters more than displacement. | `03-results-style-stacking.md` §5 |
| R6 | Stacking is where the families separate, and not in the direction the parameterization literature predicts. The kohya style LoRA is the best *single* style adapter we measured (SWD 3.336, |z| 0.616) and degrades **monotonically** when stacked with a character adapter (SWD 4.67 / 5.01 / 5.56 at α 1.0 / 1.5 / 2.0). The GPart style adapter stays at 3.59–3.71 against every character adapter tried. | `03-results-style-stacking.md` §5 |
| R7 | In our `stateless` backend, per-step cost scales with **covered weights**, almost independently of `d`: 2184 M → 20.5 s/step, 545.6 M → 8.11 s/step, and `d` 23.2 M → 50 M costs only +13%. | `04-cost-model.md` |

## Conditional claims

| # | Claim | Holds when | Stops holding when | Evidence |
|---|---|---|---|---|
| K1 | LoRA binds character identity better than GPart (0.70 vs 0.57 at each method's own optimum). | The prompt is structurally demanding — full body, crowd, explicit composition. p0 is such a prompt. | Any simpler framing. On a four-framing holdout set the gap is 0.55 vs 0.50, and a paired bootstrap puts it across zero. | `02-results-character.md` §5 |
| K2 | At equal displacement, LoRA and GPart are indistinguishable on identity. | Displacement is matched (‖ΔW‖ 68.8 both sides). | GPart is given ~1.8× LoRA's displacement, i.e. its own optimum at α 1.3: then GPart leads 0.63 vs 0.50, CI [+0.025, +0.233]. | `02-results-character.md` §5 |
| K3 | GPart learns the target style better than LoRA. | "Better" = whole-image colour statistics (|z|: 0.408 vs 0.575/0.652). | "Better" = patch-level texture (SWD). There the hybrid GPart-style + LoRA-character arm wins at 3.585, and the kohya style LoRA alone is best of all at 3.336. | `03-results-style-stacking.md` §5 |
| K4 | GPart is the better composition base. | Plain additive merge, with a style adapter on the outside. | **Not established as a mechanism.** The kohya-collapse-vs-GPart-stability contrast is real and measured; the subspace-collision explanation for it is a hypothesis and is untested. | `03-results-style-stacking.md` §5 |

The through-line in K1–K2 is worth stating plainly, because it is the most
transferable thing in this repo:

> **A 0.13 gap in either direction is what you get by changing the test set, not
> by changing the method.** Swapping the prompt set moved the identity gap by
> 0.22; adding a style adapter moved it by another 0.09. Both single-metric
> "wins" reported here were reproduced by changing inputs, at the same magnitude.

## Claims we explicitly do NOT make

| ✗ | Why not |
|---|---|
| "Low-rank is better than full-rank PEFT in general." | One character, one style, one base model, one metric family. |
| "GPart cannot do X." | We tested one configuration family. `d`, lr, and target-module choice were not swept beyond the two points listed in the protocol. |
| "GPart is slower than LoRA." | The 4.2× number is specific to this repo's stateless backend. Upstream GPart materializes the assignment table; we could not fit that in 8 GB. Stated as an implementation property, not a method property. |
| "LoRA is a better base for adapter composition." | We tested plain additive merging only. The one measured composability difference (§R6) points the other way, and it is a difference between *specific adapters*, not between the two parameterizations. |
| "GPart's advantage is composability." | That is the leading hypothesis for R6. It was not tested against an alternative (e.g. that the kohya adapter is simply saturated at the α we stacked it at, or that its style and our character share training data). |
| "The tag scores measure image quality." | They measure booru-tagger agreement with the character's canonical tags. A higher score can come with worse images; that is exactly what the α sweep shows. |
| "Style metric X is the right one." | \|z\| and SWD disagree on the ranking of the same arms, at the same magnitude as the differences between arms. Report both, or neither. |

## The one-sentence version

> In a controlled comparison on one SDXL character and one style, matching the
> training recipe removes the claimed advantage of full-rank partitioning over
> low-rank adaptation; the residual difference is a lower-saturation,
> lower-detail change at equal image displacement, and on the identity metric the
> two are indistinguishable whenever displacement is held equal.

## The one-sentence version, for people who will only read the number

> At matched recipe, matched coverage and matched displacement, GPart and LoRA
> tie on character identity; GPart moves the global palette more and the local
> texture less; and a stacked GPart style adapter is the only style adapter we
> found that does not lose quality when a character adapter is added.
