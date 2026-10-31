# What this repo does and does not claim

Written first, deliberately. Every public statement should be traceable to a row
here. If a sentence in the README or a paper draft is not supported by this list,
it is overreach.

## Claims we make

| # | Claim | Evidence |
|---|---|---|
| C1 | The open-box "GPart beats LoRA" result on this task is explained by the training recipe, not by the parameterization. The shipped LoRA ran at lr 1e-4 and reached 1/33 of GPart's per-weight displacement. | `docs/02-results-character.md` §1 |
| C2 | With recipe, layer coverage, capacity and each method's own optimal strength all aligned, LoRA scores higher on character-identity tags (0.70 vs 0.57). | `docs/02-results-character.md` §3 |
| C3 | Layer coverage is not the explanation: an all-linear GPart at matched coverage scores 0.40 vs 0.43 for attn2-only. | `docs/02-results-character.md` §2 |
| C4 | The GPart strength working point was swept (α = 1.0/1.3/1.5/1.8); 1.3–1.5 is a genuine peak, and it is still below LoRA's optimum. | `docs/02-results-character.md` §3 |
| C5 | At equal image displacement (dRGB), GPart produces lower saturation and lower edge energy than LoRA — in the style task, the stacking task, and the character task. | `docs/03-results-style-stacking.md` |
| C6 | Under-training removes identity tags non-uniformly: at 400 steps the hair colour and eye colour transfer but the accessories score 0.00; they appear by 800 steps. | `docs/02-results-character.md` §4 |
| C7 | In our `stateless` backend, per-step cost scales with **covered weights**, almost independently of `d`: 2184 M → 20.5 s/step, 545.6 M → 8.11 s/step, and `d` 23.2 M → 50 M costs only +13%. | `docs/04-cost-model.md` |

## Claims we explicitly do NOT make

| ✗ | Why not |
|---|---|
| "Low-rank is better than full-rank PEFT in general." | One character, one style, one base model, one metric family. |
| "GPart cannot do X." | We tested one configuration family. `d`, lr, and target-module choice were not swept beyond the two points listed in the protocol. |
| "GPart is slower than LoRA." | The 4.2× number is specific to this repo's stateless backend. Upstream GPart materializes the assignment table; we could not fit that in 8 GB. Stated as an implementation property, not a method property. |
| "LoRA is a better base for adapter composition." | We tested plain additive merging only. GPart's actual selling point is *exact linear composition in shared coordinates* under non-additive operations (TIES, orthogonalization). That was not built. |
| "The tag scores measure image quality." | They measure booru-tagger agreement with the character's canonical tags. A higher score can come with worse images; that is exactly what the α sweep shows. |

## The one-sentence version

> In a controlled comparison on one SDXL character and one style, matching the
> training recipe removes the claimed advantage of full-rank partitioning over
> low-rank adaptation, and the remaining difference shows up as lower saturation
> and detail retention at equal image displacement.
