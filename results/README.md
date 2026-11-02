# `results/`

Everything here was produced by the code in `src/` against the base model named
in `docs/01-protocol.md`. Each file is a copy of the numbers quoted in `docs/`;
if they ever disagree, the docs are wrong and this table is right.

## `tables/`

| file | contents | referenced by |
|---|---|---|
| `character.{csv,md}` | every character row: recipe, coverage, α, ‖ΔW‖, RMS, score | `docs/02-results-character.md` |
| `style.{csv,md}` | dRGB / S / V / edge for the five style rows | `docs/03-results-style-stacking.md` §1 |
| `stacking.{csv,md}` | character score + image stats for the four stacking rows | `docs/03-results-style-stacking.md` §2 |
| `holdout_three_family.{csv,md}` | 18 arms on the four-framing holdout set: identity, `\|z\|`, SWD, ‖ΔW‖ per adapter | `docs/02-results-character.md` §5, `docs/03-results-style-stacking.md` §5 |
| `step_curve.{csv,md}` | score by checkpoint step, with the accessory tags | `docs/02-results-character.md` §4 |
| `cost.{csv,md}` | seconds/step per adapter configuration | `docs/04-cost-model.md` §2 |
| `partition_bench.{csv,md}` | the partition-pass micro-benchmark | `docs/04-cost-model.md` §3 |

## `figures/`

ASCII-named copies of the contact sheets. The original files are named in
Chinese; `MIGRATION.md` §3 has the mapping.

| file | shows |
|---|---|
| `horizontal_full_prompt_character.png` | full prompt, 6 seeds: base / kohya LoRA / GPart attn2 / matched LoRA |
| `horizontal_alpha_sweep_top.png` | `p0`, 3 seeds: base / matched LoRA / GPart attn2 (α 1.0) / GPart all-linear (α 1.0) |
| `horizontal_alpha_sweep_bottom.png` | `p0`, 3 seeds: GPart all-linear at α 1.3 / 1.5 / 1.8 |
| `horizontal_full_prompt_style.png` | full prompt, 3 seeds: base / LoRA style / GPart all-linear style / GPart attn2 style |
| `horizontal_stacking_p0.png` | `p0`, 3 seeds: base / kohya stack / matched stack / both GPart stacks |

`readme/` holds the three showcase sheets linked from the README: base model vs
GPart at its best setting — character, style, and the style+character stack —
with no LoRA rows. They are cropped out of the sheets above and re-composited by
`src/figures/readme_sheet.py` at 1000 px JPEG; regenerate with that script rather
than editing the files.

These are the files a reuser who needs a different licensing call would remove
first — `docs/06-release-checklist.md` §1 records the considerations and the
maintainer's decision to ship them. That includes `readme/`, which is more
exposed than the rest because the README is the first thing a visitor sees.

## `logs/`

Not committed. Training logs, validation losses and the as-run shell scripts stay
in the working tree; `scripts/history/` carries the scripts when they are added.

## Empty cells

A blank or `None` cell means the quantity was not measured for that row (e.g.
‖ΔW‖ for the α-sweep rows, which reuse the α 1.0 checkpoint). It does not mean
zero.
