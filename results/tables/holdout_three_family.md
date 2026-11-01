# Three adapter families on a four-framing holdout set

4 prompts × 3 seeds = 12 images per arm. Euler ancestral, `trailing` spacing,
30 steps, CFG 5.0, 832×1216, bf16. `identity` = mean over 10 booru tags of each
tag's hit rate at p ≥ 0.35 (`src/eval/tag_eval.py`). `|z|` = mean absolute
z-score over six whole-image colour statistics against the style reference set.
`SWD` = sliced Wasserstein distance over 32×32 Oklab patch clouds. Both style
metrics are gated on the image actually containing a subject (WD14 `1girl` ≥ 0.5);
`kept` is how many of the 12 images passed that gate.

Lower is better for `|z|` and `SWD`; higher is better for `identity`.

| arm | family | style adapter | char adapter | ‖ΔW‖ style / char | identity | kept | \|z\| | SWD |
|---|---|---|---|---|---|---|---|---|
| base | — | — | — | 0 / 0 | 0.02 | 10/12 | 1.459 | 8.711 |
| **gpart_stack** | GPart | gpart attn2 @0.469 | gpart attn2 @1.3 | 44.2 / 101.9 | **0.63** | 12/12 | **0.408** | 3.713 |
| gpart_stack_a088 | GPart | gpart attn2 @0.469 | gpart attn2 @0.88 | 44.2 / 68.8 | 0.42 | 12/12 | 0.480 | 3.658 |
| gpart_stack_a100 | GPart | gpart attn2 @0.469 | gpart attn2 @1.0 | 44.2 / 78.4 | 0.54 | 12/12 | 0.475 | 3.901 |
| gpart_stack_a160 | GPart | gpart attn2 @0.469 | gpart attn2 @1.6 | 44.2 / 125.4 | 0.56 | 12/12 | 0.434 | 3.569 |
| hybrid_gpstyle_lochar | hybrid | gpart attn2 @0.469 | lora matched @1.0 | 44.2 / 68.8 | 0.50 | 12/12 | 0.575 | **3.585** |
| hybrid_kohstyle_gpchar | hybrid | kohya v3 @1.0 | gpart attn2 @1.3 | 6.50 / 101.9 | 0.52 | 12/12 | 0.530 | 5.103 |
| lora_stack_koh100 | LoRA | kohya v3 @1.0 | lora matched @1.0 | 6.50 / 68.8 | 0.57 | 12/12 | 0.652 | 4.666 |
| lora_stack_koh150 | LoRA | kohya v3 @1.5 | lora matched @1.0 | 9.75 / 68.8 | 0.47 | 12/12 | 0.583 | 5.011 |
| lora_stack_koh200 | LoRA | kohya v3 @2.0 | lora matched @1.0 | 13.00 / 68.8 | 0.47 | 12/12 | 0.726 | 5.559 |
| lora_stack_r8 | LoRA | lora matched r8 @0.469 | lora matched @1.0 | 44.2 / 68.8 | 0.50 | 12/12 | 0.721 | 4.863 |
| kohya_style_solo | solo | kohya v3 @1.5 | — | 9.75 / 0 | — | 12/12 | 0.616 | 3.336 |

Reading notes:

- **The identity column does not support a ranking.** Paired bootstrap over the
  12 (prompt, seed) cells: GPart@1.3 − LoRA is +0.133 with CI [+0.025, +0.233],
  but at *equal displacement* (GPart @0.88, ‖ΔW‖ 68.8 = LoRA's) the difference is
  −0.083 with CI [−0.225, +0.058] — not distinguishable from zero. See
  `docs/02-results-character.md` §7.
- **`|z|` prefers GPart, `SWD` prefers the hybrid.** The two style metrics
  disagree and neither is a ground truth; `SWD` is patch-level and therefore
  weights local brushwork, `|z|` is whole-image and weights the global palette.
- **`kohya_style_solo` is the best style number in the table but has no
  character adapter**, so it is not a candidate configuration — it is the
  ceiling for "style alone". Its collapse under stacking is the finding; see
  `docs/03-results-style-stacking.md` §5.
- kohya v3 ‖ΔW‖ is 6.50, computed as `up @ down * (alpha/rank)` summed in
  quadrature over 722 module pairs, from the safetensors metadata of
  `reserve1999_style_v3.safetensors` (rank 32, 1310 steps, lr 1e-4). It is
  6.8× *smaller* than the self-trained r8 style LoRA's 44.2 despite covering
  essentially the same modules (722 vs 700).
