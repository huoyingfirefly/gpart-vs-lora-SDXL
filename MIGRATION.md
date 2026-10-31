# Migration map — from the working tree to this repository

## Status (2026-10-31)

Done:

- file copies per §1
- path removals per §2 (`--data`, `--src`, `--dst` now required; `tag_eval.py`
  takes `--model`)
- `--partition-mode` wired into the CLI (it existed in the config/layer/model
  layers but was unreachable)
- `--config` two-phase parse in `src/train.py`, so `configs/*.yaml` are live
  defaults rather than documentation
- figure renames per §3, plus `results/tables/` (csv + md)
- `third_party/GPart/data/math_eval/` deleted (8.9 MB of the paper's math
  benchmark, irrelevant here)
- `scripts/history/` populated as the audit trail
- `scripts/train_gpart.py` — the one-click GPart trainer (audit → variants →
  train → manifest), verified end to end with a 4-step smoke run
- `quickstart/` — the clone-to-first-image path, `01-install` through
  `06-troubleshooting`
- `gui/app.py` — GPart Studio. The train tab shells out to
  `scripts/train_gpart.py` (so the CLI and the UI cannot drift); the sample tab
  is a rewrite of the working-tree `gui.py`, de-hardcoded, and **fixed to
  `euler_a` + `trailing`** — the old one used `leading`, which is not the
  protocol in `docs/01-protocol.md` §1

Not done:

- `reproduce_character.sh` / `reproduce_style.sh` / `build_report.sh` (§4) —
  these cover the LoRA arm, the seed grid and the scoring loop, which
  `train_gpart.py` does not
- modification headers on the patched `third_party` files (`THIRD_PARTY.md` lists
  them, but Apache-2.0 §4(b) wants the notice *in* the files)
- every item in `docs/06-release-checklist.md`
- nothing has been committed; the tree is a working copy

The experiments were run out of a single flat working directory (`D:\Gpart`)
with one-off shell scripts named by date. This file says where each piece goes
and what has to change before it is publishable.

## 1. File-by-file

| working-tree path | destination | action |
|---|---|---|
| `src/train_sdxl.py` | `src/train.py` | rename; drop `--data` default (§2) |
| `src/sample.py` | `src/sample.py` | clean docstring example paths |
| `src/data.py` | `src/data.py` | as-is |
| `src/prep_caption_variants.py` | `src/prep/caption_variants.py` | move; drop `--src`/`--dst` defaults |
| `scripts/tag_eval.py` | `src/eval/tag_eval.py` | move; `MODEL` → `--model` arg |
| `src/style_stats.py` | `src/eval/style_stats.py` | move; docstring only |
| `src/bench_d.py` | `src/bench/bench_d.py` | move (kept separate — it measures `d`, the other measures phases) |
| `src/profile_train_step.py` | `src/bench/profile_train_step.py` | move |
| `src/param_budget.py` | `src/bench/param_budget.py` | move; it produces the covered-weight table |
| `src/check_adapter.py` | `src/eval/check_adapter.py` | move; round-trip / displacement verification |
| `src/convert_ckpt.py` | `src/prep/convert_unet.py` | move; single-file → diffusers UNet |
| `scripts/compose_alpha_sheet.py` | `src/figures/alpha_sheet.py` | move; de-hardcode row paths |
| `scripts/compose_stack_sheet.py` | `src/figures/stack_sheet.py` | move; de-hardcode row paths |
| `scripts/compose_final.py` | `src/figures/final_sheet.py` | move |
| `scripts/md2html.py` | `src/figures/md_to_html.py` | move |
| `third_party/GPart/` | `third_party/GPart/` | copy; add modification headers |
| `outputs/final_2026-10-30/prompts.jsonl` | `configs/prompts/p0_p1.jsonl` | copy; rename prompt ids |
| `outputs/final_2026-10-30/prompts_p2.txt` | `configs/prompts/p2_heldout.jsonl` | wrap as jsonl |
| `outputs/final_2026-10-30/横向_*.png` | `results/figures/*.png` | copy; ASCII filenames (§3) |
| `D:/Gpart/src/gui.py` | `gui/app.py` | rewrite; drop 5 hardcoded paths, fix `leading` → `trailing` |

## 2. Hardcoded paths to remove

Only five, all in argument defaults or docstrings:

| file:line | what | fix |
|---|---|---|
| `src/train_sdxl.py:33` | `--data` default `D:/AI/lora/datasets_reserve1999_clean/6_reserve1999` | make `--data` required |
| `src/prep_caption_variants.py:73` | `--src` default | make required |
| `src/prep_caption_variants.py:74` | `--dst` default | make required |
| `scripts/tag_eval.py:28` | `MODEL = Path("D:/AI/models/wd14/...")` | `--model` argument |
| `src/style_stats.py:8`, `src/sample.py:4-6` | docstring examples | rewrite with relative paths |

Nothing else in `src/` resolves a `D:\` path at runtime. Checked with
`grep -nE '[A-Za-z]:[/\\\\]' src/ scripts/`.

## 3. Filenames

The working tree uses Chinese filenames for the contact sheets (`横向_*.png`).
These render as mojibake in some terminals and in GitHub's file listing on
Windows clients. Map them to ASCII before publishing:

| original | release name |
|---|---|
| `横向_完整提示词_角色.png` | `horizontal_full_prompt_character.png` |
| `横向_完整提示词_画风.png` | `horizontal_full_prompt_style.png` |
| `横向_叠加_p0.png` | `horizontal_stacking_p0.png` |
| `横向_全参角色_alpha扫描.png` | `horizontal_alpha_sweep_top.png` |
| `横向_全参角色_alpha扫描_下半.png` | `horizontal_alpha_sweep_bottom.png` |

The docs already reference the ASCII names.

## 4. Scripts

The `scripts/*.sh` / `*.bat` files are the actual experiment log, one per dated
run. They are useful as provenance but unusable as-is (absolute paths, dependency
on intermediate state).

- Keep them in `scripts/history/` verbatim as an audit trail.
- Write **three** clean entry points that a stranger can run:
  `scripts/reproduce_character.sh`, `scripts/reproduce_style.sh`,
  `scripts/build_report.sh`.
- Do not pretend the historical scripts are reproducible. Label them "as-run
  log, not maintained".

## 5. What does not move

| | why |
|---|---|
| `outputs/` (checkpoints, images, logs) | hundreds of MB; license-inherited |
| `data/` (latents, converted UNet) | derived from the base model and from third-party art |
| `outputs/**/samples_*.png` | see `docs/06-release-checklist.md` §1 |
| `_samplercheck/`, `gui_samples/`, `comfy_*` | one-off diagnostics; mention them in the docs, ship nothing |
| `TASKS_NEXT.md` | internal planning |

## 6. Order of work

1. Copy `third_party/`, `src/`, `scripts/tag_eval.py` per §1.
2. Strip the five hardcoded paths (§2). This is the only real code change.
3. Add `--partition-mode` to `src/train.py` — the backend is wired end-to-end but
   has no CLI entry, so it is currently unreachable.
4. Run `scripts/reproduce_character.sh` on a clean checkout; it is the gate.
5. Rename figures (§3), copy `results/tables/`, write the three clean scripts.
6. Work the checklist in `docs/06-release-checklist.md`.
