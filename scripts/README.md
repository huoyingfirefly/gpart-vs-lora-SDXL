# `scripts/`

## `train_gpart.py` — the one-click entry point

Dataset folder in, trained adapter out:

```bash
python scripts/train_gpart.py \
    --data  data/10_my_character \
    --name  mychar --trigger mychar \
    --identity "white hair,red eyes,beret" \
    --preset attn2 --steps 1200
```

It audits the dataset, builds the three caption variants, trains with the matched
recipe, writes `run_manifest.json`, and prints the sampling command.

```bash
python scripts/train_gpart.py --help          # all flags
python scripts/train_gpart.py ... --dry-run   # print the commands, run nothing
```

Design choices worth knowing:

- **`--identity` is required, with no default.** `src/prep/caption_variants.py`
  falls back to the *aima* identity set when it is omitted — fine for reproducing
  our run, silently wrong for any other character. The wrapper refuses rather
  than inheriting.
- **`--preset attn2` (280 modules, ~8.1 s/step) vs `alllinear`** (700 modules,
  ~20.5 s/step). `attn2` is the default because it is the better cost/benefit
  point in this repo; `alllinear` matches the LoRA's coverage if you are running
  your own comparison.
- **`--steps 1200` is the matched recipe, not a suggestion.** 800 is the cheapest
  budget that keeps identity accessories (`docs/02-results-character.md` §4);
  below that the tag scores drop non-uniformly.
- **`--extra`** passes anything else through, e.g. `--extra '--theta-norm-max 70'`.
- The audit fails on the three things that make a run silently useless: no
  `*_sdxl.npz`, no `*_te_outputs.npz`, no `.txt` caption. It warns under 10
  samples.

Verified end to end on 2026-11-01: 63 cached samples, 4 steps, adapter + manifest
written in 56 s.

`gui/app.py` (GPart Studio) does not reimplement any of this — its train tab
builds this exact argv and runs it as a subprocess, streaming the log. So a flag
that works here works there, and a bug here is a bug there.

## `history/` — the as-run log, not maintained

One file per dated run, exactly as it was executed. These are the provenance for
every number in `results/`.

**They do not run on your machine.** They contain absolute Windows paths
(`D:\Gpart\...`, `D:\AI\lora\venv\...`), assume intermediate state from earlier
runs, and several of them queue off log markers written by other scripts. They
are here so you can audit *what was actually run* — which flags, which
checkpoints, in which order — not so you can rerun them.

Useful ones if you want to see the exact command behind a row:

| script | row |
|---|---|
| `train_char_gpart_alllinear_2026-11-01.bat` | `gp_char_alllinear` (the coverage test) |
| `train_style_alllinear_2026-10-30.bat` | `gp_style_alllinear` |
| `sample_matched_lora_2026-10-31.sh` | `lo_char_matched` sampling |
| `sweep_gp_char_2026-11-01.sh` | the α 1.0/1.3/1.5/1.8 sweep |
| `exposure_curve_2026-10-31.sh` | the step curve (400/600/800/1200) |
| `final_test_2026-10-30.sh` | the run that produced the headline tables |

## Still missing

- [ ] `reproduce_character.sh` — the full grid (both methods, all seeds, scoring),
      not just one adapter. `train_gpart.py` covers the GPart arm only.
- [ ] `reproduce_style.sh` — same for the style task
- [ ] `build_report.sh` — tables + figures from a completed run

`train_gpart.py` makes "train one GPart adapter" reproducible; the headline table
needs the LoRA arm and the scoring loop too.
