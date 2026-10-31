# Release checklist

Things to verify before this goes public. Nothing here is a formality — items 1
and 2 can block publication entirely.

## 1. Character and style IP

The source images for `aima` and `reserve1999` are third-party artwork, and the
characters may belong to a rights-holder.

- [ ] Identify the origin of the `aima` character (design, franchise). If the
      design is from a commercial/gacha property, publish **no images** derived
      from it, or publish only heavily degraded thumbnails for illustration.
- [ ] Same for `juxueli` before that arm is released.
- [ ] Confirm `reserve1999` is a style, not an artist whose style the repo would
      be redistributing. Naming a living artist as a style target is itself a
      reputational and licensing problem — rename it to a neutral descriptor.
- [ ] **Do not redistribute the 21 + 76 source images.** They are not ours. The
      repo should describe how to build a comparable dataset, not ship one.

Mitigation if any of the above is unresolved: ship the code, protocol, and
numbers; omit `results/figures/`. The comparison does not depend on the reader
seeing the images, but its persuasiveness does — so decide deliberately.

## 2. Base model license

- [ ] Read the NoobAI-XL v1.1 license card. SDXL derivatives vary: some forbid
      certain content categories, some impose attribution or share-alike, some
      restrict commercial use of *outputs*.
- [ ] Decide whether derived adapter weights can be redistributed. Recommendation:
      **do not ship LoRA/GPart checkpoints.** They are 20–45 MB each, of no use
      without the exact base checkpoint, and they inherit the base license.
      Ship the configs and the recipe instead.
- [ ] Verify that publishing tag-score tables derived from the model is
      unproblematic (it normally is — that is just evaluation output).

## 3. Third-party code

- [ ] `third_party/GPart/` — Apache-2.0 per its `LICENSE`; `third_party/GPart/peft/LICENSE`
      is the upstream PEFT license. Both are compatible with shipping the
      repository under Apache-2.0. Keep both license files in place.
- [ ] Our patches to `third_party/` (`partition_mode` backend) must be called
      out. Apache-2.0 requires marking modified files. Add a `NOTICE` line or a
      header comment on each patched file listing the modification.
- [ ] `SMILINGWOLF WD-v1-4-moat-tagger-v2` — check the model card license before
      any script auto-downloads it. The repo should reference the download, not
      vendor the ONNX.
- [ ] diffusers / safetensors / PyTorch — Apache-2.0 or BSD; fine, but list them
      in `THIRD_PARTY.md`.

## 4. Claims review

Read `docs/00-claims.md` top to bottom against the README and any announcement
text.

- [ ] No sentence anywhere says "low-rank is better" without the
      "in this experiment" qualifier.
- [ ] No sentence says "GPart is 4× slower" without "in our stateless backend".
      The cost claim must name the implementation.
- [ ] No sentence says "GPart cannot compose" — the composition test is additive
      only.
- [ ] The α-sweep caveat is attached to the 0.57 number: the peak is bought with
      colour cast and lost detail.
- [ ] The one-character limitation appears next to the headline table, not only
      in `05-limitations.md`.

## 5. Reproducibility

- [ ] Remove or relocate the remaining hardcoded paths:
      - `src/train_sdxl.py:33` — `--data` default
      - `src/prep_caption_variants.py:73-74` — `--src` / `--dst` defaults
      - `src/style_stats.py:8` — docstring example path (cosmetic)
      - `src/sample.py:4-6` — docstring example path (cosmetic)
      - `scripts/tag_eval.py:28` — WD14 ONNX path; make it a `--model` argument
- [ ] `requirements.txt` pins versions that actually work together
      (torch / diffusers / transformers / peft / onnxruntime / pillow).
- [ ] Every config in `configs/` has been run at least once from the shipped tree.
- [ ] `scripts/reproduce*.sh` runs end-to-end on a clean machine, or is clearly
      marked as not verified.
- [ ] State the GPU and VRAM the timings came from. Timings without hardware are
      meaningless.

## 6. Repository hygiene

- [ ] `.gitignore` covers `outputs/`, `runs/`, `*.safetensors`, `*.pt`, `*.onnx`,
      latent caches, `__pycache__`, editor droppings.
- [ ] No API keys, HF tokens, ComfyUI hostnames, or absolute Windows paths in
      committed files.
- [ ] `LICENSE` present and consistent with `THIRD_PARTY.md`.
- [ ] `README.md` quick start produces something on a machine that is not the one
      it was written on.
- [ ] Large files: no file over ~10 MB. If any figure is bigger, downscale it.

## 7. Framing

The reason this repository is worth publishing is not the verdict on GPart. It
is:

1. A matched-recipe protocol for PEFT comparisons that is easy to port to other
   methods, with the specific confounds that produced a 33× displacement gap
   spelled out.
2. A measured cost model for an implementation that fits in 8 GB, including the
   finding that the implicit-partition hash is *not* the bottleneck.
3. A negative result stated as a negative result.

Lead with those. Do not lead with "LoRA beats GPart".
