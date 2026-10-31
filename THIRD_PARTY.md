# Third-party code and models

## Vendored source

### GPart / PEFT — `third_party/GPart/`

Vendored from the GPart reference implementation (Samsung), which is built on a
fork of Hugging Face PEFT.

- `third_party/GPart/LICENSE` — Apache License 2.0
- `third_party/GPart/peft/LICENSE` — Apache License 2.0 (upstream PEFT)

**Modifications made in this repository.** Apache-2.0 §4(b) requires modified
files to carry prominent notices. Each of the following files was patched and
carries a `MODIFIED for gpart-vs-lora-SDXL (2026)` notice under its Apache
2.0 header:

| file | change |
|---|---|
| `peft/tuners/gpart/config.py` | added `partition_mode` field (`"splitmix64"` \| `"int32"`) |
| `peft/tuners/gpart/model.py` | threads `partition_mode` through `update_layer` / `Linear` |
| `peft/tuners/gpart/layer.py` | threads `partition_mode` through the autograd path; stores per-adapter mode; extends the backward return tuple |
| `peft/tuners/gpart/grouping.py` | added the int32 murmur-finalizer partition and its dispatch |

No upstream file was removed. The patched tree is drop-in compatible with the
unmodified API.

## Models referenced, not redistributed

| model | use | license to verify |
|---|---|---|
| NoobAI-XL v1.1 (SDXL UNet) | base model for all runs | see the model card on the host it came from |
| `SmilingWolf/wd-v1-4-moat-tagger-v2` | ONNX WD14 booru tagger for identity scoring | see model card |

Neither is included in this repository. `scripts/tag_eval.py` expects the tagger
ONNX at a path you provide; set `--model`.

## Libraries

| package | license |
|---|---|
| PyTorch | BSD-3-Clause |
| diffusers | Apache-2.0 |
| transformers | Apache-2.0 |
| PEFT (vendored) | Apache-2.0 |
| safetensors | Apache-2.0 |
| onnxruntime | MIT |
| Pillow | MIT-CMU |
| NumPy | BSD-3-Clause |

## Data and images

No training images, generated images, or model checkpoints are redistributed.
See `docs/06-release-checklist.md` §1 and §2 for the reasoning and for what must
be verified before any image is published.
