#!/usr/bin/env bash
# Strength sweep instead of a single "matched" pair.
#
# Picking one GPart alpha and one LoRA scale and calling them equivalent was a guess
# off thumbnails; alpha and lora-scale are both just gains on Delta W, so the only
# honest way to compare them is a curve: style strength on the trigger prompts
# against content damage on the matched no-trigger prompts. 2 prompt pairs, each with
# and without the trigger, so every strength level yields 2 trigger + 2 no-trigger.
#
# Runs strictly one process at a time: each sampler needs ~11 GB of system commit and
# the machine has ~16 GB free, and orphaned samplers from a stopped task will OOM the
# next one.
set -u
cd /d/Gpart || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT=outputs/testgroup/v2b
RUN=outputs/longrun_2026-10-29
COMMON="--steps 28 --cfg 5 --sched euler --spacing leading"
SEEDS="31337"
mkdir -p "$OUT"

run() {
  local tag="$1"; shift
  echo "==== $tag ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "$@" \
    --prompts-file outputs/testgroup/prompts_v2b.jsonl --seeds $SEEDS $COMMON --grid \
    --output "$OUT/$tag" --tag "$tag" || echo "FAILED $tag"
  sync
}

run base
run gpart_a050 --adapter-dir "$RUN/d23m/adapter" --alpha 0.5
run gpart_a100 --adapter-dir "$RUN/d23m/adapter" --alpha 1.0
run gpart_a150 --adapter-dir "$RUN/d23m/adapter" --alpha 1.5
run lora_s100 --lora D:/AI/models/loras/reserve1999_style_v3.safetensors --lora-scale 1.0
run lora_s150 --lora D:/AI/models/loras/reserve1999_style_v3.safetensors --lora-scale 1.5

echo "==== TEST GROUP V2B DONE ===="
