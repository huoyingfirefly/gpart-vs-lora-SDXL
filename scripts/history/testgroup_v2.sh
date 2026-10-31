#!/usr/bin/env bash
# Strength-matched comparison on a fresh prompt set.
#   GPart d23m at alpha=1.0  vs  kohya LoRA reserve1999_style_v3 at scale 1.5
# (1.5 is the scale at which the LoRA's trigger-prompt style matched GPart alpha=1.0
#  in the 2026-10-29 test group; 0.75 was too weak to be a fair comparison)
# 4 prompt pairs, each run with and without the trigger token, 3 unseen seeds.
set -u
cd /d/Gpart || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT=outputs/testgroup/v2
RUN=outputs/longrun_2026-10-29
COMMON="--steps 28 --cfg 5 --sched euler --spacing leading"
# one seed x 8 prompts: each sampler process needs ~11 GB of system commit, so keep the
# set small enough to run one process at a time without touching the pagefile
SEEDS="31337"
mkdir -p "$OUT"

run() {
  local tag="$1"; shift
  echo "==== $tag ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "$@" \
    --prompts-file outputs/testgroup/prompts_v2.jsonl --seeds $SEEDS $COMMON --grid \
    --output "$OUT/$tag" --tag "$tag" || echo "FAILED $tag"
}

run base
run gpart_a100 --adapter-dir "$RUN/d23m/adapter" --alpha 1.0
run lora_s15 --lora D:/AI/models/loras/reserve1999_style_v3.safetensors --lora-scale 1.5

echo "==== TEST GROUP V2 DONE ===="
