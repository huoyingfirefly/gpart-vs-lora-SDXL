#!/usr/bin/env bash
# Adapter test group for the finished longrun checkpoints.
# base / d23M / d50M(alpha 1.0, 0.5, 0.25) on 4 prompts x 2 seeds,
# plus a checkpoint-progress sweep (200 / 600) on 2 prompts x 1 seed.
set -u
cd /d/Gpart || exit 1

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT=outputs/testgroup
RUN=outputs/longrun_2026-10-29
COMMON="--steps 28 --cfg 5 --sched euler --spacing leading"
mkdir -p "$OUT"

run() {
  local tag="$1"; shift
  echo "==== $tag ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "$@" \
    --prompts-file "$OUT/prompts.jsonl" --seeds 20261029 777 $COMMON --grid \
    --output "$OUT/$tag" --tag "$tag" || echo "FAILED $tag"
}

run base
run d23m_a100 --adapter-dir "$RUN/d23m/adapter" --alpha 1.0
run d50m_a100 --adapter-dir "$RUN/d50m/adapter" --alpha 1.0
run d50m_a050 --adapter-dir "$RUN/d50m/adapter" --alpha 0.5
run d50m_a025 --adapter-dir "$RUN/d50m/adapter" --alpha 0.25

runckpt() {
  local tag="$1"; shift
  echo "==== $tag ===="
  PYTHONIOENCODING=utf-8 "$PY" src/sample.py "$@" \
    --prompts-file "$OUT/prompts_ckpt.jsonl" --seeds 20261029 $COMMON --grid \
    --output "$OUT/$tag" --tag "$tag" || echo "FAILED $tag"
}

runckpt ckpt200 --adapter-dir "$RUN/d50m/step_00200" --alpha 1.0
runckpt ckpt600 --adapter-dir "$RUN/d50m/step_00600" --alpha 1.0

echo "==== TEST GROUP DONE ===="
