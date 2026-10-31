#!/usr/bin/env bash
# Sample the fairness-control LoRA (charrun_2026-10-31/aima_lora_r8_matched).
#
# Same recipe as final_test_2026-10-30.sh: same prompts.jsonl, seeds, sampler,
# spacing, negative, resolution. The only variable versus gp_char is the adapter.
#
# scale 1.0 = the kohya LoRA's nominal 1.0: this adapter was trained with
# lora_alpha=4 / rank=8, the same 0.5 scaling kohya used, so --alpha 1.0 here
# is directly comparable to `lo_char` and to `gp_char`.
#
# 6 images, ~4 min.

set -u
cd "$(dirname "$0")/.."

PY="D:/AI/lora/venv/Scripts/python.exe"
OUT="outputs/final_2026-10-30"
ADAPTER=outputs/charrun_2026-10-31/aima_lora_r8_matched/adapter

NEG="1boy, 2boys, multiple girls, worst quality, low quality, normal quality, lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, username, text, jpeg artifacts, censored, mosaic censoring, nude, topless, underwear"

PYTHONIOENCODING=utf-8 "$PY" src/sample.py \
    --steps 30 --cfg 5 --sched euler_a --spacing trailing \
    --seeds 310977364113511 12345 67890 --size 832 1216 \
    --prompts-file "$OUT/prompts.jsonl" --negative "$NEG" \
    --adapter-dir "$ADAPTER" --alpha 1.0 \
    --output "$OUT/lo_char_matched" --tag lo_char_matched

echo "==== MATCHED LORA SAMPLE DONE  $(date +%H:%M:%S) ===="
