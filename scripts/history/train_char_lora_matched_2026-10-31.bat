@echo off
rem The fairness control for the character comparison.
rem
rem The shipped LoRA (aima_noobai_v1) was trained by kohya: lr 1e-4, cosine,
rem batch 1x4, 1024 buckets. The GPart character run used this repo: lr 5e-4,
rem constant, batch 1x2, max_res 1536. In image exposures the two are nearly
rem equal (2120 vs 2400 samples), but lr differs by 5x and that, not the data,
rem is what produced the 33x displacement gap.
rem
rem This trains a LoRA with THIS repo, at the GPart settings, on the identical
rem latent/caption cache, so the only remaining difference between the two
rem adapters is the parameterization itself (low-rank vs full-rank spherical).
rem
rem Rank: kohya r=16 all-linear came out at 42.7 M params. r=8 here gives
rem ~21.4 M, which brackets GPart's d=23.2 M instead of doubling it. alpha=4
rem keeps the 0.5 scaling kohya used (16/8), so --rank/--alpha is not a
rem second variable.
rem
rem Sample it with --adapter-dir (peft), not --lora (kohya names).
rem Risk: lr 5e-4 is aggressive for a LoRA this small; if the loss goes NaN,
rem check the train.jsonl and fall back to 2.5e-4.
cd /d D:\Gpart
set LOG=outputs\charrun_2026-10-31\console.log
if not exist outputs\charrun_2026-10-31 mkdir outputs\charrun_2026-10-31
echo ==== MATCHED LORA START %DATE% %TIME% ==== >> %LOG%

D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter lora --rank 8 --alpha 4 ^
  --targets to_q to_k to_v to_out.0 ff.net.0.proj ff.net.2 ^
  --scope transformer_block --layers-pattern transformer_blocks ^
  --data D:/AI/lora/datasets_aima/10_aima_var ^
  --steps 1200 --grad-accum 2 --lr 5e-4 --lr-schedule constant --warmup 50 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/charrun_2026-10-31/aima_lora_r8_matched >> %LOG% 2>&1

echo ==== ALL DONE %DATE% %TIME% ==== >> %LOG%
