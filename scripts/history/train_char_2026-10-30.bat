@echo off
rem Character run, aima. Same protocol as the 2026-10-29 style long run so the two
rem are directly comparable: lr 5e-4 constant, warmup 50, 1200 steps, no norm cap.
rem Only the dataset and the output dir change.
rem
rem Known expectation from 2026-10-29: a constant Delta W cannot gate on the trigger,
rem so the question is whether 8x more steps pulls identity (white/pink gradient bob,
rem black beret, red eyes) out of the dataset-look, or only makes the latter stronger.
rem 63 caption variants over 21 unique images -> 1200 x 2 / 59 ~= 41 epochs, so expect
rem memorisation; checkpoints every 200 steps are the fallback.
cd /d D:\Gpart
set LOG=outputs\charrun_2026-10-30\console.log
if not exist outputs\charrun_2026-10-30 mkdir outputs\charrun_2026-10-30
echo ==== CHAR RUN START %DATE% %TIME% ==== >> %LOG%

echo ==== RUN 1/1  aima  d=23224320  steps=1200  lr=5e-4 constant ==== >> %LOG%
D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter gpart --d 23224320 ^
  --targets attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0 ^
  --scope transformer_block --layers-pattern transformer_blocks --backend stateless ^
  --data D:/AI/lora/datasets_aima/10_aima_var ^
  --steps 1200 --grad-accum 2 --lr 5e-4 --lr-schedule constant --warmup 50 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/charrun_2026-10-30/aima_d23m >> %LOG% 2>&1

echo ==== ALL DONE %DATE% %TIME% ==== >> %LOG%
