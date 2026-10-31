@echo off
cd /d D:\Gpart
set LOG=outputs\longrun_2026-10-29\console.log
echo ==== LONG RUN START %DATE% %TIME% ==== >> %LOG%

echo ==== RUN 1/2  d=23224320  steps=1200  lr=5e-4 constant ==== >> %LOG%
D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter gpart --d 23224320 ^
  --targets attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0 ^
  --scope transformer_block --layers-pattern transformer_blocks --backend stateless ^
  --data D:/AI/lora/datasets_reserve1999_clean/6_reserve1999 ^
  --steps 1200 --grad-accum 2 --lr 5e-4 --lr-schedule constant --warmup 50 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/longrun_2026-10-29/d23m >> %LOG% 2>&1

echo ==== RUN 2/2  d=50000000  steps=1200  lr=5e-4 constant ==== >> %LOG%
D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter gpart --d 50000000 ^
  --targets attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0 ^
  --scope transformer_block --layers-pattern transformer_blocks --backend stateless ^
  --data D:/AI/lora/datasets_reserve1999_clean/6_reserve1999 ^
  --steps 1200 --grad-accum 2 --lr 5e-4 --lr-schedule constant --warmup 50 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/longrun_2026-10-29/d50m >> %LOG% 2>&1

echo ==== ALL DONE %DATE% %TIME% ==== >> %LOG%
