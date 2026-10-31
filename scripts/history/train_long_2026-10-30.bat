@echo off
rem Follow-up to longrun_2026-10-29, per the 2026-10-29 test-group review:
rem   * d50m at lr 5e-4 ended at ||dW|| 94.2 and kept growing (grad_norm ~1e-3,
rem     weight_decay 0, so nothing pushed back). Step 600 (||dW|| = 70) looked best,
rem     so 70 is the ceiling for the retrain.
rem   * 1/2  retrain d50m at lr 1e-4 with --theta-norm-max 70.
rem   * 2/2  same run with --targets attn2.to_out.0 only - isolates whether the
rem     update needs q/k/v or lives entirely in the output projections.
rem Everything else identical to yesterday so the comparison is controlled.
cd /d D:\Gpart
set LOG=outputs\longrun_2026-10-30\console.log
if not exist outputs\longrun_2026-10-30 mkdir outputs\longrun_2026-10-30
echo ==== LONG RUN START %DATE% %TIME% ==== >> %LOG%

echo ==== RUN 1/2  d=50000000  steps=1200  lr=1e-4  cap=70  targets=attn2 qkv+out ==== >> %LOG%
D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter gpart --d 50000000 ^
  --targets attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0 ^
  --scope transformer_block --layers-pattern transformer_blocks --backend stateless ^
  --data D:/AI/lora/datasets_reserve1999_clean/6_reserve1999 ^
  --steps 1200 --grad-accum 2 --lr 1e-4 --lr-schedule constant --warmup 50 ^
  --theta-norm-max 70 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/longrun_2026-10-30/d50m_lr1e4_cap70 >> %LOG% 2>&1

echo ==== RUN 2/2  d=50000000  steps=1200  lr=1e-4  cap=70  targets=attn2.to_out.0 ==== >> %LOG%
D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter gpart --d 50000000 ^
  --targets attn2.to_out.0 ^
  --scope transformer_block --layers-pattern transformer_blocks --backend stateless ^
  --data D:/AI/lora/datasets_reserve1999_clean/6_reserve1999 ^
  --steps 1200 --grad-accum 2 --lr 1e-4 --lr-schedule constant --warmup 50 ^
  --theta-norm-max 70 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/longrun_2026-10-30/d50m_lr1e4_cap70_outonly >> %LOG% 2>&1

echo ==== ALL DONE %DATE% %TIME% ==== >> %LOG%
