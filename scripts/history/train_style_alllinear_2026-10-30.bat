@echo off
rem Style retrain with full linear coverage inside the transformer blocks.
rem
rem Change vs longrun_2026-10-29/d23m: --targets goes from attn2-only to attn1 +
rem attn2 + feed-forward. The suffix list below matches exactly the 700 modules
rem (2184.0M adapted weights) that param_budget.py calls [attn+ff], and that is the
rem same module set the reference kohya LoRA reserve1999_style_v3 covers (it also
rem carries 22 proj_in/proj_out tensors, 23.7M weights - those sit outside the
rem "transformer_blocks" container, so partition_scope=transformer_block cannot
rem resolve a block id for them and would raise; leaving them out costs 1% of the
rem adapted weight mass).
rem d stays at 23224320 so targets is the only changed variable.
cd /d D:\Gpart
set LOG=outputs\style_alllinear_2026-10-30\console.log
if not exist outputs\style_alllinear_2026-10-30 mkdir outputs\style_alllinear_2026-10-30
echo ==== STYLE ALL-LINEAR START %DATE% %TIME% ==== >> %LOG%

rem 600 steps, not 1200: this backend recomputes Delta W every forward over 2184M
rem adapted weights, measured 18.6 s/step, so 1200 would be ~6 h. The visual optimum of
rem the 2026-10-29 style run was at step 600 anyway; extend only if this shows promise.
echo ==== RUN 1/1  r1999  d=23224320  targets=attn1+attn2+ff  steps=600  lr=5e-4 ==== >> %LOG%
D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter gpart --d 23224320 ^
  --targets to_q to_k to_v to_out.0 ff.net.0.proj ff.net.2 ^
  --scope transformer_block --layers-pattern transformer_blocks --backend stateless ^
  --data D:/AI/lora/datasets_reserve1999_clean/6_reserve1999 ^
  --steps 600 --grad-accum 2 --lr 5e-4 --lr-schedule constant --warmup 50 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/style_alllinear_2026-10-30/r1999_d23m >> %LOG% 2>&1

echo ==== ALL DONE %DATE% %TIME% ==== >> %LOG%
