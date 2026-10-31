@echo off
rem Resolves the one confound the report could not: is all-linear style worse than
rem attn2-only style because the FFN placement is bad, or because that checkpoint was
rem simply under-trained?
rem
rem Clean 2x2 on the style domain, one variable per pair:
rem   longrun_2026-10-29/d50m            attn2 only,  d=50.0M, 1200 steps   (exists)
rem   THIS RUN                           attn1+2+ff,  d=50.0M, 1200 steps
rem   style_alllinear_2026-10-30         attn1+2+ff,  d=23.2M,  600 steps   (exists)
rem
rem d=50M matches the reference so capacity is no longer a variable; 1200 steps
rem matches its step count. Everything else (lr 5e-4 constant, warmup 50, grad-accum 2,
rem max_res 1536, noise_offset 0.05, min-snr 5, seed 42) is byte-identical to the
rem 2026-10-29 protocol, so --targets is the only difference from d50m.
rem
rem Cost: measured 23.0 s/step for all-linear at d=23.2M (style_alllinear run);
rem theta grows by 26.8M params -> ~+0.3 GiB optimizer state, peak ~6.5/7.8 GiB.
rem Expect ~7.7 h. Do not run a sampler while this holds the GPU.
cd /d D:\Gpart
set LOG=outputs\style_2x2_2026-10-31\console.log
if not exist outputs\style_2x2_2026-10-31 mkdir outputs\style_2x2_2026-10-31
echo ==== STYLE ALL-LINEAR d50M START %DATE% %TIME% ==== >> %LOG%

D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter gpart --d 50000000 ^
  --targets to_q to_k to_v to_out.0 ff.net.0.proj ff.net.2 ^
  --scope transformer_block --layers-pattern transformer_blocks --backend stateless ^
  --data D:/AI/lora/datasets_reserve1999_clean/6_reserve1999 ^
  --steps 1200 --grad-accum 2 --lr 5e-4 --lr-schedule constant --warmup 50 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/style_2x2_2026-10-31/r1999_alllinear_d50m >> %LOG% 2>&1

echo ==== ALL DONE %DATE% %TIME% ==== >> %LOG%
