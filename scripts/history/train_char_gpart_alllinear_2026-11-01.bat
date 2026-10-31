@echo off
rem Round 3, job A (the decisive one): GPart with ALL-LINEAR targets.
rem
rem The only remaining confound in REPORT.md 6.8. The matched LoRA covers
rem 700 modules / 2184 M weights; gp_char covers 280 modules / 545.6 M
rem (attn2 only). That 4x coverage gap favours the LoRA and was OUR choice.
rem
rem This run is byte-identical to charrun_2026-10-30/aima_d23m EXCEPT --targets:
rem same data, same d=23224320, same 1200 steps, lr 5e-4, seed 42, everything.
rem So the only variable is attn2-only vs all-linear.
rem
rem Expected: aima_d23m took 2.67 h for 1200 steps at 280 modules; 700 modules
rem makes the backward pass heavier, so estimate 4-5 h (not the 7.7 h quoted
rem for the d=50M style run - d, not module count, drives the partition matmul).
rem Peak was 5.85 GiB; budget < 8 GiB.
rem
rem If it reaches ~0.70 like the matched LoRA: coverage was the story, the
rem parameterization is fine. If it stays near 0.43: GPart's attn-only choice
rem was not the problem, and low-rank really does carry more character per
rem unit of displacement.
cd /d D:\Gpart
set LOG=outputs\charrun_2026-11-01\console.log
if not exist outputs\charrun_2026-11-01 mkdir outputs\charrun_2026-11-01
echo ==== GPART ALLLINEAR CHAR START %DATE% %TIME% ==== >> %LOG%

D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter gpart --d 23224320 ^
  --targets to_q to_k to_v to_out.0 ff.net.0.proj ff.net.2 ^
  --scope transformer_block --layers-pattern transformer_blocks ^
  --data D:/AI/lora/datasets_aima/10_aima_var ^
  --steps 1200 --grad-accum 2 --lr 5e-4 --lr-schedule constant --warmup 50 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/charrun_2026-11-01/aima_alllinear_d23m >> %LOG% 2>&1

echo ==== JOB A DONE %DATE% %TIME% ==== >> %LOG%
