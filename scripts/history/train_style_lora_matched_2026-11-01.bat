@echo off
rem Round 3, job B: the style-domain fairness control.
rem
rem REPORT.md never had a matched-recipe LoRA for the STYLE task - every style
rem row compares a GPart run against kohya's reserve1999_style_v3 (lr 1e-4).
rem Section 6.8 showed that on the character task the recipe difference alone
rem accounted for the whole "GPart wins" result, so the style conclusion is
rem currently unsupported in either direction.
rem
rem Settings copied from outputs/longrun_2026-10-29/d50m (the run behind
rem gp_style_attn_f, the style row the report actually relies on):
rem   data reserve1999_clean/6_reserve1999, lr 5e-4 constant, warmup 50,
rem   1200 steps, grad_accum 2, max_res 1536, seed 42, holdout 4.
rem Same rank/alpha as the character control (r8 / alpha 4 = 0.5 scaling).
rem
rem Expected ~1.5-2 h: 76 images vs 63 for aima, otherwise the same shape as
rem the 96-minute character LoRA run.
cd /d D:\Gpart
set LOG=outputs\stylerun_2026-11-01\console.log
if not exist outputs\stylerun_2026-11-01 mkdir outputs\stylerun_2026-11-01
echo ==== STYLE LORA MATCHED START %DATE% %TIME% ==== >> %LOG%

D:\AI\lora\venv\Scripts\python.exe -u src\train_sdxl.py ^
  --adapter lora --rank 8 --alpha 4 ^
  --targets to_q to_k to_v to_out.0 ff.net.0.proj ff.net.2 ^
  --scope transformer_block --layers-pattern transformer_blocks ^
  --data D:/AI/lora/datasets_reserve1999_clean/6_reserve1999 ^
  --steps 1200 --grad-accum 2 --lr 5e-4 --lr-schedule constant --warmup 50 ^
  --max-res 1536 --noise-offset 0.05 --min-snr-gamma 5 --grad-clip 1 ^
  --seed 42 --holdout 4 --val-every 200 --val-batches 4 ^
  --log-every 10 --save-every 200 ^
  --output outputs/stylerun_2026-11-01/r1999_lora_r8_matched >> %LOG% 2>&1

echo ==== JOB B DONE %DATE% %TIME% ==== >> %LOG%
