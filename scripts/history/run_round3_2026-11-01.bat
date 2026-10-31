@echo off
rem Chain job A -> job B so the second starts unattended once the first exits.
rem Serial on purpose: the GPU is 8 GB and either job alone reserves ~6 GiB.
rem Independent of each other, so B still runs if A fails - check both logs.
set LOG=D:\Gpart\outputs\round3_2026-11-01.log
echo ==== ROUND3 CHAIN START %DATE% %TIME% ==== >> %LOG%

call D:\Gpart\scripts\train_char_gpart_alllinear_2026-11-01.bat
echo ---- job A exited %ERRORLEVEL% at %TIME% ---- >> %LOG%

call D:\Gpart\scripts\train_style_lora_matched_2026-11-01.bat
echo ---- job B exited %ERRORLEVEL% at %TIME% ---- >> %LOG%

echo ==== ROUND3 CHAIN END %DATE% %TIME% ==== >> %LOG%
