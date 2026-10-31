@echo off
rem Chain: wait for the all-linear style run (r1999, 600 steps) to finish, then
rem resume the aima character run that was stopped at ~step 65 this morning.
rem
rem Waits on a log marker instead of a PID: train_style_alllinear_2026-10-30.bat
rem appends "==== ALL DONE" to its console.log once train_sdxl.py exits, whether
rem it finished, crashed or was killed, so this cannot hang on a dead process.
rem The 480-iteration cap (8 h at 60 s) keeps a stale waiter from launching the
rem character run days later if something unexpected happens.
rem
rem The partial run dir is moved aside, not deleted: train_sdxl.py opens
rem train.jsonl in append mode, so relaunching into the same dir would interleave
rem the old 60 steps with the new run and corrupt the curve. It holds no
rem checkpoint (save-every 200, stopped at step 60), only args.json / status.json
rem / train.jsonl, and those are worth keeping as the record of the aborted try.
cd /d D:\Gpart
set QUEUELOG=outputs\charrun_2026-10-30\queue.log
set STYLELOG=outputs\style_alllinear_2026-10-30\console.log
set /a WAITED=0

echo ==== QUEUE START %DATE% %TIME% ==== >> %QUEUELOG%

:wait
findstr /C:"ALL DONE" "%STYLELOG%" > nul 2>&1
if not errorlevel 1 goto ready
set /a WAITED+=1
if %WAITED% geq 480 goto timeout
rem ping, not timeout: timeout aborts when stdin is redirected, which is how a
rem detached Start-Process cmd runs
ping -n 61 127.0.0.1 > nul
goto wait

:timeout
echo ==== GAVE UP after 8 h, style run never reported done %DATE% %TIME% ==== >> %QUEUELOG%
goto end

:ready
echo ==== STYLE RUN REPORTED DONE at %DATE% %TIME% (waited %WAITED% min) ==== >> %QUEUELOG%
echo ---- style checkpoints: >> %QUEUELOG%
dir /b outputs\style_alllinear_2026-10-30\r1999_d23m>> %QUEUELOG% 2>&1
echo ---- style last lines: >> %QUEUELOG%
powershell -NoProfile -Command "Get-Content 'outputs\style_alllinear_2026-10-30\console.log' -Tail 3" >> %QUEUELOG% 2>&1

rem let the style process actually release VRAM before the next run allocates
ping -n 31 127.0.0.1 > nul

if exist "outputs\charrun_2026-10-30\aima_d23m" (
  move "outputs\charrun_2026-10-30\aima_d23m" "outputs\charrun_2026-10-30\aima_d23m_incomplete_1015" >> %QUEUELOG% 2>&1
  echo ==== moved partial char dir, relaunching clean ==== >> %QUEUELOG%
)

call scripts\train_char_2026-10-30.bat

echo ==== QUEUE DONE %DATE% %TIME% ==== >> %QUEUELOG%

:end
