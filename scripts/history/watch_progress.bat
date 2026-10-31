@echo off
title GPart training progress
chcp 65001 >nul
cd /d D:\Gpart
D:\AI\lora\venv\Scripts\python.exe -u scripts\watch_progress.py --root outputs\longrun_2026-10-29
pause
