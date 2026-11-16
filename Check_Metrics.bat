@echo off
setlocal
cd /d "%~dp0"
title DeepFake Detection - Model Metrics & Confusion Matrix

if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0scripts\check_metrics.py"
) else (
    python "%~dp0scripts\check_metrics.py"
)

pause
