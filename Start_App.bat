@echo off
setlocal
cd /d "%~dp0"
REM ============================================================
REM  DeepFake Detection - Local Gradio Application Launcher
REM  Final Year Project (MH-FFNet Inspired Explainable AI)
REM ============================================================
REM  Usage: Double-click this file to start the web application.
REM  It will launch the Gradio interface in your web browser.
REM ============================================================

title DeepFake Detection App - Gradio UI

set "CHECKPOINT=%~dp0models\checkpoints\final_frequency\calibrated.pt"

if not exist "%CHECKPOINT%" (
    echo Trained checkpoint is missing: "%CHECKPOINT%"
    echo Restore the supplied calibrated.pt file before starting the application.
    pause
    exit /b 1
)

REM Detect Python environment
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\work\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\work\venv\Scripts\python.exe"
) else (
    echo [INFO] No local virtual environment found.
    echo Running Setup_Environment.bat to configure dependencies automatically...
    echo.
    call "%~dp0Setup_Environment.bat"
    if exist "%~dp0venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
)

echo ============================================================
echo   CNN + Frequency DeepFake Detection Application
echo ============================================================
echo   Python     : %PYTHON_EXE%
echo   Checkpoint : %CHECKPOINT%
echo   Server URL : http://127.0.0.1:7860
echo   Starting server and opening browser...
echo   (Press Ctrl+C in this terminal window to stop the server)
echo ============================================================

"%PYTHON_EXE%" "%~dp0src\app.py" --checkpoint "%CHECKPOINT%" --open-browser

pause
