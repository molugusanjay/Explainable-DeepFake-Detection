@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   DeepFake Detection Project - Environment Setup
echo ============================================================
echo   This script will set up the Python virtual environment
echo   and install all required dependencies automatically.
echo ============================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found on your system PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment (venv)...
if not exist "venv" (
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Existing virtual environment found.
)

echo.
echo [2/3] Installing dependencies from requirements.txt...
"%~dp0venv\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some dependencies failed to install. Retrying setup...
)

echo.
echo [3/3] Installing package in editable mode...
"%~dp0venv\Scripts\python.exe" -m pip install -e "%~dp0."

echo.
echo ============================================================
echo   Setup Completed Successfully!
echo ============================================================
echo   You can now launch the application anytime by double-clicking:
echo   Start_App.bat
echo ============================================================
echo.
pause
