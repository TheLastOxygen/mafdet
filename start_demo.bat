@echo off
rem Starts the live demo (incremental prefill, step by step) with the project's virtual environment,
rem whatever the system's default "python" is. Double-click it; from a terminal you can add options:
rem   start_demo.bat --auto
rem   start_demo.bat --device cpu
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Cannot find .venv\Scripts\python.exe: create the virtual environment first, as described in the README.
    pause
    exit /b 1
)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
rem A NO_COLOR inherited from the terminal would take every colour away from the demo.
set NO_COLOR=
".venv\Scripts\python.exe" demo_live.py %*
if errorlevel 1 pause
