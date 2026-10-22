@echo off
rem Starts MAFDET: the launcher asks whether to open the window or the terminal app.
rem Double-click it, or run it from a terminal with a command, e.g.:  start_mafdet.bat gui
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Cannot find .venv\Scripts\python.exe: create the virtual environment first, as described in the README.
    pause
    exit /b 1
)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
".venv\Scripts\python.exe" -m mafdet %*
if errorlevel 1 pause
