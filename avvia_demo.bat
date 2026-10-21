@echo off
rem Avvia la demo interattiva con il Python dell'ambiente virtuale del progetto,
rem qualunque sia il "python" o "python3" predefinito del sistema.
rem Doppio clic per avviarla; da terminale si possono aggiungere opzioni:
rem   avvia_demo.bat --auto
rem   avvia_demo.bat --device cpu
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Non trovo .venv\Scripts\python.exe: crea prima l'ambiente virtuale come descritto nel README.
    pause
    exit /b 1
)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
rem Un NO_COLOR ereditato dal terminale toglierebbe tutti i colori alla demo.
set NO_COLOR=
".venv\Scripts\python.exe" demo_live.py %*
if errorlevel 1 pause
