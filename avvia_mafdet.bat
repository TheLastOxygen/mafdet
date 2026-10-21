@echo off
rem Apre la chat di MAFDET (mafdet watch) nella sua finestra, senza console dietro.
rem Doppio clic per avviarla. Se qualcosa non va, il registro e' in %TEMP%\mafdet-watch.log,
rem oppure lanciala da terminale per vedere i messaggi:  .venv\Scripts\python.exe -m mafdet watch
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Non trovo .venv\Scripts\pythonw.exe: crea prima l'ambiente virtuale come descritto nel README.
    pause
    exit /b 1
)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
start "" ".venv\Scripts\pythonw.exe" -m mafdet watch %*
