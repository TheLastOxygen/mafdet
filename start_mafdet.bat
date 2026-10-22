@echo off
setlocal
rem MAFDET for Windows: double-click it. The first time it prepares what it needs (a Python
rem environment in .venv with MAFDET installed, about a minute), then it opens MAFDET.
rem Options go to mafdet, from a terminal:  start_mafdet.bat gui    start_mafdet.bat doctor
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set "VENV_PY=.venv\Scripts\python.exe"

rem Outside of a block on purpose: cmd expands %PY% in a block before :find_python sets it.
if exist "%VENV_PY%" goto :install
call :find_python || goto :no_python
echo Preparing MAFDET for the first time: a Python environment in .venv...
%PY% -m venv .venv || goto :failed

:install

rem Install, and install again when pyproject.toml changes (new libraries after an update):
rem its hash is kept in .venv. certutil prints the hash on the only line without a colon.
set "HASH="
for /f "delims=" %%H in ('certutil -hashfile pyproject.toml SHA256 ^| findstr /v ":"') do set "HASH=%%H"
set "SAVED="
if exist ".venv\mafdet-installed.txt" set /p SAVED=<".venv\mafdet-installed.txt"
if not "%SAVED%"=="%HASH%" (
    echo Installing MAFDET and its libraries, about a minute...
    "%VENV_PY%" -m pip install --disable-pip-version-check -q -e . || goto :failed
    > ".venv\mafdet-installed.txt" echo %HASH%
)

"%VENV_PY%" -m mafdet %*
if errorlevel 1 pause
exit /b

:find_python
rem The Python launcher first, at a version MAFDET supports (3.12 is the one it is tested with),
rem then python on the PATH. The Microsoft Store's stand-in for python fails the check.
for %%V in (3.12 3.13 3.11 3.10 3.14) do (
    py -%%V -c "" >nul 2>&1 && (set "PY=py -%%V" & exit /b 0)
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && (set "PY=python" & exit /b 0)
exit /b 1

:no_python
echo MAFDET needs Python 3.10 or later, and none was found on this computer.
echo Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
echo then double-click this file again.
pause
exit /b 1

:failed
echo Something went wrong while preparing MAFDET: see the messages above.
pause
exit /b 1
