#!/bin/bash
# MAFDET for macOS: double-click it in the Finder. The first time it prepares what it needs (a Python
# environment in .venv with MAFDET installed, about a minute), then it opens MAFDET.
# Options go to mafdet, from a terminal:  ./start_mafdet.command gui    ./start_mafdet.command doctor
cd "$(dirname "$0")" || exit 1
VENV_PY=.venv/bin/python

fail() {
    echo "$1"
    read -r -p "Press Enter to close."
    exit 1
}

if [ ! -x "$VENV_PY" ]; then
    # A version MAFDET supports (3.12 is the one it is tested with), then python3 on the PATH.
    # /usr/bin/python3 without the Command Line Tools only offers to install them: it fails the check.
    PY=""
    for v in 3.12 3.13 3.11 3.10 3.14; do
        if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
    done
    if [ -z "$PY" ] && python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
        PY=python3
    fi
    [ -n "$PY" ] || fail "MAFDET needs Python 3.10 or later, and none was found on this Mac.
Install it from https://www.python.org/downloads/ (or: brew install python@3.12),
then double-click this file again."
    echo "Preparing MAFDET for the first time: a Python environment in .venv..."
    "$PY" -m venv .venv || fail "Something went wrong while preparing MAFDET: see the messages above."
fi

# Install, and install again when pyproject.toml changes (new libraries after an update):
# its hash is kept in .venv.
HASH=$(shasum -a 256 pyproject.toml | cut -d' ' -f1)
SAVED=$(cat .venv/mafdet-installed.txt 2>/dev/null)
if [ "$SAVED" != "$HASH" ]; then
    echo "Installing MAFDET and its libraries, about a minute. pip shows each download below,"
    echo "with its size and a progress bar."
    "$VENV_PY" -m pip install --disable-pip-version-check --progress-bar on -e . \
        || fail "Something went wrong while preparing MAFDET: see the messages above."
    echo
    echo "$HASH" > .venv/mafdet-installed.txt
fi

"$VENV_PY" -m mafdet "$@" || fail "MAFDET stopped with an error: see the messages above."
