"""
Where MAFDET keeps its files, and the settings that survive a restart.

Everything lives in one folder, ~/.mafdet (or $MAFDET_HOME):
    config.json     settings (language, model, chat options, launcher choice)
    bench.json      speeds measured on this machine, per model
    models/         GGUF files downloaded from the catalog; any other GGUF here is a model of your own
                    (models/hf/<owner>/<repo>/: the ones downloaded from the Hugging Face search)
    llama.cpp/      the llama.cpp release installed by the app
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

LANGUAGES = ("en", "it", "fr", "de")

DEFAULTS = {
    "language": "en",
    "model": "",                      # catalog id of the model to load at start ("" = none chosen yet)
    "local_models": [],               # GGUF files added from elsewhere on disk (they stay where they are)
    "launcher": "",                   # "gui" or "tui" when the launcher should not ask again
    "first_run_done": False,
    "think": True,                    # reason ahead while the user types (Qwen3 thinking mode)
    "ahead": 512,                     # answer tokens written at a pause, before Enter
    "think_ahead": 1024,              # longest reasoning kept across edits
    "max_tokens": 0,                  # longest answer (0 = as long as the context allows)
    "ctx": 8192,
    "device": "auto",                 # auto, cuda or cpu
    "recycle": False,
    "gatherer": "",                   # experimental: a small model gathering material (catalog id or path)
    "branches": False,                # experimental: reason on a guessed ending of the message (--branches 1)
    "effects": True,                  # the terminal app: boot sequence, decrypting text, interference
}


def home() -> Path:
    path = Path(os.environ.get("MAFDET_HOME") or Path.home() / ".mafdet")
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = home() / "models"
    path.mkdir(exist_ok=True)
    return path


def llama_dir() -> Path:
    return home() / "llama.cpp"


_lock = threading.Lock()


def _read(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)                  # never a half-written file, even if the app is closed mid-write


def load() -> dict:
    """The settings, with the defaults for anything missing or of the wrong type."""
    with _lock:
        saved = _read(home() / "config.json")
    out = dict(DEFAULTS)
    for k, v in saved.items():
        if k in DEFAULTS and type(v) is type(DEFAULTS[k]):          # True is not an int here
            out[k] = v
    if out["language"] not in LANGUAGES:
        out["language"] = "en"
    return out


def save(changes: dict) -> dict:
    """Merge `changes` (known keys only) into the saved settings and return them all."""
    with _lock:
        path = home() / "config.json"
        data = _read(path)
        for k, v in changes.items():
            if k in DEFAULTS and type(v) is type(DEFAULTS[k]):          # True is not an int here
                data[k] = v
        _write(path, data)
    return load()


def read_json(name: str) -> dict:
    with _lock:
        return _read(home() / name)


def write_json(name: str, data: dict):
    with _lock:
        _write(home() / name, data)
