"""
Downloads that survive an interruption: models from Hugging Face, llama.cpp from GitHub.

The file grows as <name>.part next to its final place; a new start asks the
server for the rest only (HTTP Range), and the file gets its real name once it
has its full size. No half-written model is ever loaded.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .. import __version__
from .tasks import Task

CHUNK = 1 << 20
USER_AGENT = f"mafdet/{__version__}"


def size_text(n: int) -> str:
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"


class _KeepAuthAtHome(urllib.request.HTTPRedirectHandler):
    """A token for huggingface.co is not sent on to the storage it redirects to (a signed address that
    refuses a second credential)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlparse(newurl).hostname != urlparse(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_KeepAuthAtHome)


def fetch(url: str, dest: Path, task: Task, size: int | None = None, offset: int = 0, total: int | None = None,
          headers: dict | None = None) -> Path:
    """Download `url` to `dest`, resuming a .part file; progress goes to `task`.

    `offset` and `total` place this file inside a job of several files, so the
    progress bar covers them all. `headers`: extra ones (a token for a gated model).
    """
    dest = Path(dest)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size is not None and have > size:          # a .part from another file of the same name
        part.unlink()
        have = 0
    task.detail = f"{dest.name}{f', {size_text(size)}' if size else ''}, from {urlparse(url).hostname}"
    headers = {"User-Agent": USER_AGENT, **(headers or {})}
    if have:
        headers["Range"] = f"bytes={have}-"
    with _opener.open(urllib.request.Request(url, headers=headers), timeout=60) as r:
        if have and r.status != 206:              # the server sent the whole file: start over
            have = 0
        length = int(r.headers.get("Content-Length") or 0)
        full = size or (have + length if length else None)
        grand = total or full or 0
        with open(part, "ab" if have else "wb") as f:
            done = have
            while True:
                block = r.read(CHUNK)
                if not block:
                    break
                f.write(block)
                done += len(block)
                task.bytes(offset + done, grand)
    if full is not None and part.stat().st_size != full:
        raise OSError(f"{dest.name}: the download stopped at {part.stat().st_size} of {full} bytes; "
                      "start it again to resume")
    os.replace(part, dest)
    return dest


def download_model(variant_id: str, task: Task) -> dict:
    """A catalog model into ~/.mafdet/models (nothing to do if it is already on disk)."""
    from huggingface_hub import hf_hub_url

    from . import catalog, config
    v = catalog.find(variant_id)
    if v is None:
        raise ValueError(f"Unknown model: {variant_id}")
    existing = catalog.local_path(v)
    if existing:
        task.bytes(v.bytes, v.bytes)
        return {"path": existing}
    task.stage = "downloading"
    path = fetch(hf_hub_url(v.repo, v.file), config.models_dir() / v.file, task, size=v.bytes)
    return {"path": str(path)}
