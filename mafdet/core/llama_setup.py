"""
llama.cpp, the runtime: find it, or install the right release by itself into ~/.mafdet/llama.cpp.

The release is pinned (LLAMA_TAG): the one MAFDET was tested with. The build
depends on the machine: CUDA for an NVIDIA card (CUDA 13 from driver 580 on,
except for the GTX 10 and older, which CUDA 13 no longer supports; CUDA 12
otherwise), the CPU build without one. A llama.cpp of your own still wins:
$MAFDET_LLAMA_DIR and ./llama.cpp are looked at first.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from . import config
from .downloads import USER_AGENT, fetch
from .hardware import Hardware, detect
from .tasks import Task

LLAMA_TAG = "b11747"
RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
INSTALL_INFO = "mafdet-install.json"
# Cards CUDA 13 dropped (Maxwell, Pascal, Volta): they keep the CUDA 12 build.
OLD_FOR_CUDA13 = re.compile(r"GTX (9|10)\d\d|TITAN (X|XP|V)\b|QUADRO [MP]\d|TESLA [MPV]\d|\bP\d{2,3}\b", re.I)


def build_for(hw: Hardware, cuda: str | None = None) -> dict:
    """Which build of the release fits this machine: {flavor, assets (name patterns), label}.

    `cuda` overrides the choice: "12" or "13" for that CUDA build (an older driver, a card the
    automatic choice gets wrong), "cpu" for the build without CUDA.
    """
    system, arm = hw.system, hw.arch in ("arm64", "aarch64")
    gpu = hw.gpu
    if cuda == "cpu" or arm or system not in ("windows", "linux"):
        cuda = None
    elif cuda is None and gpu:
        driver = int(re.match(r"\d+", gpu.driver).group()) if re.match(r"\d+", gpu.driver) else 0
        cuda = "13" if driver >= 580 and not OLD_FOR_CUDA13.search(gpu.name) else "12"
    if system == "windows":
        if cuda:
            ver = "13.4" if cuda == "13" else "12.4"
            return {"flavor": f"cuda-{ver}", "label": f"Windows · CUDA {ver}",
                    "assets": [f"llama-{{tag}}-bin-win-cuda-{ver}-x64.zip", f"cudart-llama-bin-win-cuda-{ver}-x64.zip"]}
        cpu = "arm64" if arm else "x64"
        return {"flavor": "cpu", "label": f"Windows · CPU ({cpu})", "assets": [f"llama-{{tag}}-bin-win-cpu-{cpu}.zip"]}
    if system == "linux":
        if cuda:
            ver = "13.4" if cuda == "13" else "12.8"
            return {"flavor": f"cuda-{ver}", "label": f"Linux · CUDA {ver}",
                    "assets": [f"llama-{{tag}}-bin-ubuntu-cuda-{ver}-x64.tar.gz",
                               f"cudart-llama-{{tag}}-bin-ubuntu-cuda-{ver}-x64.tar.gz"]}
        cpu = "arm64" if arm else "x64"
        return {"flavor": "cpu", "label": f"Linux · CPU ({cpu})", "assets": [f"llama-{{tag}}-bin-ubuntu-{cpu}.tar.gz"]}
    if system == "darwin":
        cpu = "arm64" if arm else "x64"
        return {"flavor": "metal" if arm else "cpu", "label": f"macOS · {cpu}",
                "assets": [f"llama-{{tag}}-bin-macos-{cpu}.tar.gz"]}
    raise RuntimeError(f"No llama.cpp build for {hw.os} ({hw.arch})")


def server_version(exe: str) -> str:
    """'b11747' style build of a llama-server, from --version ('' if it does not run)."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30, creationflags=flags)
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"build (\d+)", r.stdout + r.stderr)
    return f"b{m.group(1)}" if m else ""


def status() -> dict:
    """Is there a llama-server, where, which build, and is it the one MAFDET was tested with."""
    from ..llama_backend import find_llama_server
    exe = find_llama_server(required=False)
    out = {"installed": bool(exe), "path": exe, "build": "", "tested_build": LLAMA_TAG, "managed": False,
           "flavor": ""}
    if exe:
        out["build"] = server_version(exe)
        managed = Path(exe).parent == config.llama_dir()
        out["managed"] = managed
        if managed:
            out["flavor"] = config.read_json(f"llama.cpp/{INSTALL_INFO}").get("flavor", "")
    out["recommended"] = build_for(detect())["label"]
    return out


def _release_assets(tag: str) -> dict[str, dict]:
    req = urllib.request.Request(RELEASE_API.format(tag=tag),
                                 headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return {a["name"]: a for a in json.load(r)["assets"]}


def _extract(archive: Path, into: Path):
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(into, filter="data")      # no paths outside `into`, no device files
            except TypeError:                          # Python before 3.10.12 has no filter
                t.extractall(into)


def install(task: Task, tag: str = LLAMA_TAG, cuda: str | None = None) -> dict:
    """Download and unpack the release for this machine into ~/.mafdet/llama.cpp, then check it runs."""
    hw = detect()
    build = build_for(hw, cuda)
    task.stage = "release"
    assets = _release_assets(tag)
    names = [p.format(tag=tag) for p in build["assets"]]
    missing = [n for n in names if n not in assets]
    if missing:
        raise RuntimeError(f"llama.cpp {tag} has no {', '.join(missing)}")
    total = sum(assets[n]["size"] for n in names)
    cache = config.home() / "downloads"
    staging = config.home() / "llama.cpp.new"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    task.stage = "downloading"
    offset = 0
    files = []
    for n in names:
        files.append(fetch(assets[n]["browser_download_url"], cache / n, task, size=assets[n]["size"],
                           offset=offset, total=total))
        offset += assets[n]["size"]
    task.stage = "extracting"
    for f in files:
        _extract(f, staging)
        task.check()
    from ..llama_backend import SERVER_EXE
    found = next(staging.rglob(SERVER_EXE), None)
    if found is None:
        raise RuntimeError(f"{SERVER_EXE} is not in the llama.cpp archive")
    bin_dir = found.parent
    # The CUDA runtime comes in a second archive: its libraries go next to the server.
    for lib in staging.rglob("*"):
        if lib.is_file() and lib.parent != bin_dir and re.match(r"(cudart|cublas|cublasLt).*\.(dll|so.*)$", lib.name):
            shutil.move(str(lib), bin_dir / lib.name)
    if os.name != "nt":
        for exe in bin_dir.iterdir():
            if exe.is_file() and exe.name.startswith("llama-"):
                exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    task.stage = "checking"
    version = server_version(str(bin_dir / SERVER_EXE))
    if not version:
        raise RuntimeError("The llama.cpp just installed does not start: see the Hardware page for your driver")
    final = config.llama_dir()
    old = config.home() / "llama.cpp.old"
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    shutil.move(str(bin_dir), final)
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    for f in files:                                    # the archives are not needed once unpacked
        f.unlink(missing_ok=True)
    info = {"tag": tag, "flavor": build["flavor"], "label": build["label"], "build": version}
    config.write_json(f"llama.cpp/{INSTALL_INFO}", info)
    return info
