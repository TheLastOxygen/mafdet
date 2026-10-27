"""
Hugging Face search: GGUF models beyond the catalog, found, sized up and downloaded from the app.

    search("qwen3 4b")              the repositories with GGUF files, most downloaded first
    files("unsloth/Qwen3-4B-GGUF")  their files: quantization, size, what fits here
    download(repo, file, task)      into ~/.mafdet/models/hf/<owner>/<repo>/, resumable

Once downloaded a file is one of "your models" (catalog.local_variants): its
header gives the real numbers. Before that, the estimate has only what the Hub
says (parameters, architecture), and assumes the KV cache of a typical model
of today (GQA, 8 KV heads of 128).
"""
from __future__ import annotations

import os
import re
import urllib.error
from pathlib import Path

from . import catalog, config, gguf
from .downloads import fetch
from .hardware import Hardware
from .tasks import Task

SEARCH_LIMIT = 40
# The file suggested in a repository: the first of these that fits in memory (the catalog's usual choice first).
SUGGESTED = ("Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q4_0", "Q5_K_M", "Q6_K", "Q8_0")
TYPICAL = {"layers": 36, "kv_heads": 8, "head_dim": 128}      # the KV cache of a model whose header is unknown
ACTIVE = re.compile(r"(?<![A-Za-z0-9])A(\d+(?:\.\d+)?)B(?![A-Za-z0-9])")   # mixture of experts: "Qwen3-30B-A3B"


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def search(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    """Repositories with GGUF files matching `query` (empty: the most downloaded), most downloaded first."""
    found = _api().list_models(search=query.strip() or None, filter="gguf", sort="downloads", limit=limit,
                               expand=["downloads", "likes", "lastModified", "gguf", "gated"])
    out = []
    for m in found:
        g = m.gguf or {}
        out.append({"repo": m.id, "downloads": m.downloads or 0, "likes": m.likes or 0,
                    "updated": m.last_modified.strftime("%Y-%m-%d") if m.last_modified else "",
                    "arch": g.get("architecture", ""), "params_b": round((g.get("total") or 0) / 1e9, 2),
                    "gated": bool(m.gated)})
    return out


def dest_dir(repo: str) -> Path:
    return config.models_dir() / catalog.HF_DIR / Path(*repo.split("/"))


def files(repo: str, hw: Hardware, device: str = "auto", ctx: int = 8192) -> dict:
    """The models in a repository, one per quantization (a split model is one entry with all its parts),
    with where each would run here and how fast."""
    info = _api().model_info(repo, files_metadata=True)
    g = info.gguf or {}
    params_b = (g.get("total") or 0) / 1e9
    arch = g.get("architecture", "")
    thinks = arch in catalog.THINKING_ARCHS and "enable_thinking" in (g.get("chat_template") or "")
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or [] if s.rfilename.lower().endswith(".gguf")}
    # The Hub gives the parameters, not how many work for each token: a mixture of experts names them.
    active = ACTIVE.search(repo.split("/")[-1])
    active_b = min(params_b, float(active.group(1))) if active and params_b else params_b
    out = []
    for name in sorted(sizes, key=lambda n: sizes[n]):
        base = os.path.basename(name)
        if gguf.is_projector(base) or not gguf.is_first_part(base):
            continue
        parts = [str(Path(name).with_name(p.name)).replace(os.sep, "/") for p in gguf.shards(Path(name))]
        if not all(p in sizes for p in parts):
            continue
        size = sum(sizes[p] for p in parts)
        model = {**TYPICAL, "id": "", "params_b": params_b or size / 2**30, "active_b": active_b or size / 2**30,
                 "reason_ahead": thinks, "reasoning_tokens": 900}
        v = catalog.Variant(model, gguf.quant_label(base), repo, name, size)
        e = catalog.estimate(v, hw, device, ctx)
        e.update(catalog.waits(model, e["decode_tps"]))
        local = dest_dir(repo) / name
        out.append({"file": name, "parts": parts, "bytes": size, "quant": v.quant, "name": gguf.model_name(base),
                    "installed": all((dest_dir(repo) / p).is_file() for p in parts), "path": str(local), **e})
    fits = {f["quant"]: f for f in out if f["placement"] != "too_big"}
    best = next((fits[q] for q in SUGGESTED if q in fits), None)
    for f in out:
        f["suggested"] = f is best
    out.sort(key=lambda f: not f["suggested"])            # the suggestion first, then the smallest
    return {"repo": repo, "arch": arch, "params_b": round(params_b, 2), "active_b": round(active_b, 2),
            "reason_ahead": thinks,
            "gated": bool(getattr(info, "gated", False)), "license": str(getattr(info.card_data, "license", "") or ""),
            "files": out}


def _token() -> str | None:
    """The token of `hf auth login`, if any: gated repositories (Llama, Gemma) need it."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def download(repo: str, file: str, task: Task) -> dict:
    """A file of a repository (all its parts, if split) into models/hf/<owner>/<repo>/; returns its id among
    your models."""
    from huggingface_hub import HfApi, hf_hub_url
    task.stage = "downloading"
    info = HfApi().model_info(repo, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings or []}
    parts = [str(Path(file).with_name(p.name)).replace(os.sep, "/") for p in gguf.shards(Path(file))]
    missing = [p for p in parts if p not in sizes]
    if missing:
        raise FileNotFoundError(f"{repo} has no file {missing[0]}")
    total = sum(sizes[p] or 0 for p in parts)
    token = _token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    done = 0
    try:
        for p in parts:
            fetch(hf_hub_url(repo, p), dest_dir(repo) / p, task, size=sizes[p], offset=done, total=total,
                  headers=headers)
            done += sizes[p] or 0
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise PermissionError(f"{repo} is gated: accept its license on huggingface.co, then log in on this "
                                  "machine with `hf auth login`") from e
        raise
    path = dest_dir(repo) / parts[0]
    v = next((v for v in catalog.local_variants() if Path(v.file) == path), None)
    return {"path": str(path), "id": v.id if v else ""}
