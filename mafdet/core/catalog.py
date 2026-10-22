"""
The model catalog: what can be downloaded, how fast it should run here, and what was measured.

The estimate needs no download. Generating a token reads every weight once, so
its time is the bytes on the GPU over the GPU's bandwidth plus the bytes left
in system RAM over the RAM's bandwidth. How much fits on the GPU follows
llama.cpp's own split (-ngl auto): whole layers, each with its share of the KV
cache, until the free VRAM minus a reserve is full. Calibrated on the reference
PC (RTX 2060 6 GB, i5-10400, DDR4): Qwen3-4B 83 tokens/s all on the GPU,
Qwen3-8B about 13 and Qwen3-14B 4.8 with part of the model on the CPU.

From the speed comes what the user cares about: how long the answer takes
after Enter, with MAFDET reasoning ahead while they type and without it.
A measured speed (bench.py) replaces the estimate as soon as there is one.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

from . import config
from .hardware import Hardware

GPU_EFFICIENCY = 0.62          # share of the GPU's bandwidth llama.cpp reaches while generating
RAM_EFFICIENCY = 0.9           # the same for system RAM, against hardware.ram_bandwidth_gbs()
VRAM_RESERVE_MB = 900          # CUDA context, compute buffers and llama.cpp's safety margin
RAM_RESERVE_MB = 2500          # the system and the app itself
PCIE_GBS = 12.0                # weights left on the CPU are streamed to the GPU for long prefills
GPU_PREFILL_FLOPS_PER_GBS = 0.051e12   # usable prefill compute, from the card's bandwidth class (2060: 4B 2,141 tok/s)
CPU_PREFILL_FLOPS_PER_CORE = 30e9
# Typing a question takes about half a minute (measured: 30-37 s at 5 characters per second with
# pauses); while reasoning ahead the model covers at most about 60% of its reasoning, because
# the end of the message is still unknown (Qwen3-4B: 418 of 662 tokens).
TYPING_SECONDS = 34
AHEAD_SHARE_MAX = 0.6


@dataclass
class Variant:
    model: dict
    quant: str
    repo: str
    file: str
    bytes: int

    @property
    def id(self) -> str:
        return f"{self.model['id']}:{self.quant}"

    @property
    def spec(self) -> str:
        """What llama_backend.resolve_gguf takes: 'repo:file'."""
        return f"{self.repo}:{self.file}"


@lru_cache(maxsize=1)
def catalog() -> dict:
    return json.loads(files("mafdet").joinpath("catalog.json").read_text(encoding="utf-8"))


def variants() -> list[Variant]:
    return [Variant(m, v["quant"], v["repo"], v["file"], v["bytes"]) for m in catalog()["models"] for v in m["variants"]]


def find(variant_id: str) -> Variant | None:
    """A variant by id ('qwen3-4b:Q4_K_M'); a bare model id means its default quantization."""
    if ":" not in variant_id:
        m = next((m for m in catalog()["models"] if m["id"] == variant_id), None)
        variant_id = f"{variant_id}:{m['default_quant']}" if m else variant_id
    return next((v for v in variants() if v.id == variant_id), None)


# ---- installed files ---------------------------------------------------------

def local_path(v: Variant) -> str | None:
    """Where the file is on disk: the app's models folder, or the Hugging Face cache (no download twice)."""
    p = config.models_dir() / v.file
    if p.exists() and p.stat().st_size == v.bytes:
        return str(p)
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(v.repo, v.file)
    except Exception:                      # an unreadable cache is the same as an empty one
        cached = None
    return cached if isinstance(cached, str) and os.path.exists(cached) else None


# ---- measured speeds ---------------------------------------------------------

def fingerprint(hw: Hardware, device: str) -> str:
    """Measurements hold for one machine and one device: a new GPU or more RAM makes them stale."""
    gpu = hw.gpu
    parts = [hw.cpu, f"{round(hw.ram_mb / 1024)}GB", device]
    if gpu and device == "cuda":
        parts.insert(0, f"{gpu.name} {round(gpu.vram_mb / 1024)}GB")
    return " | ".join(parts)


def measured(v: Variant, hw: Hardware, device: str) -> dict | None:
    return config.read_json("bench.json").get(fingerprint(hw, device), {}).get(v.id)


def save_measurement(v: Variant, hw: Hardware, device: str, result: dict):
    data = config.read_json("bench.json")
    data.setdefault(fingerprint(hw, device), {})[v.id] = result
    config.write_json("bench.json", data)


# ---- estimate ----------------------------------------------------------------

def kv_bytes_per_token(model: dict) -> int:
    return 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * 2      # keys and values, f16


def estimate(v: Variant, hw: Hardware, device: str = "auto", ctx: int = 8192) -> dict:
    """Where the model runs here and how fast, before downloading it."""
    m = v.model
    gpu = hw.gpu if device in ("auto", "cuda") else None
    weights_mb = v.bytes / 2**20
    kv_mb = kv_bytes_per_token(m) * ctx / 2**20
    total_mb = weights_mb + kv_mb
    if gpu:
        room = max(0.0, gpu.vram_free_mb - VRAM_RESERVE_MB)
        share = min(1.0, room / total_mb)
        share = int(share * m["layers"]) / m["layers"] if share < 1 else 1.0     # whole layers, as llama.cpp
    else:
        share = 0.0
    gpu_mb, cpu_mb = weights_mb * share, weights_mb * (1 - share)
    cpu_total_mb = (weights_mb + kv_mb) * (1 - share)
    fits_ram = cpu_total_mb <= hw.ram_mb - RAM_RESERVE_MB

    # Generation: only the active experts of a mixture-of-experts are read for each token.
    active = m["active_b"] / m["params_b"]
    t = 0.0
    if gpu_mb:
        t += gpu_mb * active / 1024 / (gpu.bandwidth_gbs * GPU_EFFICIENCY)
    if cpu_mb:
        t += cpu_mb * active / 1024 / (hw.ram_bandwidth_gbs * RAM_EFFICIENCY)
    decode = 1 / t if t else 0.0

    # Reading the prompt: compute-bound, about 2 operations per active parameter and token.
    flops = 2 * m["active_b"] * 1e9
    if gpu:
        tps = gpu.bandwidth_gbs * GPU_PREFILL_FLOPS_PER_GBS / flops
        if share < 1:                          # the CPU's layers cross the bus once per batch of 512 tokens
            tps = 512 / (512 / tps + cpu_mb / 1024 / PCIE_GBS)
        prefill = tps
    else:
        prefill = hw.cores * CPU_PREFILL_FLOPS_PER_CORE / flops

    placement = ("too_big" if not fits_ram else "gpu" if share >= 1 else "partial" if share > 0 else "cpu")
    return {"placement": placement, "gpu_share": round(share, 3), "decode_tps": round(decode, 1),
            "prefill_tps": round(prefill), "size_mb": round(weights_mb), "kv_mb": round(kv_mb),
            "measured": False}


def waits(m: dict, decode_tps: float) -> dict:
    """Seconds from Enter to the answer for a typical question, with and without reasoning ahead."""
    reasoning = m["reasoning_tokens"]
    if decode_tps <= 0:
        return {"with_mafdet_s": None, "without_s": None, "ahead_tokens": 0}
    ahead = min(AHEAD_SHARE_MAX * reasoning, decode_tps * TYPING_SECONDS) if m["reason_ahead"] else 0
    return {"with_mafdet_s": round((reasoning - ahead) / decode_tps, 1),
            "without_s": round(reasoning / decode_tps, 1), "ahead_tokens": round(ahead)}


def entry(v: Variant, hw: Hardware, device: str = "auto", ctx: int = 8192) -> dict:
    """Everything an interface shows about one variant on this machine."""
    dev = "cuda" if device in ("auto", "cuda") and hw.gpu else "cpu"
    e = estimate(v, hw, device, ctx)
    got = measured(v, hw, dev)
    if got:
        e.update(decode_tps=got["decode_tps"], prefill_tps=got["prefill_tps"], measured=True,
                 measured_at=got.get("date", ""))
    e.update(waits(v.model, e["decode_tps"]))
    path = local_path(v)
    return {"id": v.id, "model_id": v.model["id"], "name": v.model["name"], "quant": v.quant,
            "default": v.quant == v.model["default_quant"], "tier": v.model["tier"],
            "params_b": v.model["params_b"], "active_b": v.model["active_b"],
            "reason_ahead": v.model["reason_ahead"], "bytes": v.bytes, "installed": bool(path),
            "path": path, **e}


def recommend(entries: list[dict]) -> str | None:
    """The largest model, at its default quantization, whose answer comes within a comfortable wait.

    Preference: all on the GPU and under 8 s after Enter; then anything under 15 s; then the
    fastest that fits at all.
    """
    pool = [e for e in entries if e["default"] and e["placement"] != "too_big" and e["with_mafdet_s"] is not None]
    for ok in (lambda e: e["placement"] == "gpu" and e["with_mafdet_s"] <= 8,
               lambda e: e["with_mafdet_s"] <= 15):
        good = [e for e in pool if ok(e)]
        if good:
            return max(good, key=lambda e: e["params_b"])["id"]
    return min(pool, key=lambda e: e["with_mafdet_s"])["id"] if pool else None


def listing(hw: Hardware, device: str = "auto", ctx: int = 8192) -> dict:
    entries = [entry(v, hw, device, ctx) for v in variants()]
    best = recommend(entries)
    for e in entries:
        e["recommended"] = e["id"] == best
    return {"models": entries, "recommended": best}
