"""
The model catalog: what can be downloaded, how fast it should run here, and what was measured.

The estimate needs no download. Generating a token reads every weight once, so
its time is the bytes on the GPU over the GPU's bandwidth plus the bytes left
in system RAM over the RAM's bandwidth. How much fits on the GPU follows
llama.cpp's own split (-ngl auto, its --fit): whole layers, each with its share
of the KV cache, until the free VRAM minus a reserve is full. The token
embeddings stay in RAM unless the output layer shares them, and generating
reads one row of them: they cost RAM, not time. Calibrated on the reference PC
(RTX 2060 6 GB, i5-10400, DDR4): Qwen3-4B 83 tokens/s all on the GPU, Qwen3-8B
about 13 and Qwen3-14B 4.8 with part of the model on the CPU; checked on an
RTX 4070 SUPER 12 GB (the 14B all on the GPU, 41 tokens/s).

From the speed comes what the user cares about: how long the answer takes
after Enter, with MAFDET reasoning ahead while they type and without it.
A measured speed (bench.py) replaces the estimate as soon as there is one.

Besides the catalog, every GGUF in ~/.mafdet/models (downloaded by hand, or
from the Hugging Face search) and every file added in the app is a model of
its own: its header gives what the estimate needs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from struct import error as struct_error

from ..llama_backend import SERVER_TOKENIZER
from . import config, gguf
from .hardware import Hardware

GPU_EFFICIENCY = 0.62          # share of the GPU's bandwidth llama.cpp reaches while generating
                               # (K-quants: 0.60 on the 2060, 0.60-0.65 on the 4070 SUPER)
RAM_EFFICIENCY = 0.9           # the same for system RAM, against hardware.ram_bandwidth_gbs()
# llama.cpp's --fit leaves 1024 MiB free (its default --fit-target) after the model, the KV cache and
# the compute buffers (~180 MiB); the CUDA context takes ~100 more than nvidia-smi counts as used.
VRAM_RESERVE_MB = 1300
RAM_RESERVE_MB = 2500          # the system and the app itself
PCIE_GBS = 12.0                # weights left on the CPU are streamed to the GPU for long prefills
GPU_PREFILL_FLOPS_PER_GBS = 0.051e12   # usable prefill compute, from the card's bandwidth class (2060: 4B 2,141 tok/s)
# Ada has about 2.5 times the tensor compute per GB/s of Turing (4070 SUPER: 4B 7,441, 14B 2,486 tok/s).
GPU_PREFILL_FLOPS_PER_GBS_ADA = 0.125e12
APPLE_PREFILL_FLOPS_PER_GBS = 0.025e12 # the same for Apple Silicon's GPU (no tensor cores)
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


def families() -> list[dict]:
    return catalog()["families"]


def variants() -> list[Variant]:
    return [Variant(m, v["quant"], v["repo"], v["file"], v["bytes"]) for m in catalog()["models"] for v in m["variants"]]


def find(variant_id: str) -> Variant | None:
    """A variant by id ('qwen3-4b:Q4_K_M', 'local-…:Q4_K_M'); a bare model id means its default quantization."""
    if variant_id.startswith(LOCAL_PREFIX):
        own = local_variants()
        return next((v for v in own if v.id == variant_id), None) or \
            next((v for v in own if v.model["id"] == variant_id), None)
    if ":" not in variant_id:
        m = next((m for m in catalog()["models"] if m["id"] == variant_id), None)
        variant_id = f"{variant_id}:{m['default_quant']}" if m else variant_id
    return next((v for v in variants() if v.id == variant_id), None)


# ---- models of your own: GGUF files outside the catalog ------------------------

LOCAL_PREFIX = "local-"
LOCAL_FAMILY = {"id": "local", "name": "Your models", "vendor": "", "license": "", "reason_ahead": False}
HF_DIR = "hf"                       # downloads from the search: models/hf/<owner>/<repo>/<file>
# Reasoning ahead is measured on Qwen3's thinking mode (the frame in watch.py): a file of your own reasons
# ahead when it is a Qwen3 with that mode, and is otherwise a plain chat model, as Llama or Gemma.
THINKING_ARCHS = ("qwen3", "qwen3moe")
_headers: dict[tuple, dict | None] = {}


def local_files() -> list[Path]:
    """The GGUF models of your own: under the models folder (not the catalog's files) and the ones added by
    hand. Only the first part of a split model, when all its parts are there; no vision projectors."""
    root = config.models_dir()
    ours = {v.file for v in variants()}
    found = [p for p in sorted(root.rglob("*.gguf")) if not (p.parent == root and p.name in ours)]
    found += [Path(p) for p in config.load()["local_models"]]
    out, seen = [], set()
    for p in found:
        key = os.path.normcase(os.path.abspath(p))
        if key in seen or not p.is_file() or gguf.is_projector(p.name) or not gguf.is_first_part(p.name):
            continue
        seen.add(key)
        if all(s.is_file() for s in gguf.shards(p)):
            out.append(p)
    return out


def header(path: Path) -> dict | None:
    """The file's header, read once per version of the file; None if it is not a readable GGUF."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _headers:
        try:
            parts = gguf.shards(path)
            info = gguf.read(path)
            for extra in parts[1:]:                # the tensors of a split model are spread over its parts
                more = gguf.read(extra)
                info["params"] += more["params"]
                info["expert_params"] += more["expert_params"]
                info["embd_params"] += more["embd_params"]
                info["output"] = info["output"] or more["output"]
            info["bytes"] = sum(p.stat().st_size for p in parts)
            _headers[key] = info
        except (OSError, ValueError, KeyError, IndexError, struct_error) as e:
            print(f"Cannot read {path.name}: {e}")
            _headers[key] = None
    return _headers[key]


def repo_of(path: Path) -> str:
    """The Hugging Face repository a file was downloaded from by the search, else ''."""
    try:
        rel = path.relative_to(config.models_dir() / HF_DIR)
    except ValueError:
        return ""
    return "/".join(rel.parts[:2]) if len(rel.parts) > 2 else ""


def local_model(path: Path, info: dict) -> dict:
    """A catalog entry for a file of your own, from its header."""
    meta = info["meta"]
    arch = meta.get("general.architecture", "")
    get = lambda k, default=0: meta.get(f"{arch}.{k}", default)
    avg = lambda v: sum(v) / len(v) if isinstance(v, list) and v else v          # per-layer values
    top = lambda v: max(v) if isinstance(v, list) and v else v
    heads = top(get("attention.head_count")) or 1
    head_dim = get("attention.key_length") or (get("embedding_length") // heads)
    experts, used = get("expert_count"), get("expert_used_count")
    active = info["params"] - info["expert_params"] * (1 - used / experts) if experts and used else info["params"]
    name = gguf.model_name(path.name)
    repo = repo_of(path)
    template = meta.get("tokenizer.chat_template", "")
    # Quantizations of one model, in one folder, are one card with a button each.
    model_id = LOCAL_PREFIX + hashlib.sha1(f"{path.parent}/{name}".encode("utf-8")).hexdigest()[:10]
    return {
        "id": model_id, "name": name, "family": LOCAL_FAMILY["id"], "tokenizer": SERVER_TOKENIZER,
        "params_b": max(info["params"], 1) / 1e9, "active_b": max(active, 1) / 1e9,
        "cpu_embd_b": info["embd_params"] / 1e9 if info["output"] else 0.0,
        "layers": get("block_count") or 1, "kv_heads": avg(get("attention.head_count_kv")) or 0,
        "head_dim": head_dim, "tier": "custom", "default_quant": gguf.quant_label(path.name),
        "reason_ahead": arch in THINKING_ARCHS and "enable_thinking" in template,
        "reasoning_tokens": 900,            # as Qwen3-4B in the catalog
        "local": True, "arch": arch, "repo": repo, "chat_template": bool(template),
        "vendor": repo or path.name, "license": meta.get("general.license", ""),
    }


def local_variants() -> list[Variant]:
    out, ids = [], set()
    for p in local_files():
        info = header(p)
        if info is None:
            continue
        m = local_model(p, info)
        v = Variant(m, gguf.quant_label(p.name), m["repo"], str(p), info["bytes"])
        if v.id in ids:                        # two files with the same name and quantization: one card each
            m["id"] = LOCAL_PREFIX + hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
        ids.add(v.id)
        out.append(v)
    return out


def add_local(path: str) -> Variant:
    """A GGUF from anywhere on disk, added to your models (it stays where it is)."""
    p = Path(path).expanduser().absolute()
    if p.suffix.lower() != ".gguf" or not p.is_file():
        raise ValueError(f"{p.name} is not a .gguf file")
    if header(p) is None:
        raise ValueError(f"{p.name} is not a GGUF model llama.cpp can read")
    if gguf.is_projector(p.name):
        raise ValueError(f"{p.name} is the vision part of a model, not a model to chat with")
    p = gguf.shards(p)[0]
    root = config.models_dir().absolute()
    if p.parent == root:                      # a catalog file: its own card
        same = next((v for v in variants() if v.file == p.name), None)
        if same is not None:
            return same
    paths = config.load()["local_models"]
    if root not in p.parents and str(p) not in paths:      # the models folder is read anyway
        config.save({"local_models": paths + [str(p)]})
    return next(v for v in local_variants() if os.path.normcase(v.file) == os.path.normcase(str(p)))


def remove_local(variant_id: str):
    """Forget a file added by hand (the file itself stays)."""
    v = find(variant_id)
    if v is not None:
        config.save({"local_models": [p for p in config.load()["local_models"] if p != v.file]})


# ---- installed files ---------------------------------------------------------

def local_path(v: Variant) -> str | None:
    """Where the file is on disk: the app's models folder, or the Hugging Face cache (no download twice)."""
    if v.model.get("local"):
        return v.file if os.path.exists(v.file) else None
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

def splits_with_cpu(path: str, hw: Hardware, ctx: int = 8192) -> bool:
    """Whether llama.cpp will put part of this GGUF on the CPU here (the same estimate as the catalog's)."""
    info = header(Path(path))
    if info is None or not hw.gpu or hw.gpu.unified:
        return False
    v = Variant(local_model(Path(path), info), gguf.quant_label(Path(path).name), "", str(path), info["bytes"])
    return estimate(v, hw, "cuda", ctx)["placement"] == "partial"


def kv_bytes_per_token(model: dict) -> int:
    return 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * 2      # keys and values, f16


def ada(name: str) -> bool:
    """An NVIDIA card of the RTX 40 series (Ada Lovelace), laptop ones included."""
    return bool(re.search(r"RTX\s*40\d0", name, re.I))


def estimate(v: Variant, hw: Hardware, device: str = "auto", ctx: int = 8192) -> dict:
    """Where the model runs here and how fast, before downloading it."""
    m = v.model
    gpu = hw.gpu if device in ("auto", "cuda") else None
    weights_mb = v.bytes / 2**20
    embd_mb = weights_mb * m.get("cpu_embd_b", 0.0) / m["params_b"]      # in RAM, one row read per token
    layers_mb = weights_mb - embd_mb
    kv_mb = kv_bytes_per_token(m) * ctx / 2**20
    total_mb = weights_mb + kv_mb
    if gpu:
        room = max(0.0, gpu.vram_free_mb - VRAM_RESERVE_MB)
        share = min(1.0, room / (layers_mb + kv_mb))
        share = int(share * m["layers"]) / m["layers"] if share < 1 else 1.0     # whole layers, as llama.cpp
    else:
        share = 0.0
    gpu_mb, cpu_mb = layers_mb * share, layers_mb * (1 - share)
    cpu_total_mb = embd_mb + (layers_mb + kv_mb) * (1 - share)
    if gpu and gpu.unified:                    # Apple Silicon: the GPU's share is in the same RAM
        cpu_total_mb = total_mb
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
        per_gbs = (APPLE_PREFILL_FLOPS_PER_GBS if gpu.unified else
                   GPU_PREFILL_FLOPS_PER_GBS_ADA if ada(gpu.name) else GPU_PREFILL_FLOPS_PER_GBS)
        tps = gpu.bandwidth_gbs * per_gbs / flops
        if share < 1 and not gpu.unified:      # the CPU's layers cross the bus once per batch of 512 tokens
            tps = 512 / (512 / tps + cpu_mb / 1024 / PCIE_GBS)
        prefill = tps
    else:
        prefill = hw.cores * CPU_PREFILL_FLOPS_PER_CORE / flops

    placement = ("too_big" if not fits_ram else "gpu" if share >= 1 else "partial" if share > 0 else "cpu")
    return {"placement": placement, "gpu_share": round(share, 3), "decode_tps": round(decode, 1),
            "prefill_tps": round(prefill), "size_mb": round(weights_mb), "kv_mb": round(kv_mb),
            "measured": False}


ANSWER_TOKENS = 300            # a typical answer, for the models that do not reason


def waits(m: dict, decode_tps: float) -> dict:
    """What the user waits for after Enter.

    A model that reasons ahead: seconds to the answer for a typical question, with and without
    MAFDET. The others start answering right away (the message is already read), so what tells
    them apart is how long a typical answer takes to write.
    """
    if decode_tps <= 0:
        return {"with_mafdet_s": None, "without_s": None, "ahead_tokens": 0, "answer_s": None}
    answer_s = round(ANSWER_TOKENS / decode_tps, 1)
    if not m["reason_ahead"]:
        return {"with_mafdet_s": None, "without_s": None, "ahead_tokens": 0, "answer_s": answer_s}
    reasoning = m["reasoning_tokens"]
    ahead = min(AHEAD_SHARE_MAX * reasoning, decode_tps * TYPING_SECONDS)
    return {"with_mafdet_s": round((reasoning - ahead) / decode_tps, 1),
            "without_s": round(reasoning / decode_tps, 1), "ahead_tokens": round(ahead), "answer_s": answer_s}


def preferred_quant(model: dict, hw: Hardware, device: str = "auto", ctx: int = 8192) -> str:
    """The quantization the app offers first for a model on this machine: its default one, or its smaller
    `fit_quant` on a GPU where the default does not fit all in VRAM (more layers on the GPU, same answers;
    measured on the 14B, see research/LOG.md). Without a GPU the default stays: the smaller one runs on
    slower CPU kernels."""
    default, fit = model["default_quant"], model.get("fit_quant")
    if not fit or not (device in ("auto", "cuda") and hw.gpu):
        return default
    mine = {v.quant: v for v in variants() if v.model["id"] == model["id"]}
    if fit not in mine or estimate(mine[default], hw, device, ctx)["placement"] == "gpu":
        return default
    return fit


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
    fam = next((f for f in families() if f["id"] == v.model["family"]), LOCAL_FAMILY)
    return {"id": v.id, "model_id": v.model["id"], "name": v.model["name"], "quant": v.quant,
            "family": fam["id"], "family_name": fam["name"], "vendor": v.model.get("vendor", fam["vendor"]),
            "license": v.model.get("license", fam["license"]), "local": bool(v.model.get("local")),
            "file": os.path.basename(v.file), "added": bool(v.model.get("local")) and v.file in config.load()["local_models"],
            "default": v.quant == preferred_quant(v.model, hw, device, ctx), "tier": v.model["tier"],
            "params_b": v.model["params_b"], "active_b": v.model["active_b"],
            "reason_ahead": v.model["reason_ahead"], "bytes": v.bytes, "installed": bool(path),
            "path": path, **e}


def recommend(entries: list[dict]) -> str | None:
    """The MAFDET experience: the largest model that reasons ahead, at its default quantization,
    whose answer comes within a comfortable wait.

    Preference: (nearly) all on the GPU and under 8 s after Enter; then anything under 15 s; then the
    fastest that fits at all. Nearly: a layer or two on the CPU (a busier desktop) costs a 4B on the
    RTX 2060 about a second, less than falling back to a model half its size.
    """
    pool = [e for e in entries if e["default"] and e["reason_ahead"] and e["placement"] != "too_big"
            and e["with_mafdet_s"] is not None and not e["local"]]
    for ok in (lambda e: e["gpu_share"] >= 0.9 and e["with_mafdet_s"] <= 8,
               lambda e: e["with_mafdet_s"] <= 15):
        good = [e for e in pool if ok(e)]
        if good:
            return max(good, key=lambda e: e["params_b"])["id"]
    return min(pool, key=lambda e: e["with_mafdet_s"])["id"] if pool else None


def best_in_family(entries: list[dict], family: str) -> str | None:
    """The largest model of a family that chats comfortably here: all on the GPU at 20 tokens/s or
    more; then anything at 10 or more; then the fastest that fits."""
    pool = [e for e in entries if e["family"] == family and e["default"] and e["placement"] != "too_big"]
    for ok in (lambda e: e["placement"] == "gpu" and e["decode_tps"] >= 20, lambda e: e["decode_tps"] >= 10):
        good = [e for e in pool if ok(e)]
        if good:
            return max(good, key=lambda e: e["params_b"])["id"]
    return max(pool, key=lambda e: e["decode_tps"])["id"] if pool else None


def listing(hw: Hardware, device: str = "auto", ctx: int = 8192) -> dict:
    own = local_variants()
    entries = [entry(v, hw, device, ctx) for v in variants() + own]
    fams = families() + ([LOCAL_FAMILY] if own else [])
    best = recommend(entries)
    # No "best of the family" among your own files: they are not a family, and you chose them.
    family_best = {f["id"]: best_in_family(entries, f["id"]) if f is not LOCAL_FAMILY else None for f in fams}
    for e in entries:
        e["recommended"] = e["id"] == best
        e["family_best"] = e["id"] == family_best[e["family"]]
    return {"models": entries, "recommended": best, "family_best": family_best, "families": fams,
            "models_dir": str(config.models_dir())}
