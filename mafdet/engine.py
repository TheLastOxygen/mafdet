"""
Shared pieces for every command: model loading, GPU timing, chat template
handling and the incremental prefill engine.
"""
from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field
from importlib.resources import files

import torch
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from . import DEFAULT_GGUF, DEFAULT_MODEL  # noqa: F401  (re-exported for the other modules)

SENTINEL = "<<<USER_TEXT>>>"


def read_sample(name: str) -> str:
    """Raw text of a bundled sample (mafdet/samples), read as package data."""
    return files("mafdet").joinpath("samples", name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Device and model
# --------------------------------------------------------------------------

def pick_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "You asked for --device cuda, but PyTorch cannot see the GPU. "
            "Run `mafdet awaken`: you probably installed the CPU-only build of PyTorch."
        )
    return requested


def pick_dtype(device: str, requested: str) -> torch.dtype:
    if requested == "auto":
        # The 2060 (Turing) has no bf16 support: fp16 on GPU, fp32 on CPU.
        return torch.float16 if device == "cuda" else torch.float32
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[requested]


def fix_sdpa_for_old_gpus(device: str):
    # With an empty cache, transformers asks SDPA for native GQA (enable_gqa=True),
    # expecting Flash Attention. Before Ampere (sm80) there is no Flash kernel and
    # the memory-efficient kernel does not support enable_gqa, so PyTorch falls back
    # to the "math" backend, which builds the n×n score matrix in fp32 (O(n²) in
    # time and memory). Without native GQA, transformers repeats K/V itself and
    # SDPA uses the fused fp16 kernel.
    if device != "cuda" or torch.cuda.get_device_capability()[0] >= 8:
        return
    try:
        from transformers.integrations import sdpa_attention
    except ImportError:
        return
    if hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        sdpa_attention.use_gqa_in_sdpa = lambda *a, **kw: False


def load(model_id: str, device: str, dtype: torch.dtype):
    fix_sdpa_for_old_gpus(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    # Since 4.56 the argument is called `dtype`; before it was `torch_dtype`.
    # Older versions silently ignore the wrong name and load in fp32, which on
    # a 2060 means running out of memory right away.
    kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_id, **{kw: dtype}, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    return tok, model


def model_label(args) -> str:
    """The model actually measured, for printouts and CSV files."""
    return args.gguf if args.backend == "llama" else args.model


def setup(args, n_slots: int = 1):
    """Return (tokenizer, model, device, dtype).

    With --backend llama the "model" is a LlamaServer, the device is
    'llama-cuda' or 'llama-cpu' and the dtype is the quantization (e.g. Q4_K_M).
    n_slots is for callers that run several engines side by side, each with
    its own cache.
    """
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    if args.backend == "llama":
        from .llama_backend import setup_llama
        return setup_llama(args, device, n_slots)
    dtype = pick_dtype(device, args.dtype)
    tok, model = load(args.model, device, dtype)
    if getattr(args, "compile", False):
        if device != "cuda":
            raise SystemExit("--compile is only supported on GPU.")
        model = compile_model(model, device)
    return tok, model, device, dtype


def compile_model(model, device: str):
    """torch.compile cuts the fixed cost of every forward pass.

    In eager mode each forward pass launches ~2000 kernels, one per operation,
    and the CPU takes longer to queue them than the GPU takes to run them.
    Inductor fuses the elementwise operations (norms, activations, rotary,
    dtype conversions) into a few kernels.

    Dynamo specializes the graph on a few variants (empty or full cache, one
    token or more), and each one costs tens of seconds the first time: they
    are all compiled here, so it never happens in the middle of a measurement.
    Lengths stay symbolic (dynamic=True), and IncrementalPrefill keeps the
    cache contiguous, because every different memory layout would be one
    more variant.
    """
    print("Compiling the model (the first time takes a few minutes) ...", flush=True)
    t0 = time.perf_counter()
    # There are about ten variants: the default limit (8) is not enough, and past
    # the limit Dynamo falls back to eager (correct, but slow).
    import torch._dynamo
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 32
    compiled = torch.compile(model, dynamic=True)
    eng = IncrementalPrefill(compiled, device)
    ids = list(range(1000, 1100))                          # arbitrary tokens
    # Dynamo treats sizes 0 and 1 as special cases. At the start of typing the
    # stable part of the text can be 1 token or none (the Qwen template opens
    # with 3 tokens, as many as the hold-back), so the cache goes through
    # lengths 1 and 0.
    sequence = [
        [ids[:1]],                          # cache with no layers, one token
        [ids[:1], ids[:2]],                 # 1-token cache, one token
        [ids[:1], ids[:3]],                 # 1-token cache, several tokens
        [ids[:1], [], ids[:1]],             # 0-token cache (full crop), one token
        [ids[:1], [], ids[:3]],             # 0-token cache, several tokens
        [ids[:40], ids[:60], ids[:61],      # empty, then full cache: several tokens, one token
         ids[:50] + ids[70:80],             # right after a crop, several tokens
         ids[:50] + ids[70:79] + [ids[95]]],  # after a crop, one token
    ]
    for steps in sequence:
        eng.reset()
        for target in steps:
            eng.sync(target)
    sync_device(device)
    print(f"Compiled in {time.perf_counter() - t0:.0f} s", flush=True)
    return compiled


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def sync_device(device: str):
    # CUDA calls are asynchronous: without synchronizing, you only measure
    # how long it takes to launch the kernels, not to run them.
    if device == "cuda":
        torch.cuda.synchronize()


def timed(device: str, fn, *a, **kw):
    sync_device(device)
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    sync_device(device)
    return out, (time.perf_counter() - t0) * 1000.0


def peak_mem_mb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    return float("nan")


def device_label(device: str) -> str:
    # With llama-server the GPU belongs to another process: no CUDA calls here,
    # they would create a context and take VRAM for nothing.
    return {"cuda": torch.cuda.get_device_name(0) if device == "cuda" else "", "cpu": "CPU",
            "llama-cuda": "llama.cpp CUDA", "llama-cpu": "llama.cpp CPU"}[device]


# --------------------------------------------------------------------------
# Chat template
# --------------------------------------------------------------------------

@dataclass
class Template:
    """The chat template split into: [prefix] user text [suffix].

    While the user types, only prefix + text go into the cache; the suffix
    (end of the message and start of the assistant's turn) is added on send.
    """
    prefix: str
    suffix: str

    @classmethod
    def from_tokenizer(cls, tok, thinking: bool = True) -> "Template":
        msgs = [{"role": "user", "content": SENTINEL}]
        s = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        if s.count(SENTINEL) != 1:
            raise RuntimeError("Unexpected chat template: cannot locate the user text.")
        pre, suf = s.split(SENTINEL)
        return cls(pre, suf)


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False).input_ids


# --------------------------------------------------------------------------
# Incremental prefill engine
# --------------------------------------------------------------------------

def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class Stats:
    forward_passes: int = 0
    tokens_computed: int = 0
    tokens_discarded: int = 0
    syncs: int = 0


@dataclass
class IncrementalPrefill:
    """Keeps the KV cache aligned with a token sequence that changes over time.

    Correctness depends only on the common-prefix check: any edit (appends,
    backspaces, edits in the middle, pastes) brings the cache to match
    `target_ids` exactly. Heuristics (hold-back, debounce) only affect how much
    work is wasted.
    """
    model: torch.nn.Module
    device: str
    max_chunk: int = 2048          # caps peak memory on long prefills
    cache: DynamicCache = field(default=None)
    ids: list[int] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self):
        self.reset()
        # A compiled model only exposes forward(*args, **kwargs): the real
        # signature has to be read on the original model.
        params = inspect.signature(getattr(self.model, "_orig_mod", self.model).forward).parameters
        # Compute logits for the last position only: with a ~150k-entry
        # vocabulary, the logits of 8000 tokens in fp32 would take ~4.8 GB.
        if "logits_to_keep" in params:
            self._keep = {"logits_to_keep": 1}
        elif "num_logits_to_keep" in params:
            self._keep = {"num_logits_to_keep": 1}
        else:
            self._keep = {}
        self._compiled = hasattr(self.model, "_orig_mod")

    def _contiguous_cache(self):
        """With torch.compile, bring the cache back to a contiguous layout.

        After a crop the cache is a view into the longer tensor (old strides);
        after the first prefill it is the transpose of the projection output.
        The values do not change, but Dynamo specializes the graph on strides:
        every different layout would cost a recompilation. .contiguous() does
        not copy when the tensor is already contiguous, as after every cat.
        """
        if not self._compiled:
            return
        for layer in getattr(self.cache, "layers", []):
            if getattr(layer, "keys", None) is not None and layer.keys.dim() == 4:
                layer.keys = layer.keys.contiguous()
                layer.values = layer.values.contiguous()

    def reset(self):
        self.cache = DynamicCache()
        self.ids = []
        self.stats = Stats()

    @torch.no_grad()
    def _forward(self, new_ids: list[int]) -> torch.Tensor:
        logits = None
        for i in range(0, len(new_ids), self.max_chunk):
            chunk = torch.tensor([new_ids[i:i + self.max_chunk]], device=self.device)
            out = self.model(input_ids=chunk, past_key_values=self.cache, use_cache=True, **self._keep)
            self.cache = out.past_key_values
            self._contiguous_cache()
            logits = out.logits[0, -1]
            self.stats.forward_passes += 1
        return logits

    def sync(self, target_ids: list[int], need_logits: bool = False) -> torch.Tensor | None:
        """Align the cache with `target_ids`.

        With need_logits=True it returns the logits of the next token: in that
        case the last token is always (re)computed, because the logits depend
        on the model's output for it.
        """
        self.stats.syncs += 1
        n = common_prefix_len(self.ids, target_ids)
        if need_logits:
            if not target_ids:
                raise ValueError("Empty sequence")
            n = min(n, len(target_ids) - 1)
        if len(self.ids) > n:
            remove = len(self.ids) - n
            self.stats.tokens_discarded += remove
            # A negative value means "drop the last k tokens" in both transformers
            # 4.x and 5.x (5.x no longer accepts a positive value, meant as the
            # final length).
            self.cache.crop(-remove)
            self._contiguous_cache()
        new = target_ids[n:]
        logits = None
        if new:
            logits = self._forward(new)
            self.stats.tokens_computed += len(new)
        self.ids = list(target_ids)
        return logits

    @torch.no_grad()
    def greedy(self, first_logits: torch.Tensor, n_tokens: int) -> list[int]:
        """Generate n_tokens greedily, starting from the logits of the first token."""
        out = []
        logits = first_logits
        for _ in range(n_tokens):
            t = int(torch.argmax(logits))
            out.append(t)
            logits = self._forward([t])
            self.ids.append(t)
        return out


def make_engine(model, device: str, **kw):
    """The right engine for the backend: IncrementalPrefill (HF) or LlamaPrefill (llama-server)."""
    if device.startswith("llama-"):
        from .llama_backend import LlamaPrefill
        return LlamaPrefill(model, slot=model.take_slot())
    return IncrementalPrefill(model, device, **kw)


# The "logits" are either a tensor over the whole vocabulary (HF) or the
# log-probabilities of the top candidates returned by llama-server.

def is_top_logprobs(logits) -> bool:
    return hasattr(logits, "logprobs")


def argmax_id(logits) -> int:
    if is_top_logprobs(logits):
        return max(logits.logprobs, key=logits.logprobs.get)
    return int(torch.argmax(logits))


def logits_finite(logits) -> bool:
    if is_top_logprobs(logits):
        return all(v == v and abs(v) != float("inf") for v in logits.logprobs.values())
    return bool(torch.isfinite(logits.float()).all())


def to_host(logits):
    """Copy to the CPU (HF) for comparison; log-probabilities are already Python data."""
    return logits if is_top_logprobs(logits) else logits.float().cpu()


def stable_ids(full_ids: list[int], holdback: int) -> list[int]:
    """The last `holdback` tokens can still change while the user types."""
    return full_ids[:-holdback] if holdback and len(full_ids) > holdback else ([] if holdback else full_ids)


def compare_logits(a: torch.Tensor, b: torch.Tensor, tol: float | None = None) -> dict:
    """Compare incremental logits (a) with logits computed from scratch (b).

    The argmax alone is not enough: a wrong cache can shift all the logits
    while leaving the top token unchanged by chance. That is why the maximum
    difference, relative to the spread of the logits, is checked too.
    """
    if is_top_logprobs(a):
        return compare_logprobs(a, b, tol)
    tol = 0.1 if tol is None else tol
    a32, b32 = a.float(), b.float()
    finite = bool(torch.isfinite(a32).all() and torch.isfinite(b32).all())
    top2 = torch.topk(b32, 2).values
    res = {
        "finite": finite,
        "same_argmax": int(torch.argmax(a32)) == int(torch.argmax(b32)),
        "max_abs_diff": float((a32 - b32).abs().max()) if finite else float("nan"),
        "baseline_margin": float(top2[0] - top2[1]),
    }
    res["rel_diff"] = res["max_abs_diff"] / max(float(b32.std()), 1e-6) if finite else float("nan")
    if not finite:
        res["verdict"] = "NaN"
    elif res["rel_diff"] > tol:
        res["verdict"] = "ERROR"
    elif res["same_argmax"]:
        res["verdict"] = "ok"
    else:
        # different argmax with nearly identical logits: the top two candidates were nearly tied
        res["verdict"] = "tie"
    return res


# Threshold for gross errors: about twice the largest noise (0.12) measured by
# computing the same 20 prompts of `mafdet judge` in two equally valid ways
# (different physical batch sizes, or half + rest).
LLAMA_TOL = 0.25


def compare_logprobs(a, b, tol: float | None = None, k: int = 10) -> dict:
    """Like compare_logits, for the log-probabilities returned by llama-server.

    With 4-bit weights llama.cpp also quantizes the activations, and the
    kernel changes with the batch size (matrix×vector below ~8 tokens, MMQ
    above): computing the same text in different chunks moves the
    log-probabilities of unlikely candidates by up to 0.5 nats, even from
    scratch with a different physical batch (no cache reused at all). So the
    criterion here is the shifted probability (total variation over the
    returned candidates), with a threshold that only catches gross errors.
    Subtle cache-reuse bugs are caught by the exact check in LlamaPrefill
    (reuse_mismatches). max_abs_diff is still the largest difference in nats
    over the top k, for information.
    """
    tol = LLAMA_TOL if tol is None else tol
    ref = sorted(b.logprobs.items(), key=lambda kv: -kv[1])
    finite = logits_finite(a) and logits_finite(b)
    diffs = [abs(a.logprobs[i] - lp) if i in a.logprobs else float("inf") for i, lp in ref[:k]]
    ids = set(a.logprobs) | set(b.logprobs)
    # A candidate missing from one side's top list has a probability below the
    # returned ones: it counts as zero (slightly overestimating the distance).
    shifted = 0.5 * sum(abs(math.exp(a.logprobs.get(i, -math.inf)) - math.exp(b.logprobs.get(i, -math.inf)))
                        for i in ids)
    res = {
        "finite": finite,
        "same_argmax": argmax_id(a) == argmax_id(b),
        "max_abs_diff": max(diffs) if finite else float("nan"),
        "baseline_margin": ref[0][1] - ref[1][1],
    }
    res["rel_diff"] = shifted if finite else float("nan")
    if not finite:
        res["verdict"] = "NaN"
    elif res["rel_diff"] > tol:
        res["verdict"] = "ERROR"
    elif res["same_argmax"]:
        res["verdict"] = "ok"
    else:
        res["verdict"] = "tie"
    return res
