"""Find out what an engine actually does, by asking it a handful of cheap questions.

PR #1 covers only the facts llama.cpp needs to decide: the spelling of a prefill-only
request, whether reuse is real and which channel reports it, and whether an edit in the
middle of the text still hits the cached prefix.  The rest (slot affinity, cancelling,
raw completions, tokenizer agreement) is deliberately deferred: a probe validated on one
engine cannot claim anything about the others, so those probes arrive with the second
engine.

Every probe is side-effect free apart from filling the engine's prefix cache with a
throwaway sentence.
"""
from __future__ import annotations

import time

from .engine import EXACT_CHANNELS, Engine, EngineCaps, WarmResult
from .profile import EngineProfile

# A filler long enough that the per-request numbers are not dominated by fixed
# overhead, and short enough to be cheap on a small GPU.
FILLER = ("The cartographer of the northern deltas meticulously recorded every shifting "
          "sandbar, tidal pool, and seasonal current along the coast. ") * 8

# Spellings tried in order, per dialect.  The first that answers with a prefill-only
# request wins; max_tokens1 is the universal fallback (one token, thrown away).
ORDER = {
    "llama": ("n_predict0", "max_tokens1"),
    "openai": ("max_tokens0", "echo0", "max_tokens1"),
}


def _spelling(engine: Engine, ids: list[int]) -> tuple[str, WarmResult]:
    """The first spelling that works, and the result of the attempt that proved it."""
    order = ORDER.get(engine.dialect, ORDER["openai"])
    last: WarmResult | None = None
    for spelling in order:
        res = engine.warm(ids, spelling=spelling)
        if res.detail.get("ok"):
            # The adapter may have fallen back inside the call (auto): record what it used.
            return res.detail.get("spelling", spelling), res
        last = res
    return "none", last or WarmResult()


def probe(engine: Engine, *, url: str, model: str, version: str = "") -> EngineProfile:
    """Run the questions and return a profile.  Never raises on a bad engine."""
    t0 = time.perf_counter()
    caps = EngineCaps()

    ids = engine.tokenize(FILLER)
    caps.tokenize = ids is not None
    if ids is None:
        # Without tokenization the proxy cannot predict reuse; it can still warm.
        ids = []

    # 1. spelling
    spelling, first = _spelling(engine, ids)
    caps.spelling = spelling
    if spelling == "none":
        caps.reuse_channel = "none"
        return EngineProfile(name=engine.name, url=url, model=model, caps=caps,
                             latencies={"probe_ms": (time.perf_counter() - t0) * 1000},
                             version=version)

    # 2. reuse evidence: the same prompt again should be (mostly) already cached
    second = engine.warm(ids, spelling=spelling)
    channel = second.channel if second.cached_tokens is not None else "none"
    if channel == "none" and engine.metrics_text() is not None:
        channel = "metrics"
    caps.reuse_channel = channel if channel in EXACT_CHANNELS else ("metrics" if channel == "metrics" else "none")

    reused = second.cached_tokens or 0
    caps.prefix_caching = reused > 0 or caps.reuse_channel == "metrics"

    # 3. does an edit in the middle still hit the prefix?  (PREFIX, PREFIX+A, PREFIX+B)
    caps.edit_branches = False
    latencies: dict = {}
    if caps.prefix_caching and ids:
        base_ms = second.ms or 1.0
        tail = engine.tokenize(" ALPHA")
        tail2 = engine.tokenize(" BETA")
        if tail and tail2:
            engine.warm(ids + tail, spelling=spelling)
            third = engine.warm(ids + tail2, spelling=spelling)
            hit = third.cached_tokens or 0
            caps.edit_branches = hit >= max(1, int(len(ids) * 0.9))
            latencies = {"base_ms": base_ms, "branch_ms": third.ms}

    return EngineProfile(name=engine.name, url=url, model=model, caps=caps,
                         latencies={**latencies, "probe_ms": (time.perf_counter() - t0) * 1000},
                         version=version)
