"""Find out what an engine actually does, by asking it a handful of cheap questions.

The questions, in order:
  1. tokens: does the engine tokenize a text, and render a chat request to token ids?  With both,
     the proxy warms ids it rendered itself and predicts the reuse exactly; without, the drafts go
     through the chat endpoint (`chat1`) and the engine's own template, with no prediction.
  2. spelling: the first prefill-only request that works (`n_predict: 0`, `max_tokens: 0`, echo, or
     one token thrown away).
  3. reuse, and which channel reports it: the same prompt again should be (mostly) already cached.
     An exact count (llama.cpp's timings, OpenAI's usage, or the prompt tokens the engine computed),
     a `/metrics` counter, or only a prefill that got faster (an engine that reports nothing).
  4. edit branches: PREFIX+A then PREFIX+B still hits the prefix (what typing is).
  5. how short a report may fall: a shorter prefix again (a cache that works in blocks, as vLLM's
     16-token blocks, reports a little less than the prediction; the audit then allows that much).
  6. a shared cache: an unrelated prompt in between, then the prefix again: is it still there?
     (llama.cpp: no, one cache per slot, which is what the second slot is for; vLLM, SGLang: yes.)
  7. tokenizer agreement (ids only): the prompt the proxy renders has the count the chat endpoint
     computes; otherwise what it warms is not the prefix of the real request, and reading ahead is off.

Every probe is side-effect free apart from filling the engine's prefix cache with throwaway
sentences, and a few one-token answers where the engine has no prefill-only request.
"""
from __future__ import annotations

import time

from .audit import TOLERANCE
from .engine import ESTIMATE_CHANNELS, Engine, EngineCaps, WarmResult, reused
from .profile import EngineProfile

# A filler long enough that a cold prefill is measurably slower than a cached one (the time
# channel), and short enough to be cheap on a small GPU: ~400 tokens.
FILLER = ("The cartographer of the northern deltas meticulously recorded every shifting "
          "sandbar, tidal pool, and seasonal current along the coast. ") * 16
# An unrelated prompt, for the shared-cache question.
OTHER = ("An unrelated inventory of the eastern warehouse lists crates, pallets and loose cargo "
         "by weight, by destination and by the day they arrived. ") * 8
WAKE = "Reply with OK."

# Spellings tried in order, per dialect.  The first that answers with a prefill-only
# request wins; max_tokens1 is the universal fallback (one token, thrown away).
ORDER = {
    "llama": ("n_predict0", "max_tokens1"),
    "openai": ("max_tokens0", "echo0", "max_tokens1"),
}

# The time channel: a prefill in less than half the cold one, and at least this much faster.
TIME_RATIO = 0.5
TIME_MIN_MS = 40


def _msg(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _spelling(engine: Engine, ids: list[int]) -> tuple[str, WarmResult]:
    """The first spelling that works, and the result of the attempt that proved it."""
    order = ORDER.get(engine.dialect, ORDER["openai"])
    last: WarmResult | None = None
    for spelling in order:
        res = engine.warm(ids, spelling=spelling)
        if res.ok:
            # The adapter may have fallen back inside the call (auto): record what it used.
            return res.detail.get("spelling", spelling), res
        last = res
    return "none", last or WarmResult()


def _prefill_ms(res: WarmResult) -> float:
    """The engine's own prompt-processing time when it gives one, else the request's wall time."""
    return float(res.detail.get("prefill_ms", res.ms))


def _faster(cold: WarmResult, warm: WarmResult) -> bool:
    a, b = _prefill_ms(cold), _prefill_ms(warm)
    return a - b >= TIME_MIN_MS and b < TIME_RATIO * a


def _channel(engine: Engine, first: WarmResult, second: WarmResult) -> tuple[str, bool]:
    """From the same prompt warmed twice: (the channel that reports reuse, whether it saw any)."""
    if second.cached_tokens is not None:
        return second.channel, second.cached_tokens > 0
    a, b = first.computed_tokens, second.computed_tokens
    if a is not None and b is not None and a > TOLERANCE + 1 and b <= max(TOLERANCE + 1, 0.1 * a):
        return "computed", True             # the engine counts only the prompt tokens it computed
    if engine.metrics_text() is not None:
        return "metrics", True              # counters only: an estimate, taken on trust
    if _faster(first, second):
        return "time", True
    return "none", False


def _seen(channel: str, cold: WarmResult, res: WarmResult, sent: int | None = None) -> bool:
    """Did `res` find (most of) its prefix cached, by the evidence `channel` gives?  `sent` is the
    number of tokens sent when the proxy knows it; otherwise the cold run's count stands for it."""
    size = sent if sent is not None else (cold.detail.get("prompt_tokens") or cold.computed_tokens or 0)
    if channel in ("timings", "usage", "metrics"):
        n = res.cached_tokens
        return n is not None and n >= 0.9 * size
    if channel == "computed":
        c = res.computed_tokens
        return c is not None and c <= max(TOLERANCE + 1, 0.1 * size)
    if channel == "time":
        return _faster(cold, res)
    return False


def _shortfall(channel: str, res: WarmResult, sent: int, predicted: int) -> int:
    """Tokens the report fell short of the prediction by (0 for an estimate)."""
    if channel in ESTIMATE_CHANNELS:
        return 0
    n = reused(res, sent, channel)
    return 0 if n is None else max(0, predicted - n)


def _allowance(shortfall: int) -> int:
    """What the audit will allow: nothing within the usual tolerance, else twice the next power of
    two (block caches work in powers of two, and the probe sees one residue of the block, not it)."""
    if shortfall <= TOLERANCE:
        return 0
    p = 1
    while p < shortfall:
        p *= 2
    return 2 * p


def _chat_count(engine: Engine, messages: list[dict]) -> int | None:
    """The prompt tokens the chat endpoint counts for `messages` (one token of answer), or None."""
    warm_chat = getattr(engine, "warm_chat", None)
    if not callable(warm_chat):
        return None
    res = warm_chat(messages)
    n = res.detail.get("prompt_tokens") if res.ok else None
    return n if isinstance(n, int) else None


def _by_ids(engine: Engine, caps: EngineCaps, ids: list[int], rendered: list[int], lat: dict) -> None:
    # 7. tokenizer agreement, cold, before the filler is cached by anything
    counted = _chat_count(engine, _msg(FILLER))
    if counted is not None:
        caps.tokenizer_agrees = abs(counted - len(rendered)) <= TOLERANCE
        lat["chat_prompt_tokens"], lat["rendered_tokens"] = counted, len(rendered)
    # 2. spelling
    spelling, first = _spelling(engine, ids)
    caps.spelling = spelling
    if spelling == "none":
        lat["why"] = first.detail.get("why", "")
        return
    caps.raw_ids = True
    # 3. reuse evidence: the same prompt again should be (mostly) already cached
    second = engine.warm(ids, spelling=spelling)
    caps.reuse_channel, caps.prefix_caching = _channel(engine, first, second)
    if not caps.prefix_caching:
        return
    n = len(ids)
    shortfall = _shortfall(caps.reuse_channel, second, n, n - 1)
    lat["base_ms"] = second.ms or 1.0
    # 4. does an edit in the middle still hit the prefix?  (PREFIX, PREFIX+A, PREFIX+B)
    tail, tail2 = engine.tokenize(" ALPHA"), engine.tokenize(" BETA")
    if not (tail and tail2):
        caps.reuse_slack = _allowance(shortfall)
        return
    engine.warm(ids + tail, spelling=spelling)
    third = engine.warm(ids + tail2, spelling=spelling)
    caps.edit_branches = _seen(caps.reuse_channel, first, third, sent=n + len(tail2))
    shortfall = max(shortfall, _shortfall(caps.reuse_channel, third, n + len(tail2), n - 1))
    lat["branch_ms"] = third.ms
    # 5. a shorter prefix: by how much may a report fall short (a cache that works in blocks)?
    if n > 20:
        shorter = engine.warm(ids[:-9], spelling=spelling)
        shortfall = max(shortfall, _shortfall(caps.reuse_channel, shorter, n - 9, n - 10))
    caps.reuse_slack = _allowance(shortfall)
    # 6. a shared cache: an unrelated prompt in between, then the prefix again
    other = engine.tokenize(OTHER)
    if other:
        engine.warm(other, spelling=spelling)
        again = engine.warm(ids + tail2, spelling=spelling)
        caps.shared_cache = _seen(caps.reuse_channel, first, again, sent=n + len(tail2))


def _by_chat(engine: Engine, caps: EngineCaps, lat: dict) -> None:
    """The engine gives no token ids: the drafts go through its chat endpoint."""
    caps.spelling = "chat1"
    engine.warm_chat(_msg(WAKE))                     # a GPU idle for a while answers its first request slowly
    first = engine.warm_chat(_msg(FILLER))
    if not first.ok:
        caps.spelling = "none"
        lat["why"] = first.detail.get("why", "")
        return
    second = engine.warm_chat(_msg(FILLER))
    caps.reuse_channel, caps.prefix_caching = _channel(engine, first, second)
    if not caps.prefix_caching:
        return
    lat["base_ms"] = second.ms or 1.0
    engine.warm_chat(_msg(FILLER + " ALPHA"))
    third = engine.warm_chat(_msg(FILLER + " BETA"))
    caps.edit_branches = _seen(caps.reuse_channel, first, third)
    lat["branch_ms"] = third.ms
    engine.warm_chat(_msg(OTHER))
    again = engine.warm_chat(_msg(FILLER + " BETA"))
    caps.shared_cache = _seen(caps.reuse_channel, first, again)


def probe(engine: Engine, *, url: str, model: str, version: str = "") -> EngineProfile:
    """Run the questions and return a profile.  Never raises on a bad engine."""
    t0 = time.perf_counter()
    caps = EngineCaps()
    lat: dict = {}
    version = version or str(getattr(engine, "version", "") or "")

    # 1. tokens: ids of a text, and of the prompt the engine builds for a chat request
    ids = engine.tokenize(FILLER)
    caps.tokenize = ids is not None
    rendered = engine.render(_msg(FILLER)) if ids else None
    caps.render = bool(rendered)

    if caps.render:
        _by_ids(engine, caps, ids, rendered, lat)
    elif callable(getattr(engine, "warm_chat", None)):
        _by_chat(engine, caps, lat)
    # else: no ids and no chat warm: nothing the proxy can do (spelling "none", tier T-1)

    lat["probe_ms"] = (time.perf_counter() - t0) * 1000
    return EngineProfile(name=engine.name, url=url, model=model, caps=caps, latencies=lat, version=version)
