"""What the proxy asks of an engine, and what an engine answers back.

An engine adapter is thin on purpose: it knows how to spell one request for one
backend (llama.cpp's `/completion`, an OpenAI-style `/v1/completions`) and how to
read that backend's answer.  Every *decision* -- which spelling to use, whether
reuse is real, which channel reports it -- comes from `probe.py` and lands in an
`EngineProfile`, so the rest of the proxy never asks an engine what it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol


# The canonical spelling names a profile can carry.  The first four prefill token ids
# the proxy rendered itself; "chat1" sends the draft as a chat request with one token
# of answer, for an engine that applies its own template and gives the proxy no token
# ids (Ollama, LM Studio).  "none" means the probe found no spelling that works.
SPELLINGS = ("n_predict0", "max_tokens0", "echo0", "max_tokens1", "chat1", "none")

# Where a `WarmResult.cached_tokens` number came from.  "timings" (llama.cpp) and
# "usage" (OpenAI's prompt_tokens_details.cached_tokens) are exact; "computed" means
# the engine reports only the prompt tokens it computed this time (the reuse is what
# was sent minus that); "metrics" is a global counter delta (an estimate, never
# grounds for a demotion); "time" means the engine reports nothing and the only
# evidence of reuse is a prefill that got faster; "none" means reuse is unknown.
CHANNELS = ("timings", "usage", "computed", "metrics", "time", "none")

EXACT_CHANNELS = frozenset({"timings", "usage", "computed"})
ESTIMATE_CHANNELS = frozenset({"metrics", "time"})


@dataclass
class WarmResult:
    """What one read-ahead request told us."""

    #: Tokens the engine says it reused from what it already had cached.
    cached_tokens: int | None = None
    #: Tokens it had to compute for this request (or the prompt tokens it counted, when
    #: it does not say which: the probe tells the two apart).
    computed_tokens: int | None = None
    #: Wall time of the request, in milliseconds.
    ms: float = 0.0
    #: Provenance of `cached_tokens` (one of CHANNELS).
    channel: str = "none"
    #: True when the number is a global-counter delta rather than per-request truth.
    estimate: bool = False
    #: Free-form detail for logs (status codes tried, error text, `prefill_ms` when the
    #: engine times its own prompt processing, `prompt_tokens` as it counted them).
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.detail.get("ok"))


@dataclass
class EngineCaps:
    """The facts the probe established for one (engine, model) pair."""

    prefix_caching: bool = False
    spelling: str = "none"            # one of SPELLINGS
    reuse_channel: str = "none"       # one of CHANNELS
    edit_branches: bool = False       # an edited tail still hits the cached prefix
    tokenize: bool = False            # the engine gives token ids for a text
    render: bool = False              # and the ids of the prompt it builds for a chat request
    # The cache holds more than one prompt: another request (a front end's title, another
    # conversation) does not evict this one.  llama.cpp's cache is per slot (False, which is
    # what the second slot is for); vLLM's and SGLang's are shared by every request.
    shared_cache: bool = False
    # Tokens a reuse report may fall short of the prediction by, when the cache works in
    # blocks (vLLM caches whole blocks of 16 tokens): the audit allows that much.
    reuse_slack: int = 0
    # The prompt the proxy renders has the token count the chat endpoint computes: if not,
    # what it warms is not the prefix of the real request, and reading ahead is off.
    tokenizer_agrees: bool = True
    # A completion request takes token ids (thinking ahead needs that).
    raw_ids: bool = False

    @property
    def tier(self) -> str:
        """T1 when read-ahead is exact; T0 when only whole-prompt reuse works."""
        if self.spelling == "none" or self.reuse_channel == "none":
            return "T-1"
        if not self.prefix_caching or not self.tokenizer_agrees:
            return "T-1"
        return "T1" if self.edit_branches else "T0"

    @property
    def by_chat(self) -> bool:
        """The drafts go through the chat endpoint (the engine applies its own template)."""
        return self.spelling == "chat1"


def reused(res: WarmResult, sent: int | None, channel: str) -> int | None:
    """Tokens reused by one warm, from what the engine reported through `channel`:
    the number it gives, or what was sent minus what it computed."""
    if res.cached_tokens is not None:
        return res.cached_tokens
    if channel == "computed" and sent is not None and res.computed_tokens is not None:
        return max(0, sent - res.computed_tokens)
    return None


class Engine(Protocol):
    """The adapter contract.  No method may raise on an engine that answers badly:
    return a status/None instead, because the probe needs to see failures."""

    name: str
    dialect: str          # "llama" or "openai"

    def tokenize(self, text: str) -> list[int] | None:
        """Token ids for `text`, or None when the engine cannot tokenize."""
        ...

    def render(self, messages: list[dict], template_kwargs: dict | None = None,
               tools: list | None = None) -> list[int] | None:
        """The ids of the prompt the engine would build for these messages, with the
        request's template options (`chat_template_kwargs`) and tool definitions.

        Warming anything else warms nothing: the chat template's own tokens are the
        prefix of the real request, so the draft must go through the same template."""
        ...

    def warm(self, ids: list[int], *, spelling: str = "auto") -> WarmResult:
        """Prefill `ids` without generating (or generating one token and dropping it)."""
        ...

    def warm_chat(self, messages: list[dict], template_kwargs: dict | None = None,
                  tools: list | None = None, *, spelling: str = "chat1") -> WarmResult:
        """Optional: the draft as a chat request with one token of answer, for an engine
        that gives the proxy no token ids (it applies its own template)."""
        ...

    def chat(self, body: dict, *, stream: bool = False,
             timeout: float = 600.0) -> tuple[int, dict[str, str], Iterable[bytes]]:
        """Forward an OpenAI-style chat request.  Streaming yields raw bytes."""
        ...

    def models(self) -> tuple[int, dict]:
        """GET /v1/models, as (status, parsed-json-or-{})."""
        ...

    def slots(self) -> int | None:
        """How many requests the engine runs side by side, each with its own cache (llama-server's
        slots), or None when it does not say.  With two or more, the proxy keeps the
        conversation in one and sends a front end's side requests (titles, tags) to another."""
        ...

    def metrics_text(self) -> str | None:
        """GET /metrics if the engine has it, else None (used for the estimate channel)."""
        ...
