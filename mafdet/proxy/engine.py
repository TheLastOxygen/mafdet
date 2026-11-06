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


# The canonical spelling names a profile can carry.  "none" means the probe found
# no prefill-only spelling and the proxy must fall back to a one-token request.
SPELLINGS = ("n_predict0", "max_tokens0", "echo0", "max_tokens1", "none")

# Where a `WarmResult.cached_tokens` number came from.  Only "timings" and "usage"
# are exact; "metrics" is a global counter delta (an estimate, never grounds for a
# demotion); "none" means the engine reports nothing and reuse is unknown.
CHANNELS = ("timings", "usage", "metrics", "none")

EXACT_CHANNELS = frozenset({"timings", "usage"})


@dataclass
class WarmResult:
    """What one read-ahead request told us."""

    #: Tokens the engine says it reused from what it already had cached.
    cached_tokens: int | None = None
    #: Tokens it had to compute for this request.
    computed_tokens: int | None = None
    #: Wall time of the request, in milliseconds.
    ms: float = 0.0
    #: Provenance of `cached_tokens` (one of CHANNELS).
    channel: str = "none"
    #: True when the number is a global-counter delta rather than per-request truth.
    estimate: bool = False
    #: Free-form detail for logs (status codes tried, error text).
    detail: dict = field(default_factory=dict)


@dataclass
class EngineCaps:
    """The facts the probe established for one (engine, model) pair."""

    prefix_caching: bool = False
    spelling: str = "none"            # one of SPELLINGS
    reuse_channel: str = "none"       # one of CHANNELS
    edit_branches: bool = False       # an edited tail still hits the cached prefix
    tokenize: bool = False

    @property
    def tier(self) -> str:
        """T1 when read-ahead is exact; T0 when only whole-prompt reuse works."""
        if self.spelling == "none" or self.reuse_channel == "none":
            return "T-1"
        if not self.prefix_caching:
            return "T-1"
        return "T1" if self.edit_branches else "T0"


class Engine(Protocol):
    """The adapter contract.  No method may raise on an engine that answers badly:
    return a status/None instead, because the probe needs to see failures."""

    name: str
    dialect: str          # "llama" or "openai"

    def tokenize(self, text: str) -> list[int] | None:
        """Token ids for `text`, or None when the engine cannot tokenize."""
        ...

    def render(self, messages: list[dict], template_kwargs: dict | None = None) -> list[int] | None:
        """The ids of the prompt the engine would build for these messages, with the
        request's template options (`chat_template_kwargs`).

        Warming anything else warms nothing: the chat template's own tokens are the
        prefix of the real request, so the draft must go through the same template."""
        ...

    def warm(self, ids: list[int], *, spelling: str = "auto") -> WarmResult:
        """Prefill `ids` without generating (or generating one token and dropping it)."""
        ...

    def chat(self, body: dict, *, stream: bool = False,
             timeout: float = 600.0) -> tuple[int, dict[str, str], Iterable[bytes]]:
        """Forward an OpenAI-style chat request.  Streaming yields raw bytes."""
        ...

    def models(self) -> tuple[int, dict]:
        """GET /v1/models, as (status, parsed-json-or-{})."""
        ...

    def metrics_text(self) -> str | None:
        """GET /metrics if the engine has it, else None (used for the estimate channel)."""
        ...
