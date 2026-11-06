"""Engine adapters, one per spelling of the same three questions.

PR #1 ships only llama.cpp.  The second adapter (an OpenAI-compatible one, for vLLM and
SGLang) arrives with PR #2, together with the probes a second engine makes meaningful:
slot affinity, cancelling mid-generation, raw completions, tokenizer agreement.
"""
from __future__ import annotations

__all__ = ["llama", "make"]


def make(name: str, url: str, **kwargs):
    """Build an adapter by name.  Unknown names are an error, not a guess."""
    from . import llama

    if name in ("llama", "llama.cpp", "llama-server"):
        return llama.LlamaEngine(url, **kwargs)
    raise SystemExit(f"No adapter called {name!r} in PR #1: llama.cpp only (see PROXY.md).")
