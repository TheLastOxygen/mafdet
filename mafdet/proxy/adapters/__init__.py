"""Engine adapters, one per spelling of the same questions.

llama.cpp has its own (`llama.py`: /tokenize, /apply-template, /completion, the one measured in
research/LOG.md); every other engine goes through the OpenAI-compatible one (`openai.py`: vLLM,
SGLang, Ollama, LM Studio, anything with /v1/chat/completions).  `auto` asks the server which it is.
What each engine answered when probed is data, in its `EngineProfile`, never a code branch here.
"""
from __future__ import annotations

__all__ = ["llama", "openai", "make", "NAMES"]

NAMES = ("auto", "llama", "openai", "vllm", "sglang", "ollama", "lmstudio")
LLAMA = ("llama", "llama.cpp", "llama-server")


def make(name: str, url: str, *, model: str = "", api_key: str = "", slot: int = 0, timeout: float = 600.0):
    """Build an adapter by name.  Unknown names are an error, not a guess; `auto` detects."""
    from . import llama, openai

    version = ""
    if name == "auto":
        found = openai.detect(url, api_key=api_key)
        if found is None:
            raise SystemExit(f"Nothing answers at {url}: is the engine running?")
        name, version = found
    if name in LLAMA:
        engine = llama.LlamaEngine(url, slot=slot, timeout=timeout)
        engine.version = version
        return engine
    if name in openai.FLAVORS:
        return openai.OpenAIEngine(url, model=model, api_key=api_key, flavor=name, version=version, timeout=timeout)
    raise SystemExit(f"No adapter called {name!r}: one of {', '.join(NAMES)} (see PROXY.md).")
