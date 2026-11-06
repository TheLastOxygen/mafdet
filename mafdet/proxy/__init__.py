"""The MAFDET proxy: reading ahead for any engine, behind an OpenAI-compatible server.

See PROXY.md at the repository root for what this does and PROXY.md "What the engine
must offer" for the requirements.  This package is deliberately separate from the app:
it never imports `mafdet.watch` or `mafdet.engine`, and it adds no runtime dependency.  Only
`launch.py` (`mafdet proxy`) uses the app's `LlamaServer` and catalog, to start llama-server.
"""
from __future__ import annotations

__all__ = ["ENGINE_NAME", "TIERS"]

#: What the proxy calls itself in logs and in /proxy/stats.
ENGINE_NAME = "mafdet-proxy"

#: Capability tiers, best first.  T2 is thinking ahead (--think-ahead, llama.cpp only so far).
TIERS = ("T2", "T1", "T0", "T-1")


def run(args) -> int:
    """`mafdet proxy` (the CLI imports mafdet.<command> and calls run)."""
    from .launch import run as launch
    return launch(args)
