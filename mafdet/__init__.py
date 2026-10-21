"""MAFDET: you type, it already knows.

Incremental prefill for local LLMs: the KV cache follows the text while the
user types, so only a handful of tokens are left to process when they hit enter.
"""
__version__ = "0.1.0"

# Kept here (and not in engine.py) so the CLI can show them in --help without
# importing PyTorch, which takes several seconds.
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_GGUF = "Qwen/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf"     # for --backend llama
