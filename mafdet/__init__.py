"""MAFDET: you type, it already knows.

Local LLMs that read and think while you type: the KV cache follows the text
as it is written, and a reasoning model starts thinking about the message
before it is sent. On Enter, little is left to do.
"""
__version__ = "0.1.0"

# Kept here (and not in engine.py) so the CLI can show them in --help without
# importing PyTorch, which takes several seconds.
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_GGUF = "Qwen/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf"     # for --backend llama
