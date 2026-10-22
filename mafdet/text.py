"""
The pieces of the engine that need no PyTorch: chat template, tokenization,
common prefix, hold-back and the counters of a prefill engine.

The chat (GUI and TUI) runs on llama.cpp and only needs a tokenizer, so it
imports from here; engine.py re-exports these names for the measurement
commands, which also run on PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

SENTINEL = "<<<USER_TEXT>>>"


def read_sample(name: str) -> str:
    """Raw text of a bundled sample (mafdet/samples), read as package data."""
    return files("mafdet").joinpath("samples", name).read_text(encoding="utf-8")


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


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def stable_ids(full_ids: list[int], holdback: int) -> list[int]:
    """The last `holdback` tokens can still change while the user types."""
    return full_ids[:-holdback] if holdback and len(full_ids) > holdback else ([] if holdback else full_ids)


@dataclass
class Stats:
    forward_passes: int = 0
    tokens_computed: int = 0
    tokens_discarded: int = 0
    syncs: int = 0
