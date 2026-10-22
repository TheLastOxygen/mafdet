"""
A short benchmark of an installed model on this machine: how fast it reads a prompt and writes.

About half a minute: llama-server starts with the same split between GPU and CPU
as the chat (-ngl auto, the same context), reads a prompt of about 800 tokens
and writes 128. The speeds come from the server's own timings, as in
`mafdet race`, and replace the catalog's estimate for this model on this
machine (bench.json).
"""
from __future__ import annotations

import time
from typing import Callable

from . import catalog
from .hardware import detect

BENCH_PORT = 8091                  # not the chat's 8089: the two never share a server
DECODE_TOKENS = 128


def run(variant_id: str, device: str = "auto", ctx: int = 8192,
        progress: Callable[[str, float], None] = lambda stage, done: None) -> dict:
    """Measure one variant; returns (and saves) {prefill_tps, decode_tps, date}."""
    from ..llama_backend import LlamaServer, pick_device
    from ..text import read_sample

    v = catalog.find(variant_id)
    if v is None:
        raise ValueError(f"Unknown model: {variant_id}")
    path = catalog.local_path(v)
    if path is None:
        raise FileNotFoundError(f"{v.file} is not downloaded")
    dev = pick_device(device)
    progress("starting", 0.05)
    server = LlamaServer(path, dev, ctx=ctx, port=BENCH_PORT)
    try:
        prompt = read_sample("documento.txt")
        progress("prefill", 0.35)
        prefill = 0.0
        for _ in range(2):                 # the first read also warms up the kernels
            t = server.call("/completion", {"prompt": prompt, "n_predict": 0, "cache_prompt": False})["timings"]
            prefill = max(prefill, t["prompt_per_second"])
        progress("decode", 0.65)
        decode = 0.0
        for _ in range(2):
            t = server.call("/completion", {"prompt": "Write a long story about a lighthouse keeper.",
                                            "n_predict": DECODE_TOKENS, "temperature": 0, "ignore_eos": True,
                                            "cache_prompt": False})["timings"]
            decode = max(decode, t["predicted_per_second"])
    finally:
        server.close()
    result = {"prefill_tps": round(prefill), "decode_tps": round(decode, 1), "date": time.strftime("%Y-%m-%d")}
    catalog.save_measurement(v, detect(), dev, result)
    progress("done", 1.0)
    return result
