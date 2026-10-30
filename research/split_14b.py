"""Ways to split Qwen3-14B between a small GPU and the CPU, all within the same VRAM.

On a 12 GB card, llama.cpp's --fit-target stands in for a smaller one: with -fitt 7100 the fit leaves
the 14B Q4_K_M 15 of 41 layers on the GPU (3903 MiB), as on the RTX 2060 6 GB with the desktop's 1.4 GB.
Each configuration is a set of llama-server options; the ones that place tensors by hand (-ngl all
-ncffn N: attention of every layer on the GPU, the FFNs of the first N layers on the CPU) take an N
that keeps the same VRAM. Measured, alternating configurations each round: prompt reading (an ~800-token
text), and 128 tokens written after a short prompt and after a long one (~1600 tokens, about a chat's
notes and reasoning).

    python research/split_14b.py --rounds 2 --out research/data/split_14b_4070s.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mafdet.llama_backend import LlamaServer, resolve_gguf
from mafdet.text import read_sample

Q4 = "Qwen/Qwen3-14B-GGUF:Qwen3-14B-Q4_K_M.gguf"
IQ4 = "unsloth/Qwen3-14B-GGUF:Qwen3-14B-IQ4_XS.gguf"
FIT = ["-fitt", "7100"]
KV8 = ["-ctk", "q8_0", "-ctv", "q8_0"]
CONFIGS = {
    "layers": (Q4, FIT),
    "layers kv8": (Q4, FIT + KV8),
    "ffn-cpu": (Q4, ["-fit", "off", "-ngl", "all", "-ncffn", "38"]),
    "ffn-cpu kv8": (Q4, ["-fit", "off", "-ngl", "all", "-ncffn", "34"] + KV8),
    "iq4xs layers": (IQ4, FIT),
    "iq4xs layers kv8": (IQ4, FIT + KV8),
    "iq4xs ffn-cpu kv8": (IQ4, ["-fit", "off", "-ngl", "all", "-ncffn", "32"] + KV8),
}


def vram(log: str) -> int:
    """MiB the server put on the GPU: model, KV cache and compute buffers."""
    return round(sum(float(x) for x in re.findall(r"CUDA0 (?:model|KV|compute) buffer size =\s*([\d.]+) MiB", log)))


def run(name: str, gguf: str, extra: list[str], doc: str) -> dict:
    server = LlamaServer(resolve_gguf(gguf), "cuda", ctx=8192, port=8098, extra_args=("-lv", "4", *extra),
                         log_name="server-split")
    try:
        prefill = server.call("/completion", {"prompt": doc, "n_predict": 0, "cache_prompt": False})
        short = server.call("/completion", {"prompt": "Write a long story about a lighthouse keeper.",
                                            "n_predict": 128, "temperature": 0, "ignore_eos": True,
                                            "cache_prompt": False})
        long = server.call("/completion", {"prompt": doc + "\n\n" + doc + "\n\nWrite a long story about it.",
                                           "n_predict": 128, "temperature": 0, "ignore_eos": True,
                                           "cache_prompt": False})
    finally:
        server.close()
    log = open(server.log_path, encoding="utf-8", errors="replace").read()
    layers = re.search(r"offloaded (\d+/\d+) layers", log)
    return {"config": name, "gguf": gguf, "args": " ".join(extra), "gpu_mib": vram(log),
            "layers_on_gpu": layers.group(1) if layers else "?",
            "prefill_tps": round(prefill["timings"]["prompt_per_second"]),
            "decode_short_tps": round(short["timings"]["predicted_per_second"], 2),
            "long_prompt_tokens": long["timings"]["prompt_n"],
            "decode_long_tps": round(long["timings"]["predicted_per_second"], 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    doc = read_sample("documento.txt")
    rows = []
    for r in range(a.rounds):
        for name in a.configs.split(","):
            gguf, extra = CONFIGS[name]
            row = {"round": r, **run(name, gguf, extra, doc)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if a.out:
                json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1)
