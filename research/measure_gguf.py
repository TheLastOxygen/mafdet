"""The app's Measure (mafdet/core/bench.py) on any GGUF, not only the catalog's: prompt of ~800 tokens read
twice, 128 tokens written twice at temperature 0, best of each; plus where llama.cpp put the layers.
Same split as the chat (-ngl auto, ctx 8192). Used to compare quantizations of one model:

    python research/measure_gguf.py Qwen/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf unsloth/Qwen3-4B-GGUF:Qwen3-4B-IQ4_XS.gguf
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mafdet.core.bench import BENCH_PORT, DECODE_TOKENS
from mafdet.llama_backend import LlamaServer, resolve_gguf
from mafdet.text import read_sample


def measure(spec: str, ctx: int, extra: list[str]) -> dict:
    server = LlamaServer(resolve_gguf(spec), "cuda", ctx=ctx, port=BENCH_PORT, extra_args=("-lv", "4", *extra),
                         log_name="server-measure")
    try:
        prompt = read_sample("documento.txt")
        prefill = max(server.call("/completion", {"prompt": prompt, "n_predict": 0, "cache_prompt": False})
                      ["timings"]["prompt_per_second"] for _ in range(2))
        decode = max(server.call("/completion", {"prompt": "Write a long story about a lighthouse keeper.",
                                                 "n_predict": DECODE_TOKENS, "temperature": 0, "ignore_eos": True,
                                                 "cache_prompt": False})["timings"]["predicted_per_second"]
                     for _ in range(2))
    finally:
        server.close()
    log = open(server.log_path, encoding="utf-8", errors="replace").read()
    layers = re.search(r"offloaded (\d+/\d+) layers", log)
    return {"gguf": spec, "prefill_tps": round(prefill), "decode_tps": round(decode, 1),
            "layers_on_gpu": layers.group(1) if layers else "?"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", nargs="+", help="models, as for mafdet strike --gguf")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--llama-args", default="", help="extra llama-server options, e.g. '-ctk q8_0 -ctv q8_0'")
    a = ap.parse_args()
    for spec in a.gguf:
        r = measure(spec, a.ctx, a.llama_args.split())
        print(f"{r['gguf']}: prefill {r['prefill_tps']} tok/s, decode {r['decode_tps']} tok/s, "
              f"{r['layers_on_gpu']} layers on the GPU", flush=True)
