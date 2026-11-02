"""Qwen3-30B-A3B (mixture of experts) on a 12 GB card and the CPU: the largest Qwen3 that runs at a usable
speed with 12 GB of VRAM and 16 GB of RAM. Only ~3B of its 30B parameters are read for each token, so the
experts left in RAM cost much less than the layers of a dense model would.

llama.cpp's own fit already keeps attention, router and KV of every layer on the GPU and moves experts to
the CPU until 1024 MiB are left free (its --fit-target). Against it: a smaller margin (-fitt) and the
experts placed by hand (-ngl all -ncmoe N: the experts of the first N layers on the CPU), with and without
a q8_0 KV cache. The file (17.7 GB) does not fit in 16 GB of RAM next to the system, so the experts on the
CPU are read from disk the first time: each configuration reads the text once before measuring. Then
prompt reading (an ~800-token text, best of 2) and 128 tokens written after a short and after a
~1600-token prompt; VRAM in use is nvidia-smi's, the desktop included. Alternating configurations each round:

    python research/moe_30b.py --rounds 2 --out research/data/moe_30b_3060.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mafdet.llama_backend import LlamaServer, resolve_gguf
from mafdet.text import read_sample

GGUF = "Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf"
KV8 = ["-ctk", "q8_0", "-ctv", "q8_0"]
HAND = ["-fit", "off", "-ngl", "all", "-ncmoe"]
CONFIGS = {
    "fit": [],
    "fit kv8": KV8,
    "fitt 256": ["-fitt", "256"],
    "fitt 256 kv8": ["-fitt", "256"] + KV8,
    "ncmoe 21": HAND + ["21"],
    "ncmoe 20 kv8": HAND + ["20"] + KV8,
}


def vram_used() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def run(name: str, extra: list[str], doc: str) -> dict:
    server = LlamaServer(resolve_gguf(GGUF), "cuda", ctx=8192, port=8098, extra_args=("-lv", "4", *extra),
                         log_name="server-moe")
    story = {"n_predict": 128, "temperature": 0, "ignore_eos": True, "cache_prompt": False}
    try:
        server.call("/completion", {"prompt": doc, "n_predict": 0, "cache_prompt": False})     # experts into RAM
        prefill = max(server.call("/completion", {"prompt": doc, "n_predict": 0, "cache_prompt": False})
                      ["timings"]["prompt_per_second"] for _ in range(2))
        short = server.call("/completion", {"prompt": "Write a long story about a lighthouse keeper.", **story})
        long = server.call("/completion", {"prompt": doc + "\n\n" + doc + "\n\nWrite a long story about it.", **story})
        used = vram_used()
    finally:
        server.close()
    log = open(server.log_path, encoding="utf-8", errors="replace").read()
    # The fit's last choice: (layers, layers whose experts stay on the CPU); by hand, the -ncmoe given.
    fit = re.findall(r"set ngl_per_device\[0\]\.\(n_layer, n_part[^=]*=\((\d+), (\d+)", log)
    moe_cpu = int(extra[extra.index("-ncmoe") + 1]) if "-ncmoe" in extra else int(fit[-1][1]) if fit else None
    return {"config": name, "args": " ".join(extra), "vram_used_mib": used, "expert_layers_on_cpu": moe_cpu,
            "prefill_tps": round(prefill), "decode_short_tps": round(short["timings"]["predicted_per_second"], 2),
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
            row = {"round": r, **run(name, CONFIGS[name], doc)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if a.out:
                json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1)
