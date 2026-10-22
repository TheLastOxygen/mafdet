"""
Builds mafdet/catalog.json: the models the app offers, with exact file sizes and the shape of
each model (layers, KV heads, head size, sliding window) read from Hugging Face.

    python tools/build_catalog.py

To add a model, add a line to MODELS: the GGUF repository and file name, a tokenizer
repository that is not gated (the app downloads it without a login), and its tier. A family
reasons ahead only if its frame was measured (see the README): today, Qwen3. The others read
while you type and write the answer ahead at a pause.
"""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

FAMILIES = [
    {"id": "qwen3", "name": "Qwen3", "vendor": "Alibaba", "license": "Apache-2.0", "reason_ahead": True},
    {"id": "llama", "name": "Llama 3", "vendor": "Meta", "license": "Llama Community License", "reason_ahead": False},
    {"id": "gemma", "name": "Gemma 3", "vendor": "Google", "license": "Gemma Terms of Use", "reason_ahead": False},
    {"id": "mistral", "name": "Mistral", "vendor": "Mistral AI", "license": "Apache-2.0", "reason_ahead": False},
    {"id": "phi", "name": "Phi-4", "vendor": "Microsoft", "license": "MIT", "reason_ahead": False},
]

QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

# id, name, family, GGUF repo, file stem ({stem}-{quant}.gguf), tokenizer repo, params (B), active (B),
# tier, extras. reasoning_tokens: a typical reasoning on a word problem (measured: 4B ~900, 14B ~500).
MODELS = [
    ("qwen3-0.6b", "Qwen3 0.6B", "qwen3", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B", "Qwen/Qwen3-0.6B", 0.6, 0.6, "tiny",
     {"reasoning_tokens": 1100}),
    ("qwen3-1.7b", "Qwen3 1.7B", "qwen3", "Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B", "Qwen/Qwen3-1.7B", 1.7, 1.7, "light",
     {"reasoning_tokens": 1000}),
    ("qwen3-4b", "Qwen3 4B", "qwen3", "Qwen/Qwen3-4B-GGUF", "Qwen3-4B", "Qwen/Qwen3-4B", 4.0, 4.0, "balanced",
     {"reasoning_tokens": 900}),
    ("qwen3-8b", "Qwen3 8B", "qwen3", "Qwen/Qwen3-8B-GGUF", "Qwen3-8B", "Qwen/Qwen3-8B", 8.2, 8.2, "strong",
     {"reasoning_tokens": 700}),
    ("qwen3-14b", "Qwen3 14B", "qwen3", "Qwen/Qwen3-14B-GGUF", "Qwen3-14B", "Qwen/Qwen3-14B", 14.8, 14.8, "very_strong",
     {"reasoning_tokens": 500}),
    ("qwen3-30b-a3b", "Qwen3 30B-A3B", "qwen3", "Qwen/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B",
     30.5, 3.3, "moe", {"reasoning_tokens": 600}),
    ("qwen3-32b", "Qwen3 32B", "qwen3", "Qwen/Qwen3-32B-GGUF", "Qwen3-32B", "Qwen/Qwen3-32B", 32.8, 32.8, "best",
     {"reasoning_tokens": 450}),
    ("llama-3.2-1b", "Llama 3.2 1B", "llama", "unsloth/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct",
     "unsloth/Llama-3.2-1B-Instruct", 1.2, 1.2, "tiny", {}),
    ("llama-3.2-3b", "Llama 3.2 3B", "llama", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct",
     "unsloth/Llama-3.2-3B-Instruct", 3.2, 3.2, "light", {}),
    ("llama-3.1-8b", "Llama 3.1 8B", "llama", "unsloth/Llama-3.1-8B-Instruct-GGUF", "Llama-3.1-8B-Instruct",
     "unsloth/Llama-3.1-8B-Instruct", 8.0, 8.0, "strong", {}),
    # Gemma 3 attends to a sliding window in five layers out of six. llama.cpp then keeps only the
    # window in cache, and a cut further back than the window means reading everything again:
    # --swa-full keeps the whole cache, so the cache can follow the text like with any model.
    ("gemma-3-1b", "Gemma 3 1B", "gemma", "unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it", "unsloth/gemma-3-1b-it",
     1.0, 1.0, "tiny", {"llama_args": ["--swa-full"]}),
    ("gemma-3-4b", "Gemma 3 4B", "gemma", "unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it", "unsloth/gemma-3-4b-it",
     3.9, 3.9, "balanced", {"llama_args": ["--swa-full"]}),
    ("gemma-3-12b", "Gemma 3 12B", "gemma", "unsloth/gemma-3-12b-it-GGUF", "gemma-3-12b-it", "unsloth/gemma-3-12b-it",
     11.8, 11.8, "very_strong", {"llama_args": ["--swa-full"]}),
    ("gemma-3-27b", "Gemma 3 27B", "gemma", "unsloth/gemma-3-27b-it-GGUF", "gemma-3-27b-it", "unsloth/gemma-3-27b-it",
     27.0, 27.0, "best", {"llama_args": ["--swa-full"]}),
    ("mistral-7b", "Mistral 7B v0.3", "mistral", "bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3",
     "unsloth/mistral-7b-instruct-v0.3", 7.2, 7.2, "strong", {}),
    # The tokenizer of Mistral Small 3.x ships with a wrong pre-tokenizer pattern unless told otherwise.
    ("mistral-small-3.2", "Mistral Small 3.2", "mistral", "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
     "Mistral-Small-3.2-24B-Instruct-2506", "unsloth/Mistral-Small-3.2-24B-Instruct-2506", 23.6, 23.6, "best",
     {"tokenizer_kwargs": {"fix_mistral_regex": True}}),
    ("phi-4-mini", "Phi-4 mini", "phi", "unsloth/Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct",
     "microsoft/Phi-4-mini-instruct", 3.8, 3.8, "balanced", {}),
    ("phi-4", "Phi-4", "phi", "unsloth/phi-4-GGUF", "phi-4", "microsoft/phi-4", 14.7, 14.7, "very_strong", {}),
]


def shape(tokenizer_repo: str) -> dict:
    """Layers, KV heads and head size from the model's config.json (text part of multimodal ones)."""
    cfg = json.loads(Path(hf_hub_download(tokenizer_repo, "config.json")).read_text(encoding="utf-8"))
    c = cfg.get("text_config", cfg)
    heads = c["num_attention_heads"]
    return {"layers": c["num_hidden_layers"], "kv_heads": c.get("num_key_value_heads", heads),
            "head_dim": c.get("head_dim") or c["hidden_size"] // heads}


def main():
    api = HfApi()
    fams = {f["id"]: f for f in FAMILIES}
    out = {"version": 2, "families": FAMILIES, "models": []}
    for mid, name, fam, repo, stem, tok, params, active, tier, extra in MODELS:
        info = api.model_info(repo, files_metadata=True)
        files = {s.rfilename: s.size for s in info.siblings}
        variants = []
        for q in QUANTS:
            for f in (f"{stem}-{q}.gguf", f"{stem}.{q}.gguf"):      # unsloth writes Phi-4-mini's Q8_0 with a dot
                if f in files:
                    variants.append({"quant": q, "repo": repo, "file": f, "bytes": files[f]})
                    break
        default = "Q4_K_M" if any(v["quant"] == "Q4_K_M" for v in variants) else variants[0]["quant"]
        if mid in ("qwen3-0.6b", "qwen3-1.7b"):
            default = "Q8_0"                       # the only quantization of the official repositories
        model = {"id": mid, "name": name, "family": fam, "tokenizer": tok, "params_b": params, "active_b": active,
                 **shape(tok), "reason_ahead": fams[fam]["reason_ahead"], "tier": tier, "default_quant": default,
                 "variants": variants}
        model.update({k: v for k, v in extra.items()})
        out["models"].append(model)
        print(f"{mid:20s} {len(variants)} variants, {model['layers']} layers, {model['kv_heads']} KV heads")
    path = Path(__file__).resolve().parents[1] / "mafdet" / "catalog.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("written", path)


if __name__ == "__main__":
    main()
