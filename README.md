<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/mafdet-logo-dark.svg">
  <img src="assets/mafdet-logo.svg" alt="" width="112">
</picture>

<h1>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/mafdet-wordmark-dark.svg">
  <img src="assets/mafdet-wordmark.svg" alt="MAFDET" height="30">
</picture>
</h1>

### You type. It already knows.

MAFDET reads your message while you type it and gets your local LLM ready to answer.<br>
Hit enter, and the response starts right away.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-EEEAE2?labelColor=0E0D0C)](LICENSE)
[![Runtime: local](https://img.shields.io/badge/runtime-local-D4892A?labelColor=0E0D0C)](#requirements)
[![Status: prototype](https://img.shields.io/badge/status-prototype-6A655D?labelColor=0E0D0C)](#roadmap)

[Italiano](README.it.md)

</div>

---

## Why

Chat interfaces make the model wait for you to press enter before it does anything. While you type, the GPU sits idle. When you send, the model first has to *read* your whole message (prefill), and only then can it start answering.

On expensive hardware, reading is fast enough that nobody notices. On a mid-range GPU, a laptop or a CPU, reading a long prompt can take whole seconds, and that is exactly what MAFDET hides.

MAFDET is built for the hardware most people actually own. The reference machine is an **RTX 2060 with 6 GB of VRAM**, and everything also runs on **CPU only**.

> Named after Mafdet, the ancient Egyptian feline goddess known as "the runner", who struck snakes and scorpions before they could attack.

## How it works

An LLM processes text in two phases:

| Phase | What happens | Speed |
|---|---|---|
| **Prefill** | reads the prompt; all tokens are processed in parallel | fast |
| **Decode** | generates the answer, one token at a time | slow |

While reading, the model stores intermediate results for every token in the **KV cache**. Those results never change for a given prefix, so they can be computed ahead of time.

MAFDET keeps the KV cache in sync with the text **while you type**:

1. On every pause, it tokenizes the current text.
2. It holds back the last few tokens, because a word you are still typing can still change how the text is split into tokens.
3. It finds the **longest common prefix** between the new tokens and the ones already in cache, trims the cache there, and prefills only what is new.
4. When you hit enter, only a handful of tokens are left to process.

**Correctness does not depend on how you type.** Backspaces, edits in the middle of the text, pasted blocks, missing punctuation or spaces: the common-prefix check always brings the cache back to exactly what a from-scratch computation would produce. Heuristics such as hold-back and debounce only affect how much work is wasted, never the result.

## Commands

Run `mafdet` with no arguments for the table of commands, `mafdet --help` for the full guide (what each command does, with examples), or `mafdet <command> --help` for the options of one.

| | Command | | What it does | Status |
|---|---|---|---|---|
| <img src="assets/icons/awaken.svg" width="22" alt=""> | `mafdet awaken` | the awakening | checks the GPU, downloads and loads the model | available |
| <img src="assets/icons/watch.svg" width="22" alt=""> | `mafdet watch` | the guard | opens the chat window: the model reads while you type | available |
| <img src="assets/icons/strike.svg" width="22" alt=""> | `mafdet strike` | the strike | thinks ahead and answers the moment you hit enter | *planned* |
| <img src="assets/icons/hunt.svg" width="22" alt=""> | `mafdet hunt` | the hunt | simulates people typing and measures the wait after sending | available |
| <img src="assets/icons/race.svg" width="22" alt=""> | `mafdet race` | the race | benchmarks prefill and decode on your machine | available |
| <img src="assets/icons/judge.svg" width="22" alt=""> | `mafdet judge` | the judgement | checks results against a from-scratch computation | available |

Every command accepts the common options `--model`, `--device`, `--dtype` and `--threads`. `python -m mafdet` works too.

## Requirements

- **NVIDIA GPU** with up-to-date drivers (tested target: RTX 2060, 6 GB), or just a CPU
- **Python 3.12** (3.11 and 3.14 also work). PyTorch does not ship wheels for Python 3.15 yet.
- About **7 GB of disk space** (PyTorch + model)
- **16 GB of RAM** recommended for CPU-only runs

Default model: **Qwen3-1.7B in fp16**. Turing GPUs such as the 2060 have no native bf16 support.

## Installation

Order matters: **install PyTorch from the CUDA index first**, then MAFDET.

> **PyTorch is not a dependency of the `mafdet` package, on purpose.** If pip installed it from PyPI, on Windows it would pick the CPU-only build and the GPU would sit unused. Install it yourself from the CUDA index, as below, before installing MAFDET.

```bash
git clone https://github.com/TheLastOxygen/mafdet.git
cd mafdet

python -m venv .venv          # Windows: py -3.12 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

This installs the `mafdet` command in the virtual environment (`pip install -e .` keeps it linked to the source folder, so edits take effect right away).

## Quick start

```bash
mafdet awaken --load    # GPU check, model download (~3.5 GB), test generation
mafdet watch            # the chat window
mafdet judge            # must end with 0 errors
mafdet race             # prefill and decode cost on your hardware
mafdet hunt             # wait time with and without incremental prefill
```

### Chat

```bash
mafdet watch
```

`mafdet watch` opens the MAFDET chat window. While you type, a status line under the message shows how much of it the model has already read; when you pause it also reads the end of the prompt, so on Enter a single token is left to compute. The conversation stays in the cache from one turn to the next. Answers render as Markdown and scroll freely while they are written.

It runs on llama.cpp: extract a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) into `./llama.cpp` (or set `MAFDET_LLAMA_DIR`). The first start downloads Qwen3-4B in Q4_K_M (about 2.5 GB). On Windows, double-click `avvia_mafdet.bat`. `--browser` opens the chat in the default browser instead of its own window.

`demo_live.py` (`avvia_demo.bat`) is the step-by-step view for debugging: tokens, cache, hold-back and timings, live in the terminal.

### Measurements

Add `--device cpu` to any command to measure the no-GPU case. Results are saved as CSV files in `results/` (change it with `--out`).

### Typing scenarios

`mafdet hunt` replays three scenarios (pick them with `--scenario`):

- **clean**: a message typed from start to finish
- **messy**: typos fixed with backspace, a word deleted and retyped, an insertion in the middle, little punctuation
- **paste**: a long document pasted in, followed by a typed question. This is where the gain is largest.

The typing is not replayed in real time. Each prefill really runs on the model, and its measured duration is placed on the typing timeline. If the model falls behind the typist, the delay piles up and is added to the final wait, just as in a real system.

## Results

*Coming soon: measurements on an RTX 2060 and on CPU.*

What to expect: on short messages the gain is small in absolute terms, because prefilling a few dozen tokens is already fast. The gain grows with message length and is largest when pasting long documents and on CPU. MAFDET hides the cost of *reading* the prompt, not the cost of *generating* the answer.

## Roadmap

- [x] Incremental prefill engine with common-prefix sync and token hold-back
- [x] Benchmarks, typing simulation and correctness checks
- [x] Unified `mafdet` CLI
- [x] `mafdet watch`: the chat window
- [ ] llama.cpp backend with quantized (GGUF) models
- [ ] `mafdet strike`: speculative thinking on partial input, re-read as notes when you hit enter

## Related research

- [LiveMind: Low-latency Large Language Models with Simultaneous Inference](https://arxiv.org/abs/2406.14319)
- [StreamingThinker: Large Language Models Can Think While Reading](https://arxiv.org/abs/2510.17238)
- [SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models](https://arxiv.org/abs/2510.06917)
- [Asynchronous Reasoning: Training-Free Interactive Thinking LLMs](https://arxiv.org/abs/2512.10931)

## License

[Apache-2.0](LICENSE)
