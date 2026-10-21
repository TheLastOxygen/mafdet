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
| <img src="assets/icons/strike.svg" width="22" alt=""> | `mafdet strike` | the strike | starts reasoning while you type, and writes the answer when you pause | available |
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

It runs on llama.cpp: extract a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) into `./llama.cpp` (or set `MAFDET_LLAMA_DIR`). The first start downloads Qwen3-4B in Q4_K_M (about 2.5 GB). On Windows, double-click `avvia_mafdet.bat` (it opens `mafdet strike`, below). `--browser` opens the chat in the default browser instead of its own window.

### Reasoning ahead

```bash
mafdet strike
```

`mafdet strike` is the same chat, one step further: the model **starts thinking about your message while you are still typing it**. It runs Qwen3 in thinking mode, so before answering the model reasons in a hidden block (shown folded above the answer). That reasoning is usually the longest wait, and `strike` moves as much of it as possible to before Enter:

1. **While you type**, the model reasons in short chunks of 32 tokens, on the message as it is at that moment, on whole words (after a space or punctuation, or at a pause).
2. **When the text changes**, the reasoning is not thrown away. It stays in the prompt as notes, after the new version of the message. Moving it there only costs a prefill: the fast, parallel phase, about 60 times faster than writing the same tokens again for the 14B on the RTX 2060. A short sentence tells the model that the message has changed, so it rereads it and goes on from what it had worked out.
3. **On Enter**, a last sentence tells the model that the message is now complete. It finishes reasoning from where it got to and answers.

The notes start with a frame: *"The user is still typing... I will not guess the rest: I will work out what I can from the part already written."* Without it, the model spent its notes remarking that the message was unfinished, and after every edit it copied its last paragraph again (measured: 70% more reasoning in total, and two answers out of 24 that never came). With it, the model does useful partial work, such as `3 × 12 = 36` before the question has even been typed.

**If you pause and send the message unchanged**, `strike` goes further and also writes the answer itself. That part is exact: the answer only holds for the exact text it was written for, and any change drops it. With `--no-think` (no reasoning) this is all `strike` does, and with greedy decoding the answer is identical token by token to the one started after Enter (checked).

Reasoning ahead is **not exact**: the model reasons on intermediate versions of the message, so the answer can differ from the one it would give starting from scratch. That is why it was measured on answers that can be checked (below).

| Option | What it does |
|---|---|
| `--no-think` | no reasoning: answer right away, and write ahead only the exact answer |
| `--think-ahead N` | longest reasoning kept across edits (default 1024 tokens) |
| `--ahead N` | answer tokens written ahead when the text does not change (default 512) |
| `--recycle` | reuse the answers written for earlier versions of the message as n-gram drafts for speculative decoding (+4% decode speed measured with natural typing, +20% on code) |
| `--draft repo:file` | a small model of the same family drafts tokens for speculative decoding (no gain measured on the RTX 2060, see below) |

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

All on an RTX 2060 (6 GB), llama.cpp, Q4_K_M models, real-time typing (5 to 8 characters per second, a few natural pauses, Enter right after the last character, no rereading).

**Reasoning ahead, Qwen3-4B** (fits in VRAM, 83 tok/s): 24 word problems with a known numeric answer, typed in Italian.

| | Without reasoning ahead | `mafdet strike` |
|---|---|---|
| Correct answers | 22/24 | 22/24 |
| Median wait for the first word of the answer | 10.2 s | **4.4 s** |
| Mean wait | 11.8 s | **6.2 s** |
| Faster | | in 20 of 22 problems |
| Reasoning in total | 914 tokens | 907 tokens |
| Reasoning left after Enter | 914 tokens | **462 tokens** |

Same quality, same amount of reasoning, half of it done before Enter. In both columns, 2 answers never finished (the 4B sometimes reasons in circles until the token limit, with or without reasoning ahead).

**Exact answer ahead, Qwen3-4B, no reasoning** (pause before Enter, text unchanged):

| Answer | Written before Enter | Full answer on screen after Enter, strike | Without strike |
|---|---|---|---|
| 300 tokens | 200 | 1.2 s | 3.5 s |
| 168 tokens | 168 (all of it) | instant | 1.9 s |

**Larger models.** What the model can do before Enter is bounded by typing time × its writing speed:

| Model | Fits in 6 GB? | Writes | Reasoning written while typing a 30 s message |
|---|---|---|---|
| Qwen3-4B | yes | 83 tok/s | most of it |
| Qwen3-8B | no, partly on the CPU | about 13 tok/s | 130 to 280 tokens |
| Qwen3-14B | no, partly on the CPU | 5 tok/s | 130 tokens |

On the 8B and 14B the reasoning is 400 to 1500 tokens, so at most a fraction of it is done before Enter, and in single runs the variation in reasoning length between two samples is larger than the gain (examples: 8B 32 s → 20 s on one problem, 74 s → 83 s on another; 14B 235 s → 146 s). Reading ahead, on the other hand, gains more the larger the model: the 14B reads a 3,000-token pasted document in about 10 s, all of it hidden while you type.

**Tried, and why they are not on by default:**

- `--recycle` (answers to earlier versions of the message as n-gram drafts): +4% decode speed with natural typing, +20% on code, slightly later first token.
- `--draft` (Qwen3-0.6B drafting for the 14B): on the GPU it takes VRAM from the 14B, which then runs more of its layers on the CPU (4.8 → 3.5 tok/s); on the CPU it gains 8% with 4-token drafts and halves the reading speed.

## Roadmap

- [x] Incremental prefill engine with common-prefix sync and token hold-back
- [x] Benchmarks, typing simulation and correctness checks
- [x] Unified `mafdet` CLI
- [x] `mafdet watch`: the chat window
- [x] llama.cpp backend with quantized (GGUF) models
- [x] `mafdet strike`: reasoning ahead while you type, kept across edits; the exact answer ahead at a pause
- [ ] A small model reasons while you type, a large one answers: reasoning ahead for models larger than the VRAM
- [ ] Repeated measurements on models larger than the VRAM

## Related research

- [LiveMind: Low-latency Large Language Models with Simultaneous Inference](https://arxiv.org/abs/2406.14319)
- [StreamingThinker: Large Language Models Can Think While Reading](https://arxiv.org/abs/2510.17238)
- [SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models](https://arxiv.org/abs/2510.06917)
- [Asynchronous Reasoning: Training-Free Interactive Thinking LLMs](https://arxiv.org/abs/2512.10931)

## License

[Apache-2.0](LICENSE)
