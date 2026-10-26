# MAFDET research log

Working notes, kept up to date as the work goes. Read this first in a new session: where the research
stands, what was measured, what is next. Newest first in each section. Results that are final also go
into the README (Results, "Tried, and why they are not on by default").

## Where we are (2026-10-23)

Goal: make the hardware run the models as fast as it can, beyond reasoning ahead (which works: Qwen3-4B
wait after Enter 12.2 s → 3.2 s, Qwen3-14B 108 s → 47 s, see README).

Reference machine: RTX 2060 6 GB (about 1.1-1.4 GB taken by the desktop), i5-10400 (6 cores), 16 GB DDR4,
Windows 11, llama.cpp b11747 in `./llama.cpp`. Qwen3-4B Q4_K_M fits in VRAM (~75-83 tok/s);
Qwen3-14B Q4_K_M does not (38% on GPU, ~4.5-5 tok/s through llama-server).

Key fact learned: on a model split between GPU and CPU, the CPU's layers *compute* every token they are
given. Anything that processes several tokens at once (parallel branches, speculative-decoding drafts)
costs the 14B almost proportionally: 2 sequences run at 77% of the speed of one each, a 9-token draft
costs 2.4 decoding steps. To speed up the 14B, take work away from the CPU.

## Next

1. **Put more of the 14B on the GPU** (configuration only, ~1 hour of measurements). Measure decode tok/s
   and wait after Enter with `research/thinking_ahead.py --gguf Qwen/Qwen3-14B-GGUF:Qwen3-14B-Q4_K_M.gguf
   --max-tokens 1800 --warm-up --problems 0,4 --config "name=--llama-args='...'"`:
   - KV cache quantized: `-ctk q8_0 -ctv q8_0` (frees ~0.6 GB at ctx 8192: a few more layers on the GPU)
   - smaller quantizations of the 14B: IQ4_XS, Q3_K_M (more layers on the GPU; check accuracy)
   - tensor placement instead of layer split: `-ot` (override-tensor) to keep attention on the GPU and
     put part of the FFN weights on the CPU
   - CPU threads (`-t`), batch sizes
   Then: if one wins, make it the app's default for models larger than the VRAM (catalog / chat.py).

2. **Activation sparsity on dense models (the "mathematical trick"), feasibility study first.**
   Idea: in a dense model most FFN neurons contribute almost nothing for a given token. Training-free
   methods (e.g. TEAL: magnitude thresholding of the hidden states, also on SwiGLU models) report 1.5-1.8x
   decode speedups with little quality loss — numbers not verified here. For MAFDET it would skip exactly
   the work the CPU does on the 14B. Real use means C++ kernels in llama.cpp (weeks), so first:
   - in Python (PyTorch, HF transformers, Qwen3-4B then 14B if it fits in RAM): measure the distribution of
     FFN activations per layer; how many can be zeroed at 25/40/50% sparsity
   - simulate the sparsity (zero the smallest activations) and run the 24 word problems: accuracy
     against the dense model
   - estimate the speedup: bytes of weights skipped on the CPU's layers
   Only if accuracy holds: plan the llama.cpp kernel work (CPU matmul skipping zeroed columns).

3. Other ideas discussed, not started:
   - show the answer written ahead live under the input field (zero perceived wait; a product/UX change,
     test with people, not a benchmark)
   - harder benchmark (problems the 4B gets wrong 30-50% of the time) to see accuracy effects at all
   - reasoning ahead beyond Qwen3 (each reasoning model's frame must be measured); MiniCPM, newer Qwen
     and Gemma were asked for on Reddit

## Done and measured

- **2026-10-26 · macOS on Apple Silicon** — the app runs on a MacBook Pro M2 Pro 16 GB, macOS 26.2, llama.cpp
  b11747 macOS arm64 (Metal). The GPU is detected through Metal (its recommended working set, 12,124 MB here,
  is the "VRAM"; the unified memory counts once). `mafdet bench` Qwen3-4B Q4_K_M: prefill 593 tok/s, decode
  56.9 (catalog estimate 625 / 53). A typed question: all 265 reasoning tokens done before Enter. Not yet run
  here: `research/thinking_ahead.py` (the 24 problems) and the 14B (it should fit on the GPU, ~15 tok/s).
- **2026-10-23 · Notes as a speculative-decoding draft after Enter** — negative. Offline replay
  (`research/notes_as_draft.py`, data `notes_as_draft_4b.json`): prompt lookup would write 1.5-1.7 tokens
  per step on the 4B (1.3-1.5 without notes). Real, llama.cpp `--spec-type ngram-simple`: 4B 73 tok/s
  either way; 14B 5.0 → 3.1 tok/s (11% of drafted tokens accepted; `ngram_4b.json`, `ngram_14b.json`).
- **2026-10-23 · `--branches` (guess the end of the message, reason on the guess in a parallel slot)** —
  shipped as an experimental toggle ("Guess the ending"). 4B, 2 runs × 24 problems: mean wait 4.17 → 3.51 s,
  slowest tenth 7.9 → 5.3 s, 48/48 both, 14/48 exact guesses, not significant (p = 0.15). Only exact guesses
  are adopted (close ones gave 2 wrong answers out of 4). 2 branches slower than none. 14B: guesses too
  slow, no branch ever reasoned (4 problems), so the app gives a branch only to models that fit in VRAM.
  Data: `branches_4b.json`, `branches_14b.json`. llama-server does not share prefixes across slots; with
  several slots use `-kvu` (4 slots: 139 tok/s in all with it, 103 without).
- **2026-10-22 · Gatherer (small model writes notes for the 14B)** — closed, no gain on the RTX 2060
  (README). Do not reopen unless asked. Old data in `research/data/14b/`.
- **Terminal app restyle (2026-10-23)**: amber CRT, boot sequence, decrypting text (`mafdet/tui/fx.py`).

## How to work here

- Python: `.venv/Scripts/python` (Windows), `.venv/bin/python` (Mac). Tests: `<python> -m pytest -q`.
- Benchmark: `research/thinking_ahead.py` (24 Italian word problems typed in real time; `--problems`,
  `--config "name=strike options"`, `--gguf`, `--append`). The README's 4B table is `thinking_ahead_4b.json`.
- Runs longer than ~10 minutes get killed: split them with `--problems` and `--append`. A 4B run is
  ~25 s per problem; a 14B run ~2 min per problem.
- Single runs vary a lot (the same 24 problems: 3.2 s or 4.2 s mean wait). Compare configs measured the
  same day; prefer decode tok/s (rows' `tps`) and repeated runs over one mean.
- After every change: commit and push (the user asked for it). The user writes in Italian.
