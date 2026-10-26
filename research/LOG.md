# MAFDET research log

Working notes, kept up to date as the work goes. Read this first in a new session: where the research
stands, what was measured, what is next. Newest first in each section. Results that are final also go
into the README (Results, "Tried, and why they are not on by default").

## Where we are (2026-10-26)

Goal: make the hardware run the models as fast as it can, beyond reasoning ahead (which works: Qwen3-4B
wait after Enter 12.2 s → 3.2 s, Qwen3-14B 108 s → 47 s, see README).

Reference machine: RTX 2060 6 GB (about 1.1-1.4 GB taken by the desktop), i5-10400 (6 cores), 16 GB DDR4,
Windows 11, llama.cpp b11747 in `./llama.cpp`. Qwen3-4B Q4_K_M fits in VRAM (~75-83 tok/s);
Qwen3-14B Q4_K_M does not (38% on GPU, ~4.5-5 tok/s through llama-server).

Second machine (since 2026-10-26): MacBook Pro M2 Pro 16 GB, macOS 26.2, llama.cpp b11747 macos-arm64 (Metal,
installed by the app in ~/.mafdet/llama.cpp). Metal may use 12,124 MB of the unified memory: Qwen3-4B 54 tok/s,
Qwen3-14B all on the GPU at 18 tok/s. Reasoning ahead: 4B 18.1 → 5.6 s, 14B 32.7 → 12.1 s (README, "On a Mac").
It gets hot under long runs (thermal pressure Heavy after ~25 min of the 4B benchmark): check
`sudo powermetrics --samplers thermal -n 1` before a run, and compare configs measured close together.

AMD Radeon (since 2026-10-26): supported through llama.cpp's Vulkan build, no Radeon here to measure on. The README
asks for volunteers ("Help wanted: benchmarks on an AMD Radeon"): `mafdet doctor`, Measure, and
`research/thinking_ahead.py radeon_4b.json`, sent as an issue.

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

3. **Mac follow-ups** (cheap, the M2 Pro is set up): the README's no-ahead 4B column is from the run before
   the empty-answer fix; repeat both configs in one run when the Mac is cool. The 8B on the Mac (should fit
   on the GPU, ~27 tok/s estimated). The SENT frame ("...and answer it") may invite the answer inside the
   reasoning: count `closed` in future runs (now in the rows) before touching it.
   The language: an English question can get an Italian answer (above). Next try, if wanted: SEED's example
   line in English ("How much does the Moon weigh compared to") and SENT unchanged, which removes the cue
   without an instruction; measure with `research/language_check.py` and the 24 Italian problems.

4. **AMD Radeon results**, when someone sends them (README, help wanted): a "On an AMD Radeon" table next to
   "On a Mac", and the catalog's estimates checked against Measure (GPU_PREFILL_FLOPS_PER_GBS and
   GPU_EFFICIENCY are calibrated on the RTX 2060's CUDA). Integrated Radeons (Ryzen APUs) are left on the CPU
   build (AMD_MIN_VRAM_MB): whether Vulkan on one beats the CPU is open.

5. Other ideas discussed, not started:
   - show the answer written ahead live under the input field (zero perceived wait; a product/UX change,
     test with people, not a benchmark)
   - harder benchmark (problems the 4B gets wrong 30-50% of the time) to see accuracy effects at all
   - reasoning ahead beyond Qwen3 (each reasoning model's frame must be measured); MiniCPM, newer Qwen
     and Gemma were asked for on Reddit

## Done and measured

- **2026-10-26 · AMD Radeon through Vulkan: in place, checked on the RTX 2060, not measured on a Radeon.**
  Detection: DXGI on Windows (vendor 0x1002, dedicated memory; the registry also lists cards removed long
  ago), amdgpu's sysfs on Linux; 2 GB or more of its own, so integrated Radeons are left out. Bandwidths of
  RX 400-9000 in the table. `build_for` picks `win-vulkan-x64` / `ubuntu-vulkan-x64`; `mafdet doctor
  --install --build vulkan` forces it on any card (`--cuda` still works). On Windows the VRAM in use is not
  known (it counts as free); llama.cpp's own fit asks Vulkan. Checked with the Vulkan build on the RTX 2060
  (MAFDET_HOME and MAFDET_LLAMA_DIR in a scratch folder): llama-bench Qwen3-4B pp512 2103 tok/s, tg64 82.8
  (CUDA 2141 / 87); llama.cpp sees the Intel UHD 630 too (Vulkan1) but uses only Vulkan0, the 14B splits
  17/41 layers on Vulkan0 and the rest on the CPU, never on the iGPU; `-dev none` keeps the CPU mode.
  `research/thinking_ahead.py --problems 0:3`: 3/3 both ways, mean wait 9.05 → 4.42 s, ~70 tok/s.
- **2026-10-26 · An English question answered in Italian: found, fix tried, not kept.** Recording the app on
  the M2 Pro, one of 3 runs of an English word problem got a right answer in Italian, after reasoning ahead.
  Likely cause: the only Italian in the frame is SEED's example line ("Quanto pesa la Luna rispetto alla").
  All benchmarks were Italian, so it never showed. `research/language_check.py` (3 English problems typed
  in real time): 0 of 12 Italian answers as released (`language_4b_m2pro_before.json`), so it is rare (1 in
  15 counting the app runs). Tried: SENT "...answer it in the language it is written in, using...": 0 of 24
  (`language_4b_m2pro_after.json`), but the reasoning got longer on both models: 4B, 24 Italian problems,
  24/24, 773 → 875 reasoning tokens, median 4.1 → 5.0 s, mean 5.6 → 7.2 s
  (`thinking_ahead_4b_m2pro_language.json`); 14B, 6 problems, 6/6, 455 → 636 tokens, mean 12.1 → 22.7 s
  (median 11.7 → 11.9; `thinking_ahead_14b_m2pro_language.json`). After Enter the model rereads the Italian
  message and argues over its translation. Reverted, the user stopped here; the README's numbers are the
  released frame.
- **2026-10-26 · Empty answer after reasoning ahead: fixed.** Qwen3-4B sometimes ends its turn inside the
  reasoning, the answer written there, with no </think>: the page got only thoughts and an empty answer
  (`answer_ms` 0). Seen once in the M2 Pro run (problem 5), 1 in 6 repeats of it, and once before on the
  RTX 2060 (`branches_4b.json`, row 17). Now `_answer` closes the reasoning as the template does
  ("\n</think>\n\n") and the model writes the answer; the `done` event says `closed`. Forced with
  </think> banned on 3 problems: 3/3 correct answers in Italian, 4.4-6.3 s. The SENT frame ("...and answer
  it") may invite it; not changed (it would need the measurements again).
- **2026-10-26 · macOS on Apple Silicon** — the app runs on a MacBook Pro M2 Pro 16 GB, macOS 26.2, llama.cpp
  b11747 macOS arm64 (Metal). The GPU is detected through Metal (its recommended working set, 12,124 MB here,
  is the "VRAM"; the unified memory counts once). `mafdet bench` Qwen3-4B Q4_K_M: prefill 593 tok/s, decode
  56.9 (catalog estimate 625 / 53).
  `research/thinking_ahead.py` (in the README, "On a Mac"):
  - 4B, 24 problems, no ahead (`thinking_ahead_4b_m2pro.json`): 24/24, mean 18.1 s, median 16.7 s, 966
    reasoning tokens, 54 tok/s. Reason ahead in that run, before the fix: 23/24 (the empty answer), mean 6.7 s.
  - 4B reason ahead after the fix (`thinking_ahead_4b_m2pro_fixed.json`): 24/24, mean 5.6 s, median 4.1 s,
    slowest tenth 8.6 s, faster in 24/24, 773 reasoning tokens of which 525 ahead (248 left after Enter).
    `closed` was not recorded in the rows then (it is now), so whether the fix fired in it is unknown.
  - 14B, the 6 problems of the README, --warm-up, --max-tokens 1800 (`thinking_ahead_14b_m2pro.json`):
    mean 32.7 → 12.1 s, median 26.5 → 11.7 s, 6/6 both; 18.3 tok/s, all on the GPU; 593 reasoning tokens
    without, 455 with (272 ahead). RTX 2060: 108 → 47 s.
  The Mac reached thermal pressure "Heavy" for a few minutes during the first 4B run (54 → 51-53 tok/s).
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
