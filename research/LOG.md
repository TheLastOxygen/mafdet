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

Third machine (since 2026-10-26): RTX 4070 SUPER 12 GB (504 GB/s, ~1.9 GB taken by the desktop), i5-14600KF (14 cores),
32 GB DDR (~28 GB/s by `doctor`), Windows 11, llama.cpp b11747 CUDA 13. Its C: has 3 GB free: everything is on E:,
`MAFDET_HOME=E:/mafdet-home HF_HOME=E:/hf-cache` in front of every command (the app itself would still use C:).
Qwen3-14B Q4_K_M goes all on the GPU here (41 tok/s); the catalog now says the same (39 tok/s) and recommends it
(2026-10-30, "Done and measured").

AMD Radeon (since 2026-10-26): supported through llama.cpp's Vulkan build, no Radeon here to measure on. The README
asks for volunteers ("Help wanted: benchmarks on an AMD Radeon"): `mafdet doctor`, Measure, and
`research/thinking_ahead.py radeon_4b.json`, sent as an issue. First one in (issue #1, 2026-10-28): RX 5700 XT,
reasoning ahead works but prefill is only 93 tok/s there, see "Done and measured".

Key fact learned: on a model split between GPU and CPU, the CPU's layers *compute* every token they are
given. Anything that processes several tokens at once (parallel branches, speculative-decoding drafts)
costs the 14B almost proportionally: 2 sequences run at 77% of the speed of one each, a 9-token draft
costs 2.4 decoding steps. To speed up the 14B, take work away from the CPU.

## Next

0. **IQ4_XS where the 14B does not fit, and the estimate fixed: done (2026-10-30, below).** Left from it:
   - **Problem 19 on the 2060's 14B** (below): wrong with reasoning ahead in 2 of 2 runs (one per quantization),
     right without it in 2 of 2, right with it on the 4070 SUPER and the M2 Pro. Rows keep only the answer's tail:
     save the reasoning of that problem and read where the numbers that are not in the text (38 motorbikes,
     84 wheels) come from, before calling it a limit of reasoning ahead on slow models.
   - the 2060's numbers under the new estimate are 1-2 layers short of the measured split (14/40 against 15/41
     with ~1.4 GB taken): check `VRAM_RESERVE_MB` there with the server log's `common_params_fit_impl` lines
     (projected use against free memory) next time the 2060 runs anything.

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

4. **AMD Radeon results** (README, help wanted; first one: RX 5700 XT, issue #1, below): a "On an AMD Radeon"
   table next to "On a Mac", and the catalog's estimates checked against Measure (GPU_PREFILL_FLOPS_PER_GBS and
   GPU_EFFICIENCY are calibrated on the RTX 2060's CUDA). On the 5700 XT both are far off (prefill 93 tok/s
   measured, ~2,850 estimated; decode 69, ~115 estimated). Before changing the catalog: is 93 tok/s llama.cpp's
   Vulkan on RDNA1 with AMD's Windows driver, or something in the app? `llama-bench -m <4B gguf> -p 512 -n 128`
   from ~/.mafdet/llama.cpp on that card would tell.
   README: "On an AMD Radeon" table in (2026-10-28), marked as a first data point to investigate; the help-wanted
   steps now ask for that llama-bench too. Integrated Radeons (Ryzen APUs) are left on the CPU
   build (AMD_MIN_VRAM_MB): whether Vulkan on one beats the CPU is open.

5. **Proxy** (2026-10-28, proposed on Reddit by a user of SGLang, llama.cpp and other engines, with Open WebUI and Hermes):
   an OpenAI-compatible server in front of any engine, fed debounced drafts by the front end. `PROXY.md` has
   the scope and the contract; contributions as pull requests to `main`, code in `mafdet/proxy/`. Nothing
   built yet. Reading ahead ports to any engine with prefix caching; thinking ahead needs a raw completion
   endpoint, cancelling, and a frame measured per model family.

6. Other ideas discussed, not started:
   - **past reasoning out of the history** (2026-10-28, idea from github.com/GionaGranchelli/context-watcher, a
     context compactor for coding agents: young repo, marketing-heavy numbers, take the idea, not the code).
     `Session._answer` keeps notes + reasoning + answer in `self.history` (`history = final + generated + sep`),
     so every turn starts from the cache, but the 8192-token context fills after a few problems ("no longer fits
     in the context") and Qwen3's own template drops past reasoning (the model is trained without it). Dropping
     the last turn's reasoning means rereading its answer and the new message, which mafdet already does while
     the next message is typed: nearly free. To measure before adopting: conversations of 5-6 problems in a row
     with `thinking_ahead.py`, with and without, for accuracy, wait after Enter and turns per 8192 tokens.
     Not started on purpose (the user's call, 2026-10-28): keep it in mind.
   - show the answer written ahead live under the input field (zero perceived wait; a product/UX change,
     test with people, not a benchmark)
   - harder benchmark (problems the 4B gets wrong 30-50% of the time) to see accuracy effects at all
   - reasoning ahead beyond Qwen3 (each reasoning model's frame must be measured); MiniCPM, newer Qwen
     and Gemma were asked for on Reddit

## Done and measured

- **2026-10-30 · The catalog's estimate follows llama.cpp's fit; the 14B's IQ4_XS where its Q4_K_M does not fit.**
  Measured on the 4070 SUPER with the server log (`-lv 4`): `--fit` projects model + KV cache + compute buffers
  (14B: 8162 + 1280 + 150 = 9591 MiB) and keeps 1024 MiB free (its default `--fit-target`); the CUDA context takes
  ~100 MiB more than nvidia-smi counts. Simulating a busier card with `-fitt 2500` and `-fitt 5000`: 36/41 and 24/41
  layers, ~232 MiB per layer with its KV. The token embeddings stay in RAM (`CPU_Mapped`) unless the output layer
  shares them (Qwen3 0.6-4B, Llama 3.2, Gemma 3, Phi-4-mini do, and keep them on the GPU); generating reads one row
  of them, so they cost RAM, not time. Changes: `cpu_embd_b` per model (from config.json; from the tensors for files
  of your own), not counted against VRAM nor in the decode time; `VRAM_RESERVE_MB` 900 → 1300 (it was not the
  reserve that was too large but the embeddings counted on the GPU); `GPU_EFFICIENCY` unchanged: without the
  embeddings, K-quants read 0.60-0.65 of the bandwidth on both the 2060 and the 4070 SUPER (the 0.66-0.78 noted
  before was the embeddings and IQ4_XS); prefill on Ada 0.125e12 FLOPS per GB/s (Turing 0.051e12). Estimate against
  measured: 2060 4B 89.6 / 87.1 tok/s, 8B 17.0 / 16.2, 14B 5.0 / 4.9; 4070 SUPER 4B 134 / 130 (prefill 7875 / 7441),
  14B all on the GPU 39.3 / 40.9 (prefill 2128 / 2486); M2 Pro 4B 53 / 54, 14B 15.6 / 18. The 4070 SUPER now
  recommends the 14B (was the 8B). The 1.9 GB of desktop noted on 2026-10-26 cannot have been taken when the 14B
  loaded 41/41 (fit would have dropped layers): today 1.1 GB, and fit leaves 1477 MiB free.
  `fit_quant` in the catalog (14B: IQ4_XS from unsloth's repo) is what the app offers on a GPU where the default
  does not fit all in VRAM (`catalog.preferred_quant`; the CPU-only build keeps Q4_K_M); `mafdet models --download
  qwen3-14b` follows it. `recommend` takes "all on the GPU" as 90% or more: on the 2060 with 1.4 GB taken the 4B
  has one layer on the CPU (4.8 s instead of 4.0) and the pick fell to the 1.7B.
  `tests/test_local_models.py::test_files_of_your_own` fails on Windows before and after this change (paths
  against file names), not looked into.
- **2026-10-28 · First Radeon: RX 5700 XT (8 GB, 448 GB/s), Qwen3-4B Q4_K_M, Vulkan, from issue #1.** Sent by
  a volunteer: Ryzen 5 3600, 16 GB, Windows 11, driver 31.0.23013.1023, llama.cpp b11747. Measure: decode 69 tok/s,
  prefill **93 tok/s**. `thinking_ahead.py`, 24 problems (`thinking_ahead_4b_rx5700xt.json`): no ahead 24/24,
  mean 14.00 s, median 10.94 s, 920 reasoning tokens, 67.8 tok/s; reasoning ahead 24/24, mean **7.69 s**, median
  **6.81 s**, 595 tokens (343 ahead), 63.6 tok/s; faster in 22 of 24. Reasoning ahead works, but the wait is
  twice the 2060's (3.2-4.2 s) with about the same reasoning left after Enter (252 tokens against 244). Where the
  rest goes: without reasoning ahead the wait is all decode (wait - left/tps ≈ 0); with it, 2.2-5.5 s more on
  every problem, and the extra time over the tokens reasoned ahead is exactly 93 tok/s. The notes come after
  the message in the prompt, so the last words typed make the model reread all of them after Enter: 343 tokens,
  0.17 s on the 2060, 3.7 s here. Decode is ~0.37 of the bandwidth (0.62 on the 2060); prefill ~30 times below
  the catalog's estimate. Asked next: llama-bench on that card (see Next, 4).
- **2026-10-27 · Any GGUF: the Hugging Face search and files of your own.** The app now runs models outside
  the catalog: searched and downloaded in the app (`mafdet/core/hub.py`), put in `~/.mafdet/models`, or added
  from anywhere. Their tokenizer is the GGUF's own, through llama-server's `/tokenize`, `/detokenize` and
  `/apply-template` (`ServerTokenizer` in `llama_backend.py`): on Qwen3-4B identical to `Qwen/Qwen3-4B`'s
  (ids with special tokens and odd spacing, every prefix decoded the same, emoji cut mid-character
  included, chat template thinking on and off); 0.16 ms a call, also while the server generates. The
  catalog's models keep the Hugging Face tokenizer. Reasoning ahead stays on Qwen3 only (architecture
  qwen3/qwen3moe and `enable_thinking` in the template). Checked end to end on the M2 Pro: SmolLM2-135M
  downloaded from the search (interrupted and resumed), answer written ahead at the pause; the Qwen3-4B
  file as a model of its own reasoned ahead (342 of 637 reasoning tokens before Enter), right answer.
- **2026-10-27 · IQ4_XS against Q4_K_M on the RTX 2060: no gain on the 4B, +11% decode on the 14B.** Same files as
  on the 4070 SUPER (Q4_K_M from Qwen's repo, IQ4_XS from unsloth's). `research/measure_gguf.py` (the app's
  Measure on any GGUF, with the layers on the GPU from the server log; new), two alternated rounds, desktop
  holding ~1.4 GB of VRAM:
  - 4B, 37/37 layers on the GPU both: prefill 1861-2076 vs **2318-2347** tok/s, decode **86.0-86.3** vs 83.9-84.1.
    IQ4_XS's 9% fewer bytes do not make Turing write faster (Ada: +31%).
  - 14B: Q4_K_M 15/41 layers on the GPU, prefill 305-339, decode 5.2; IQ4_XS 17/41, prefill 396-406, decode
    **5.8-5.9** (+11%, about what the bytes and the 2 extra layers explain).
  `thinking_ahead.py`, reasoning ahead only, configs measured one after the other:
  - 4B, 24 problems (`quant_4b_2060.json`): Q4_K_M 23/24, mean 3.43 s, median 3.15 s (correct answers), 1078
    reasoning tokens, 74.8 tok/s; IQ4_XS 23/24, mean 3.90 s, median 3.65 s, 1120 tokens, 73.5 tok/s. The one miss
    of each is a reasoning cut at --max-tokens 3000 with no answer (problem 16 for Q4_K_M, 8 for IQ4_XS), not a
    wrong number.
  - 14B, the README's 6 problems, --warm-up, --max-tokens 1800 (`quant_14b_2060.json`): Q4_K_M 5/6, mean 63.5 s,
    median 65.3 s, 502 tokens (178 ahead), 4.9 tok/s; IQ4_XS 5/6, mean **43.4 s**, median 44.8 s, 406 tokens
    (180 ahead), 5.6 tok/s. Single runs vary a lot on this model (the README's 47 s mean is one run).
  - Problem 19 (30 vehicles, 100 wheels, how many motorbikes: 10) was wrong in both, with numbers that are not
    in the text: "38" with "38 × 2 + 2 × 4 = 84" (Q4_K_M), "30" from "x + y = 60" (IQ4_XS). Rerun without
    reasoning ahead (IQ4_XS, `--ahead 0`): right twice, 155 and 132 s. The run with it was killed for low
    memory (the 14B takes most of the 16 GB). See Next, 0.
- **2026-10-27 · Qwen3-14B IQ4_XS against Q4_K_M on 24 problems (4070 SUPER): same answers, +8% in the chat, not
  adopted as the default yet.** Reasoning ahead, --warm-up, --max-tokens 1800 (`quant_14b_4070s.json`): IQ4_XS
  24/24, mean wait 6.03 s, median 4.98 s, 43-47 tok/s in the chat; Q4_K_M 24/24, 5.65 / 5.03 s, 40-45 tok/s. The
  +31% of the Measure-style decode (short prompt, 128 tokens) shrinks to +8% with the chat's context (notes and
  600-1000 reasoning tokens: the attention's share of each token grows, the weights' share falls), and the IQ4_XS
  reasoned longer in this run (903 against 847 tokens, single runs vary that much), so the wait is the same. What it
  does give on a 12 GB card is room: 830 MB of VRAM (the Q4_K_M leaves 800 MB free with the desktop's 1.9 GB taken).
  Not the default, because the 4B lost an answer with it (below) and the 2060's 14B runs its CPU layers on the IQ4_XS
  CPU kernels; a candidate as a second variant of the 14B for 12 GB cards, to decide with the 2060 numbers.
  **The checker was wrong on trailing zeros**: problem 18's "22,50 euro" (22,5) counted as a miss; `correct()` now
  accepts trailing zeros after the decimal separator (not after an integer: 80 is not 8). Every saved run recounted:
  only that row changes (the README's numbers stand).
- **2026-10-26 · Qwen3-4B IQ4_XS against Q4_K_M (4070 SUPER): 4% faster, one answer worse — not worth it on the 4B.**
  24 problems, reasoning ahead (`quant_4b_4070s.json`): Q4_K_M 24/24, mean wait 2.83 s, median 2.14 s, 106-119 tok/s;
  IQ4_XS 23/24, 2.37 / 1.73 s, 117-130 tok/s. The miss is problem 8 (Anna's age, the ambiguous "il doppio della sua
  età"): read as her own age, answer 10. Repeated 4 times each (`quant_4b_4070s_p8.json`): IQ4_XS 3/4 (the miss a
  reasoning loop cut at 3000 tokens, no answer), Q4_K_M 4/4. In all 26/28 against 28/28. Measure-style: prefill
  7441 / 7027 tok/s, decode **129.8 / 134.9 tok/s** (2376 / 2160 MiB on the GPU). The 14B's +31% does not carry
  over to the 4B: +4% clean, +9% in the chat benchmark. During the repeats the tok/s had dropped to 90-104 on both
  quantizations (the GPU after 40 minutes of runs): compare rows measured one after the other, as on the Mac.
- **2026-10-26 · Qwen3-14B on a 12 GB card (RTX 4070 SUPER): all on the GPU, IQ4_XS 31% faster than Q4_K_M, same
  answers.** The plan was "the largest model entirely in VRAM, with a smaller quantization, against the same model
  split with the CPU"; on this card the 14B Q4_K_M turned out to be entirely in VRAM already (`-lv 5` in the server
  log: `offloaded 41/41 layers`, CUDA0 model buffer 8162 MiB, CPU_Mapped 417 MiB = the embeddings, which llama.cpp
  never offloads; KV 1280 MiB at ctx 8192; 11.5 GB of VRAM in use with the desktop's 1.9), so the comparison became
  three all-GPU configurations. `thinking_ahead.py`, 6 problems, --warm-up, --max-tokens 1800, reasoning ahead
  (`thinking_ahead_14b_4070s.json`): Q4_K_M 6/6, mean wait 5.25 s, median 3.99 s, 33-42 tok/s; Q4_K_M with
  `-ctk q8_0 -ctv q8_0` 6/6, 4.54 / 4.12 s, 38-41 tok/s (KV 680 MiB, nothing else changes); IQ4_XS (unsloth,
  7335 MiB on the GPU) 6/6, 3.66 / 3.97 s, 43-47 tok/s. Measure-style (prompt of 800 tokens, 128 tokens at
  temperature 0): prefill 2486 / 2400 / 2567 tok/s, decode **40.9 / 41.1 / 53.5 tok/s**. The IQ4_XS gain (+31%) is
  three times what its bytes explain (−10%): on Ada the IQ4_XS kernel runs better than Q4_K_M's. The catalog's
  estimate for this machine was "partial, 95% on the GPU, 23.8 tok/s", and it recommends the 8B (66.7 tok/s
  estimated): the 14B at 41-53 tok/s is the right pick here (see Next, 0).
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
