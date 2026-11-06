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

Fourth machine (since 2026-11-02): RTX 3060 12 GB (360 GB/s, Ampere, ~0.8 GB taken by the desktop), i5-12400F
(6 cores, 12 threads), 16 GB DDR4 (~29 GB/s by `doctor`), Windows 11, llama.cpp b11747 CUDA 13 in ~/.mafdet/llama.cpp,
`.venv` on Python 3.14. The whole 12 GB used: the 14B all on the GPU (35 tok/s), Qwen3-30B-A3B with its experts
split with the CPU (39.5 tok/s, 11.8-12.0 GB in use), see "Done and measured" (2026-11-02).

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

1. **Put more of the 14B on the GPU: done on a simulated 6 GB budget (2026-10-31, below), in the app.** A model
   split with the CPU now gets IQ4_XS (14B) and a q8_0 KV cache: 24/24, mean wait 34.0 → 21.1 s. Left:
   - started on 2026-11-06, stopped by the user after 4 of 24 (enough measurements for now): IQ4_XS with the app's
     defaults (q8_0 KV chosen by itself), --warm-up, --max-tokens 1800, problems 0-3: 4/4 right, waits 27.7,
     100.9, 46.1, 33.3 s (mean 52.0 s), 4.6-5.1 tok/s, 152-222 reasoning tokens ahead
     (`split_14b_2060_chat24.json`). Too few to compare with the README's 47 s; the f16 Q4_K_M not run.
   - confirm on the RTX 2060 itself: `research/thinking_ahead.py` on the 24 problems with the app's defaults
     (the KV choice is automatic there) against `--llama-args='-ctk f16 -ctv f16'` with the Q4_K_M; this machine's
     CPU and RAM run the split twice as fast as the 2060's (10 against 5 tok/s), the ratios should hold. Then the
     README's 14B row (47 s) and "Reasoning ahead for models larger than the VRAM" in the Roadmap.
   - threads (`-t`) and batch sizes: not measured, they depend on the CPU (the 2060's i5-10400 has 6 cores, this
     one 6+8): only on the 2060.
   - FFNs on the CPU (`-ngl all -ncffn N`) lost in the chat (below): keep it in mind only for long contexts.

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
   the scope and the contract; contributions as pull requests to `main`, code in `mafdet/proxy/`. Reading
   ahead for llama.cpp merged (PR #4, That80-20, 2026-11-06); roadmap step 0 (a solid core) done and measured
   the same day (issue #5, "Done and measured" 2026-11-06, both entries). The roadmap is in `PROXY.md`.
   Steps 1-2 on Open WebUI done the same day ("Done and measured"). Next: a timed multi-turn benchmark through
   Open WebUI, the Tampermonkey install tried by a person, then step 3 (one command, a 5-minute guide).
   Left from step 0:
   - the audit cannot see requests that bypass the proxy (the bench's own OFF and reset runs): its
     mismatches there are the bench's, not the proxy's.
   - the GPU's clocks: on the 2060 the first request after seconds of idle is slow (one token 45 ms
     instead of 12, a short prompt ~300 ms instead of ~45). The drafts keep the GPU awake, most of the gain
     on short messages. Check on another card (the 4070 SUPER, the 3060) before writing it as a result.
   - thinking ahead needs a raw completion endpoint, cancelling, and a frame measured per model family.

6. Other ideas discussed, not started:
   - **past reasoning out of the history: done (2026-11-04, "Done and measured").** Left: the same conversations
     on the 14B (its slower reading makes the reread answer cost more, but it happens while typing); a miss at
     turn 6 (problem 17, 20:30 for 20:24) to watch in later multi-turn runs.
   - show the answer written ahead live under the input field (zero perceived wait; a product/UX change,
     test with people, not a benchmark)
   - harder benchmark (problems the 4B gets wrong 30-50% of the time) to see accuracy effects at all
   - reasoning ahead beyond Qwen3 (each reasoning model's frame must be measured); MiniCPM, newer Qwen
     and Gemma were asked for on Reddit

7. **RTX 3060 follow-ups (2026-11-02, above).**
   - The catalog on Ampere: a prefill coefficient between Turing's and Ada's (~0.08e12 from the 3060's 4B and 8B),
     and decode on the 8B/14B 20-26% low; one card only, check on a second Ampere before changing
     GPU_EFFICIENCY.
   - The catalog's MoE estimate: with the experts in RAM (what llama.cpp's fit does) the 30B-A3B is all-GPU except
     the experts, ~34 tok/s here, not 27 at 54%.
   - A MoE split with the CPU: a smaller fit target (`-fitt 256`: +9%) or `-ncmoe` sized from the free VRAM (+17%)
     in `LlamaServer`; why the fit ignores the room freed by a q8_0 KV cache; the 30B-A3B's IQ4_XS (unsloth).
   - The README: a "RTX 3060 12 GB" row (4B, 14B, 30B-A3B) when the user wants these numbers there.

## Done and measured

- **2026-11-06 · Thinking ahead in the proxy (`--think-ahead`, roadmap step 5): Open WebUI with Qwen3 thinking on,
  Enter → first word of the answer 7.55 s → 4.31 s mean (median 6.34 → 4.06 s), faster in 20 of 22 turns
  (RTX 2060, Qwen3-4B Q4_K_M).** `mafdet/proxy/think.py`: the app's frame (SEED, BRIDGE, SENT, META_BIAS,
  copied from `watch.py`) run by the proxy on the draft, in chunks of 32 tokens on slot 0; plain read-ahead off
  while it reasons; on Enter the final prompt + notes + SENT through `/completion`, streamed back as an OpenAI
  chat completion (notes and the rest of the reasoning as `reasoning_content`, the answer as `content`, Qwen3's
  `<tool_call>` as `tool_calls`). First try by hand: 566 tokens reasoned while typing, answer 4.3 s after Enter
  (the run before without it: 2000 tokens of reasoning, cut, no answer). Found there: the drafts of the first
  request after a restart had no tools (the saved request had none) and the answer followed them: the prompt
  is now rendered again when the request's tools or template options differ from the draft's.
  `research/proxy_openwebui.py --think --max-tokens 3000` (Open WebUI's default: thinking on; it now also
  records the first word of the answer), same 8 conversations, one warm-up each block, through `mafdet proxy`:
  | | first word of the answer: mean | median | p90 | max | whole answer | tokens after Enter |
  |---|---|---|---|---|---|---|
  | reading ahead only | 7.55 s | 6.34 s | 11.7 s | 21.0 s | 9.37 s | 338 |
  | thinking ahead | **4.31 s** | **4.06 s** | 5.1 s | 10.4 s | 7.49 s | 230 |
  Not checked: whether the answers are right (the bench has no answer key; the app's `thinking_ahead.py` does,
  24/24 on the 4B). The run with llama-server's defaults and thinking on was stopped by Claude Code for low
  memory (with Open WebUI): not measured. Data: `proxy_openwebui_think_4b_2060.json` (arms t1, t2).

- **2026-11-06 · Open WebUI, Enter → first token, 8 conversations each way: llama-server with its default options
  5.40 s mean (median 3.31 s), through `mafdet proxy` 0.16 s (median 0.14 s); RTX 2060, Qwen3-4B Q4_K_M.**
  `research/proxy_openwebui.py` drives Open WebUI 0.13.0 as its frontend does (POST /api/chat/completions and a
  socket.io session): its tools, history, title, tags and follow-ups reach the engine as for a person. 8
  conversations of 3 messages (2 of them a ~1.5k-token paste, then a typed question, then a follow-up: 2
  counted turns), typed at 12 chars/s (drafts at most 0.6 s apart, as the user script), 1 s before Enter, 3 s
  reading after each answer; thinking off for the answers (`chat_template_kwargs` as a custom param),
  max_tokens 200; Open WebUI's side requests as it sends them (no tools, Qwen3 thinking). One warm-up
  conversation per block, not counted. Time from the POST to the first token on the socket, and llama-server's
  own timings relayed by Open WebUI (`prompt_n`, `cache_n`, `prompt_ms`). 22 counted turns each:
  | | mean | median | p90 | max | read on Enter | waiting besides the read |
  |---|---|---|---|---|---|---|
  | llama-server defaults (`-c 16384 -ngl 99`: 4 slots, kvu, RAM cache) | 5.40 s | 3.31 s | 10.8 s | 20.7 s | 199 tok, 767 ms | 4.63 s |
  | llama-server as `mafdet proxy` starts it, no proxy (2 slots, kvu, `--cache-ram 0`) | 11.66 s | 9.50 s | 16.1 s | 58.3 s | 6206 tok, 4430 ms | 7.23 s |
  | `mafdet proxy` | **0.16 s** | **0.14 s** | 0.16 s | 0.4 s | 1 tok, 31 ms | 0.13 s |
  With the defaults llama-server mostly keeps the conversation (4 slots, prompt-similarity selection, RAM cache):
  the wait is queueing behind the title, tags and follow-ups, which think at length on Qwen3 and share the 16k
  pool. With 2 slots and no proxy the follow-ups land on the conversation's slot (LRU) and it is read again
  (~6.2k tokens, 4.4 s) on top of the queue. Through the proxy every turn had 1 token left to read, first turns
  and pastes included (the tools come from the saved last request). First design alternated the arms on one
  llama-server: the direct arm's requests took the proxy's slot 0 (a first turn waited 4.4 s with 43 tokens to
  read), so each arm now runs on its own, with a warm-up. Not measured: the answer's writing speed (it shares
  the GPU with the side requests on slot 1), thinking on for the answers, other cards and models. Data:
  `proxy_openwebui_4b_2060.json` (proxy, 2 slots), `proxy_openwebui_4b_2060_defaults.json`.

- **2026-11-06 · `mafdet proxy`: one command starts llama-server for the proxy and the proxy (roadmap step 3).**
  `mafdet/proxy/launch.py`, through the app's `LlamaServer` and catalog: the model chosen in the app or
  `--model` (catalog id, file, repo:file), 2 slots with one 16k pool (`-kvu`; LlamaServer already passes
  `--cache-ram 0`, which keeps idle slots from being cleared), `--alias` the catalog id (Open WebUI showed the
  file path), engine on port 8078 (fixed, so the profile and the saved last request are found again), CORS
  for Open WebUI on this machine by default. Found on the first run: the profile's file name kept the model
  id's ":" (`qwen3-4b:Q4_K_M`), which Windows refuses (WinError 87; it had made an alternate data stream):
  now only `[A-Za-z0-9._-]`. Through it, in Chrome with Open WebUI: a chat's second message 1 token on
  Enter (`n_past = 7171`); the first was cold (the saved request was under the old file name). Two proxy
  tests that read the cache or the history right after an answer were flaky in the full suite (the proxy
  updates them after sending the answer): they wait for it now. The guide: "Open WebUI + MAFDET in 5
  minutes" (`mafdet/proxy/clients/README.md`).

- **2026-11-06 · Proxy on Open WebUI tried by hand, with Tampermonkey: two fixes, then a new chat's first
  message 23 tokens to read on Enter and the second 1 (RTX 2060, Qwen3-4B).** The user installed the script
  in Chrome through Tampermonkey: drafts arrived (97 warms, no mismatch). Two problems showed up:
  - A follow-up suggestion clicked is sent with no drafts: the proxy took it for a side request (it matched
    no draft) and sent it to slot 1. Now a side request must also be not streamed (Open WebUI streams every
    turn and asks for titles, tags and follow-ups whole), and a streamed turn that continues the conversation
    no draft has named yet is kept for the next id. The last request's tools, template options and system
    prompt are saved (`*.last.json` next to the profile), so the first drafts after a restart carry them.
  - Without `--cache-ram 0` (the first run had it), the second turn's drafts found slot 0 empty (audit:
    predicted 7087, reported 0) and reread 7117 tokens in 4.9 s while typing. llama-server's
    `--cache-idle-slots` (on by default with a unified KV cache) saves an idle slot to RAM and clears it when
    another slot gets a task: the title on slot 1 emptied the conversation. `--no-cache-idle-slots` fixes it.
  Then, `-np 2 -kvu --no-cache-idle-slots`, a new chat typed in three bursts: the drafts read 7068 tokens
  (tools included, from the saved request) while typing, **23 tokens on Enter (37 ms)**; second message,
  drafts reused 7185, **1 token on Enter**; a suggestion clicked: slot 0, 63 tokens. Side requests: 9 on
  slot 1.

- **2026-11-06 · Proxy steps 1-2 on Open WebUI: second turn of a chat, 7158 tokens (4.7 s) to read on Enter
  without the proxy, 1 with it (RTX 2060, Qwen3-4B Q4_K_M).** Open WebUI 0.13.0 in its own venv
  (`E:\openwebui`, Python 3.12), llama-server `-np 2 -kvu -c 16384`.
  - What Open WebUI sends (captured through a logging pass-through, 10 conversations): `stream, model,
    messages, tools`. The history is rebuilt on its server; the previous answer comes back as its content,
    byte-identical to the stream, reasoning stripped. **30 built-in tools on every request, ~5.8k tokens,
    which Qwen3's template writes first**: a draft rendered without them shares nothing with the request.
    Side requests, all to the chat's model, `stream: false`, no tools: a title 60 ms *before* the answer on a
    new chat, follow-ups after every answer, tags after the first. On one slot they evict the conversation.
    Long pastes become a file only with "Paste Large Text as File" (off by default); a file changes the
    prompt from the first user message every turn (RAG template), so it is not read ahead.
  - Proxy (step 1): a draft may carry only `{conversation_id, text}`; the proxy appends it to the
    conversation it saw for that id (or the last system prompt for "new", handed to the chat's real id on
    its next draft), renders it with the last request's `tools` and `chat_template_kwargs`, and matches
    the request on Enter by its user message or its history. With 2+ slots the conversation is pinned to
    slot 0 (`id_slot`) and a request matching no conversation while drafting goes to slot 1. 41 tests.
    The draft cap went from 4096 to 65536 tokens (with the tools, 4096 never reached the message).
  - `mafdet/proxy/clients/open-webui.user.js` (step 2): reads Open WebUI's TipTap box (`#chat-input`) as
    its turndown does, posts drafts debounced 250 ms (at most 600 ms apart, at once on a paste). Tried by
    running its code in the page from Chrome, not installed through Tampermonkey.
  - In Chrome, a new chat and a second turn typed in three bursts. With the proxy: turn 1 read 7065 tokens
    (the proxy did not know the tools yet: first request since it started); turn 2 drafts of 7176 tokens
    (predicted 7161, reported 7162 reused), on Enter `n_past = 7176`, 1 token read; title, follow-ups and
    tags (3) on slot 1. Without the proxy (Open WebUI straight to llama-server, same 2 slots): llama-server
    picks slots by LRU, the follow-ups landed on the conversation's slot, turn 1 read 7065 tokens (4.3 s)
    and turn 2 **7158 tokens in 4692 ms** before the first token. One conversation each way: the size of the
    gain is the tools reread, not a statistic.
  - Not done: a timed multi-turn benchmark through Open WebUI; Tampermonkey install by a person; rich text
    (lists, code) in the box; what one slot gives with the proxy (it still loses the conversation to the
    side requests).

- **2026-11-06 · Proxy step 0, a solid core: short typed messages 302 → 102 ms, a paste 2 s before Enter
  1365 → 54 ms (RTX 2060, Qwen3-4B Q4_K_M).** Same setup as the entry below. What changed in `mafdet/proxy/`:
  - The draft is rendered with the request's `chat_template_kwargs` (sent with the draft, or those of the last
    chat request). With `enable_thinking: false` Qwen3's template ends the prompt with an empty think block;
    the drafts were rendered without it, so on Enter the engine still had 6-7 tokens to read, and a batch that
    small costs ~6 ms a token on the 2060 (as much as 30 tokens). Now 1 token is left on Enter.
  - The generation lock and the warm lock are the slot's (`SharedCache`), not the conversation's: an answer
    from a client with no `conversation_id` holds every draft. On Enter every pending draft is dropped (a
    stale one warmed after the answer would evict it); a draft typed during the answer is warmed when it
    ends. After an answer the cache model holds its prompt, and the next prediction is a lower bound.
  - `--exact` (no read-ahead, `cache_prompt: false`), `--cors <origins>` (off by default), one write per
    streamed chunk. 33 proxy tests (11 new).
  - The bench: the OFF run now waits as long as the typing lasts, then the same pause, before Enter (the
    entry below sent it right after the reset). That is what made the proxy look ~25 ms slower: the first
    request after seconds of little work runs at lower GPU clocks. In the server log the drafts' warms of
    ~11 tokens go from 21 to 80 ms as the typing goes on, and one token on Enter costs 45 ms instead of 12.
    The proxy's own hop costs ~1 ms (direct 31.8-32.3 ms, through the proxy 32.2-34.2 ms, no drafts).
  Enter → first token, without → with the proxy:
  - 24 typed problems, 12 chars/s, Enter 0.35 s after the last key: **302 → 102 ms** mean (median 287 → 85),
    faster in 24/24. Without the proxy the GPU sits idle while the user types; with it the drafts keep it
    awake and the message is read. How much of it is the clocks is a property of the card: see Next 5.
  - 4 pasted prompts (~2k and ~4k tokens), Enter 0.35 s after the paste: 1171 → 880 ms; Enter 2 s after the
    paste: **1365 → 54 ms** (944 → 54, 1782 → 51, 963 → 41, 1771 → 68).
  Answers: 19/24 typed and 3/4 pasted byte-identical; 26 of 27 comparable agree on the number. The one that
  does not is again the printer problem (133 without, 132 with the proxy, the right reading), as on the V620:
  llama.cpp computes the prefix read ahead in other batches. engine_stability identical. Audit 493 warms,
  4 mismatches (requests the proxy cannot see). Data: `proxy_read_ahead_4b_2060.json`,
  `proxy_read_ahead_4b_2060_settle2.json` (both rewritten by this run).

- **2026-11-06 · Proxy read-ahead on the RTX 2060, with the bench fixed: a pasted prompt is read while the user
  waits to press Enter (1181 → 49 ms with 2 s before Enter).** The typed numbers here are superseded by the
  entry above (the OFF run did not wait for the typing; the "~25 ms added" was the GPU's clocks). PR #4
  (2026-11-03 entry below) measured the proxy right after the same prompt sent without it, cache on: with one
  slot the ON run found its prompt already cached by the OFF run, drafts or not. And the clock started when
  the response headers arrived: llama-server sends them before reading the prompt, the proxy only after the
  warm still running is done, so that wait was not counted. `mafdet/proxy/bench.py` now leaves the slot
  holding the system prompt and an unrelated message before each run (the state after a previous turn) and
  starts the clock before the request goes out. RTX 2060, llama.cpp b11747, Qwen3-4B Q4_K_M all on the GPU,
  greedy, thinking off, 8192 ctx, `-np 1 --cache-ram 0 --cache-reuse 0`, proxy and engine on the same PC.
  Probe: T1 (`n_predict: 0`, `timings.cache_n`, edits hit the prefix), as on the V620. Enter → first token,
  without → with the proxy:
  - 24 typed problems, 12 chars/s, Enter 0.35 s after the last key: **36 → 59 ms** mean (median 32 → 62),
    slower in 23/24 (explained in the entry above: GPU clocks and the missing think block).
  - 4 pasted prompts (~2k and ~4k tokens), Enter 0.35 s after the paste: **1160 → 873 ms** (771 → 466,
    1553 → 1274): the gain is the 0.35 s, the rest of the reading is still running at Enter.
  - The same with Enter 2 s after the paste: **1181 → 49 ms** (760 → 39, 1661 → 56).
  - First run, clock at the headers (the PR's clock, cache reset fixed): typed 31 → 32 ms, pasted 1213 →
    22 ms, so the PR's 492 → 23 was mostly the cache left by the OFF run plus the uncounted wait.
  Answers: 21/24 typed and 4/4 pasted byte-identical, all 27 comparable agree on the number (one hit
  max_tokens); engine_stability identical. Audit 493 warms, 4 mismatches. Data:
  not kept: the same files hold the run of the entry above, in git history at b0be007.

- **2026-11-04 · Past turns' reasoning out of the history: the context lasts, the wait does not grow.** With
  `--think`, `Session._next_history` keeps message and answer of each turn (what follows the last </think>, its
  leading blank lines dropped), as Qwen3's template renders past turns; the next turn's cache rereads the answer
  while the message is typed. Checked on the 4B: the history after two turns decodes to exactly the template's
  text. `--keep-reasoning` keeps the old behavior. `thinking_ahead.py --turns 6` (new: problems asked in
  conversations of 6, a new chat when the context is full), Qwen3-4B on the 4070 SUPER, --warm-up, configs one
  after the other (`history_4b_4070s.json`):
  - drop (the new default): 23/24, mean wait **1.94 s**, median 1.41 s, 119 tok/s, longest history **1342 tokens**
    after 6 turns, context never full.
  - keep: 24/24, mean 2.45 s, median 2.47 s, 108 tok/s (93-91 at turns 5-6 against 121 at turn 1: attention over a
    longer context), history 7000-8015 tokens by turn 5-6, **context full twice** at turn 6 ("Start a new chat").
  - The miss is problem 17 at turn 6 (the clock: 20:30 for 20:24, an arithmetic slip, right in 'keep' as turn 1
    of a new chat). One sample: not read as an effect of the shorter history.
  On by default in the app (the setting reaches `Session` through `getattr`, False unless given).

- **2026-11-02 · RTX 3060 12 GB (Ampere), the whole VRAM: the 14B all on the GPU, the 30B-A3B at 39.5 tok/s with
  its experts in RAM; reasoning ahead 20.2 → 7.0 s (14B) and 18.9 → 7.3 s (30B-A3B).** New machine (see Where we
  are). Measure-style (`research/measure_gguf.py`, two rounds for the 14B, identical to ±2 tok/s), prefill / decode
  tok/s, against the catalog's estimate here:
  - 4B Q4_K_M 37/37: **3356 / 98.8** (estimate 2295 / 96.0); 8B Q4_K_M 37/37: **2010 / 62.0** (1120 / 51.6).
  - 14B Q4_K_M 41/41: **1120 / 35.4** (620 / 28.1); IQ4_XS 41/41: **1240 / 39.4** (620 / 31.1): +11% decode on Ampere
    (Ada +31%, Turing +0% on the 4B).
  - So on Ampere the catalog is low: prefill 1.5-1.8 times (GPU_PREFILL_FLOPS_PER_GBS is Turing's 0.051e12; the
    3060 measures ~0.075-0.09e12), decode by 3% on the 4B, 20-26% on the 8B and 14B (they read 0.74-0.78 of the
    bandwidth, the 4B 0.64). It recommends the 14B here, which is right.
  Qwen3-30B-A3B Q4_K_M (17.7 GB, 48 layers, ~3B active; `research/moe_30b.py`, new, two alternated rounds,
  `moe_30b_3060.json`). llama.cpp's fit already does the right thing for a MoE (server log: "all MoE tensors moved
  to system memory", then "filling dense-only layers"): attention, router and KV of all 49 layers on the GPU, the
  experts of 24 layers on the CPU, 1 GB left free. The file does not fit in 16 GB of RAM next to the system: the
  first prompt after loading reads the CPU's experts from disk (31-71 tok/s), so the script reads the text once
  before measuring. VRAM in use (nvidia-smi, desktop included) / prefill / decode after a short prompt / after 1607
  tokens, both rounds:
  - fit (the app's default): 10.9 GB, 508-527 / **33.7** / 37.2-37.7. fit with q8_0 KV: the same VRAM and the same
    24 expert layers on the CPU (the fit does not use the 384 MiB freed: not looked into), same speed.
  - `-fitt 256`: 11.6 GB, 22 on the CPU, 555-570 / 36.7-37.3 / 39.7-39.8.
  - `-fit off -ngl all -ncmoe 21`: 11.5-11.7 GB, 462-519 / 31.3-38.3 / 34.1-39.0 (one round slow: too close to full).
  - `-fit off -ngl all -ncmoe 20 -ctk q8_0 -ctv q8_0`: 11.8-12.0 GB, 528-532 / **39.5-39.6** / 39.4-40.6: +17% over
    the fit, the best; `-ncmoe 19` not tried (it would leave nothing for the desktop).
  The catalog estimated 54% on the GPU, 27.3 / 611 tok/s; it counts the 30B as layers, not as experts in RAM.
  `research/thinking_ahead.py`, 24 problems, --warm-up, --max-tokens 1800 for the 14B and the 30B (3000 for the 4B),
  no ahead → reasoning ahead:
  - 4B Q4_K_M (`thinking_ahead_4b_3060.json`): 24/24 both, mean **10.25 → 3.29 s**, median 8.78 → 2.26 s, 920 →
    1102 reasoning tokens (814 ahead), 91 tok/s, faster in 24 of 24.
  - 14B Q4_K_M, all on the GPU (`thinking_ahead_14b_3060.json`): 23/24 → 24/24, mean **20.19 → 7.03 s**, median
    19.00 → 6.32 s, 764 → 726 tokens (500 ahead), 34.5 tok/s, faster in 23 of 24.
  - 30B-A3B, `-ncmoe 20` and q8_0 KV (`thinking_ahead_30b_3060.json`): 23/24 → 24/24, mean **18.94 → 7.33 s**,
    median 17.76 → 6.99 s, 791 → 602 tokens (346 ahead), 38-40 tok/s, faster in 23 of 24.
  Both misses are problem 16 (the snail) cut at 1800 tokens with no answer, not a wrong number. Reasoning ahead
  works on the MoE as on the dense models (first run of it on qwen3moe). Problem 19 right everywhere. The 30B-A3B
  writes faster than the 14B and waits about the same: it reasoned less ahead (346 against 500 tokens), with less
  room for its cache. Not adopted in the app: the hand placement (`-ncmoe`) needs the expert sizes and the free
  VRAM, see Next, 7.

- **2026-10-31 · Qwen3-14B on a 6 GB budget: IQ4_XS and a q8_0 KV cache, wait after Enter 34.0 → 21.1 s, 24/24.**
  On the 4070 SUPER, `-fitt 7100` leaves the 14B Q4_K_M 15/41 layers in 3903 MiB, as on the 2060 with the
  desktop's 1.4 GB; every configuration kept within that VRAM (`research/split_14b.py`, two alternated rounds,
  `split_14b_4070s.json` and `_b.json`). Prefill tok/s / decode tok/s after a short prompt / after 1607 tokens,
  mean of all rounds:
  - layers (today's app): 583 / 10.4 / 9.2. With q8_0 KV: 16/41 layers, 626 / 10.8 / 9.4.
  - FFNs on the CPU, attention and KV of every layer on the GPU (`-ngl all -ncffn 38`; llama.cpp's fit does not
    do this by itself, with `-ngl all` it ignores its target): 660 / 9.1 / 9.1; with q8_0 KV (`-ncffn 34`) 718 /
    10.0 / 9.9. Its decode does not drop with the context, the layer split's does.
  - IQ4_XS layers: 16/41, 680 / 11.7 / 10.3; with q8_0 KV 18/41, 728 / 12.4 / 10.8.
  - IQ4_XS, FFNs on the CPU, q8_0 KV (`-ncffn 32`): 838 / 11.9 / 11.7: best on the long prompt.
  In the chat (`thinking_ahead.py`, 6 problems, --warm-up, --max-tokens 1800, `split_14b_4070s_chat.json`): layers
  6/6, mean 27.5 s, median 26.0 s, 9.9 tok/s; IQ4_XS FFNs on the CPU q8_0 6/6, 21.8 / 17.0 s, 11.9 tok/s; IQ4_XS
  layers q8_0 6/6, 17.1 / 14.7 s, 12.2 tok/s. The chat's context (~1000 tokens) is too short for the FFN split
  to pay, and it needs the app to place tensors by hand: not adopted. All 24 problems
  (`split_14b_4070s_chat24.json`): IQ4_XS layers q8_0 24/24, mean **21.1 s**, median 18.4 s, 519 reasoning tokens
  (271 ahead), 12.1 tok/s; Q4_K_M layers 24/24, 34.0 / 31.5 s, 578 tokens (252 ahead), 9.8 tok/s; faster in 21
  of 24. Problem 19 right in every run here (it was wrong with reasoning ahead on the 2060's 14B).
  In the app: `LlamaServer` adds `-ctk q8_0 -ctv q8_0` when `catalog.splits_with_cpu` (the catalog's estimate on
  the GGUF's header and the VRAM free at that moment) says the model will not fit all in VRAM, unless the
  options given by hand set the cache type; checked in the server log (Qwen3-4B, 612 MiB of KV instead of 1152;
  with `-ctk f16 -ctv f16` given, f16). IQ4_XS for the 14B came the day before (`fit_quant`). Not on Apple
  Silicon (unified memory, unmeasured) nor the CPU build.
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
- **2026-11-03 · Proxy read-ahead, first measurement (`mafdet/proxy/`)** — one V620 (gfx1030,
  ROCm 7.2.4, `llama.cpp-rdna2` e194cbd), Qwen3-4B-Q4_K_M, greedy, thinking off, 8192 ctx,
  `-np 1 --cache-ram 0 --cache-reuse 0`; the proxy on another machine, over the LAN. Probe:
  tier T1 in 0.4 s (`n_predict: 0`, `timings.cache_n`, an edited tail still hits the prefix).
  Enter → first token, drafts on vs off: 24 typed problems at 12 chars/s **33 → 25 ms mean**
  (median 33 → 19); pasted prompts of ~2k and ~4k tokens **492 → 23 ms mean** (median 633 → 23).
  Audit: 493 warms, 5 prediction mismatches (1%); it was 28/493 when the cache was modelled per
  conversation instead of per engine slot (`SharedCache` in `proxy/session.py`). Answers: 20/28
  byte-identical, 27 of 28 complete enough to compare, 26 agree on the final number. The one
  that differs (133 → 132, the printer problem) is the model reading the question differently,
  and the proxy's run is the correct reading. A warm prefix is computed in different batches,
  so greedy decoding is not bit-reproducible against a cold run — same class as the 8
  wording-only rows. Control, no proxy involved (`engine_stability` in the data): cold vs cold
  and cold vs warm of the same prompt are identical, so the divergence needs a *partial* prefix
  (a draft) and belongs to the engine, not to the proxy. Data:
  `research/data/proxy_read_ahead_4b.json`. **The timings are not valid** (the OFF run cached the
  prompt for the ON run, and the proxy's wait before its headers was not counted): see 2026-11-06 above.
- **Terminal app restyle (2026-10-23)**: amber CRT, boot sequence, decrypting text (`mafdet/tui/fx.py`).

## How to work here

- Python: `.venv/Scripts/python` (Windows), `.venv/bin/python` (Mac). Tests: `<python> -m pytest -q`.
- Proxy: `python -m mafdet.proxy --url <engine> --model <id>` (front end posts drafts to
  `/v1/drafts`, stats at `/proxy/stats`); bench as `python -m mafdet.proxy.bench --url <engine>
  --model <id> --prompts 24 --long 4` (add `--append-log research/LOG.md` to write the block).
- Benchmark: `research/thinking_ahead.py` (24 Italian word problems typed in real time; `--problems`,
  `--config "name=strike options"`, `--gguf`, `--append`). The README's 4B table is `thinking_ahead_4b.json`.
- Runs longer than ~10 minutes get killed: split them with `--problems` and `--append`. A 4B run is
  ~25 s per problem; a 14B run ~2 min per problem.
- Single runs vary a lot (the same 24 problems: 3.2 s or 4.2 s mean wait). Compare configs measured the
  same day; prefer decode tok/s (rows' `tps`) and repeated runs over one mean.
- After every change: commit and push (the user asked for it). The user writes in Italian.
