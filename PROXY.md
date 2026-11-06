# MAFDET proxy (work in progress)

An OpenAI-compatible server that sits in front of an inference engine and brings MAFDET's reading and
thinking ahead to any front end (Open WebUI, Hermes, agents) and any engine (SGLang, vLLM, llama.cpp,
others). It lives in its own module (`mafdet/proxy/`), apart from the app, and stays experimental until it
works and is measured.

Status: reading ahead is built for llama.cpp in `mafdet/proxy/` (see its `README.md`); thinking ahead is
not built yet. Where it stands and what is left: "Where we are" and "Roadmap" below.

## Where we are (2026-11-06)

The goal is thinking ahead in the front end people already use: in the app, the wait after Enter goes from
12.2 to 3.2 s on Qwen3-4B. The proxy has the first step, reading ahead on llama.cpp (PR #4).

- **Built:** `POST /v1/drafts`, `GET /proxy/stats`, the chat request passed through unchanged, a probe that
  finds what the engine supports (tier T1 on llama.cpp: `n_predict: 0`, `timings.cache_n`, edits still hit
  the cached prefix), and an audit of the reuse on every warm. Standard library only, 22 tests.
- **Measured** (RTX 2060, Qwen3-4B Q4_K_M, `research/LOG.md` 2026-11-06), wait from Enter to the first token,
  without → with the proxy:
  - a pasted prompt of 2-4k tokens, Enter 2 s after the paste: **1181 → 49 ms**;
  - the same, Enter 0.35 s after the paste: 1160 → 873 ms (it hides only the time before Enter);
  - short typed messages: 36 → 59 ms. Nothing to read ahead there, and the proxy adds ~25 ms (to fix).
  - The final answers agree; the wording can differ when llama.cpp computes a cached prefix in other batches.
- **Not yet usable from a real front end:** no front end sends drafts today, and a browser page cannot call
  the proxy (no CORS headers).

## Roadmap

To take a piece, say so in an issue first, so that two people do not build the same thing. Each step ends
measured, with the numbers in `research/LOG.md`.

**0. A solid core.** The proxy must never make the wait longer.
- [ ] Find and remove the ~25 ms added on short messages.
- [ ] Hold the drafts' warms while *any* answer is being written on the engine's slot, not only one of the
      same conversation (a plain OpenAI client sends no `conversation_id`).
- [ ] On Enter, cancel the draft still waiting, so that it cannot evict the answer just written.
- [ ] Track the requests that go through the proxy in its model of the cache, so that the audit knows what is
      really there.
- [ ] An `exact` option: answers bit-identical to no proxy, without the gain.
- [ ] CORS headers, so that a web page can post drafts.

**1. Drafts without knowing the conversation.** What makes the proxy work with front ends not written for it.
- [ ] The front end sends only the text being typed; the proxy rebuilds the rest (system prompt, history,
      last answer) from the last request it passed through.
- [ ] Tell conversations apart without `conversation_id`, by the start of the message list.
- [ ] Handle the front ends' side requests (titles, tags, follow-up suggestions), which today evict the
      conversation from a one-slot engine.

**2. Front ends.**
- [ ] Open WebUI first: a browser user script (Tampermonkey) that posts the drafts, with Open WebUI untouched.
- [ ] `mafdet-drafts.js`: a few dozen lines for anyone writing a front end ("call this while the user types").
- [ ] Support inside Open WebUI itself (a pull request or an option there), so that no script is needed.
- [ ] A guide per front end, each one tried for real.

**3. Easy to install.**
- [ ] One command, `mafdet proxy`, that finds or starts llama-server and starts the proxy. Starting the engine
      means sharing the app's `LlamaServer`: a pull request of its own (see Contributing).
- [ ] A Docker image and a `docker-compose` file next to Open WebUI.
- [ ] A page "Open WebUI + MAFDET in 5 minutes".
- [ ] `/proxy/stats` as a small readable page: is it working, how much waiting it saved.

**4. More engines.**
- [ ] Ollama and LM Studio (what most people run): check first what they expose (prefix reuse, a request that
      only reads).
- [ ] vLLM and SGLang (long prompts, shared servers: where reading ahead gains most).
- [ ] The rest of the probe (slot affinity, cancelling, raw completions, tokenizer agreement), checked against
      the second engine.
- [ ] A table "engine, version, what works", measured and dated.

**5. Thinking ahead in the proxy.** The big gain, and the hardest part.
- [ ] llama.cpp first: raw completion, cancelling a generation in the middle, the Qwen3 frame the app uses.
- [ ] Opt-in: it changes the answer (it is not exact), and whoever turns it on must know that.
- [ ] A frame measured per model family, the most used first.
- [ ] Engines without a raw completion endpoint (Ignis for now) keep reading ahead only.

**6. Show that it works.**
- [ ] Realistic benchmarks: multi-turn chats, a pasted document followed by a question, through a real Open
      WebUI and not only `bench.py`.
- [ ] A "Proxy" section in the README with the results.
- [ ] Measurements from other cards and engines, asked of the community as for the Radeons.

Order: 0 and 1 first (the core), then 2 and 3 on Open WebUI, so that there is something people can try.
4 and 5 can go in parallel when someone works on them.

Still to decide (by the maintainer): whether the proxy may start llama-server itself, and whether to aim at
support inside Open WebUI or stay with the user script.

## What it does

1. **The front end sends drafts.** While the user types, it posts the conversation with the message as it
   stands, debounced (a few hundred ms), to an endpoint of the proxy, e.g. `POST /v1/drafts` with
   `{conversation_id, model, messages}`. On Enter it sends the usual `POST /v1/chat/completions`.
2. **Reading ahead (exact).** For each draft the proxy has the engine read the prompt so far, so the prefix
   cache is warm on Enter. Any engine with prefix caching does this (a prefill-only request:
   `max_tokens: 0` or `n_predict: 0` where the engine accepts it, otherwise `max_tokens: 1`
   with the token thrown away -- the proxy probes which one; SGLang's
   radix cache keeps the branches an edit leaves). The answer is the same as without the proxy.
3. **Thinking ahead (not exact).** The proxy has the model reason on the draft, keeps that reasoning across
   edits, and on Enter tells the model the message is complete and lets it go on from there. It needs
   more of the engine (below) and a frame measured for each model family.

## What the engine must offer

- Prefix caching (both).
- For thinking ahead: a raw completion endpoint where the proxy controls the exact prompt (the reasoning
  goes after the partial message, inside the assistant turn, so a chat API is not enough), streaming, and
  cancelling a generation in the middle.
- Tokenization that matches the model's (the app uses llama-server's `/tokenize` and `/apply-template`).

## What we know from the app

- The frame is model-specific: `SEED`, `BRIDGE` and `SENT` in `mafdet/watch.py`, measured on Qwen3 only.
  Another family needs its own frame, checked with `research/thinking_ahead.py` (answers that can be
  checked, with and without thinking ahead).
- The reasoning comes after the message in the prompt (`Session._prompt`), so on Enter the engine rereads it:
  ~0.2 s at 2,000 tokens/s, 3-4 s on a card that reads at 93 (RX 5700 XT, README "On an AMD Radeon").
- How edits, cancels and reasoning kept across edits are handled: `Session.draft`, `_start_chunk`,
  `_absorb` and `_answer` in `mafdet/watch.py`. The llama.cpp side is `LlamaServer` and `LlamaPrefill` in
  `mafdet/llama_backend.py`.
- A system prompt and tool definitions that are the same every turn are already reused by prefix caching
  after the first request. The gain is on the user's message and the reasoning about it.

## Contributing

Fork the repository and open pull requests against `main`. Keep the proxy's code in `mafdet/proxy/`; a
change to the app's engine (`mafdet/watch.py`, `mafdet/llama_backend.py`) to share code with it goes in a
pull request of its own. Measured results go in `research/LOG.md` along with the rest of the research.
