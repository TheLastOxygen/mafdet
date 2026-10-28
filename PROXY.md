# MAFDET proxy (work in progress)

An OpenAI-compatible server that sits in front of an inference engine and brings MAFDET's reading and
thinking ahead to any front end (Open WebUI, Hermes, agents) and any engine (SGLang, vLLM, llama.cpp,
others). It lives in its own module (`mafdet/proxy/`), apart from the app, and stays experimental until it
works and is measured.

Nothing is built yet. This page is the starting point: what the proxy has to do, what we already know from
the app, and how to contribute.

## What it does

1. **The front end sends drafts.** While the user types, it posts the conversation with the message as it
   stands, debounced (a few hundred ms), to an endpoint of the proxy, e.g. `POST /v1/drafts` with
   `{conversation_id, model, messages}`. On Enter it sends the usual `POST /v1/chat/completions`.
2. **Reading ahead (exact).** For each draft the proxy has the engine read the prompt so far, so the prefix
   cache is warm on Enter. Any engine with prefix caching does this (a request with `max_tokens: 1`; SGLang's
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
