# The proxy

An OpenAI-compatible server that sits in front of an inference engine and warms its
prefix cache while the user types, so that Enter is cheaper.  The idea, the engine
requirements and the contribution rules are in `PROXY.md` at the repository root; this
file is the short version of how the code is laid out.

It is separate from the app on purpose: nothing here imports `mafdet.watch` or
`mafdet.engine`, and the proxy adds no runtime dependency (stdlib only).

## Run it

The engine (`llama-server`) must already be running.  Then:

```bash
python -m mafdet.proxy --url http://127.0.0.1:8081 --model Qwen3-4B-Q4_K_M
# probed in 1.4s -> tier T1, spelling n_predict0, channel timings (saved to ~/.mafdet/proxy/...)
# llama.cpp http://127.0.0.1:8081 -> proxy on http://127.0.0.1:8077 (tier T1)
```

Point the front end at the proxy: `POST /v1/drafts` while the user types, the usual
`POST /v1/chat/completions` on Enter.  `GET /proxy/stats` shows what the proxy believes
(it is the "is it working" line), `GET /healthz` is liveness.

A draft can carry the whole request (`messages`) or only the message being typed (`text`,
with the `conversation_id` the front end uses). With `text`, the proxy puts it after the
conversation it saw pass through for that id (the last request's messages and the answer);
for `"new"` it starts from the last system prompt, and hands the conversation to the first id
that drafts after the answer. On Enter the request is matched to the session by its user message
(equal to the last draft, or extending it) or by its history.

With two or more slots (`/props` total_slots), the conversation stays in slot 0 and a request that
matches no conversation, while a front end is drafting, goes to slot 1 (`id_slot`): Open WebUI's
titles, tags and follow-ups, which would otherwise evict the conversation. `/proxy/stats` counts
them under `requests`. A request that matches no conversation but is streamed (a suggestion
clicked, which comes with no drafts) stays in slot 0. Start llama-server with
`--no-cache-idle-slots`: with `-kvu` it otherwise clears an idle slot whenever another one gets a
task. The tools, template options and system prompt of the last request are saved next to the
profile (`*.last.json`), so the first drafts after a restart carry them.

A draft may carry the request's `chat_template_kwargs` (e.g. `{"enable_thinking": false}` for
Qwen3) and `tools`; without them the proxy uses those of the last chat request it passed
through (Open WebUI sends 30 tools, ~5.8k tokens, that the template puts first). The
draft must render exactly as the request on Enter will, or the engine still has a few
tokens of template to read then.

- `--cors http://localhost:3000` (comma-separated, or `*`): lets a web page in those origins
  call the proxy. Off by default: with it on, any page open in the browser can reach the
  local engine through the proxy.
- `--exact`: no read-ahead, and every prompt recomputed (`cache_prompt: false`): answers
  bit-identical to a cold run, none of the gain.

The profile is cached per (url, model) in `~/.mafdet/proxy/`; `--reprobe` forces a fresh
probe, `--no-probe` starts with read-ahead off (tier T-1).

## What the probe decides (PR #1: only what llama.cpp needs)

| Question | Answer stored in `EngineProfile` |
|---|---|
| which request prefills without generating | `spelling`: `n_predict0`, else `max_tokens1` |
| is prefix reuse real, and which channel reports it | `reuse_channel`: `timings` (exact), `usage` (exact), `metrics` (estimate), `none` |
| does an edit in the middle still hit the prefix | `edit_branches` |
| tier | `T1` exact read-ahead, `T0` whole-prompt only, `T-1` passthrough |

The remaining probes (slot affinity, cancelling mid-generation, raw completions,
tokenizer agreement) are deliberately left for the second engine: a probe checked on
one engine cannot claim anything about the others.

## Files

| file | what it holds |
|---|---|
| `engine.py` | the adapter contract (`Engine` Protocol), `WarmResult`, `EngineCaps` |
| `profile.py` | `EngineProfile` -- the bit of data the rest of the code reads instead of an engine name |
| `probe.py` | the questions, and the tier they imply |
| `session.py` | one conversation: debounce, latest draft, generation lock, "generation goes first" |
| `audit.py` | predicted prefix reuse vs reported, mismatch counter, estimate handling |
| `server.py` | the HTTP front door and the routes |
| `adapters/llama.py` | llama-server: `/tokenize`, `/completion`, `/v1/chat/completions` pass-through |
| `bench.py` | Enter -> first token, drafts on vs off, with an exactness check |

## Read-ahead does not change the prompt

A draft is only ever a *prefix* of what the user will send, so the prompt the engine
answers is the same.  The answer can still differ in wording, rarely in substance:
llama.cpp computes a prefix read ahead in other batches than a cold prompt, and greedy
decoding is not bit-reproducible across them (Qwen3-4B, on two cards: 25 of 28 answers
byte-identical, one final number different, research/LOG.md 2026-11-06).  `bench.py`
compares the texts and the final numbers and exits with 1 when a number changes; `--exact`
gives bit-identical answers at the price of the gain.
