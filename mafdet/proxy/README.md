# The proxy

An OpenAI-compatible server that sits in front of an inference engine and warms its
prefix cache while the user types, so that Enter is cheaper.  The idea, the engine
requirements and the contribution rules are in `PROXY.md` at the repository root; this
file is the short version of how the code is laid out.

It is separate from the app on purpose: nothing here imports `mafdet.watch` or
`mafdet.engine`, and the proxy adds no runtime dependency (stdlib only).

## Run it

The engine must already be running: `llama-server`, or Ollama, vLLM, SGLang, LM Studio, any server
with `/v1/chat/completions` (`--engine auto`, the default, asks which).  Then:

```bash
python -m mafdet.proxy --url http://127.0.0.1:8081 --model Qwen3-4B-Q4_K_M
# probed in 1.4s -> tier T1, spelling n_predict0, channel timings (saved to ~/.mafdet/proxy/...)
# MAFDET proxy on http://127.0.0.1:8077 (tier T1)
#   engine       llama.cpp at http://127.0.0.1:8081: drafts as token ids, reuse seen in timings, cache one per slot
python -m mafdet.proxy --url http://127.0.0.1:11434 --model qwen3:4b        # Ollama
python -m mafdet.proxy --url http://127.0.0.1:8000 --model Qwen/Qwen3-4B     # vLLM (--api-key if it wants one)
```

Without `--model` the first model the engine lists is used.  `mafdet proxy --url ...` is the same
through the app's command.

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

## Other engines (vLLM, SGLang, Ollama, LM Studio)

llama.cpp has its own adapter (`adapters/llama.py`, the one measured); every other engine goes through
the OpenAI-compatible one (`adapters/openai.py`), and the probe picks one of two ways to read ahead:

- **by token ids**, when the engine renders the chat prompt for the proxy (vLLM's `/tokenize` takes the
  messages, with the template options and the tools): the draft goes through `/v1/completions` with the
  ids and one token of answer (vLLM refuses `max_tokens: 0`; SGLang takes it), and the proxy predicts and
  audits the reuse exactly, as on llama.cpp. The exact count comes from
  `usage.prompt_tokens_details.cached_tokens`: vLLM with `--enable-prompt-tokens-details`, SGLang with
  `--enable-cache-report`; without, the `/metrics` prefix-cache counters are an estimate.
- **through the chat endpoint**, when it does not (Ollama and LM Studio have no tokenizer over HTTP,
  SGLang's `/tokenize` takes no messages): the draft is sent as a chat request with `max_tokens: 1`, the
  engine applies its own template, and the proxy counts nothing. Reuse is then seen in the prefill time
  the engine reports (Ollama's `prompt_eval_duration` on `/api/chat`, LM Studio's `time_to_first_token`
  on `/api/v0/chat/completions`), or in `prompt_tokens` when the engine counts only what it computed.
  The audit records these warms as unknown: they never demote the session.

A cache shared by every request (vLLM, SGLang, Ollama with `OLLAMA_NUM_PARALLEL=2`) needs no second slot:
a front end's titles and follow-ups do not evict the conversation, and the proxy says so
(`cache shared` in the `engine` line, `/proxy/stats` under `cache`). On one cache (Ollama's default) it
warns, as for a one-slot llama-server. `--exact` on these engines only stops the drafts (they have no
per-request switch for their cache), which is already the answer the engine gives without the proxy.

On the status page, "read on Enter" and "already read" are filled from llama-server's `timings`, or
from an OpenAI `usage` block that carries `cached_tokens` (in a stream only when the front end asks for
usage); otherwise those columns stay empty and the time to the first token is what there is.

All of this is written against the engines' documented APIs and checked on fakes of vLLM and Ollama
(`tests/proxy_fake_servers.py`); which engine has really been run through it is the table in `PROXY.md`.

## Your own front end

Point its OpenAI connection at the proxy, and post the message being typed to `POST /v1/drafts`:
`{"conversation_id": "<your chat id, or \"new\">", "text": "<the whole message box>"}`. The text must be
the user message exactly as the request on Enter will carry it. In a web page,
`clients/mafdet-drafts.js` does the debouncing (a pause of 250 ms, at least every 600 ms while typing, a
paste at once) and stays quiet when the proxy is down:

```js
const drafts = mafdetDrafts({ proxy: 'http://127.0.0.1:8077' });
box.addEventListener('input', () => drafts.typed(chatId, box.value));
box.addEventListener('paste', () => setTimeout(() => drafts.pasted(chatId, box.value)));
// on Enter, after sending the chat request through the proxy:
drafts.sent();
```

The page's origin must be allowed with `--cors`. Open WebUI's user script (`clients/open-webui.user.js`)
is the same thing, reading Open WebUI's editor.

## What the probe decides

| Question | Answer stored in `EngineProfile` |
|---|---|
| does the engine give token ids for a text, and for a chat request | `tokenize`, `render` (without `render`, the drafts go through the chat endpoint) |
| which request prefills without generating | `spelling`: `n_predict0` (llama.cpp), `max_tokens0`, `echo0`, `max_tokens1` (one token thrown away), `chat1` (the chat endpoint, one token) |
| is prefix reuse real, and which channel reports it | `reuse_channel`: `timings` (exact), `usage` (exact), `computed` (the engine counts what it computed: exact), `metrics` (estimate), `time` (a faster prefill), `none` |
| does an edit in the middle still hit the prefix | `edit_branches` |
| by how much may a report fall short of the prediction (a block cache) | `reuse_slack`, allowed by the audit |
| does an unrelated request in between evict the prefix | `shared_cache` (no second slot needed when true) |
| does the rendered prompt have the count the chat endpoint computes | `tokenizer_agrees` (reading ahead is off when not) |
| does the completion route take token ids (thinking ahead needs it) | `raw_ids` |
| tier | `T1` exact read-ahead, `T0` whole-prompt only, `T-1` passthrough |

Cancelling mid-generation is not probed: nothing needs it until thinking ahead reaches the other
engines (it is llama.cpp only, `think.py`).

## Files

| file | what it holds |
|---|---|
| `engine.py` | the adapter contract (`Engine` Protocol), `WarmResult`, `EngineCaps` |
| `profile.py` | `EngineProfile` -- the bit of data the rest of the code reads instead of an engine name |
| `probe.py` | the questions, and the tier they imply |
| `session.py` | one conversation: debounce, latest draft, generation lock, "generation goes first" |
| `audit.py` | predicted prefix reuse vs reported, mismatch counter, estimate handling |
| `server.py` | the HTTP front door and the routes |
| `adapters/llama.py` | llama-server: `/tokenize`, `/apply-template`, `/completion`, `/v1/chat/completions` pass-through, raw generation for `think.py` |
| `adapters/openai.py` | vLLM, SGLang, Ollama, LM Studio and any OpenAI-compatible server: `/tokenize`, `/v1/completions`, the chat endpoint as a warm, `/metrics`; `detect()` asks a server which it is |
| `think.py` | thinking ahead (T2): the Qwen3 frame on the draft, the answer after the notes |
| `launch.py` | `mafdet proxy`: llama-server started the right way, or an engine at `--url` detected, probed and served |
| `bench.py` | Enter -> first token, drafts on vs off, with an exactness check |

## Read-ahead does not change the prompt

A draft is only ever a *prefix* of what the user will send, so the prompt the engine
answers is the same.  The answer can still differ in wording, rarely in substance:
llama.cpp computes a prefix read ahead in other batches than a cold prompt, and greedy
decoding is not bit-reproducible across them (Qwen3-4B, on two cards: 25 of 28 answers
byte-identical, one final number different, research/LOG.md 2026-11-06).  `bench.py`
compares the texts and the final numbers and exits with 1 when a number changes; `--exact`
gives bit-identical answers at the price of the gain.
