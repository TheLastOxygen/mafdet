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

## Read-ahead is exact

A draft is only ever a *prefix* of what the user will send, so prewarming it cannot
change the answer.  `bench.py` checks that the text of every answer with the proxy is
identical to the answer without it, and fails the run if not.
