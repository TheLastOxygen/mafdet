# Open WebUI + MAFDET in 5 minutes

Open WebUI with a local model, where the answer starts as soon as you press Enter. Measured on an
RTX 2060 with Qwen3-4B, 8 conversations of 2-3 messages each way, typed at 12 characters a second
(`research/proxy_openwebui.py`, 2026-11-06), time from Enter to the first word:

| | mean | median | slowest |
|---|---|---|---|
| Open WebUI on llama-server (its default options) | 5.4 s | 3.3 s | 20.7 s |
| Open WebUI through MAFDET | **0.16 s** | **0.14 s** | 0.4 s |

Two things make you wait without it. Open WebUI asks the model for a title, tags and follow-up
suggestions around every answer, and your message queues behind them (4.6 s on average here); and
it sends its 30 built-in tools (~5.8k tokens) with every request, which llama-server sometimes has
to read again. MAFDET sends those requests to a slot of their own, keeps the conversation in the
other, and reads your message while you type: on Enter, 1 token is left to read. It is reading
ahead only: the prompt the model answers is the same.

## 1. MAFDET and a model

`mafdet proxy` comes with MAFDET 0.5.0 and later (`mafdet --version` says which you have):

```bash
pipx install mafdet           # or pipx upgrade mafdet; Windows with python.org's Python: pip install -U mafdet
mafdet doctor --install       # llama.cpp for this machine (CUDA, Vulkan, Metal or CPU)
mafdet models                 # which models run well here
mafdet models --download qwen3-4b
```

From the source instead: `git clone https://github.com/Mafdet-AI/mafdet.git`, then `pip install -e .` in a
virtual environment (the README's "From the source").

## 2. Start it

```bash
mafdet proxy --model qwen3-4b
```

It starts llama-server with the options reading ahead needs (two slots sharing one 16k-token cache,
nothing cleared behind its back) and the proxy at `http://127.0.0.1:8077/v1`. It prints the address,
where the user script is and where to see that it works. Without `--model` it uses the model chosen
in the MAFDET app. Ctrl+C stops both.

Open WebUI in Docker: `mafdet proxy --host 0.0.0.0`, and use `http://host.docker.internal:8077/v1`
below. Or all of it in Docker (llama.cpp, the proxy and Open WebUI): [`docker/`](../../../docker/README.md).

With a Qwen3 and thinking on (Open WebUI's default), `--think-ahead` also lets the model reason about
your message while you type: the first word of the answer came 7.6 → 4.3 s after Enter here. It is not
exact (the reasoning starts on an unfinished message), so it is off unless you ask for it.

## 3. Point Open WebUI at it

Admin Settings > Connections > OpenAI API: add a connection with the URL `http://127.0.0.1:8077/v1`
and any key (`none`). The model appears as `qwen3-4b:Q4_K_M`; choose it.

## 4. The user script

It sends what you type to the proxy (Open WebUI itself stays untouched).

1. Install [Tampermonkey](https://www.tampermonkey.net/) (or Violentmonkey) in the browser. In Chrome,
   open `chrome://extensions`, Tampermonkey's Details, and turn on "Allow user scripts" (or the
   Developer mode switch, top right, on older versions).
2. Tampermonkey's icon > "Create a new script": replace everything with the content of
   `open-webui.user.js` (the path `mafdet proxy` prints), and save (Ctrl+S).
3. Open Open WebUI at `http://localhost:8080`.

## 5. Check that it works

Open `http://127.0.0.1:8077/proxy` and type a message without sending it: under *Conversations*, a
`new` row with drafts read. After a few messages, *Last messages* lists them with the time to their
first token and how much of the prompt was already read on Enter, and the titles, tags and follow-ups
sent to the other slot are counted. (`/proxy/stats` has the same as JSON.)

Nothing there: the browser console (F12) shows `[mafdet] proxy unreachable` when the proxy is down
or refuses the page. The address in the bar must be one `mafdet proxy` accepts (`--cors`, by default
`http://localhost:8080` and `http://127.0.0.1:8080`: two different origins for the browser).

## The script's settings

| setting | default | what |
|---|---|---|
| `@match` lines | `localhost` and `127.0.0.1`, ports 8080 (`open-webui serve`) and 3000 (Docker) | where Open WebUI runs; add yours |
| `PROXY` | `http://127.0.0.1:8077` | the proxy, as the browser reaches it |
| `USE_GM` | `false` | `true` sends with `GM_xmlhttpRequest`, no CORS needed: also replace `@grant none` with `@grant GM_xmlhttpRequest` and add `@connect 127.0.0.1` (the comment under the header says how) |
| `DEBOUNCE_MS`, `MAX_WAIT_MS` | 250, 600 | after a pause of 250 ms, and at least every 600 ms while typing |
| `INPUT` | `#chat-input` | the message box |

If Open WebUI is started with a `CONTENT_SECURITY_POLICY` environment variable, its `connect-src`
must allow the proxy, or use `USE_GM`.

## Your own llama-server

`mafdet proxy --url http://127.0.0.1:8081` uses a llama-server already running. Give it two slots and
a shared cache, and keep it from clearing an idle slot: `-np 2 -kvu --no-cache-idle-slots` (or
`--cache-ram 0`), with a context of at least 16k. On one slot, or with idle slots cleared (llama-server's
default with `-kvu`), the title on the other slot empties the conversation.

## Ollama, vLLM, SGLang, LM Studio

`mafdet proxy --url` in front of an engine you already run, instead of llama-server:

```bash
mafdet proxy --url http://127.0.0.1:11434 --model qwen3:4b          # Ollama
mafdet proxy --url http://127.0.0.1:8000 --model Qwen/Qwen3-4B       # vLLM, SGLang (--api-key if it wants one)
mafdet proxy --url http://127.0.0.1:1234                             # LM Studio (the first model it lists)
```

The rest is the same: Open WebUI's connection at `http://127.0.0.1:8077/v1`, the user script, the status
page. The `engine` line the proxy prints says what it found: how the drafts go (as token ids, or through the
engine's chat endpoint), where it sees the reuse, and whether the engine's cache is shared (then Open WebUI's
titles and follow-ups cannot evict the conversation; on Ollama's one cache, start it with
`OLLAMA_NUM_PARALLEL=2`). vLLM reports the exact count with `--enable-prompt-tokens-details`, SGLang with
`--enable-cache-report`; without, the proxy goes by its metrics or by the prefill time. These engines are
supported on their documented APIs and have not been tried for real yet ([PROXY.md](../../../PROXY.md),
"More engines"): if you run one, the `engine` line and `/proxy/stats` in an issue are what fills its row.

## When the read-ahead does not count

The proxy ties the request on Enter to your drafts by the user message (equal to the last draft, or
extending it), or to the conversation by its history. The script rebuilds the text the way Open WebUI
does (its editor's HTML through turndown: one newline between paragraphs, `"  \n"` for a Shift+Enter
line break, outer spaces trimmed), but:

- **Paste Large Text as File** (Settings > Interface, off by default): a paste over 1000 characters
  becomes a file attachment instead of text in the box, and is not read ahead. Hold Shift while
  pasting, or leave the setting off.
- Attached files, knowledge, web search: Open WebUI wraps the message in its RAG template and changes
  the earlier messages too, so the prompt differs from the warmed one.
- Rich text (lists, tables, code blocks): the script approximates the markdown Open WebUI makes of
  them; where it differs, only the part before the difference is reused.
- The first message after `mafdet proxy` is started for the first time: the proxy learns Open WebUI's
  tools from the first request (then keeps them on disk).
