# Front-end clients

Small pieces that make a chat front end send drafts to the proxy (`POST /v1/drafts`) while
the user types. The front end itself stays untouched.

## Open WebUI: `open-webui.user.js`

A browser user script. While you type in Open WebUI's message box it posts
`{"conversation_id", "text"}` to the proxy: 250 ms after the last change, at least every
600 ms while you keep typing, at once on a paste. It never sends an empty or unchanged text.
`conversation_id` is the chat id from the URL (`/c/<id>`), or `"new"` on a new chat.
If the proxy is down it prints one warning in the console and retries quietly.

### Install

1. Install Tampermonkey (or Violentmonkey) in the browser.
2. Its dashboard: "Create a new script", paste the whole content of `open-webui.user.js`,
   save. Or drag the file onto the dashboard.
3. Start llama-server with two slots, `-np 2 -kvu`. Open WebUI asks the model for a title, tags
   and follow-up suggestions around every answer; on one slot they evict the conversation, and
   the next turn rereads it all (with Open WebUI's 30 built-in tools, ~7k tokens: 4.7 s on an RTX
   2060). With two, the proxy keeps the conversation in slot 0 and sends those to slot 1.
   Then start the proxy with CORS for the page's origin, and point Open WebUI at it:

   ```bash
   python -m mafdet.proxy --url http://127.0.0.1:8081 --model Qwen3-4B-Q4_K_M \
       --cors http://localhost:8080,http://127.0.0.1:8080
   ```

   In Open WebUI, Admin Settings > Connections: an OpenAI connection with the base URL
   `http://127.0.0.1:8077/v1` (from Open WebUI in Docker: `http://host.docker.internal:8077/v1`).
   The chat requests go from Open WebUI's server to the proxy; the drafts go from the browser,
   which is why only they need CORS.
4. Reload Open WebUI.

### Settings (top of the script)

| setting | default | what |
|---|---|---|
| `@match` lines | `http://localhost:8080/*`, `http://127.0.0.1:8080/*` | where Open WebUI runs; add yours |
| `PROXY` | `http://127.0.0.1:8077` | the proxy, as the browser reaches it |
| `USE_GM` | `false` | `true` sends with `GM_xmlhttpRequest`, no CORS needed: also replace `@grant none` with `@grant GM_xmlhttpRequest` and add `@connect 127.0.0.1` (the comment under the header says how) |
| `DEBOUNCE_MS`, `MAX_WAIT_MS` | 250, 600 | the debounce |
| `INPUT` | `#chat-input` | the message box |

If Open WebUI is started with a `CONTENT_SECURITY_POLICY` environment variable, its
`connect-src` must allow the proxy, or use `USE_GM`.

### Check that it works

Type something in a chat (without Enter), then open `http://127.0.0.1:8077/proxy/stats`.
`sessions` should have one entry with your chat id (or `"new"`), `draft_tokens` above zero and
`warms` going up. After Enter, `requests` counts the chat request. Nothing there: the browser
console (F12) shows `[mafdet] proxy unreachable` when the proxy is down or CORS refuses
the origin (the address bar's origin must be in `--cors` exactly: `localhost` and `127.0.0.1`
are two origins).

### What it gains (RTX 2060, Qwen3-4B, Open WebUI 0.13.0, 2026-11-06)

Second turn of a chat, tokens llama-server still had to read on Enter: **7158 (4.7 s) without
the proxy, 1 with it**. Tried in Chrome with the script's code run in the page, not yet installed
through Tampermonkey. Details in `research/LOG.md`.

### When the read-ahead does not count

The proxy ties the request on Enter to the drafts by the user message: it must equal the last
draft (or extend it). The script rebuilds the text the way Open WebUI does (its editor's HTML
goes through turndown: one newline between paragraphs, `"  \n"` for a Shift+Enter line break,
outer spaces trimmed), but:

- **Paste Large Text as File** (Settings > Interface, off by default): a paste over 1000
  characters becomes a file attachment instead of text in the box. That text is not read
  ahead. Hold Shift while pasting, or turn the setting off.
- Attached files, knowledge, web search, and system prompt variables: Open WebUI changes what
  it sends around the message, so the prompt differs from the warmed one.
- Rich text formatting (lists, tables, code blocks, typed with "Rich Text Input for Chat" on):
  the script approximates the markdown Open WebUI makes of them; where it differs, only the
  part before the difference is reused.
