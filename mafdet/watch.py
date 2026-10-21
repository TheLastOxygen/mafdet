"""
mafdet watch: the chat window.

A desktop window (pywebview) shows the conversation; behind it a small local
HTTP server connects the page to llama-server. While the user types, the page
sends the draft on every change and the session keeps the slot's KV cache
aligned with it: common prefix, hold-back, threshold and debounce, the same
rules measured by `mafdet hunt`. On Enter only the tail is left to compute,
then the answer streams into the window.

The whole conversation stays in the cache: after an answer the next user turn
is appended to the tokens already there, so every turn starts where the
previous one ended instead of rereading the history.

    mafdet watch
    mafdet watch --device cpu
    mafdet watch --browser          # in the default browser instead of a window
"""
from __future__ import annotations

import http.client
import http.server
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import webbrowser
from importlib.resources import files

from .identity import INK, SHAPES, WORDMARK, WORDMARK_SIZE

UI = files("mafdet").joinpath("ui")

# Files the page may load: nothing else under ui/ (or outside it) is served.
STATIC = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "fonts/archivo.woff2": "font/woff2",
    "fonts/jetbrains-mono.woff2": "font/woff2",
}

# Qwen3 in non-thinking mode: the sampling settings recommended by its authors.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0}


def turn_separator(tok) -> str:
    """What the chat template puts between an answer and the next user message.

    For Qwen3: '<|im_end|>\\n<|im_start|>user\\n'.
    """
    a, b, c = "<<<U1>>>", "<<<A1>>>", "<<<U2>>>"
    msgs = [{"role": "user", "content": a}, {"role": "assistant", "content": b}, {"role": "user", "content": c}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return s.split(b, 1)[1].split(c, 1)[0]


class Session:
    """One conversation on one llama-server slot, its cache following the draft."""

    def __init__(self, args):
        self.args = args
        self.state, self.error, self.label = "loading", "", ""
        self.holdback, self.min_new, self.debounce = args.holdback, args.min_new, args.debounce / 1000
        self.slot_lock = threading.Lock()          # one request at a time on the slot
        self.tok_lock = threading.Lock()           # the fast tokenizer is not safe across threads
        self.text, self.text_t = "", 0.0
        self.target: list[int] = []                # what the cache should hold now (draft minus hold-back)
        self.final_ids: list[int] = []             # the prompt if the user pressed Enter now
        self.history: list[int] = []               # the conversation so far, up to the start of the next user text
        self.busy = self.generating = False
        self.stop = threading.Event()
        threading.Thread(target=self._load, daemon=True).start()

    # ---- start-up ------------------------------------------------------------
    def _load(self):
        try:
            # Imported here: PyTorch and transformers take seconds, and the window is already open.
            from .engine import Template, common_prefix_len, encode, make_engine, setup, stable_ids
            self._prefix_len, self._encode, self._stable = common_prefix_len, encode, stable_ids
            self.tok, self.server, device, quant = setup(self.args, n_slots=1)
            self.eng = make_engine(self.server, device)
            tpl = Template.from_tokenizer(self.tok, thinking=False)
            self.start_ids = encode(self.tok, tpl.prefix)
            self.suffix = tpl.suffix
            self.between_ids = encode(self.tok, turn_separator(self.tok))
            # How far the cache can run past the stable text after a read at a pause.
            self.tail_slack = len(encode(self.tok, tpl.suffix)) + self.holdback + 2
            self.history = list(self.start_ids)
            name = os.path.basename(self.server.model_path).removesuffix(".gguf").removesuffix(f"-{quant}")
            self.label = f"{name} · {quant} · {'GPU' if device == 'llama-cuda' else 'CPU'}"
            self._retarget()
            self.state = "ready"
        except BaseException as e:                 # SystemExit included: it carries the message for the user
            self.state, self.error = "error", str(e) or type(e).__name__
            return
        self._prefill_loop()

    # ---- draft and prefill -------------------------------------------------------
    def draft(self, text: str):
        self.text, self.text_t = text, time.perf_counter()
        if self.state == "ready":
            self._retarget()

    def _retarget(self):
        with self.tok_lock:
            msg = self._encode(self.tok, self.text)
            final = self.history + self._encode(self.tok, self.text + self.suffix)
        self.target = self._stable(self.history + msg, self.holdback)
        self.final_ids = final

    def _prefill_loop(self):
        """Reads the draft in the background: a burst of new tokens at the end, or a pause.

        At a pause it reads the whole prompt, held-back tokens and end of the
        template included, as if the user had pressed Enter: if they do, a
        single token is left to compute; if they type on, the common-prefix
        check drops those few tokens on the next read.
        """
        while True:
            time.sleep(0.02)
            target, final, done = self.target, self.final_ids, self.eng.ids
            paused = time.perf_counter() - self.text_t >= self.debounce
            goal = final if paused and self.text.strip() else target
            if self.generating or not goal or goal == done:
                continue
            n = self._prefix_len(done, target)
            # Edits in the middle wait for a pause: recomputing the tail on every key would be a waste.
            # The tail read at the last pause (end of the template, held-back tokens) does not count.
            burst = n >= len(done) - self.tail_slack and len(target) - n >= self.min_new
            if burst or paused:
                if burst and not paused:
                    goal = target
                with self.slot_lock:
                    if self.generating or target is not self.target:
                        continue
                    self.busy = True
                    try:
                        self.eng.sync(goal)
                    except Exception as e:     # llama-server gone: say so in the window
                        self.state, self.error = "error", f"llama-server stopped answering ({e})."
                        return
                    finally:
                        self.busy = False

    def status(self) -> dict:
        s = {"state": self.state, "error": self.error, "model": self.label, "ctx": self.args.ctx,
             "generating": self.generating}
        if self.state != "ready":
            return s
        final, target, cache = self.final_ids, self.target, self.eng.cached_tokens
        history = len(self.history)
        reused = min(self._prefix_len(cache, final), len(final) - 1)
        s.update(has_text=bool(self.text.strip()), busy=self.busy, context=len(final),
                 message_tokens=len(final) - history, read=max(0, reused - history), left=len(final) - reused,
                 pending=max(0, len(target) - self._prefix_len(cache, target)))
        return s

    # ---- send ----------------------------------------------------------------------
    def send(self, text: str):
        """Answer `text`, as a stream of events for the page."""
        if self.state != "ready":
            yield {"type": "error", "message": "The model is not ready yet."}
            return
        t_enter = time.perf_counter()
        self.stop.clear()
        self.generating = True
        try:
            with self.slot_lock:                   # waits for a prefill still running
                yield from self._answer(text, t_enter)
        finally:
            self.generating = False
            self._retarget()

    def _answer(self, text: str, t_enter: float):
        history = len(self.history)
        with self.tok_lock:
            final = self.history + self._encode(self.tok, text + self.suffix)
        reused = min(self._prefix_len(self.eng.cached_tokens, final), len(final) - 1)
        room = self.args.ctx - len(final) - 8
        if room < 32:
            yield {"type": "error", "message": f"The conversation no longer fits in the context "
                                               f"({self.args.ctx} tokens). Start a new chat."}
            return
        limit = min(self.args.max_tokens or room, room)
        body = {"prompt": final, "id_slot": self.eng.slot, "cache_prompt": True, "stream": True,
                "n_predict": limit, "return_tokens": True, **SAMPLING}
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=600)
        generated, ttft, info, stopped = [], None, {}, False
        try:
            conn.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            for raw in resp:
                if self.stop.is_set():
                    stopped = True                 # closing the connection stops llama-server too
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                d = json.loads(line[6:])
                generated += d.get("tokens") or []
                if d.get("content"):
                    if ttft is None:
                        ttft = (time.perf_counter() - t_enter) * 1000
                    yield {"type": "token", "text": d["content"]}
                if d.get("stop"):
                    info = d
                    break
        finally:
            conn.close()
        # The slot now holds the prompt and the answer: the next turn continues from there.
        cached = final + generated
        self.eng.assume_cached(cached[:info.get("tokens_cached", len(cached))])
        sep = self.between_ids[1:] if generated and generated[-1] == self.between_ids[0] else self.between_ids
        self.history = cached + sep
        t = info.get("timings") or {}
        yield {"type": "done", "ttft_ms": round(ttft or 0), "tokens": t.get("predicted_n", len(generated)),
               "tps": round(t.get("predicted_per_second", 0), 1), "message_tokens": len(final) - history,
               "read": max(0, reused - history), "left": len(final) - reused,
               "cut": info.get("stop_type") == "limit", "stopped": stopped}

    def new_chat(self):
        with self.slot_lock:
            if self.state == "ready":
                self.history = list(self.start_ids)
                self._retarget()


# --------------------------------------------------------------------------
# Local HTTP server: the page, its files and a small JSON API
# --------------------------------------------------------------------------

def identity_js() -> bytes:
    data = {"shapes": {k: [[d, a, o] for d, a, o in v] for k, v in SHAPES.items()},
            "wordmark": {"size": WORDMARK_SIZE, "letters": WORDMARK}}
    return f"window.MAFDET_SHAPES = {json.dumps(data)};\n".encode("utf-8")


def make_handler(session: Session, token: str, port_box: list):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code: int, body: bytes = b"", ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: dict):
            self._reply(200, json.dumps(data).encode("utf-8"))

        def _allowed(self, api: bool) -> bool:
            # Only this machine, by its own address (no DNS rebinding), and the API only with the page's token.
            if self.headers.get("Host") != f"127.0.0.1:{port_box[0]}":
                return False
            return not api or secrets.compare_digest(self.headers.get("X-Mafdet-Token", ""), token)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if not self._allowed(api=path.startswith("/api/")):
                return self._reply(403)
            if path == "/":
                page = UI.joinpath("index.html").read_text(encoding="utf-8").replace("{{TOKEN}}", token)
                return self._reply(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/identity.js":
                return self._reply(200, identity_js(), "text/javascript; charset=utf-8")
            if path == "/favicon.ico":
                return self._reply(200, UI.joinpath("mafdet.ico").read_bytes(), "image/x-icon")
            if path.startswith("/static/") and path[8:] in STATIC:
                return self._reply(200, UI.joinpath(*path[8:].split("/")).read_bytes(), STATIC[path[8:]])
            if path == "/api/status":
                return self._json(session.status())
            self._reply(404)

        def do_POST(self):
            if not self._allowed(api=True):
                return self._reply(403)
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/draft":
                session.draft(str(data.get("text", "")))
                return self._json(session.status())
            if self.path == "/api/stop":
                session.stop.set()
                return self._json({})
            if self.path == "/api/new":
                session.new_chat()
                return self._json({})
            if self.path == "/api/open":
                url = str(data.get("url", ""))
                if url.startswith(("http://", "https://")):
                    webbrowser.open(url)           # links open in the browser, not in the chat window
                return self._json({})
            if self.path == "/api/send":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    for event in session.send(str(data.get("text", ""))):
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    session.stop.set()             # the window went away
                return
            self._reply(404)

    return Handler


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------

def open_window(url: str, browser: bool):
    if not browser:
        try:
            import webview
        except ImportError:
            print("pywebview is not installed (pip install pywebview): opening the browser instead.")
            browser = True
    if browser:
        webbrowser.open(url)
        print(f"MAFDET is running at {url}\nPress Ctrl+C to stop.", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return
    webview.create_window("MAFDET", url, width=1180, height=840, min_size=(760, 560),
                          background_color=INK, text_select=True)
    webview.start(icon=str(UI.joinpath("mafdet.ico")))


def run(args) -> int:
    if args.backend != "llama":
        raise SystemExit("mafdet watch runs on llama.cpp: drop --backend hf.")
    if sys.stdout is None or sys.stderr is None:
        # Started without a console (pythonw): keep the output somewhere for when something goes wrong.
        log = open(os.path.join(tempfile.gettempdir(), "mafdet-watch.log"), "w", encoding="utf-8")
        sys.stdout = sys.stderr = log
    session = Session(args)
    token, port_box = secrets.token_urlsafe(24), [0]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session, token, port_box))
    server.daemon_threads = True
    port_box[0] = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    open_window(f"http://127.0.0.1:{port_box[0]}/", args.browser)
    server.shutdown()
    if getattr(session, "server", None):
        session.server.close()
    return 0
