"""The OpenAI-compatible front door: POST /v1/drafts, POST /v1/chat/completions.

The proxy does not interpret the conversation (no templating, no token juggling on the
message): it keeps the draft's token ids to know what it already warmed, and otherwise
passes every real request through unchanged.  That is what keeps read-ahead exact --
the answer to a committed message is computed by the engine exactly as if the proxy
were not there.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import Engine
from .profile import EngineProfile
from .session import DraftSession, SharedCache


class ProxyState:
    def __init__(self, engine: Engine, profile: EngineProfile, model: str, *,
                 debounce_ms: int = 300, min_new: int = 8, max_sessions: int = 32,
                 exact: bool = False, cors: list[str] | None = None):
        self.engine = engine
        self.profile = profile
        self.model = model
        self.debounce_ms = debounce_ms
        self.min_new = min_new
        self.max_sessions = max_sessions
        # exact: no read-ahead, and the engine recomputes every prompt from scratch
        # (cache_prompt: false), so an answer is bit-identical to a cold run.  The price is
        # all of the gain, the engine's own reuse of the previous turn included.
        self.exact = exact
        # Origins allowed to call the proxy from a web page ("*" for any).  Off by default:
        # with it on, any page open in the browser can send requests to the local engine.
        self.cors = [o.rstrip("/") for o in (cors or [])]
        self.sessions: dict[str, DraftSession] = {}
        self.cache = SharedCache()          # one engine slot -> one shared prefix cache
        self.lock = threading.Lock()
        # The template options of the last real request (e.g. Qwen3's enable_thinking):
        # used for a draft that does not send its own, since the draft must be rendered
        # exactly as the request on Enter will be.
        self.template_kwargs: dict | None = None
        self.started = 0

    def session(self, conversation_id: str) -> DraftSession:
        with self.lock:
            sess = self.sessions.get(conversation_id)
            if sess is None:
                if len(self.sessions) >= self.max_sessions:
                    oldest = min(self.sessions.values(), key=lambda s: s.warmed_at or 0)
                    del self.sessions[oldest.conversation_id]
                sess = DraftSession(conversation_id, self.engine, self.profile, self.model,
                                    debounce_ms=self.debounce_ms, min_new=self.min_new,
                                    cache=self.cache)
                if self.exact:
                    sess.tier = "T-1"
                self.sessions[conversation_id] = sess
            return sess

    # ---- an answer holds the slot for every conversation -----------------------------
    def begin_generation(self) -> None:
        """Enter, from any client: the drafts still waiting are the message just sent (or
        older ones), so they are dropped; then the answer takes the slot."""
        with self.lock:
            sessions = list(self.sessions.values())
        for sess in sessions:
            sess.drop_pending()
        self.cache.begin_generation()

    def end_generation(self, body: dict | None = None) -> None:
        if body is not None and not self.exact and self.profile.tier != "T-1":
            # The slot now holds this prompt and its answer; the proxy knows the prompt.
            # Rendering it is off the critical path (the answer has been sent already) and
            # happens before the slot is released, so no warm can land in between.
            try:
                ids = self.engine.render(body.get("messages") or [], body.get("chat_template_kwargs"))
            except Exception:
                ids = None
            if ids:
                self.cache.set(ids, tail_unknown=True)
        self.cache.end_generation()
        with self.lock:
            sessions = list(self.sessions.values())
        for sess in sessions:
            sess.catch_up()

    def stats(self) -> dict:
        with self.lock:
            sessions = [s.stats() for s in self.sessions.values()]
        prof = self.profile.to_dict()
        return {
            "proxy": {"engine": self.engine.name, "model": self.model,
                      "tier": "T-1" if self.exact else self.profile.tier,
                      "spelling": self.profile.spelling,
                      "reuse_channel": self.profile.channel, "exact": self.exact},
            "profile": prof,
            "sessions": sessions,
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: ProxyState

    # ---- plumbing ---------------------------------------------------------------
    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            # A client that closes a keep-alive connection is normal, not an error.
            self.close_connection = True

    def log_message(self, format, *args):          # keep the console for the status line
        pass

    def end_headers(self):
        self._cors_headers()
        super().end_headers()

    def _cors_headers(self) -> None:
        allowed = self.state.cors
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if not allowed or not origin:
            return
        if "*" in allowed:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin in allowed:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode() or "{}")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"bad JSON body: {exc}")
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _bad(self, status: int, message: str) -> None:
        self._json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    # ---- routes -----------------------------------------------------------------
    def do_OPTIONS(self):  # noqa: N802 -- a browser's CORS preflight
        self.send_response(204)
        if self.state.cors:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802 (http.server naming)
        if self.path in ("/healthz", "/health"):
            self._json(200, {"ok": True, "tier": self.state.profile.tier})
        elif self.path == "/proxy/stats":
            self._json(200, self.state.stats())
        elif self.path.startswith("/v1/models"):
            status, data = self.state.engine.models()
            self._json(status if status else 502, data or {})
        else:
            self._bad(404, f"no route {self.path}")

    def do_POST(self):  # noqa: N802
        if self.path == "/v1/drafts":
            self._drafts()
        elif self.path.startswith("/v1/chat/completions"):
            self._chat()
        else:
            self._bad(404, f"no route {self.path}")

    def _drafts(self) -> None:
        try:
            body = self._body()
        except ValueError as exc:
            return self._bad(400, str(exc))
        conversation_id = str(body.get("conversation_id") or "default")
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            return self._bad(400, "messages must be a list")
        sess = self.state.session(conversation_id)
        if self.state.exact:
            return self._json(200, {"conversation_id": conversation_id, "tier": "T-1",
                                    "draft_tokens": 0, "cached_tokens": 0})
        kwargs = body.get("chat_template_kwargs")
        if kwargs is None:
            kwargs = self.state.template_kwargs
        tokens = sess.submit(messages, kwargs)
        self._json(200, {"conversation_id": conversation_id, "tier": sess.tier,
                         "draft_tokens": tokens, "cached_tokens": len(sess.cached)})

    def _chat(self) -> None:
        try:
            body = self._body()
        except ValueError as exc:
            return self._bad(400, str(exc))
        stream = bool(body.get("stream"))
        if "chat_template_kwargs" in body:
            self.state.template_kwargs = body.get("chat_template_kwargs")
        forward = dict(body, cache_prompt=False) if self.state.exact else body
        self.state.begin_generation()
        ok = False
        try:
            status, headers, chunks = self.state.engine.chat(forward, stream=stream)
            ok = status < 400
            if status >= 400 or not stream:
                raw = b"".join(chunks)
                self.send_response(status)
                self.send_header("Content-Type", headers.get("content-type", "application/json"))
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_response(status)
            self.send_header("Content-Type", headers.get("content-type", "text/event-stream"))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in chunks:
                if not chunk:
                    continue
                # One write per chunk: three small writes can wait on each other in TCP.
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.write(b"0\r\n\r\n")
        except (ConnectionError, OSError) as exc:
            try:
                self._bad(502, f"engine unreachable: {exc}")
            except Exception:
                pass
        finally:
            self.state.end_generation(body if ok else None)


def make_server(state: ProxyState, host: str = "127.0.0.1", port: int = 8077) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
