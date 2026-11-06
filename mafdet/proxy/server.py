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
                 debounce_ms: int = 300, min_new: int = 8, max_sessions: int = 32):
        self.engine = engine
        self.profile = profile
        self.model = model
        self.debounce_ms = debounce_ms
        self.min_new = min_new
        self.max_sessions = max_sessions
        self.sessions: dict[str, DraftSession] = {}
        self.cache = SharedCache()          # one engine slot -> one shared prefix cache
        self.lock = threading.Lock()
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
                self.sessions[conversation_id] = sess
            return sess

    def stats(self) -> dict:
        with self.lock:
            sessions = [s.stats() for s in self.sessions.values()]
        prof = self.profile.to_dict()
        return {
            "proxy": {"engine": self.engine.name, "model": self.model,
                      "tier": self.profile.tier, "spelling": self.profile.spelling,
                      "reuse_channel": self.profile.channel},
            "profile": prof,
            "sessions": sessions,
        }


def _text_of(messages: list[dict]) -> str:
    """The last user message, flattened — kept for callers that only need the text."""
    parts = []
    for msg in messages or []:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        parts.append(content or "")
    return parts[-1] if parts else ""


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
        tokens = sess.submit(messages)
        self._json(200, {"conversation_id": conversation_id, "tier": sess.tier,
                         "draft_tokens": tokens, "cached_tokens": len(sess.cached)})

    def _chat(self) -> None:
        try:
            body = self._body()
        except ValueError as exc:
            return self._bad(400, str(exc))
        conversation_id = str(body.get("conversation_id") or
                              (body.get("messages") or [{}])[0].get("conversation_id") or "default")
        sess = self.state.session(conversation_id)
        stream = bool(body.get("stream"))
        sess.begin_generation()
        try:
            status, headers, chunks = self.state.engine.chat(body, stream=stream)
            if status >= 400:
                raw = b"".join(chunks)
                self.send_response(status)
                self.send_header("Content-Type", headers.get("content-type", "application/json"))
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if not stream:
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
                self.wfile.write(b"%x\r\n" % len(chunk))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        except (ConnectionError, OSError) as exc:
            try:
                self._bad(502, f"engine unreachable: {exc}")
            except Exception:
                pass
        finally:
            sess.end_generation()


def make_server(state: ProxyState, host: str = "127.0.0.1", port: int = 8077) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
