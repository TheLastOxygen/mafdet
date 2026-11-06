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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import Engine
from .profile import EngineProfile
from . import think
from .session import DraftSession, SharedCache, _text_of, same_messages


class ProxyState:
    def __init__(self, engine: Engine, profile: EngineProfile, model: str, *,
                 debounce_ms: int = 300, min_new: int = 8, max_sessions: int = 32,
                 exact: bool = False, cors: list[str] | None = None, think_ahead: bool = False):
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
        self.tools: list | None = None             # and its tool definitions, for the same reason
        # For drafts that carry only the text: the system prompt of the last request (a
        # new conversation starts with it), and the conversation a "new" chat became, until
        # the front end's next draft names the chat's real id.
        self.system: dict | None = None
        self.unclaimed: tuple[list[dict], float] | None = None
        # Slots: the conversation lives in the adapter's slot (0); with two or more, a
        # request that matches no conversation (a front end's title, tags, follow-ups...)
        # goes to the next one, so it does not evict the conversation from the cache.
        try:
            self.slots = engine.slots() or 1
        except Exception:
            self.slots = 1
        self.side_slot = 1 if self.slots >= 2 and getattr(engine, "dialect", "") == "llama" else None
        self.counts = {"matched": 0, "unmatched": 0, "side": 0, "thought_ahead": 0}
        self.started = 0
        self._load_last_request()
        # Thinking ahead (T2, --think-ahead): only for a model whose template has a reasoning block.
        self.frame = None
        if think_ahead and not exact and hasattr(engine, "generate"):
            try:
                if engine.thinks():
                    frame = think.Frame(engine)
                    self.frame = frame if frame.usable else None
            except Exception:
                self.frame = None

    def thinking(self) -> bool:
        """Think ahead now?  Not when the front end turned the model's thinking off."""
        return self.frame is not None and think.thinking_on(self.template_kwargs)

    # ---- what the last request looked like, kept across restarts --------------------------
    def _last_request_path(self):
        try:
            return self.profile.path().with_suffix(".last.json")
        except Exception:
            return None

    def _load_last_request(self) -> None:
        """The tools, template options and system prompt of the last request, saved by an
        earlier run: without them the first drafts after a restart miss Open WebUI's ~5.8k
        tokens of tools, and the first message is read on Enter (4.4 s on an RTX 2060)."""
        path = self._last_request_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path else {}
        except (OSError, ValueError):
            data = {}
        self.tools = data.get("tools")
        self.template_kwargs = data.get("chat_template_kwargs")
        self.system = data.get("system")

    def _save_last_request(self) -> None:
        path = self._last_request_path()
        if path is None:
            return
        data = {"tools": self.tools, "chat_template_kwargs": self.template_kwargs, "system": self.system}
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass

    def note_request(self, body: dict, messages: list[dict], matched: bool) -> None:
        """A conversation turn went through: remember what the next drafts must carry."""
        kwargs, tools = body.get("chat_template_kwargs"), body.get("tools")
        system = next((m for m in messages if m.get("role") == "system"), None)
        if system is None and not matched:
            system = self.system                  # an unknown request without one: keep ours
        changed = (kwargs, tools, system) != (self.template_kwargs, self.tools, self.system)
        self.template_kwargs, self.tools, self.system = kwargs, tools, system
        if changed:
            self._save_last_request()

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
                elif self.frame is not None:
                    sess.thinker = think.Thinker(sess, self.frame, enabled=self.thinking)
                if conversation_id != "new" and self.unclaimed is not None:
                    # The first draft from a chat the proxy has not seen, right after a
                    # new chat was answered: that chat's id, now that the front end has one.
                    history, at = self.unclaimed
                    self.unclaimed = None
                    if time.time() - at < 600:
                        sess.history = history
                self.sessions[conversation_id] = sess
            return sess

    # ---- which conversation a request on Enter belongs to ------------------------------
    def match(self, messages: list[dict]) -> DraftSession | None:
        with self.lock:
            sessions = sorted(self.sessions.values(), key=lambda s: s.drafted_at, reverse=True)
        for sess in sessions:
            if sess.matches(messages):
                return sess
        return None

    def continues_unclaimed(self, messages: list[dict]) -> bool:
        """Does this request continue the conversation no draft has named yet (a new chat
        answered, then a follow-up suggestion clicked instead of a message typed)?"""
        with self.lock:
            unclaimed = self.unclaimed
        return unclaimed is not None and same_messages(messages[:-1], unclaimed[0])

    def remember_unclaimed(self, messages: list[dict], answer: str) -> None:
        """A conversation turn no draft was for: keep it for the first id that drafts next."""
        with self.lock:
            self.unclaimed = (list(messages) + [{"role": "assistant", "content": answer}], time.time())

    def drafting(self, within: float = 600.0) -> bool:
        """Has a front end been sending drafts lately?  Only then can a request that matches
        no conversation be told apart as a side request."""
        now = time.time()
        with self.lock:
            return any(now - s.drafted_at < within for s in self.sessions.values())

    def remember(self, sess: DraftSession, messages: list[dict], answer: str) -> None:
        """The conversation after this answer, for the next turn's drafts."""
        history = list(messages) + [{"role": "assistant", "content": answer}]
        if sess.thinker is not None:
            sess.thinker.reset()                   # a new turn: new notes
        if sess.conversation_id == "new":
            self.unclaimed = (history, time.time())
            sess.history = None
        else:
            sess.history = history

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
                ids = self.engine.render(body.get("messages") or [], body.get("chat_template_kwargs"),
                                         body.get("tools"))
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
                      "reuse_channel": self.profile.channel, "exact": self.exact,
                      "think_ahead": self.frame is not None},
            "slots": {"total": self.slots, "conversation": 0, "side": self.side_slot},
            "requests": dict(self.counts),
            "profile": prof,
            "sessions": sessions,
        }


SSE_END = bytes([10, 10])                     # the blank line after a server-sent event
CRLF = bytes([13, 10])


def _chunked(data: bytes) -> bytes:
    """One HTTP/1.1 chunk (an empty one ends the body)."""
    return b"%x" % len(data) + CRLF + data + CRLF


def _answer_of(chunks_seen: list[bytes], stream: bool) -> str:
    """The assistant's visible answer, from the bytes relayed to the client."""
    raw = b"".join(chunks_seen)
    if not stream:
        try:
            msg = (json.loads(raw.decode()).get("choices") or [{}])[0].get("message") or {}
        except (ValueError, UnicodeDecodeError, AttributeError):
            return ""
        return msg.get("content") or ""
    parts = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            delta = ((json.loads(data.decode()).get("choices") or [{}])[0].get("delta") or {})
        except (ValueError, UnicodeDecodeError, AttributeError):
            continue
        parts.append(delta.get("content") or "")
    return "".join(parts)


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
        messages = body.get("messages")
        text = body.get("text")
        if messages is None and not isinstance(text, str):
            return self._bad(400, "a draft needs messages (the whole request) or text (the message being typed)")
        if messages is not None and not isinstance(messages, list):
            return self._bad(400, "messages must be a list")
        sess = self.state.session(conversation_id)
        if self.state.exact or (messages is None and not text.strip()):
            return self._json(200, {"conversation_id": conversation_id, "tier": sess.tier,
                                    "draft_tokens": 0, "cached_tokens": 0})
        if messages is None:
            # Only the text: the proxy puts it where the front end will, after the
            # conversation it saw pass through (or the last system prompt, for a new chat).
            messages = sess.messages_for(text, self.state.system)
        kwargs = body.get("chat_template_kwargs")
        if kwargs is None:
            kwargs = self.state.template_kwargs
        tools = body["tools"] if "tools" in body else self.state.tools
        tokens = sess.submit(messages, kwargs, text=text, tools=tools)
        self._json(200, {"conversation_id": conversation_id, "tier": sess.tier,
                         "draft_tokens": tokens, "cached_tokens": len(sess.cached)})

    def _chat(self) -> None:
        try:
            body = self._body()
        except ValueError as exc:
            return self._bad(400, str(exc))
        stream = bool(body.get("stream"))
        state = self.state
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        sess = state.match(messages)
        # A side request (a title, tags, follow-ups) matches no conversation and is not
        # streamed: Open WebUI asks for those whole, and streams every turn of the chat,
        # including one sent by clicking a suggestion, which comes with no drafts.  With a
        # second slot it goes there and leaves the conversation's cache alone.
        side = (sess is None and not stream and state.side_slot is not None and state.drafting()
                and not state.continues_unclaimed(messages))
        forward = dict(body)
        if state.exact:
            forward["cache_prompt"] = False
        if state.side_slot is not None:
            forward["id_slot"] = state.side_slot if side else 0
        thinker = None
        if side:
            state.counts["side"] += 1
        else:
            state.counts["matched" if sess is not None else "unmatched"] += 1
            state.note_request(body, messages, sess is not None)
            thinker = sess.thinker if sess is not None and not state.exact else None
            if thinker is not None:
                with sess.lock:                       # the draft's prompt, before Enter drops it
                    draft_ids, draft_text = list(sess.latest), sess.latest_text
                    draft_with = sess.rendered_with
            state.begin_generation()
        # Thinking ahead: the notes written while the user typed, if the request leaves thinking on.
        notes = prompt = None
        if not side and thinker is not None and think.thinking_on(body.get("chat_template_kwargs")):
            final_text = _text_of(messages)
            notes = thinker.take(final_text)
            if notes:
                # The draft's prompt only if it was rendered as this request is: same text, same template
                # options, same tools (the drafts of a first request after a restart may have none).
                same = (draft_ids and draft_text.strip() == final_text.strip()
                        and draft_with == (body.get("chat_template_kwargs"), body.get("tools")))
                prompt = draft_ids if same else state.engine.render(messages, body.get("chat_template_kwargs"),
                                                                     body.get("tools"))
                if not prompt:
                    notes = None
        ok = False
        seen: list[bytes] = []
        try:
            if notes:
                state.counts["thought_ahead"] += 1
                ok = True
                events = think.answer(state.engine, prompt, notes, state.frame, body)
                if not stream:
                    raw = json.dumps(think.collect(events)).encode()
                    seen.append(raw)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for event in events:
                    data = b"data: " + json.dumps(event).encode() + SSE_END
                    seen.append(data)
                    self.wfile.write(_chunked(data))
                done = b"data: [DONE]" + SSE_END
                self.wfile.write(_chunked(done) + _chunked(b""))
                return
            status, headers, chunks = state.engine.chat(forward, stream=stream)
            ok = status < 400
            if status >= 400 or not stream:
                raw = b"".join(chunks)
                seen.append(raw)
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
                seen.append(chunk)
                # One write per chunk: three small writes can wait on each other in TCP.
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.write(b"0\r\n\r\n")
        except (ConnectionError, OSError) as exc:
            ok = False
            try:
                self._bad(502, f"engine unreachable: {exc}")
            except Exception:
                pass
        finally:
            if not side:
                if ok and not state.exact and sess is not None:
                    state.remember(sess, messages, _answer_of(seen, stream))
                elif ok and not state.exact and stream and messages:
                    # a turn typed nowhere the proxy saw (a suggestion clicked, another tab)
                    state.remember_unclaimed(messages, _answer_of(seen, stream))
                state.end_generation(body if ok else None)


def make_server(state: ProxyState, host: str = "127.0.0.1", port: int = 8077) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
