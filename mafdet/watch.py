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
import socket
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

# Qwen3 in non-thinking and thinking mode: the sampling settings recommended by its authors.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0}
THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0}

# Reasoning ahead (--think) is written by the model as its own thoughts, framed by a few
# sentences of ours. Without the frame it spends them remarking that the message is
# unfinished and, after every edit, copies its last paragraph again (measured).
SEED = ("<think>\nThe user is still typing, so the message above may be unfinished. I will not guess the "
        "rest: I will work out what I can from the part already written, so I am ready when it is complete.\n\n")
# When the message changed under the notes: reread it and go on instead of starting over.
BRIDGE = "\n\nThe user has typed more since then. Let me reread the message and continue from what I have worked out.\n\n"
# On Enter: the message is complete now.
SENT = "\n\nThe user has now sent the message, so this is the complete version. Let me reread it and finish.\n\n"
THINK_MIN_WORDS = 3                 # no reasoning ahead on the first couple of words
AHEAD_CHUNK = 128                   # tokens per request written ahead; the next one carries on from there
THINK_CHUNK = 32                    # the same while reasoning: short, so it follows the text as it is typed
BRIDGE_EVERY = 64                   # new reasoning tokens before an edit earns a bridge


def turn_separator(tok) -> str:
    """What the chat template puts between an answer and the next user message.

    For Qwen3: '<|im_end|>\\n<|im_start|>user\\n'.
    """
    a, b, c = "<<<U1>>>", "<<<A1>>>", "<<<U2>>>"
    msgs = [{"role": "user", "content": a}, {"role": "assistant", "content": b}, {"role": "user", "content": c}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return s.split(b, 1)[1].split(c, 1)[0]


class Stream:
    """One streamed generation on the slot, run in its own thread and read by whoever needs it.

    An answer written ahead (mafdet strike) is a Stream started at a pause,
    before Enter: it fills `tokens` while nobody shows them. On Enter, if the
    prompt is still the same, the page gets all of them at once and then
    follows the stream live. Sampling has no penalties, so each token depends
    only on the tokens before it: the answer written ahead is drawn from the
    same distribution as one started after Enter.
    """

    def __init__(self, session: "Session", prompt: list[int], limit: int, text: str = ""):
        self.session, self.prompt, self.limit, self.text = session, prompt, limit, text
        self.tokens: list[int] = []
        self.info: dict = {}                       # the last event: stop_type, timings, tokens_cached
        self.done = self.cancelled = False
        self.cond = threading.Condition()
        self._conn: http.client.HTTPConnection | None = None
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def hit_limit(self) -> bool:
        return self.info.get("stop_type") == "limit"

    def _run(self):
        s = self.session
        body = {"prompt": self.prompt, "id_slot": s.eng.slot, "cache_prompt": True, "stream": True,
                "n_predict": self.limit, "return_tokens": True, **s.sampling}
        with s.slot_lock:
            try:
                if not self.cancelled:
                    self._conn = http.client.HTTPConnection("127.0.0.1", s.server.port, timeout=600)
                    self._conn.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
                    for raw in self._conn.getresponse():
                        if self.cancelled:
                            break                  # closing the connection stops llama-server too
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data: "):
                            continue
                        d = json.loads(line[6:])
                        with self.cond:
                            self.tokens += d.get("tokens") or []
                            if d.get("stop"):
                                self.info = d
                            self.cond.notify_all()
                        if d.get("stop"):
                            break
            except (OSError, http.client.HTTPException, ValueError) as e:
                if not self.cancelled:
                    self.info = {"error": str(e)}
            finally:
                if self._conn is not None:
                    self._conn.close()
                    # The slot now holds the prompt and what was generated: the next request starts from there.
                    cached = self.prompt + self.tokens
                    s.eng.assume_cached(cached[:self.info.get("tokens_cached", len(cached))])
                with self.cond:
                    self.done = True
                    self.cond.notify_all()

    def cancel(self):
        """Stop now: shutting the socket down also wakes a read waiting for the next token."""
        self.cancelled = True
        conn = self._conn
        if conn is not None and conn.sock is not None:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def wait(self):
        with self.cond:
            while not self.done:
                self.cond.wait()

    def follow(self, stop: threading.Event):
        """The tokens as they arrive: what is already there in one go, then as they come."""
        i = 0
        while True:
            with self.cond:
                while i == len(self.tokens) and not self.done and not stop.is_set():
                    self.cond.wait(0.1)
                new, done = self.tokens[i:], self.done
            i += len(new)
            if new:
                yield new
            if done or stop.is_set():
                return


class Detokenizer:
    """Text from a growing list of tokens, never cutting a character in two.

    The text streamed by the server is cut where each request ends: a letter
    whose bytes fall across two chunks (an accent, a typographic quote) comes
    out as two replacement characters. Decoding the tokens here, a short window
    at a time, holds such a character back until its last byte arrives.
    """

    def __init__(self, tok, lock: threading.Lock, skip: int):
        self.tok, self.lock, self.skip = tok, lock, skip
        self.ids: list[int] = []
        self.start = self.read = 0

    def push(self, tokens: list[int]) -> str:
        self.ids += [t for t in tokens if t != self.skip]
        with self.lock:
            head = self.tok.decode(self.ids[self.start:self.read])
            text = self.tok.decode(self.ids[self.start:])
        if text.endswith("�"):
            return ""                              # a character still incomplete: wait for its other bytes
        self.start, self.read = self.read, len(self.ids)
        return text[len(head):]


class Session:
    """One conversation on one llama-server slot, its cache following the draft."""

    def __init__(self, args):
        self.args = args
        self.state, self.error, self.label = "loading", "", ""
        self.holdback, self.min_new, self.debounce = args.holdback, args.min_new, args.debounce / 1000
        self.ahead = getattr(args, "ahead", 0)     # mafdet strike: tokens written at a pause, before Enter
        self.think = getattr(args, "think", False)  # mafdet strike: reason ahead, keeping the notes across edits
        self.think_cap = getattr(args, "think_ahead", 0)
        self.slot_lock = threading.Lock()          # one request at a time on the slot
        self.tok_lock = threading.Lock()           # the fast tokenizer is not safe across threads
        self.spec_lock = threading.Lock()          # starting an answer ahead vs. pressing Enter
        self.text, self.text_t = "", 0.0
        self.target: list[int] = []                # what the cache should hold now (draft minus hold-back)
        self.final_ids: list[int] = []             # the prompt if the user pressed Enter now
        self.history: list[int] = []               # the conversation so far, up to the start of the next user text
        self.spec: Stream | None = None            # the answer being written ahead for final_ids
        self._failed: list[int] = []               # a prompt whose answer ahead failed
        # Written ahead, in the prompt after the message. The reasoning (--think) survives edits:
        # notes. The answer only holds for the exact text it was written for: answer_ahead
        # (with --think it starts with </think>).
        self.notes: list[int] = []
        self.answer_ahead: list[int] = []
        self.notes_text = ""                       # the text notes and answer_ahead were last extended on
        self.notes_fresh = 0                       # notes tokens written since the last bridge
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
            tpl = Template.from_tokenizer(self.tok, thinking=self.think)
            self.sampling = THINK_SAMPLING if self.think else SAMPLING
            self.start_ids = encode(self.tok, tpl.prefix)
            self.suffix = tpl.suffix
            self.between_ids = encode(self.tok, turn_separator(self.tok))
            self.end_id = self.between_ids[0]      # end of a turn: <|im_end|> for Qwen3
            self.think_end = self.tok.convert_tokens_to_ids("</think>")
            self.bridge_ids = encode(self.tok, BRIDGE)
            self.seed_ids = encode(self.tok, SEED)
            self.sent_ids = encode(self.tok, SENT)
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

    def _prompt(self, text: str) -> list[int]:
        """The prompt on Enter: history, message, start of the answer, then what was written ahead."""
        # Trailing spaces and newlines are dropped: one typed after a pause must not throw away the answer written ahead.
        with self.tok_lock:
            ids = self.history + self._encode(self.tok, text.rstrip() + self.suffix) + self.notes
        if self.answer_ahead and text.rstrip() == self.notes_text.rstrip():
            ids += self.answer_ahead
        return ids

    def _retarget(self):
        with self.tok_lock:
            msg = self._encode(self.tok, self.text)
        self.target = self._stable(self.history + msg, self.holdback)
        self.final_ids = self._prompt(self.text)

    def _take_spec(self, spec: Stream) -> bool:
        """Take `spec` away from the session, unless Enter has claimed it first."""
        with self.spec_lock:
            if self.spec is not spec:
                return False
            self.spec = None
        return True

    def _absorb(self, spec: Stream, changed: bool, bridge_after: int = BRIDGE_EVERY):
        """Keep what `spec` wrote: reasoning (before </think>) to the notes, answer to answer_ahead.

        If the message changed meanwhile, the answer is dropped. A bridge then
        follows the notes, telling the model to reread the message, so it goes
        on from its earlier thoughts instead of starting over: once it had
        finished thinking, or after `bridge_after` tokens of new thoughts (not
        after every chunk written while the user types on).
        """
        spec.wait()
        tokens = spec.tokens
        if self.think and not self.answer_ahead:
            cut = tokens.index(self.think_end) if self.think_end in tokens else len(tokens)
            thought, answer = tokens[:cut], tokens[cut:]
        else:
            thought, answer = [], tokens
        if thought:
            self.notes = self.notes + thought
            self.notes_fresh += len(thought)
        if changed:
            had_answer = bool(answer or self.answer_ahead)
            self.answer_ahead = []
            self._bridge(1 if had_answer else bridge_after)
        else:
            self.answer_ahead = self.answer_ahead + answer
        self.notes_text = spec.text

    def _bridge(self, after: int = 1):
        if self.notes_fresh >= after:
            self.notes = self.notes + self.bridge_ids
            self.notes_fresh = 0

    def _ahead_room(self, extra: int = 0) -> bool:
        """Still room to write ahead: reasoning up to --think-ahead, then answer up to --ahead (or its end)."""
        if self.answer_ahead and self.answer_ahead[-1] == self.end_id:
            return False
        if self.think and not self.answer_ahead:
            return len(self.notes) + extra < self.think_cap
        return len(self.answer_ahead) + extra < self.ahead

    def _notes_stale(self) -> bool:
        """The message was emptied or mostly rewritten: notes about the old one would mislead."""
        if not self.notes:
            return False
        kept = len(os.path.commonprefix([self.text, self.notes_text]))
        return not self.text.strip() or kept < len(self.notes_text) // 2

    def _prefill_loop(self):
        """Reads the draft in the background: a burst of new tokens at the end, or a pause.

        At a pause it reads the whole prompt, held-back tokens and end of the
        template included, as if the user had pressed Enter: if they do, a
        single token is left to compute; if they type on, the common-prefix
        check drops those few tokens on the next read.

        With --ahead (mafdet strike), once the whole prompt is read and the
        pause goes on, it also starts writing the answer. An answer only holds
        for the exact prompt it was written for, so any change cancels it.

        With --think the model reasons while the user types, not only at pauses,
        in short chunks: a chunk runs on the text as it was when it started, then
        the cache reads the new text and the next chunk goes on from there. The
        reasoning is kept across edits: it becomes notes that follow the new
        text, a prefill away (parallel and fast) rather than a decode away.
        """
        while True:
            time.sleep(0.02)
            target, final, done = self.target, self.final_ids, self.eng.ids
            spec = self.spec
            if spec is not None:
                changed, failed = spec.prompt != final, bool(spec.info.get("error"))
                if changed and self.think and not failed and not spec.done and not self._notes_stale():
                    continue                       # reasoning on a slightly older text is still worth finishing
                if (changed or failed) and self._take_spec(spec):
                    spec.cancel()                  # the text changed: that answer no longer holds
                    if failed:
                        self._failed = spec.prompt     # do not retry the same prompt in a loop
                    elif self.think:
                        self._absorb(spec, changed=True)
                    self._retarget()
                    continue
                if spec.done and spec.hit_limit and self._ahead_room(len(spec.tokens)) and self._take_spec(spec):
                    self._absorb(spec, changed=False)  # the pause goes on: carry on from there at the next turn
                    self._retarget()
                    continue
            if self.generating or self.spec is not None:
                continue
            if self.notes_text and self.text.rstrip() != self.notes_text.rstrip():
                if self.answer_ahead:              # written for a text that has changed since
                    self.answer_ahead = []
                    self._bridge()
                    self._retarget()
                    continue
                if self._notes_stale():
                    self.notes, self.notes_text, self.notes_fresh = [], "", 0
                    self._retarget()
                    continue
            paused = time.perf_counter() - self.text_t >= self.debounce
            has_text = bool(self.text.strip())
            # Reasoning starts on whole words: a pause, or a space or punctuation just typed.
            can_start = self._ahead_room() and (not self.think or (
                len(self.text.split()) >= THINK_MIN_WORDS and (paused or not self.text[-1:].isalnum())))
            # Reasoning goes on while the user types; an answer (no --think) waits for a pause.
            go = paused or (self.think and self.ahead and can_start)
            if (self.ahead and go and has_text and can_start and final and done == final
                    and final != self._failed):
                with self.spec_lock:
                    if not self.generating and self.final_ids is final:
                        room = self.args.ctx - len(final) - 8
                        if room >= 32 and self.think and not self.notes:
                            self.notes = list(self.seed_ids)     # the frame first, then the thoughts
                            self._retarget()
                        elif room >= 32:
                            if self.think and not self.answer_ahead:
                                chunk = min(THINK_CHUNK, self.think_cap - len(self.notes))
                            else:
                                chunk = min(AHEAD_CHUNK, self.ahead - len(self.answer_ahead))
                            self.spec = Stream(self, final, min(chunk, room), self.text)
                continue
            goal = final if go and has_text else target
            if not goal or goal == done:
                continue
            n = self._prefix_len(done, target)
            # Edits in the middle wait for a pause: recomputing the tail on every key would be a waste.
            # The tail read at the last pause (end of the template, held-back tokens) does not count.
            burst = n >= len(done) - self.tail_slack and len(target) - n >= self.min_new
            if burst or go:
                if burst and not go:
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
        history, notes = len(self.history), len(self.notes)
        message = len(final) - history - notes
        reused = min(self._prefix_len(cache, final), len(final) - 1)
        spec = self.spec
        thought, answer = self._split_ahead(list(spec.tokens) if spec else [])
        answer_ahead = len(self.answer_ahead) if self.text.rstrip() == self.notes_text.rstrip() else 0
        message -= answer_ahead
        s.update(has_text=bool(self.text.strip()), busy=self.busy, context=len(final),
                 message_tokens=message, read=min(message, max(0, reused - history)),
                 left=max(0, len(final) - reused), pending=max(0, len(target) - self._prefix_len(cache, target)),
                 strike=bool(self.ahead), think=self.think, ahead=answer_ahead + answer,
                 ahead_writing=bool(spec and not spec.done), thinking_ahead=notes + thought)
        return s

    def _split_ahead(self, tokens: list[int]) -> tuple[int, int]:
        """How many of `tokens`, written after the current prompt, are reasoning and how many answer."""
        if not self.think or self.answer_ahead:
            return 0, len(tokens)
        cut = tokens.index(self.think_end) if self.think_end in tokens else len(tokens)
        return cut, len(tokens) - cut

    # ---- send ----------------------------------------------------------------------
    def send(self, text: str):
        """Answer `text`, as a stream of events for the page."""
        if self.state != "ready":
            yield {"type": "error", "message": "The model is not ready yet."}
            return
        t_enter = time.perf_counter()
        self.stop.clear()
        with self.spec_lock:                       # from here on no answer ahead can start
            self.generating = True
            spec, self.spec = self.spec, None
        try:
            yield from self._answer(text, t_enter, spec)
        finally:
            self.notes, self.answer_ahead, self.notes_fresh, self.notes_text = [], [], 0, ""
            self.generating = False
            self._retarget()

    def _answer(self, text: str, t_enter: float, spec: Stream | None):
        history = len(self.history)
        final = self._prompt(text)
        reused = min(self._prefix_len(self.eng.cached_tokens, final), len(final) - 1)
        if self.think and self.notes:
            # The notes were written as for a message still being typed: keep them, say it is complete now.
            if spec is not None:
                spec.cancel()
                if not spec.info.get("error"):
                    self._absorb(spec, changed=spec.text.rstrip() != text.rstrip(), bridge_after=10 ** 9)
                spec = None
            if text.rstrip() != self.notes_text.rstrip():
                self.answer_ahead = []
            if len(self.notes) <= len(self.seed_ids):
                self.notes = []                    # nothing but the frame: no reasoning ahead after all
            elif not self.answer_ahead:
                if self.notes[-len(self.bridge_ids):] == self.bridge_ids:
                    self.notes = self.notes[:-len(self.bridge_ids)]
                self.notes = self.notes + self.sent_ids
            final = self._prompt(text)
        if spec is not None and (spec.prompt != final or spec.cancelled or spec.info.get("error")):
            spec.cancel()                          # written for another text: it does not hold
            if self.think and not spec.info.get("error"):
                self._absorb(spec, changed=True, bridge_after=1)   # but its reasoning does
                final = self._prompt(text)
            spec = None
        room = self.args.ctx - len(final) - 8
        if room < 32:
            yield {"type": "error", "message": f"The conversation no longer fits in the context "
                                               f"({self.args.ctx} tokens). Start a new chat."}
            return
        notes = list(self.notes)
        answer_pre = self.answer_ahead if text.rstrip() == self.notes_text.rstrip() else []
        limit = min(self.args.max_tokens or room, room) - len(answer_pre)
        thought, answer = self._split_ahead(list(spec.tokens) if spec else [])
        thinking = self.think and not answer_pre
        ttft = answer_ms = None
        state = {"thinking": thinking, "started": False}

        def pieces(piece: str):
            """Reasoning and answer events for a piece of text, leading blank lines of the answer dropped."""
            if state["thinking"]:
                head, end, piece = piece.partition("</think>")
                head = head.replace("<think>", "")
                if head:
                    yield {"type": "think", "text": head if state["started"] else head.lstrip()}
                    state["started"] = True
                if not end:
                    return
                state["thinking"], state["started"] = False, False
            if not state["thinking"] and "</think>" in piece:
                piece = piece.split("</think>", 1)[1]
            if not state["started"]:
                piece = piece.lstrip()
            if piece:
                state["started"] = True
                yield {"type": "token", "text": piece}

        # What was written ahead first, all at once: the notes, then the answer if it holds.
        detok = Detokenizer(self.tok, self.tok_lock, self.end_id)
        pre = detok.push(notes + answer_pre)
        if pre:
            state["thinking"] = self.think
        for ev in pieces(pre):
            if ev["type"] == "token" and answer_ms is None:
                answer_ms = (time.perf_counter() - t_enter) * 1000
            yield ev
        # The answer written ahead, if it holds; otherwise a new one (the stream waits for a prefill still running).
        # An answer ahead that reached its end is complete: nothing is left to generate.
        complete = bool(answer_pre) and answer_pre[-1] == self.end_id
        stream = None if complete else spec or Stream(self, final, limit, text)
        first, generated, stopped = stream, [], False
        while stream is not None:
            for tokens in stream.follow(self.stop):
                if ttft is None:
                    ttft = (time.perf_counter() - t_enter) * 1000
                for ev in pieces(detok.push(tokens)):
                    if ev["type"] == "token" and answer_ms is None:
                        answer_ms = (time.perf_counter() - t_enter) * 1000
                    yield ev
            if self.stop.is_set():
                stream.cancel()
                stopped = True
            stream.wait()
            generated += stream.tokens
            if stream.info.get("error"):
                yield {"type": "error", "message": f"llama-server stopped answering ({stream.info['error']})."}
                break
            if stopped or not stream.hit_limit or len(generated) >= limit:
                break
            # The answer ahead stopped at --ahead tokens: carry on from there, with the slot's cache already holding it.
            stream = Stream(self, final + generated, limit - len(generated), text)
        written = answer_pre + generated
        sep = self.between_ids[1:] if written and written[-1] == self.end_id else self.between_ids
        self.history = final + generated + sep
        if first is not None and "cache_n" in (first.info.get("timings") or {}):
            reused = first.info["timings"]["cache_n"]   # what the server actually reused
        reasoning = len(notes) + (self._split_ahead(generated)[0] if not answer_pre else 0)
        t = (stream.info.get("timings") or {}) if stream is not None else {}
        message = len(final) - history - len(notes) - len(answer_pre)
        yield {"type": "done", "ttft_ms": round(ttft if ttft is not None else answer_ms or 0),
               "answer_ms": round(answer_ms or 0), "tokens": len(answer_pre) + len(generated),
               "tps": round(t.get("predicted_per_second", 0), 1),
               "message_tokens": message, "read": min(message, max(0, reused - history)),
               "left": max(0, len(final) - reused), "ahead": max(0, len(answer_pre) - self.think) + answer,
               "think_tokens": reasoning if self.think else 0,
               "think_ahead": len(notes) + thought if self.think else 0,
               "cut": stream is not None and stream.hit_limit and len(generated) >= limit, "stopped": stopped}

    def new_chat(self):
        with self.spec_lock:
            self.generating = True                 # no answer ahead may start meanwhile
            spec, self.spec = self.spec, None
        if spec is not None:
            spec.cancel()
        try:
            with self.slot_lock:
                if self.state == "ready":
                    self.history = list(self.start_ids)
                    self.notes, self.answer_ahead, self.notes_fresh, self.notes_text = [], [], 0, ""
                    self._retarget()
        finally:
            self.generating = False


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
