"""Thinking ahead (tier T2): the model reasons about the draft while the user types.

Not exact: the answer comes from reasoning started on a message that was not finished yet.  It is
opt-in (`--think-ahead`), for models whose chat template has a reasoning block (Qwen3), and only
for requests that leave thinking on.

The frame is the MAFDET app's, measured on Qwen3 (mafdet/watch.py, research/LOG.md), copied here
because the proxy does not import the app; sharing it is a separate change, as agreed in #2:
  * after the draft, inside the assistant's turn, SEED opens the reasoning: a list of material for
    the answer, one line per key term, name or number already typed;
  * the model writes it in short chunks (THINK_CHUNK tokens), each on the draft as it is then;
  * when the user typed more, a BRIDGE line names the new words, once enough new reasoning was
    written (BRIDGE_EVERY) or the reasoning had ended: it goes on from its notes, not from scratch;
  * on Enter: the final message, the notes, SENT ("the user has now sent the complete message"),
    and the model finishes reasoning from there and answers.
The proxy streams that answer back as an OpenAI chat completion: the reasoning (the notes, then
what follows) as `reasoning_content`, the answer as `content`, and Qwen3's <tool_call> blocks as
`tool_calls`.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Iterable

SEED = ("<think>\nThe user is still typing. I will not wonder what they will ask, nor remark that the text is "
        "unfinished. While I wait, I collect material for the answer: for each key term, name or number already "
        "written, one line with what it is and the facts I know about it, or a partial result. For example, from "
        "\"Quanto pesa la Luna rispetto alla\" I would write \"- Luna (the Moon): mass 7.35e22 kg, radius "
        "1,737 km\".\n\n- ")
BRIDGE = "\n- New words \"{new}\": "
BRIDGE_EDIT = "\n- The text was edited, so: "
BRIDGE_NEW_CHARS = 300
BRIDGE_MIN_WORDS = 2
SENT = "\n\nThe user has now sent the complete message. Let me reread it and answer it, using the material above.\n\n"
META_WORDS = ("user", "User", "incomplete", "unfinished", "typing", "typed", "intent", "intends", "asking",
              "wondering", "seems")
META_BIAS = -4.0
THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 1.5}
THINK_MIN_WORDS = 3          # no reasoning on the first couple of words
THINK_CHUNK = 32             # tokens per chunk: short, so it follows the text as it is typed
BRIDGE_EVERY = 64            # new reasoning tokens before an edit earns a bridge
THINK_CAP = 1024             # longest reasoning written ahead
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
                 "repeat_penalty", "seed")


class Frame:
    """The frame's token ids for this engine's tokenizer, computed once."""

    def __init__(self, engine):
        tok = engine.tokenize
        self.seed = tok(SEED) or []
        self.sent = tok(SENT) or []
        self.bridge_edit = tok(BRIDGE_EDIT) or []
        end = tok("</think>") or []
        self.think_end = end[0] if len(end) == 1 else None
        meta = set()
        for w in META_WORDS:
            for v in (w, " " + w):
                ids = tok(v) or []
                if len(ids) == 1:
                    meta.add(ids[0])
        self.meta_bias = [[t, META_BIAS] for t in sorted(meta)] if META_BIAS else []
        self.tokenize = tok

    @property
    def usable(self) -> bool:
        return bool(self.seed and self.sent and self.think_end is not None)


def thinking_on(template_kwargs: dict | None) -> bool:
    return (template_kwargs or {}).get("enable_thinking", True) is not False


class Thinker:
    """One conversation's reasoning ahead: the notes after the draft, grown chunk by chunk."""

    def __init__(self, session, frame: Frame, enabled=lambda: True):
        self.session, self.frame, self.enabled = session, frame, enabled
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopped = False
        self._clear()
        threading.Thread(target=self._loop, daemon=True).start()

    def _clear(self):
        self.notes: list[int] = []             # SEED, reasoning and bridges, after the draft's prompt
        self.notes_text = ""                   # the draft the notes last followed
        self.bridge_text = ""                  # up to where the bridges named the new words
        self.fresh = 0                         # reasoning tokens since the last bridge
        self.finished = False                  # the model closed its reasoning (</think>)

    def reset(self):
        """A new turn: no notes."""
        with self.lock:
            self._clear()

    @property
    def active(self) -> bool:
        """Reasoning on this draft (then plain read-ahead would only evict the notes from the cache)."""
        return self.enabled() and len(self.session.latest_text.split()) >= THINK_MIN_WORDS

    def poke(self):
        self.wake.set()

    @property
    def reasoned(self) -> int:
        return max(0, len(self.notes) - len(self.frame.seed))

    # ---- while the user types -----------------------------------------------------------
    def _loop(self):
        while not self.stopped:
            self.wake.wait(0.25)
            self.wake.clear()
            try:
                while self._step():
                    pass
            except Exception:                  # thinking ahead is best-effort, never a crash
                time.sleep(0.5)

    def _new_words(self, text: str) -> tuple[str, int]:
        start = len(os.path.commonprefix([self.bridge_text, text]))
        new = text[start:]
        if new[-1:].isalnum():                 # the last word may still be being typed
            new = new[:max(new.rfind(" "), 0)]
        return new.strip(), start + len(new)

    def _bridge(self, text: str, force: bool = False) -> list[int]:
        new, end = self._new_words(text)
        if new and len(new.split()) < BRIDGE_MIN_WORDS and not force:
            return []
        ids = self.frame.tokenize(BRIDGE.format(new=new[-BRIDGE_NEW_CHARS:])) if new else self.frame.bridge_edit
        self.bridge_text = text[:end] if new else text
        return ids or []

    def _step(self) -> bool:
        """Write one chunk if there is something to do.  True when it wrote one."""
        s = self.session
        if self.stopped or not self.enabled() or s.cache.generating:
            return False
        with s.lock:
            prompt, text = list(s.latest), s.latest_text
        if not prompt or len(text.split()) < THINK_MIN_WORDS:
            return False
        with self.lock:
            if self.notes and len(os.path.commonprefix([text, self.notes_text])) < len(self.notes_text) // 2:
                self._clear()                  # the message was mostly rewritten: old notes would mislead
            if not self.notes:
                self.notes, self.notes_text, self.bridge_text = list(self.frame.seed), text, ""
            elif text != self.notes_text and (self.finished or self.fresh >= BRIDGE_EVERY):
                bridge = self._bridge(text, force=self.finished)
                if bridge:
                    self.notes += bridge
                    self.fresh, self.finished = 0, False
            if self.finished and text == self.notes_text:
                return False                   # done reasoning on this text: wait for more typing
            if self.reasoned >= THINK_CAP:
                return False
            ids = prompt + self.notes
        if not s.cache.warm_lock.acquire(timeout=0.05):
            return False
        try:
            if s.cache.generating or self.stopped:
                return False
            res = s.engine.generate(ids, n_predict=THINK_CHUNK, sampling=THINK_SAMPLING,
                                    logit_bias=self.frame.meta_bias)
            s.cache.set(ids + list(res.get("tokens") or []))
        finally:
            s.cache.warm_lock.release()
        tokens = list(res.get("tokens") or [])
        if not tokens:
            return False
        with self.lock:
            if self.notes_text and ids[len(prompt):] != self.notes:
                return False                   # reset or bridged meanwhile: the chunk is stale
            end = self.frame.think_end
            if end in tokens:
                tokens = tokens[:tokens.index(end)]
                self.finished = True
            self.notes += tokens
            self.fresh += len(tokens)
            self.notes_text = text
        return True

    # ---- on Enter -------------------------------------------------------------------------
    def take(self, final_text: str) -> list[int] | None:
        """The notes for the message as sent, ready to follow its prompt (bridge and SENT included),
        or None if nothing was reasoned yet (then the request goes through as it is)."""
        with self.lock:
            if self.reasoned < 1:
                return None
            notes = list(self.notes)
            if final_text.rstrip() != self.notes_text.rstrip():
                saved = self.bridge_text
                notes += self._bridge(final_text + " ", force=True)
                self.bridge_text = saved
            return notes + self.frame.sent


def _sampling(body: dict) -> dict:
    out = dict(THINK_SAMPLING)
    out.update({k: body[k] for k in SAMPLING_KEYS if body.get(k) is not None})
    return out


class ToolCalls:
    """Qwen3 writes a tool call as <tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>:
    split the answer's text into plain content and OpenAI tool calls."""

    OPEN, CLOSE = "<tool_call>", "</tool_call>"

    def __init__(self):
        self.buf = ""
        self.calls: list[dict] = []

    def feed(self, text: str) -> tuple[str, list[dict]]:
        """Content safe to show now, and tool calls completed by this text."""
        self.buf += text
        out, done = "", []
        while True:
            i = self.buf.find(self.OPEN)
            if i < 0:
                # keep a tail that could be the start of "<tool_call>"
                keep = max((k for k in range(1, len(self.OPEN)) if self.buf.endswith(self.OPEN[:k])), default=0)
                out += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                return out, done
            out += self.buf[:i]
            j = self.buf.find(self.CLOSE, i)
            if j < 0:
                self.buf = self.buf[i:]
                return out, done
            raw = self.buf[i + len(self.OPEN):j].strip()
            self.buf = self.buf[j + len(self.CLOSE):]
            try:
                call = json.loads(raw)
                args = call.get("arguments", {})
                done.append({"index": len(self.calls), "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                             "function": {"name": call.get("name", ""),
                                          "arguments": args if isinstance(args, str) else json.dumps(args)}})
                self.calls.append(done[-1])
            except (ValueError, AttributeError):
                out += self.OPEN + raw + self.CLOSE

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


def answer(engine, prompt: list[int], notes: list[int], frame: Frame, body: dict) -> Iterable[dict]:
    """The answer after the notes, as OpenAI chat-completion chunks (dicts), the last with
    finish_reason, usage and llama-server's timings.  The notes are shown first as reasoning."""
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    model = body.get("model", "")

    def chunk(delta: dict, finish: str | None = None, **extra) -> dict:
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

    yield chunk({"role": "assistant", "content": None})
    seen = engine.detokenize(notes[len(frame.seed):len(notes) - len(frame.sent)])
    if seen.strip():
        yield chunk({"reasoning_content": "- " + seen.lstrip()})
    limit = body.get("max_completion_tokens") or body.get("max_tokens") or -1
    thinking, tools, produced, final = True, ToolCalls(), 0, {}
    pending = ""
    for ev in engine.stream(prompt + notes, n_predict=limit, sampling=_sampling(body)):
        if ev.get("error"):
            final = {"stop_type": "error"}
            break
        produced += len(ev.get("tokens") or [])
        text = ev.get("text", "")
        if thinking:
            pending += text
            if "</think>" in pending:
                before, after = pending.split("</think>", 1)
                if before:
                    yield chunk({"reasoning_content": before})
                thinking, pending = False, ""
                text = after.lstrip("\n")
            else:
                safe = len(pending) - max((k for k in range(1, 8) if pending.endswith("</think>"[:k])), default=0)
                if safe > 0:
                    yield chunk({"reasoning_content": pending[:safe]})
                    pending = pending[safe:]
                text = ""
        if not thinking and text:
            content, calls = tools.feed(text)
            if content:
                yield chunk({"content": content})
            for call in calls:
                yield chunk({"tool_calls": [call]})
        if ev.get("stop"):
            final = ev
            break
    rest = tools.flush() if not thinking else pending
    if rest:
        yield chunk({"content": rest} if not thinking else {"reasoning_content": rest})
    finish = "tool_calls" if tools.calls else ("length" if final.get("stop_type") == "limit" else "stop")
    t = final.get("timings") or {}
    yield chunk({}, finish, usage={"prompt_tokens": len(prompt) + len(notes), "completion_tokens": produced,
                                   "total_tokens": len(prompt) + len(notes) + produced},
                timings=t)


def collect(chunks: Iterable[dict]) -> dict:
    """A non-streamed chat completion from the chunks."""
    content, reasoning, calls, last = "", "", [], {}
    for c in chunks:
        d = c["choices"][0]["delta"]
        content += d.get("content") or ""
        reasoning += d.get("reasoning_content") or ""
        calls += d.get("tool_calls") or []
        last = c
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
    return {"id": last.get("id"), "object": "chat.completion", "created": last.get("created"), "model": last.get("model"),
            "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
            "usage": last.get("usage"), "timings": last.get("timings")}
