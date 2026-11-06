"""One conversation's draft state: debounce, the latest draft, and the generation lock.

Rules taken from the app (mafdet/watch.py):
  * generation goes first: a read-ahead never overlaps an answer being written;
  * a warm only ever prefills a prefix of what the user will send, so the answer
    cannot change -- read-ahead is exact;
  * a draft is warmed when it has grown by `min_new` tokens, or after `debounce_ms`
    of quiet, whichever comes first.  Pure debounce would never fire while the user
    types steadily.
"""
from __future__ import annotations

import threading
import time

from ..text import common_prefix_len
from .audit import Auditor
from .engine import Engine, WarmResult
from .profile import EngineProfile


def content_text(content) -> str:
    """A message's content as plain text (a string, or OpenAI's list of parts)."""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _text_of(messages: list[dict]) -> str:
    parts = [content_text(m.get("content")) for m in messages or [] if m.get("role") == "user"]
    return parts[-1] if parts else ""


def same_messages(a: list[dict], b: list[dict]) -> bool:
    """Same roles and texts, whatever else the front end attaches to each message."""
    if len(a) != len(b):
        return False
    return all(x.get("role") == y.get("role") and
               content_text(x.get("content")).strip() == content_text(y.get("content")).strip()
               for x, y in zip(a, b))


class SharedCache:
    """The engine's slot: what its prefix cache holds, and who is using it.

    llama-server keeps one cache per slot, so two conversations sharing a slot share a
    cache: whoever warmed last decides what is there.  Predicting reuse from a per-session
    idea of the cache would be wrong the moment a second conversation types -- which is
    exactly the audit's job to notice, so the prediction has to model the engine, not the
    session.  For the same reason the generation lock lives here: an answer being written
    holds the slot for every conversation, and a front end that sends no
    `conversation_id` must still stop the drafts' warms.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.ids: list[int] = []
        # After an answer the slot holds the prompt *and* the answer, and the proxy knows
        # only the prompt: the prediction is then a lower bound, not an exact number.
        self.tail_unknown = False
        self.warm_lock = threading.Lock()     # one read-ahead request at a time on the slot
        self._generating = 0                  # answers being written right now
        self._gen_cond = threading.Condition(self.lock)

    def predict(self, ids: list[int]) -> tuple[int, bool]:
        """(tokens already held minus the one the engine always recomputes, whether that
        number is only a lower bound because the slot holds more than the proxy knows)."""
        with self.lock:
            common = common_prefix_len(self.ids, ids)
            at_least = self.tail_unknown and common == len(self.ids) and len(ids) > common
            return max(0, common - 1), at_least

    def set(self, ids: list[int], *, tail_unknown: bool = False) -> None:
        with self.lock:
            self.ids = list(ids)
            self.tail_unknown = tail_unknown

    # ---- the answer being written goes first ---------------------------------------
    @property
    def generating(self) -> bool:
        with self.lock:
            return self._generating > 0

    def begin_generation(self, wait_for_warm: float = 5.0) -> None:
        """An answer takes the slot: wait out a warm in flight (bounded), then hold it."""
        if self.warm_lock.acquire(timeout=wait_for_warm):
            self.warm_lock.release()
        with self.lock:
            self._generating += 1

    def end_generation(self) -> None:
        with self.lock:
            self._generating = max(0, self._generating - 1)


class DraftSession:
    def __init__(self, conversation_id: str, engine: Engine, profile: EngineProfile, model: str,
                 *, debounce_ms: int = 300, min_new: int = 8, max_draft_tokens: int = 65536,
                 cache: "SharedCache | None" = None):
        self.conversation_id = conversation_id
        self.engine = engine
        self.profile = profile
        self.model = model
        self.cache = cache or SharedCache()
        self.debounce_ms = debounce_ms
        self.min_new = min_new
        self.max_draft_tokens = max_draft_tokens

        self.auditor = Auditor(profile.channel)
        self.tier = profile.tier
        self.lock = threading.Lock()          # protects the fields below
        self.latest: list[int] = []           # newest draft not yet warmed or committed, as tokens
        self.cached: list[int] = []           # what this session last warmed
        self.warmed_at = 0.0
        self.warms = 0
        self.skipped = 0                      # drafts skipped because a generation was running
        self._timer: threading.Timer | None = None
        # For a front end that sends only the text being typed: the text of the newest
        # draft (to recognise the request on Enter) and the conversation so far (the last
        # request's messages and the answer), which the draft is appended to.
        self.latest_text = ""
        self.drafted_at = 0.0
        self.history: list[dict] | None = None
        self.thinker = None                    # a think.Thinker when thinking ahead is on (--think-ahead)
        self.rendered_with: tuple = (None, None)   # the template options and tools the draft was rendered with

    def messages_for(self, text: str, system: dict | None = None) -> list[dict]:
        """The request the front end will send on Enter, as far as the proxy can tell: the
        conversation so far and the text as the next user message.  A new conversation
        starts with the system prompt the front end sent last."""
        head = list(self.history) if self.history is not None else ([system] if system else [])
        return head + [{"role": "user", "content": text}]

    def matches(self, messages: list[dict]) -> bool:
        """Is this request the one this session's drafts were for?  The user message is the
        newest draft (or it, plus what was typed after the last draft was posted), or the
        request continues the conversation this session holds."""
        last = _text_of(messages).strip()
        draft = self.latest_text.strip()
        if draft and last:
            if last == draft:
                return True
            if len(draft) >= 8 and (last.startswith(draft) or draft.startswith(last)):
                return True
        return self.history is not None and same_messages(messages[:-1], self.history)

    # ---- draft submission -------------------------------------------------------
    def submit(self, messages: list[dict], template_kwargs: dict | None = None,
               text: str | None = None, tools: list | None = None) -> int:
        """Record a draft and schedule its warm.  Returns the token count kept.

        The draft is rendered through the engine's own chat template, with the same
        template options as the real request (`chat_template_kwargs`): the real prompt
        starts with those tokens, so only they are worth warming.  A Qwen3 asked not to
        think gets an empty think block after the generation prompt; without it the
        engine has 6-7 tokens left to read on Enter, and a batch that small costs more
        than a larger one on a GPU.  The same goes for `tools`: Open WebUI sends 30 of
        them (~5.8k tokens) with every request, and the template puts them first."""
        ids = self.engine.render(messages, template_kwargs, tools)
        self.rendered_with = (template_kwargs, tools)
        if ids is None:                       # engine cannot render: fall back to the user text
            ids = self.engine.tokenize(_text_of(messages))
        ids = (ids or [])[: self.max_draft_tokens]
        with self.lock:
            self.latest_text = text if text is not None else _text_of(messages)
            self.drafted_at = time.time()
            if ids == self.latest:
                return len(ids)
            self.latest = ids
            grew = len(ids) - len(self.cached)
            self._schedule(0.02 if grew >= self.min_new else self.debounce_ms / 1000.0)
        if self.thinker is not None:
            self.thinker.poke()
        return len(ids)

    def _schedule(self, delay: float) -> None:
        """(Re)start the warm timer.  The caller holds self.lock."""
        self._cancel_timer()
        # Typing steadily (the draft grew by min_new): warm now rather than at a pause.
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _fire(self) -> None:
        try:
            self.warm_now()
        except Exception:                      # a warm is best-effort by construction
            pass

    # ---- the warm itself --------------------------------------------------------
    def warm_now(self) -> WarmResult | None:
        """Prefill the latest draft, if no answer is being written.  Blocking."""
        with self.lock:
            ids = list(self.latest)
        if not ids or self.tier == "T-1" or ids == self.cached:
            return None
        if self.thinker is not None and self.thinker.active:
            return None                        # its chunks read the draft; a warm would evict the notes
        if not self.cache.warm_lock.acquire(timeout=0.05):
            self.skipped += 1
            return None
        try:
            if self.cache.generating:
                # One slot: the answer being written goes first (mafdet rule 1).  The
                # draft stays in `latest`; end_generation() warms it once the slot is free.
                self.skipped += 1
                return None
            with self.lock:
                if self.latest != ids:         # dropped on Enter, or a newer draft: not this one
                    return None
            predicted, at_least = self.cache.predict(ids)
            res = self.engine.warm(ids, spelling=self.profile.spelling)
            self.auditor.record(predicted, res, at_least=at_least)
            if self.auditor.should_demote() and self.tier != "T0":
                self.tier = "T0"
            self.cache.set(ids)
            self.cached = ids
            self.warmed_at = time.time()
            self.warms += 1
            return res
        finally:
            self.cache.warm_lock.release()

    def flush(self) -> WarmResult | None:
        """Run a pending warm synchronously (tests, and the debounce timer)."""
        with self.lock:
            self._cancel_timer()
        return self.warm_now()

    # ---- the answer -------------------------------------------------------------
    def drop_pending(self) -> None:
        """Enter was pressed: the draft waiting to be warmed is the message just sent (or
        older).  Warming it after the answer would evict the answer from the slot, which
        is the start of the next turn's prompt."""
        with self.lock:
            self._cancel_timer()
            self.latest = []
            self.latest_text = ""
            self.cached = []                   # the next draft (next turn) warms at once

    def catch_up(self) -> None:
        """After an answer: warm a draft typed while it was being written."""
        with self.lock:
            if self.latest and self.latest != self.cached:
                self._schedule(0.02)
        if self.thinker is not None:
            self.thinker.poke()

    def begin_generation(self) -> None:
        """Commit from this session: drop its pending draft and take the slot."""
        self.drop_pending()
        self.cache.begin_generation()

    def end_generation(self) -> None:
        self.cache.end_generation()
        self.catch_up()

    def stats(self) -> dict:
        with self.lock:
            draft = len(self.latest)
        out = {
            "conversation_id": self.conversation_id,
            "model": self.model,
            "tier": self.tier,
            "draft_tokens": draft,
            "cached_tokens": len(self.cached),
            "warms": self.warms,
            "skipped": self.skipped,
            "history_messages": None if self.history is None else len(self.history),
        }
        if self.thinker is not None:
            out["thinking"] = {"active": self.thinker.active, "reasoned": self.thinker.reasoned,
                               "finished": self.thinker.finished}
        out.update(self.auditor.stats())
        out["last_warm_age_s"] = round(time.time() - self.warmed_at, 1) if self.warmed_at else None
        return out
