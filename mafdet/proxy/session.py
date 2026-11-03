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


def _text_of(messages: list[dict]) -> str:
    parts = []
    for msg in messages or []:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            parts.append(content or "")
    return parts[-1] if parts else ""


class SharedCache:
    """What the engine's prefix cache holds.

    llama-server keeps one cache per slot, so two conversations sharing a slot share a
    cache: whoever warmed last decides what is there.  Predicting reuse from a per-session
    idea of the cache would be wrong the moment a second conversation types -- which is
    exactly the audit's job to notice, so the prediction has to model the engine, not the
    session.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.ids: list[int] = []

    def predict(self, ids: list[int]) -> int:
        """Tokens already held, minus the one the engine always recomputes (its logits)."""
        with self.lock:
            return max(0, common_prefix_len(self.ids, ids) - 1)

    def set(self, ids: list[int]) -> None:
        with self.lock:
            self.ids = list(ids)


class DraftSession:
    def __init__(self, conversation_id: str, engine: Engine, profile: EngineProfile, model: str,
                 *, debounce_ms: int = 300, min_new: int = 8, max_draft_tokens: int = 4096,
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
        self.gen_lock = threading.Lock()      # held while an answer is being written
        self.warm_lock = threading.Lock()     # held while a read-ahead request runs
        self.abort = threading.Event()        # set on commit: a warm in flight gives up
        self.latest: list[int] = []           # newest draft, as tokens
        self.cached: list[int] = []           # what the engine's cache holds
        self.warmed_at = 0.0
        self.warms = 0
        self.skipped = 0                      # drafts skipped because a generation was running
        self._timer: threading.Timer | None = None

    # ---- draft submission -------------------------------------------------------
    def submit(self, messages: list[dict]) -> int:
        """Record a draft and schedule its warm.  Returns the token count kept.

        The draft is rendered through the engine's own chat template: the real request's
        prompt starts with those tokens, so only they are worth warming."""
        ids = self.engine.render(messages)
        if ids is None:                       # engine cannot render: fall back to the user text
            ids = self.engine.tokenize(_text_of(messages))
        ids = (ids or [])[: self.max_draft_tokens]
        with self.lock:
            if ids == self.latest:
                return len(ids)
            self.latest = ids
            grew = len(ids) - len(self.cached)
            self._cancel_timer()
            if grew >= self.min_new:
                # Typing steadily: warm now rather than waiting for a pause.
                self._timer = threading.Timer(0.02, self._fire)
            else:
                self._timer = threading.Timer(self.debounce_ms / 1000.0, self._fire)
            self._timer.daemon = True
            self._timer.start()
        return len(ids)

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
        if not ids or self.tier == "T-1":
            return None
        if not self.warm_lock.acquire(timeout=0.05):
            self.skipped += 1
            return None
        try:
            if self.gen_lock.locked():
                # One slot: the answer being written goes first (mafdet rule 1).
                self.skipped += 1
                return None
            if self.abort.is_set():
                return None
            predicted = self.cache.predict(ids)
            res = self.engine.warm(ids, spelling=self.profile.spelling)
            self.auditor.record(predicted, res)
            if self.auditor.should_demote() and self.tier != "T0":
                self.tier = "T0"
            self.cache.set(ids)
            self.cached = ids
            self.warmed_at = time.time()
            self.warms += 1
            return res
        finally:
            self.warm_lock.release()

    def flush(self) -> WarmResult | None:
        """Run a pending warm synchronously (tests, and the debounce timer)."""
        with self.lock:
            self._cancel_timer()
        return self.warm_now()

    # ---- the answer -------------------------------------------------------------
    def begin_generation(self) -> None:
        """Commit: abort any warm, then take the slot for the answer."""
        self.abort.set()
        self.warm_lock.acquire(timeout=5.0)    # waits out a warm in flight, bounded
        self.warm_lock.release()
        self.gen_lock.acquire()

    def end_generation(self) -> None:
        self.abort.clear()
        self.gen_lock.release()

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
        }
        out.update(self.auditor.stats())
        out["last_warm_age_s"] = round(time.time() - self.warmed_at, 1) if self.warmed_at else None
        return out
