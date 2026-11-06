"""The audit: predicted prefix reuse vs what the engine reported.

Read-ahead is exact, so a mismatch means the proxy's idea of the engine is wrong --
not that the answer is wrong.  The consequence is a demotion to the static tier and a
counter in /proxy/stats, never a silent lie.

A `/metrics` counter can only ever produce an *estimate*: it is global to the engine,
so two conversations warming at the same time cannot be told apart.  Per the author's
ruling, an estimate never demotes a session on its own.  The same goes for the time
channel (reuse seen only as a faster prefill), and for a draft the proxy could not
count (it went through the engine's own template): nothing to compare, nothing to
demote.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from .engine import ESTIMATE_CHANNELS, WarmResult

# llama.cpp always recomputes the last token (it needs its logits), and tokenizers
# can disagree by a token at a chat-template boundary: allow a small slack.
TOLERANCE = 2


@dataclass
class AuditEntry:
    predicted: int | None
    reported: int | None
    channel: str
    estimate: bool
    mismatch: bool
    ms: float


class Auditor:
    """Per-conversation record of every warm, with a mismatch counter.

    `slack`: tokens a report may fall short of the prediction by, on top of the
    tolerance, for a cache that works in blocks (the probe measures it)."""

    def __init__(self, channel: str, slack: int = 0):
        self.channel = channel
        self.slack = max(0, int(slack))
        self.lock = threading.Lock()
        self.entries: list[AuditEntry] = []
        self.mismatches = 0
        self.estimates = 0
        self.unknowns = 0
        self.demoted = False

    def record(self, predicted: int | None, warm: WarmResult, *, at_least: bool = False) -> AuditEntry:
        """`at_least`: the slot holds more than the proxy knows (an answer it passed
        through, or a cache shared with other prompts), so only reusing *less* than
        predicted is a mismatch.  `predicted` None: the proxy could not count the draft."""
        estimate = warm.channel in ESTIMATE_CHANNELS or warm.estimate
        reported = warm.cached_tokens
        if reported is None or predicted is None:
            self.unknowns += 1
            entry = AuditEntry(predicted, reported, warm.channel, estimate, False, warm.ms)
        elif estimate:
            self.estimates += 1
            entry = AuditEntry(predicted, reported, warm.channel, True, False, warm.ms)
        else:
            short = reported < predicted - TOLERANCE - self.slack
            mismatch = short if at_least else (short or reported > predicted + TOLERANCE)
            if mismatch:
                self.mismatches += 1
            entry = AuditEntry(predicted, reported, warm.channel, False, mismatch, warm.ms)
        with self.lock:
            self.entries.append(entry)
            if len(self.entries) > 256:
                del self.entries[:128]
        return entry

    def should_demote(self) -> bool:
        """Two exact-channel mismatches in a row: stop trusting the profile."""
        with self.lock:
            exact = [e for e in self.entries if not e.estimate and e.reported is not None and e.predicted is not None]
        return len(exact) >= 2 and all(e.mismatch for e in exact[-2:])

    def stats(self) -> dict:
        with self.lock:
            last = self.entries[-1] if self.entries else None
        return {
            "channel": self.channel,
            "warms": len(self.entries),
            "mismatches": self.mismatches,
            "estimates": self.estimates,
            "unknown": self.unknowns,
            "last": None if last is None else {
                "predicted": last.predicted,
                "reported": last.reported,
                "channel": last.channel,
                "estimate": last.estimate,
                "ms": round(last.ms, 2),
            },
        }
