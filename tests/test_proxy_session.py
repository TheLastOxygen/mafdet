"""The conversation's draft state: debounce, the generation lock, and demotion."""
import time

from mafdet.proxy.probe import probe
from mafdet.proxy.session import DraftSession
from proxy_fakes import FakeEngine


def make_session(**engine_kw) -> tuple[DraftSession, FakeEngine]:
    # The profile comes from a well-behaved engine; the session then serves from the
    # engine under test (that's how a server whose behaviour later changes is modelled).
    profile = probe(FakeEngine(), url="http://127.0.0.1:1", model="fake-model")
    engine = FakeEngine(**engine_kw)
    engine.warms = 0                       # the probe's own warms don't count here
    return DraftSession("conv", engine, profile, "fake-model", debounce_ms=40, min_new=4), engine


def wait_until(predicate, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_rapid_small_drafts_coalesce_into_one_warm():
    sess, engine = make_session()
    sess.submit([{"role": "user", "content": "alpha"}])
    sess.submit([{"role": "user", "content": "alpha beta"}])              # grew by 1 token: still under min_new
    assert wait_until(lambda: engine.warms == 1), "the debounce never fired"
    time.sleep(0.12)                       # nothing else should fire
    assert engine.warms == 1


def test_a_draft_that_grew_by_min_new_warms_without_waiting_for_a_pause():
    sess, engine = make_session()
    sess.submit([{"role": "user", "content": "one two three four five six seven eight nine ten"}])
    assert wait_until(lambda: engine.warms == 1, timeout=0.5)


def test_a_warm_never_overlaps_a_generation():
    sess, engine = make_session()
    sess.begin_generation()
    try:
        sess.submit([{"role": "user", "content": "one two three four five six seven eight nine ten"}])
        time.sleep(0.15)
        assert engine.warms == 0
        assert sess.skipped >= 1
    finally:
        sess.end_generation()
    sess.flush()
    assert engine.warms == 1               # the draft is caught up once the slot is free


def test_two_exact_mismatches_demote_the_session_to_static():
    sess, engine = make_session(report_cached=0)   # an engine that lies about reuse
    sess.submit([{"role": "user", "content": "one two three four five six seven eight nine ten"}])
    sess.flush()
    sess.flush()
    sess.flush()
    assert sess.auditor.mismatches >= 2
    assert sess.tier == "T0"


def test_nothing_is_warmed_without_a_draft():
    sess, engine = make_session()
    assert sess.flush() is None
    assert engine.warms == 0
