"""The audit: exact channels can demote a session, estimates can only be labelled."""
from mafdet.proxy.audit import Auditor
from mafdet.proxy.engine import WarmResult


def test_an_exact_mismatch_is_counted():
    audit = Auditor("timings")
    entry = audit.record(predicted=90, warm=WarmResult(cached_tokens=10, channel="timings", ms=3.0))
    assert entry.mismatch and entry.reported == 10
    assert audit.mismatches == 1


def test_two_exact_mismatches_in_a_row_ask_for_a_demotion():
    audit = Auditor("timings")
    for _ in range(2):
        audit.record(90, WarmResult(cached_tokens=10, channel="timings"))
    assert audit.should_demote()


def test_a_match_clears_the_demotion():
    audit = Auditor("timings")
    audit.record(90, WarmResult(cached_tokens=10, channel="timings"))
    audit.record(90, WarmResult(cached_tokens=90, channel="timings"))
    audit.record(90, WarmResult(cached_tokens=91, channel="timings"))   # within tolerance
    assert not audit.should_demote()


def test_a_metrics_estimate_never_demotes():
    audit = Auditor("metrics")
    for _ in range(5):
        audit.record(90, WarmResult(cached_tokens=0, channel="metrics", estimate=True))
    assert audit.estimates == 5
    assert audit.mismatches == 0
    assert not audit.should_demote()
    assert audit.stats()["last"]["estimate"] is True


def test_no_signal_is_reported_as_unknown():
    audit = Auditor("none")
    audit.record(90, WarmResult(cached_tokens=None, channel="none"))
    assert audit.unknowns == 1
    assert not audit.should_demote()
    assert audit.stats()["last"]["reported"] is None
