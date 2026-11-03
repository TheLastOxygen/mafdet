"""The probe: the spelling it picks, and the tier that follows from what an engine answers."""
from mafdet.proxy.probe import probe
from proxy_fakes import FakeEngine

URL, MODEL = "http://127.0.0.1:9999", "fake-model"


def test_a_capable_engine_is_tier_1():
    profile = probe(FakeEngine(), url=URL, model=MODEL)
    assert profile.spelling == "n_predict0"
    assert profile.channel == "timings"
    assert profile.caps.prefix_caching and profile.caps.edit_branches
    assert profile.tier == "T1"


def test_an_engine_without_prefill_only_falls_back_to_one_token():
    profile = probe(FakeEngine(n_predict0=False), url=URL, model=MODEL)
    assert profile.spelling == "max_tokens1"      # what the adapter actually used
    assert profile.tier == "T1"


def test_an_exact_match_cache_is_only_static_warm():
    profile = probe(FakeEngine(branch=False), url=URL, model=MODEL)
    assert profile.caps.prefix_caching is True
    assert profile.caps.edit_branches is False
    assert profile.tier == "T0"


def test_an_engine_that_never_reuses_is_passthrough():
    profile = probe(FakeEngine(reuse=False), url=URL, model=MODEL)
    assert profile.caps.prefix_caching is False
    assert profile.tier == "T-1"


def test_usage_counters_are_an_exact_channel_too():
    profile = probe(FakeEngine(channel="usage"), url=URL, model=MODEL)
    assert profile.channel == "usage"
    assert profile.tier == "T1"


def test_metrics_only_is_marked_as_the_estimate_channel():
    engine = FakeEngine(channel="none", metrics="vllm:prefix_cache_hits_total 41\n")
    profile = probe(engine, url=URL, model=MODEL)
    assert profile.channel == "metrics"
