"""The HTTP front door: drafts, the chat pass-through, streaming, stats, liveness."""
import http.client
import json
import threading
import time

import pytest

from mafdet.proxy.probe import probe
from mafdet.proxy.server import ProxyState, make_server
from proxy_fakes import FakeEngine

MESSAGES = [{"role": "system", "content": "be brief"},
            {"role": "user", "content": "one two three four five six seven eight nine ten"}]


@pytest.fixture()
def proxy():
    engine = FakeEngine()
    profile = probe(engine, url="http://127.0.0.1:1", model="fake-model")
    engine.warms = 0                       # the probe's own warms don't count here
    state = ProxyState(engine, profile, "fake-model", debounce_ms=20, min_new=4)
    server = make_server(state, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, state, engine
    finally:
        server.shutdown()
        server.server_close()


def call(port, path, body=None, method=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method or ("POST" if payload else "GET"), path, body=payload,
                 headers={"Content-Type": "application/json"} if payload else {})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, resp.getheader("Content-Type") or "", raw


def test_healthz(proxy):
    port, _, _ = proxy
    status, _, raw = call(port, "/healthz")
    assert status == 200 and json.loads(raw)["ok"] is True


def test_a_draft_is_accepted_and_warms_the_cache(proxy):
    port, state, engine = proxy
    status, _, raw = call(port, "/v1/drafts",
                          {"conversation_id": "c1", "model": "fake-model", "messages": MESSAGES})
    assert status == 200
    body = json.loads(raw)
    assert body["conversation_id"] == "c1" and body["tier"] == "T1"
    assert body["draft_tokens"] >= 10
    state.session("c1").flush()
    assert engine.warms >= 1


def test_stats_report_the_tier_and_the_audit(proxy):
    port, state, _ = proxy
    call(port, "/v1/drafts", {"conversation_id": "c1", "model": "fake-model", "messages": MESSAGES})
    state.session("c1").flush()
    status, _, raw = call(port, "/proxy/stats")
    assert status == 200
    stats = json.loads(raw)
    assert stats["proxy"]["tier"] == "T1"
    session = stats["sessions"][0]
    assert session["conversation_id"] == "c1"
    assert session["warms"] == 1
    assert session["channel"] == "timings"


def test_chat_is_passed_through(proxy):
    port, _, engine = proxy
    status, ctype, raw = call(port, "/v1/chat/completions",
                              {"model": "fake-model", "messages": MESSAGES})
    assert status == 200 and "json" in ctype
    assert json.loads(raw)["choices"][0]["message"]["content"].startswith("echo:")
    assert engine.chats == 1


def test_streaming_is_relayed(proxy):
    port, _, _ = proxy
    status, ctype, raw = call(port, "/v1/chat/completions",
                              {"model": "fake-model", "messages": MESSAGES, "stream": True})
    assert status == 200 and "event-stream" in ctype
    assert b"data: " in raw and b"[DONE]" in raw


def test_an_unknown_route_is_a_404(proxy):
    port, _, _ = proxy
    status, _, _ = call(port, "/v1/nope", method="GET")
    assert status == 404


def serve(state):
    server = make_server(state, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def make_state(**kw):
    engine = FakeEngine()
    profile = probe(engine, url="http://127.0.0.1:1", model="fake-model")
    engine.warms = 0
    return ProxyState(engine, profile, "fake-model", debounce_ms=20, min_new=4, **kw), engine


def test_an_answer_without_conversation_id_holds_every_draft():
    # A plain OpenAI client sends no conversation_id: its answer must still hold the slot.
    state, engine = make_state()
    state.begin_generation()
    try:
        state.session("c1").submit(MESSAGES)
        time.sleep(0.1)
        assert engine.warms == 0
    finally:
        state.end_generation()
    assert wait_until(lambda: engine.warms == 1)


def test_after_an_answer_the_cache_holds_its_prompt():
    state, engine = make_state()
    server, port = serve(state)
    try:
        call(port, "/v1/chat/completions", {"model": "fake-model", "messages": MESSAGES,
                                            "chat_template_kwargs": {"enable_thinking": False}})
        assert state.cache.ids == engine.render(MESSAGES, {"enable_thinking": False})
        assert state.cache.tail_unknown
        # and a later draft is rendered with the options the last request used
        call(port, "/v1/drafts", {"conversation_id": "c1", "messages": MESSAGES})
        assert engine.last_template_kwargs == {"enable_thinking": False}
    finally:
        server.shutdown()
        server.server_close()


def test_exact_mode_warms_nothing_and_turns_the_cache_off():
    state, engine = make_state(exact=True)
    seen = {}
    real_chat = engine.chat

    def chat(body, **kw):
        seen.update(body)
        return real_chat(body, **kw)

    engine.chat = chat
    server, port = serve(state)
    try:
        status, _, raw = call(port, "/v1/drafts", {"conversation_id": "c1", "messages": MESSAGES})
        assert status == 200 and json.loads(raw)["tier"] == "T-1"
        state.session("c1").flush()
        call(port, "/v1/chat/completions", {"model": "fake-model", "messages": MESSAGES})
        assert engine.warms == 0
        assert seen.get("cache_prompt") is False
    finally:
        server.shutdown()
        server.server_close()


def preflight(port, origin):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("OPTIONS", "/v1/drafts", headers={"Origin": origin,
                                                   "Access-Control-Request-Method": "POST"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp.status, resp.getheader("Access-Control-Allow-Origin")


def test_cors_is_off_by_default():
    state, _ = make_state()
    server, port = serve(state)
    try:
        assert preflight(port, "http://localhost:3000")[1] is None
    finally:
        server.shutdown()
        server.server_close()


def test_cors_allows_only_the_origins_given():
    state, _ = make_state(cors=["http://localhost:3000"])
    server, port = serve(state)
    try:
        assert preflight(port, "http://localhost:3000") == (204, "http://localhost:3000")
        assert preflight(port, "http://evil.example")[1] is None
    finally:
        server.shutdown()
        server.server_close()


def wait_until(predicate, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False
