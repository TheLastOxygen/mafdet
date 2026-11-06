"""The OpenAI-compatible adapter (vLLM, SGLang, Ollama, LM Studio...): detection, the probe's two ways
(token ids, the chat endpoint), the audit on a block cache, and the proxy in front of it."""
import http.client
import json
import threading
import time

import pytest

from mafdet.proxy.adapters import make
from mafdet.proxy.adapters.openai import OpenAIEngine, _split, detect
from mafdet.proxy.launch import _first_model
from mafdet.proxy.probe import probe
from mafdet.proxy.server import ProxyState, make_server
from mafdet.proxy.session import DraftSession
from proxy_fake_servers import FakeLoading, FakeLocked, FakeOllama, FakeVLLM, start

SYSTEM = {"role": "system", "content": "be brief"}


@pytest.fixture()
def vllm():
    fake = FakeVLLM()
    server, url = start(fake)
    try:
        yield fake, url
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def ollama():
    fake = FakeOllama()
    server, url = start(fake)
    try:
        yield fake, url
    finally:
        server.shutdown()
        server.server_close()


def serve(fake):
    server, url = start(fake)
    return server, url


def call(port, path, body=None, method=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"} if payload else {}
    h.update(headers or {})
    conn.request(method or ("POST" if payload else "GET"), path, body=payload, headers=h)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, resp.getheader("Content-Type") or "", raw


def wait_until(predicate, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---- which engine, and where ----------------------------------------------------------
def test_the_url_may_carry_the_v1_of_the_routes():
    assert _split("http://host:8000/v1") == ("http", "host", 8000, "")
    assert _split("host:11434") == ("http", "host", 11434, "")
    assert _split("https://x.example/api/v1/") == ("https", "x.example", 443, "/api")


def test_detection_tells_the_engines_apart(vllm, ollama):
    assert detect(vllm[1]) == ("vllm", "0.11.0")
    assert detect(ollama[1]) == ("ollama", "0.12.6")
    assert detect("http://127.0.0.1:1") is None
    assert make("auto", vllm[1]).name == "vLLM"
    engine = make("auto", ollama[1])
    assert engine.name == "Ollama" and engine.version == "0.12.6"


def test_a_server_still_loading_is_not_detected_yet():
    # llama-server answers 503 on every route until the model is loaded: not "some OpenAI server".
    server, url = start(FakeLoading())
    try:
        assert detect(url) is None
    finally:
        server.shutdown()


def test_a_server_that_wants_a_key_is_an_openai_engine():
    server, url = start(FakeLocked())
    try:
        assert detect(url) == ("openai", "")
        assert _first_model(make("auto", url, api_key="secret"), url) == "m"
    finally:
        server.shutdown()


def test_an_unknown_adapter_name_is_an_error():
    with pytest.raises(SystemExit):
        make("exotic", "http://127.0.0.1:1")


def test_the_first_model_listed_stands_in_for_a_missing_one(vllm):
    engine = OpenAIEngine(vllm[1])
    assert _first_model(engine, vllm[1]) == "Qwen/Qwen3-4B"


# ---- the probe, by token ids (vLLM) ------------------------------------------------------
def test_vllm_is_read_ahead_by_token_ids(vllm):
    fake, url = vllm
    engine = OpenAIEngine(url, model="Qwen/Qwen3-4B", flavor="vllm")
    profile = probe(engine, url=url, model="Qwen/Qwen3-4B")
    caps = profile.caps
    assert caps.tokenize and caps.render and caps.raw_ids
    assert profile.spelling == "max_tokens1"            # vLLM refuses max_tokens 0, with or without echo
    assert profile.channel == "usage"
    assert caps.prefix_caching and caps.edit_branches and caps.tokenizer_agrees
    assert caps.shared_cache                            # the unrelated prompt in between evicted nothing
    assert caps.reuse_slack == 16                       # blocks of 16: a report may fall short by up to 15
    assert profile.tier == "T1"
    spellings = [b.get("max_tokens") for p, b in fake.requests if p == "/v1/completions"]
    assert spellings[:3] == [0, 0, 1]                   # max_tokens0, echo0, then the one that works


def test_vllm_without_its_cache_report_is_read_from_the_metrics():
    fake = FakeVLLM(report_cached=False, metrics=True)
    server, url = serve(fake)
    try:
        profile = probe(OpenAIEngine(url, flavor="vllm"), url=url, model="m")
        assert profile.channel == "metrics"
        assert profile.caps.prefix_caching and profile.caps.edit_branches and profile.caps.shared_cache
        assert profile.tier == "T1"
    finally:
        server.shutdown()


def test_an_engine_that_reports_nothing_and_answers_at_once_is_passthrough():
    # No cached_tokens, no /metrics, and a prefill no faster the second time: no evidence of reuse.
    fake = FakeVLLM(report_cached=False, metrics=False)
    server, url = serve(fake)
    try:
        profile = probe(OpenAIEngine(url, flavor="vllm"), url=url, model="m")
        assert profile.channel == "none" and profile.tier == "T-1"
    finally:
        server.shutdown()


def test_the_audit_allows_a_block_cache_to_fall_short(vllm):
    fake, url = vllm
    engine = OpenAIEngine(url, model="Qwen/Qwen3-4B", flavor="vllm")
    profile = probe(engine, url=url, model="Qwen/Qwen3-4B")
    sess = DraftSession("c1", engine, profile, "Qwen/Qwen3-4B", debounce_ms=20, min_new=4)
    words = ("w%d" % i for i in range(200))
    text = ""
    for _ in range(4):
        text += " " + " ".join(next(words) for _ in range(40))
        sess.submit([SYSTEM, {"role": "user", "content": text.strip()}])
        res = sess.flush()
        assert res is not None and res.channel == "usage"
    stats = sess.auditor.stats()
    assert stats["warms"] == 4 and stats["mismatches"] == 0
    assert stats["last"]["reported"] % 16 == 0 or stats["last"]["reported"] == stats["last"]["predicted"]
    assert sess.tier == "T1"


# ---- the probe, through the chat endpoint (Ollama) -----------------------------------------
def test_ollama_is_read_ahead_through_its_chat_endpoint(ollama):
    fake, url = ollama
    engine = OpenAIEngine(url, model="qwen3:4b", flavor="ollama")
    profile = probe(engine, url=url, model="qwen3:4b")
    caps = profile.caps
    assert not caps.tokenize and not caps.render and not caps.raw_ids
    assert profile.spelling == "chat1" and caps.by_chat
    assert profile.channel == "time"                    # only prompt_eval_duration says the prefix was there
    assert caps.prefix_caching and caps.edit_branches
    assert not caps.shared_cache                        # one slot: the unrelated prompt evicted the filler
    assert profile.tier == "T1"
    chats = [b for p, b in fake.requests if p == "/api/chat"]
    assert chats and all(b["options"]["num_predict"] == 1 for b in chats)


def test_ollama_with_two_slots_keeps_the_conversation():
    fake = FakeOllama(parallel=2)
    server, url = serve(fake)
    try:
        profile = probe(OpenAIEngine(url, model="qwen3:4b", flavor="ollama"), url=url, model="qwen3:4b")
        assert profile.caps.shared_cache and profile.tier == "T1"
    finally:
        server.shutdown()


def test_an_engine_that_counts_only_what_it_computed():
    fake = FakeOllama(counts_computed=True)
    server, url = serve(fake)
    try:
        profile = probe(OpenAIEngine(url, model="qwen3:4b", flavor="ollama"), url=url, model="qwen3:4b")
        assert profile.channel == "computed" and profile.tier == "T1"
    finally:
        server.shutdown()


def test_a_session_on_ollama_sends_the_draft_as_a_chat_request(ollama):
    fake, url = ollama
    engine = OpenAIEngine(url, model="qwen3:4b", flavor="ollama")
    profile = probe(engine, url=url, model="qwen3:4b")
    del fake.requests[:]
    sess = DraftSession("c1", engine, profile, "qwen3:4b", debounce_ms=20, min_new=4)
    messages = [SYSTEM, {"role": "user", "content": "how many apples are left in the basket"}]
    sess.submit(messages, {"enable_thinking": False})
    assert sess.flush() is not None
    path, body = fake.requests[-1]
    assert path == "/api/chat" and body["messages"] == messages and body["think"] is False
    assert sess.flush() is None                         # the same draft is not sent twice
    sess.submit(messages)                               # other template options: another draft
    assert sess.flush() is not None
    stats = sess.auditor.stats()
    assert stats["warms"] == 2 and stats["mismatches"] == 0 and stats["unknown"] == 2
    assert sess.tier == "T1"


# ---- the proxy in front of it ---------------------------------------------------------------
def proxy_on(fake, url, **kw):
    engine = make("auto", url, model=fake.model, api_key=kw.pop("api_key", ""))   # detected, as `mafdet proxy --url` does
    profile = probe(engine, url=url, model=fake.model)
    del fake.requests[:]
    del fake.headers[:]
    state = ProxyState(engine, profile, fake.model, debounce_ms=20, min_new=4, **kw)
    server = make_server(state, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


def test_the_proxy_in_front_of_vllm(vllm):
    fake, url = vllm
    server, port, state = proxy_on(fake, url, api_key="secret")
    try:
        status, _, raw = call(port, "/v1/drafts", {"conversation_id": "c1", "text": "one two three four five six"})
        assert status == 200 and json.loads(raw)["tier"] == "T1"
        state.session("c1").flush()
        assert [p for p, _ in fake.requests][-1] == "/v1/completions"
        assert fake.headers[-1].get("authorization") == "Bearer secret"
        messages = [SYSTEM, {"role": "user", "content": "one two three four five six"}]
        status, ctype, raw = call(port, "/v1/chat/completions", {"model": fake.model, "messages": messages, "stream": True})
        assert status == 200 and "event-stream" in ctype and b"[DONE]" in raw
        assert wait_until(lambda: not state.cache.generating)
        assert state.counts["matched"] == 1
        body = [b for p, b in fake.requests if p == "/v1/chat/completions"][-1]
        assert "id_slot" not in body and "cache_prompt" not in body     # llama-server's options, not vLLM's
        assert fake.requests[-1][0] == "/tokenize"                      # the answer's prompt, for the cache model
        status, _, raw = call(port, "/v1/models")
        assert status == 200 and json.loads(raw)["data"][0]["id"] == fake.model
        stats = json.loads(call(port, "/proxy/stats")[2])
        assert stats["proxy"]["engine"] == "vLLM" and stats["proxy"]["version"] == "0.11.0"
        assert stats["cache"] == {"shared": True, "drafts": "ids", "slack": 16}
    finally:
        server.shutdown()
        server.server_close()


def test_the_request_log_reads_the_usage_of_an_openai_engine(vllm):
    fake, url = vllm
    server, port, state = proxy_on(fake, url)
    try:
        messages = [SYSTEM, {"role": "user", "content": "one two three four five six seven eight"}]
        call(port, "/v1/chat/completions", {"model": fake.model, "messages": messages})
        assert wait_until(lambda: len(state.recent) == 1)
        row = state.recent[0]
        assert row["already_read"] is not None and row["read_on_enter"] is not None
    finally:
        server.shutdown()
        server.server_close()


def test_exact_mode_on_an_openai_engine_only_stops_the_drafts(vllm):
    fake, url = vllm
    server, port, state = proxy_on(fake, url, exact=True)
    try:
        call(port, "/v1/drafts", {"conversation_id": "c1", "text": "one two three four five six"})
        state.session("c1").flush()
        call(port, "/v1/chat/completions", {"model": fake.model, "messages": [{"role": "user", "content": "hi there"}]})
        paths = [p for p, _ in fake.requests]
        assert "/v1/completions" not in paths
        assert "cache_prompt" not in fake.requests[-1][1]    # vLLM has no such switch, and would refuse it
    finally:
        server.shutdown()
        server.server_close()


def test_the_proxy_in_front_of_ollama(ollama):
    fake, url = ollama
    server, port, state = proxy_on(fake, url)
    try:
        call(port, "/v1/drafts", {"conversation_id": "c1", "text": "how many apples are left in the basket today"})
        state.session("c1").flush()
        assert fake.requests[-1][0] == "/api/chat"
        messages = [SYSTEM, {"role": "user", "content": "how many apples are left in the basket today"}]
        status, _, raw = call(port, "/v1/chat/completions", {"model": fake.model, "messages": messages})
        assert status == 200 and json.loads(raw)["choices"][0]["message"]["content"].startswith("echo:")
        assert wait_until(lambda: not state.cache.generating)
        assert state.counts["matched"] == 1
        stats = json.loads(call(port, "/proxy/stats")[2])
        assert stats["cache"]["drafts"] == "chat" and stats["proxy"]["engine"] == "Ollama"
        # the next turn's draft follows the conversation, through the chat endpoint again
        call(port, "/v1/drafts", {"conversation_id": "c1", "text": "and the pears"})
        state.session("c1").flush()
        path, body = fake.requests[-1]
        assert path == "/api/chat" and body["messages"][-1] == {"role": "user", "content": "and the pears"}
        assert body["messages"][-2]["role"] == "assistant"
    finally:
        server.shutdown()
        server.server_close()
