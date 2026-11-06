"""Thinking ahead (T2): notes written while the user types, the answer on Enter from them."""
import http.client
import json
import threading
import time

from mafdet.proxy import think
from mafdet.proxy.probe import probe
from mafdet.proxy.server import ProxyState, make_server
from proxy_fakes import FakeEngine

SYSTEM = {"role": "system", "content": "be brief"}
TEXT = "how many apples are left if Luca buys three"


def start(**engine_attrs):
    engine = FakeEngine()
    for k, v in engine_attrs.items():
        setattr(engine, k, v)
    profile = probe(engine, url="http://127.0.0.1:1", model="fake-model")
    state = ProxyState(engine, profile, "fake-model", debounce_ms=20, min_new=4, think_ahead=True)
    server = make_server(state, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state, engine


def post(port, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, raw


def wait_until(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def events_of(raw: bytes) -> list[dict]:
    out = []
    for line in raw.split(b"\n"):
        if line.startswith(b"data: ") and b"[DONE]" not in line:
            out.append(json.loads(line[6:]))
    return out


def test_the_model_reasons_while_the_user_types():
    server, port, state, engine = start()
    try:
        assert state.frame is not None
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": TEXT})
        sess = state.session("c1")
        assert wait_until(lambda: sess.thinker.reasoned >= 8)
        notes = sess.thinker.notes
        assert notes[:len(state.frame.seed)] == state.frame.seed        # the frame opens the reasoning
        # each chunk runs on the draft's prompt followed by the notes so far
        assert engine.gen_prompts[0][:len(sess.latest)] == sess.latest
    finally:
        server.shutdown()


def test_plain_read_ahead_stays_off_while_it_reasons():
    # A warm would leave the slot holding the prompt alone and throw the notes out of the cache.
    server, port, state, engine = start()
    try:
        engine.warms = 0
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": TEXT})
        sess = state.session("c1")
        assert wait_until(lambda: sess.thinker.reasoned >= 4)
        assert sess.flush() is None
        assert engine.warms == 0
    finally:
        server.shutdown()


def test_on_enter_the_answer_follows_the_notes():
    server, port, state, engine = start(think_after=6)
    try:
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": TEXT})
        sess = state.session("c1")
        assert wait_until(lambda: sess.thinker.finished)
        status, raw = post(port, "/v1/chat/completions",
                           {"model": "fake-model", "stream": True,
                            "messages": [SYSTEM, {"role": "user", "content": TEXT}]})
        assert status == 200
        events = events_of(raw)
        reasoning = "".join(e["choices"][0]["delta"].get("reasoning_content") or "" for e in events)
        content = "".join(e["choices"][0]["delta"].get("content") or "" for e in events)
        assert "fact1" in reasoning and "more reasoning" in reasoning      # the notes, then what followed
        assert content.strip() == "The answer is 42"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
        # the prompt was the draft's, then the notes, then SENT
        assert engine.streamed[-len(state.frame.sent):] == state.frame.sent
        assert state.counts["thought_ahead"] == 1
        assert engine.chats == 0                                          # not passed through
        # the conversation is kept, and the next turn starts new notes
        assert wait_until(lambda: sess.history is not None)
        assert sess.history[-1]["content"].strip() == "The answer is 42"
        assert sess.thinker.notes == []
    finally:
        server.shutdown()


def test_a_request_with_thinking_off_goes_through_as_it_is():
    server, port, state, engine = start()
    try:
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": TEXT})
        assert wait_until(lambda: state.session("c1").thinker.reasoned >= 4)
        post(port, "/v1/chat/completions", {"model": "fake-model", "chat_template_kwargs": {"enable_thinking": False},
                                            "messages": [SYSTEM, {"role": "user", "content": TEXT}]})
        assert engine.chats == 1 and state.counts["thought_ahead"] == 0
    finally:
        server.shutdown()


def test_without_a_reasoning_template_think_ahead_is_off():
    server, port, state, engine = start(thinking=False)
    try:
        assert state.frame is None
        assert state.session("c1").thinker is None
    finally:
        server.shutdown()


def test_a_non_streamed_request_gets_one_message():
    server, port, state, engine = start(think_after=4)
    try:
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": TEXT})
        assert wait_until(lambda: state.session("c1").thinker.finished)
        status, raw = post(port, "/v1/chat/completions",
                           {"model": "fake-model", "messages": [SYSTEM, {"role": "user", "content": TEXT}]})
        msg = json.loads(raw)["choices"][0]["message"]
        assert msg["content"].strip() == "The answer is 42" and "fact1" in msg["reasoning_content"]
    finally:
        server.shutdown()


def test_qwen3_tool_calls_become_openai_tool_calls():
    calls = think.ToolCalls()
    shown = ""
    for piece in ["Let me check. <tool", "_call>\n{\"name\": \"get_time\", ", "\"arguments\": {\"tz\": \"UTC\"}}\n</tool_",
                  "call>"]:
        content, done = calls.feed(piece)
        shown += content
    shown += calls.flush()
    assert shown == "Let me check. "
    assert len(calls.calls) == 1
    assert calls.calls[0]["function"]["name"] == "get_time"
    assert json.loads(calls.calls[0]["function"]["arguments"]) == {"tz": "UTC"}


def test_a_tool_call_ends_the_answer_with_tool_calls():
    server, port, state, engine = start(think_after=4, final_text="ok </think> <tool_call> "
                                                                   "{\"name\": \"get_time\", \"arguments\": {}} </tool_call>")
    try:
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": TEXT})
        assert wait_until(lambda: state.session("c1").thinker.finished)
        status, raw = post(port, "/v1/chat/completions",
                           {"model": "fake-model", "stream": True, "messages": [SYSTEM, {"role": "user", "content": TEXT}]})
        events = events_of(raw)
        calls = [c for e in events for c in (e["choices"][0]["delta"].get("tool_calls") or [])]
        assert calls and calls[0]["function"]["name"] == "get_time"
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    finally:
        server.shutdown()


def test_the_prompt_is_rendered_again_when_the_request_has_other_tools():
    # Drafts rendered before the proxy knew the front end's tools (first request after a restart):
    # the answer must follow the prompt the request asks for, tools included.
    server, port, state, engine = start(think_after=4)
    tools = [{"type": "function", "function": {"name": "get_time", "parameters": {}}}]
    try:
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": TEXT})
        assert wait_until(lambda: state.session("c1").thinker.finished)
        post(port, "/v1/chat/completions", {"model": "fake-model", "stream": True, "tools": tools,
                                            "messages": [SYSTEM, {"role": "user", "content": TEXT}]})
        with_tools = engine.render([SYSTEM, {"role": "user", "content": TEXT}], None, tools)
        assert engine.streamed[:len(with_tools)] == with_tools
    finally:
        server.shutdown()
