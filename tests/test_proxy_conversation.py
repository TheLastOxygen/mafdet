"""Step 1: drafts that carry only the text, the request on Enter matched to them, the
conversation kept for the next turn, and side requests sent to another slot."""
import http.client
import json
import threading
import time

from mafdet.proxy.probe import probe
from mafdet.proxy.server import ProxyState, make_server
from proxy_fakes import FakeEngine

SYSTEM = {"role": "system", "content": "be brief"}


def start(n_slots=1):
    engine = FakeEngine()
    engine.n_slots = n_slots
    profile = probe(engine, url="http://127.0.0.1:1", model="fake-model")
    engine.warms = 0
    state = ProxyState(engine, profile, "fake-model", debounce_ms=20, min_new=4)
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


def chat(port, messages, stream=False):
    return post(port, "/v1/chat/completions", {"model": "fake-model", "messages": messages, "stream": stream})


def test_a_text_draft_in_a_new_chat_starts_with_the_last_system_prompt():
    server, port, state, engine = start()
    try:
        chat(port, [SYSTEM, {"role": "user", "content": "earlier question"}])
        status, _ = post(port, "/v1/drafts", {"conversation_id": "new", "text": "how many apples"})
        assert status == 200
        assert engine.last_messages == [SYSTEM, {"role": "user", "content": "how many apples"}]
    finally:
        server.shutdown()


def test_the_request_on_enter_is_matched_and_the_conversation_kept():
    server, port, state, engine = start()
    try:
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": "how many apples are left"})
        messages = [SYSTEM, {"role": "user", "content": "how many apples are left"}]
        chat(port, messages, stream=True)
        assert state.counts["matched"] == 1
        sess = state.session("c1")
        assert sess.history[:2] == messages
        assert sess.history[2]["role"] == "assistant" and sess.history[2]["content"].startswith("echo:")
        # the next turn's draft goes after the answer
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": "and pears"})
        assert engine.last_messages == sess.history + [{"role": "user", "content": "and pears"}]
    finally:
        server.shutdown()


def test_text_typed_after_the_last_draft_still_matches():
    server, port, state, _ = start()
    try:
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": "how many apples are"})
        chat(port, [SYSTEM, {"role": "user", "content": "how many apples are left?"}])
        assert state.counts["matched"] == 1
    finally:
        server.shutdown()


def test_a_new_chat_hands_its_conversation_to_the_id_it_gets():
    server, port, state, engine = start()
    try:
        post(port, "/v1/drafts", {"conversation_id": "new", "text": "first question here"})
        messages = [SYSTEM, {"role": "user", "content": "first question here"}]
        chat(port, messages)
        post(port, "/v1/drafts", {"conversation_id": "abc123", "text": "second"})
        assert engine.last_messages[:2] == messages
        assert engine.last_messages[-1] == {"role": "user", "content": "second"}
        assert state.session("new").history is None        # the next new chat starts clean
    finally:
        server.shutdown()


def test_a_side_request_goes_to_the_other_slot_and_leaves_the_cache():
    server, port, state, engine = start(n_slots=2)
    try:
        post(port, "/v1/drafts", {"conversation_id": "c1", "text": "how many apples are left"})
        chat(port, [SYSTEM, {"role": "user", "content": "how many apples are left"}])
        assert engine.last_chat["id_slot"] == 0
        held = list(state.cache.ids)
        chat(port, [{"role": "user", "content": "### Task: write a title for this chat"}])
        assert engine.last_chat["id_slot"] == 1
        assert state.counts["side"] == 1
        assert state.cache.ids == held
    finally:
        server.shutdown()


def test_without_drafts_every_request_stays_in_the_conversation_slot():
    server, port, state, engine = start(n_slots=2)
    try:
        chat(port, [{"role": "user", "content": "a plain client, no drafts"}])
        assert engine.last_chat["id_slot"] == 0
        assert state.counts["unmatched"] == 1 and state.counts["side"] == 0
    finally:
        server.shutdown()


def test_a_draft_needs_text_or_messages():
    server, port, _, _ = start()
    try:
        assert post(port, "/v1/drafts", {"conversation_id": "c1"})[0] == 400
        assert post(port, "/v1/drafts", {"conversation_id": "c1", "text": "   "})[0] == 200
    finally:
        server.shutdown()


def test_drafts_carry_the_tools_of_the_last_request():
    # Open WebUI sends its built-in tools with every request, and the template writes them
    # at the start of the prompt: a draft without them would share almost nothing.
    server, port, state, engine = start()
    tools = [{"type": "function", "function": {"name": "get_time", "parameters": {}}}]
    try:
        post(port, "/v1/chat/completions", {"model": "fake-model", "tools": tools,
                                            "messages": [SYSTEM, {"role": "user", "content": "hello there"}]})
        post(port, "/v1/drafts", {"conversation_id": "new", "text": "what time is it"})
        assert engine.last_tools == tools
    finally:
        server.shutdown()
