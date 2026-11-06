"""The llama.cpp adapter: one `http.client` connection to a llama-server the user already runs.

This talks to llama-server's native routes, the same ones the app's `LlamaPrefill` uses:
`/tokenize` for exact ids, `/completion` with `n_predict: 0` for a prefill-only request
(the cache is warm when it returns), and `timings.cache_n` for the exact number of
tokens reused.  `/v1/chat/completions` is passed through untouched, so the answer to a
real request is whatever the engine would have said anyway.
"""
from __future__ import annotations

import http.client
import json
import time
from typing import Iterable

from ..engine import WarmResult


class LlamaEngine:
    name = "llama.cpp"
    dialect = "llama"
    flavor = "llama"
    version = ""          # a label from detection (/props build_info), never a code branch

    def __init__(self, url: str, *, slot: int = 0, timeout: float = 600.0):
        if "://" in url:
            url = url.split("://", 1)[1]
        host, _, port = url.partition(":")
        self.host = host or "127.0.0.1"
        self.port = int(port or 8080)
        self.slot = slot
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"LlamaEngine({self.host}:{self.port}, slot {self.slot})"

    # ---- plumbing ---------------------------------------------------------------
    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _call(self, path: str, body: dict | None = None, method: str = "POST") -> tuple[int, dict, bytes]:
        conn = self._conn()
        try:
            data = json.dumps(body).encode() if body is not None else None
            headers = {"Content-Type": "application/json"} if data else {}
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            try:
                parsed = json.loads(raw.decode())
            except (ValueError, UnicodeDecodeError):
                parsed = {}
            return resp.status, parsed, raw
        finally:
            conn.close()

    # ---- Engine -------------------------------------------------------------
    def tokenize(self, text: str) -> list[int] | None:
        status, data, _ = self._call("/tokenize", {"content": text, "add_special": False})
        if status != 200:
            return None
        tokens = data.get("tokens")
        return list(tokens) if isinstance(tokens, list) else None

    def render(self, messages: list[dict], template_kwargs: dict | None = None,
               tools: list | None = None) -> list[int] | None:
        """The prompt llama-server builds for a chat request: its own template, with the
        request's `chat_template_kwargs` (e.g. Qwen3's enable_thinking) and `tools` (the
        template writes them into the system turn, at the start), then tokens."""
        body = {"messages": messages, "add_generation_prompt": True}
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        if tools:
            body["tools"] = tools
        status, data, _ = self._call("/apply-template", body)
        if status != 200 or not data.get("prompt"):
            return None
        # The template's text is already in the model's vocabulary; no BOS is added twice.
        return self.tokenize(data["prompt"])

    def warm(self, ids: list[int], *, spelling: str = "auto") -> WarmResult:
        """One prefill request.  `n_predict: 0` is exact (no token is generated);
        `max_tokens1` asks for one token and throws it away -- the universal fallback."""
        if not ids:
            return WarmResult(channel="none", detail={"ok": False, "why": "empty prompt"})
        body = {"prompt": ids, "id_slot": self.slot, "cache_prompt": True, "temperature": 0.0}
        if spelling in ("auto", "n_predict0"):
            body["n_predict"] = 0
            t0 = time.perf_counter()
            status, data, raw = self._call("/completion", body)
            ms = (time.perf_counter() - t0) * 1000
            if status == 200:
                timings = data.get("timings") or {}
                return WarmResult(cached_tokens=timings.get("cache_n"),
                                  computed_tokens=timings.get("prompt_n"),
                                  ms=ms, channel="timings",
                                  detail={"ok": True, "spelling": "n_predict0"})
            if spelling == "n_predict0":
                return WarmResult(ms=ms, channel="none",
                                  detail={"ok": False, "status": status, "why": raw[:200].decode(errors="replace")})
        # Fallback: one token, discarded.
        body["n_predict"] = 1
        t0 = time.perf_counter()
        status, data, raw = self._call("/completion", body)
        ms = (time.perf_counter() - t0) * 1000
        if status != 200:
            return WarmResult(ms=ms, channel="none",
                              detail={"ok": False, "status": status, "why": raw[:200].decode(errors="replace")})
        timings = data.get("timings") or {}
        return WarmResult(cached_tokens=timings.get("cache_n"), computed_tokens=timings.get("prompt_n"),
                          ms=ms, channel="timings", detail={"ok": True, "spelling": "max_tokens1"})

    def chat(self, body: dict, *, stream: bool = False,
             timeout: float = 600.0) -> tuple[int, dict[str, str], Iterable[bytes]]:
        payload = dict(body)
        payload["stream"] = bool(stream)
        data = json.dumps(payload).encode()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        conn.request("POST", "/v1/chat/completions", body=data,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        headers = {k.lower(): v for k, v in resp.getheaders()}
        if not stream:
            raw = resp.read()
            conn.close()
            return resp.status, headers, [raw]

        def chunks() -> Iterable[bytes]:
            try:
                while True:
                    # Line at a time: a 1024-byte read() blocks until the buffer fills,
                    # which would add latency to every streamed answer.
                    line = resp.readline()
                    if not line:
                        break
                    yield line
            finally:
                conn.close()

        return resp.status, headers, chunks()

    def ready(self) -> bool:
        """llama-server answers /health with ok (not while it loads or downloads the model)."""
        try:
            status, data, _ = self._call("/health", method="GET")
        except OSError:
            return False
        return status == 200 and data.get("status") == "ok"

    def slots(self) -> int | None:
        """How many slots the server has (`/props` total_slots), or None if it does not say."""
        try:
            status, data, _ = self._call("/props", method="GET")
        except OSError:
            return None
        n = data.get("total_slots") if status == 200 else None
        return int(n) if isinstance(n, int) and n > 0 else None

    # ---- thinking ahead: raw generation on token ids -------------------------------
    def thinks(self) -> bool:
        """Does the model's chat template have a reasoning block (Qwen3's <think>)?"""
        try:
            status, data, _ = self._call("/props", method="GET")
        except OSError:
            return False
        return status == 200 and "<think>" in str(data.get("chat_template") or "")

    def detokenize(self, ids: list[int]) -> str:
        if not ids:
            return ""
        status, data, _ = self._call("/detokenize", {"tokens": ids})
        return data.get("content", "") if status == 200 else ""

    def generate(self, ids: list[int], *, n_predict: int, sampling: dict | None = None,
                 logit_bias: list | None = None, abort=None) -> dict:
        """Up to n_predict tokens after `ids`: {"tokens", "text", "stop_type", "timings", "aborted"}.

        Streamed, so that `abort` (a threading.Event) stops it between two tokens: closing the
        connection stops llama-server too.  The tokens written until then are returned."""
        tokens, text, last = [], "", {}
        sampling = dict(sampling or {})
        if logit_bias:
            sampling["logit_bias"] = logit_bias
        aborted = False
        events = self.stream(ids, n_predict=n_predict, sampling=sampling)
        try:
            for ev in events:
                if ev.get("error"):
                    return {"tokens": [], "text": "", "error": ev["error"]}
                tokens += ev.get("tokens") or []
                text += ev.get("text", "")
                last = ev
                if abort is not None and abort.is_set() and not ev.get("stop"):
                    aborted = True
                    break
        finally:
            events.close()
        return {"tokens": tokens, "text": text, "stop_type": "abort" if aborted else last.get("stop_type"),
                "timings": last.get("timings") or {}, "aborted": aborted}

    def stream(self, ids: list[int], *, n_predict: int, sampling: dict | None = None,
               timeout: float = 600.0) -> Iterable[dict]:
        """Generate after `ids`, one event per token: {"text", "tokens", "stop", ...}.  Closing the
        generator closes the connection, which stops llama-server."""
        body = {"prompt": ids, "id_slot": self.slot, "cache_prompt": True, "n_predict": n_predict,
                "stream": True, "return_tokens": True, **(sampling or {})}
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request("POST", "/completion", body=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                yield {"error": resp.read()[:200].decode(errors="replace"), "stop": True}
                return
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                d = json.loads(line[5:].decode())
                yield {"text": d.get("content", ""), "tokens": d.get("tokens") or [], "stop": bool(d.get("stop")),
                       "stop_type": d.get("stop_type"), "timings": d.get("timings") or {},
                       "tokens_evaluated": d.get("tokens_evaluated")}
                if d.get("stop"):
                    break
        finally:
            conn.close()

    def models(self) -> tuple[int, dict]:
        status, data, _ = self._call("/v1/models", method="GET")
        return status, data

    def metrics_text(self) -> str | None:
        status, _, raw = self._call("/metrics", method="GET")
        return raw.decode(errors="replace") if status == 200 else None
