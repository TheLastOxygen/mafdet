"""
llama.cpp backend: the same incremental prefill, run by llama-server.

llama-server keeps one KV cache per slot. On every request it compares the new
prompt with the tokens already in cache, keeps the common prefix, drops the
rest and computes only the new tokens: the same logic as
IncrementalPrefill.sync(), but in C++, with fused CUDA kernels, CUDA graphs
and quantized weights.

LlamaPrefill has the same interface as IncrementalPrefill, so the commands
(simulated typing, hold-back, debounce, measurements) stay identical and the
numbers comparable. Statistics come from the server's `timings`, not from
estimates.
"""
from __future__ import annotations

import atexit
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .engine import DEFAULT_MODEL, Stats, common_prefix_len

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PROBS = 20                        # candidates returned for the comparison with a from-scratch run

# Draft recycling (mafdet strike): n-gram speculative decoding whose n-gram pool
# outlives the request. The answers written at earlier pauses, for earlier
# versions of the message, feed the pool; the final answer reuses their phrases
# wherever the model agrees with them, several tokens per decode step. Lookup
# of 4 tokens, drafts of 1 to 8: the best of the settings measured on Qwen3-4B
# with the chat sampling (temperature 0.7).
RECYCLE_ARGS = ("--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "4",
                "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", "8")


def find_llama_server() -> str:
    """llama-server from $MAFDET_LLAMA_DIR, ./llama.cpp or <project>/llama.cpp, in this order."""
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    dirs = [os.environ.get("MAFDET_LLAMA_DIR", ""), os.path.join(os.getcwd(), "llama.cpp"),
            os.path.join(PROJECT_ROOT, "llama.cpp")]
    for d in filter(None, dirs):
        exe = os.path.join(d, name)
        if os.path.exists(exe):
            return exe
    raise SystemExit(f"Cannot find {name}. Extract a llama.cpp release into ./llama.cpp, "
                     f"or set MAFDET_LLAMA_DIR. Looked in: {', '.join(filter(None, dirs))}")


def resolve_gguf(spec: str) -> str:
    """A local path, or 'repo:file' to download from (or find in the cache of) Hugging Face."""
    if os.path.exists(spec):
        return spec
    repo, fname = spec.split(":", 1)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname)


def tokenizer_id(gguf: str) -> str:
    """The matching HF tokenizer: 'Qwen/Qwen3-4B-GGUF:...' → 'Qwen/Qwen3-4B'."""
    repo = gguf.split(":", 1)[0]
    if repo.endswith("-GGUF"):
        return repo[:-len("-GGUF")]
    raise SystemExit("With a local GGUF file, pass the tokenizer with --model (e.g. Qwen/Qwen3-4B).")


class LlamaServer:
    """Starts llama-server, waits until it is ready and shuts it down on exit."""

    def __init__(self, gguf: str, device: str, ctx: int = 8192, n_slots: int = 1, port: int = 8089,
                 extra_args: tuple[str, ...] = ()):
        self.url = f"http://127.0.0.1:{port}"
        self.port = port
        self._conn: http.client.HTTPConnection | None = None
        self.device = device
        self.n_slots = n_slots
        self.next_slot = 0
        if self._alive():
            raise SystemExit(f"Something is already listening on port {port}: close it first "
                             "(e.g. Stop-Process -Name llama-server).")
        exe = find_llama_server()
        self.model_path = resolve_gguf(gguf)
        args = [exe, "-m", self.model_path,
                "-c", str(ctx * n_slots), "-np", str(n_slots),   # each slot gets its own cache of `ctx` tokens
                "-fa", "on",
                # No copies of prompts in system RAM: they could restore a cache
                # and spoil the from-scratch measurements.
                "--cache-ram", "0",
                # Reusing shifted chunks (KV shifting) is not exact after an edit in
                # the middle of the text: the following tokens depend on the changed ones.
                "--cache-reuse", "0",
                "--host", "127.0.0.1", "--port", str(port)]
        args += ["-ngl", "all"] if device == "cuda" else ["-ngl", "0", "-dev", "none"]
        args += list(extra_args)
        self.log_path = os.path.join(os.path.dirname(exe), "server.log")
        self._log = open(self.log_path, "w", encoding="utf-8")
        # No console window of its own when started from a windowed app (mafdet watch); the output goes to the log.
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(args, stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags)
        atexit.register(self.close)
        t0 = time.perf_counter()
        while not self._alive():
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server exited right away: see {self.log_path}")
            if time.perf_counter() - t0 > 600:
                raise SystemExit(f"llama-server is not responding after 10 minutes: see {self.log_path}")
            time.sleep(0.5)
        print(f"llama-server ready in {time.perf_counter() - t0:.0f} s", flush=True)

    def _alive(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                return json.loads(r.read()).get("status") == "ok"
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            return False

    def call(self, path: str, body: dict) -> dict:
        # Persistent connection: opening a new one for every request added a
        # variable latency of 1 to 16 ms on Windows (measured). Reusing it keeps
        # the overhead below 1 ms.
        for attempt in range(2):
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3600)
                self._conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                resp = self._conn.getresponse()
                data = resp.read()
            except (http.client.HTTPException, OSError):
                self._conn = None                 # the server closed the connection: open a new one
                if attempt:
                    raise
                continue
            if resp.status >= 400:
                raise RuntimeError(f"llama-server answered {resp.status}: {data[:300]!r}")
            return json.loads(data)

    def take_slot(self) -> int:
        if self.next_slot >= self.n_slots:
            raise RuntimeError(f"More than {self.n_slots} slots needed: start the server with a higher n_slots.")
        self.next_slot += 1
        return self.next_slot - 1

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=30)
        self._log.close()


@dataclass
class TopLogprobs:
    """The "logits" returned by the server: log-probabilities of the top candidates."""
    logprobs: dict[int, float]


@dataclass
class LlamaPrefill:
    """Keeps the cache of one llama-server slot aligned with a changing sequence.

    The server does the common-prefix check and the crop itself; this class
    only decides when to send it a request.
    """
    server: LlamaServer
    slot: int = 0
    ids: list[int] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self):
        self._server_tokens: list[int] = []     # what the slot's cache holds
        self._fresh = True
        # Exact reuse check: on every request the server must reuse the common
        # prefix between its cache and the prompt (minus the last token, which it
        # always recomputes). Integers, no noise: any difference is a bug.
        self.reuse_checks = 0
        self.reuse_mismatches = 0
        self.last_timings: dict = {}            # timings of the last request (cache_n, prompt_n, prompt_ms)

    @property
    def cached_tokens(self) -> list[int]:
        """The tokens the slot's cache holds right now (according to the server)."""
        return self._server_tokens

    def assume_cached(self, tokens: list[int]):
        """After a request made outside this class (a streamed answer), the slot's cache holds `tokens`.

        The next sync then reuses them instead of starting from scratch: in a
        conversation the history stays in cache from one turn to the next.
        """
        self._server_tokens = list(tokens)
        self.ids = list(tokens)
        self._fresh = False

    def reset(self):
        # A slot cannot be erased without --slot-save-path: the next request
        # ignores the cache (cache_prompt=false) and computes everything from scratch.
        self.ids = []
        self.stats = Stats()
        self._fresh = True

    def _request(self, prompt: list[int], **kw) -> dict:
        body = {"prompt": prompt, "id_slot": self.slot, "cache_prompt": not self._fresh, **kw}
        expected = 0 if self._fresh else min(common_prefix_len(self._server_tokens, prompt), len(prompt) - 1)
        r = self.server.call("/completion", body)
        t = r["timings"]
        self.last_timings = t
        self.reuse_checks += 1
        if t["cache_n"] != expected:
            self.reuse_mismatches += 1
        self.stats.forward_passes += 1
        self.stats.tokens_computed += t["prompt_n"]
        if not self._fresh:
            self.stats.tokens_discarded += max(0, len(self._server_tokens) - t["cache_n"])
        # The cache keeps the prompt plus the generated tokens already evaluated (not the last one).
        self._server_tokens = (list(prompt) + list(r.get("tokens") or []))[:r["tokens_cached"]]
        self._fresh = False
        return r

    def sync(self, target_ids: list[int], need_logits: bool = False) -> TopLogprobs | None:
        """Align the slot's cache with `target_ids`.

        The server always recomputes at least the last token, because it needs
        its logits. That is why a shorter text (backspace) sends no request at
        all: the server will crop on the next one, at no extra cost.
        """
        self.stats.syncs += 1
        target = list(target_ids)
        if not need_logits and target == self.ids[:len(target)]:
            self.ids = target
            return None
        if not target:
            raise ValueError("Empty sequence")
        if not need_logits:
            self._request(target, n_predict=0)
            self.ids = target
            return None
        r = self._request(target, n_predict=1, n_probs=N_PROBS, temperature=0)
        self.ids = target
        top = r["completion_probabilities"][0]["top_logprobs"]
        return TopLogprobs({p["id"]: p["logprob"] for p in top})

    def greedy(self, first_logits: TopLogprobs, n_tokens: int) -> list[int]:
        """Generate n_tokens greedily (temperature 0) after `ids`."""
        r = self._request(self.ids, n_predict=n_tokens, temperature=0, return_tokens=True)
        out = list(r["tokens"])
        self.ids = self.ids + out
        return out


def setup_llama(args, device: str, n_slots: int = 1):
    from transformers import AutoTokenizer

    tok_id = tokenizer_id(args.gguf) if args.model == DEFAULT_MODEL else args.model
    tok = AutoTokenizer.from_pretrained(tok_id)
    extra = (RECYCLE_ARGS if getattr(args, "recycle", False) else ()) + tuple(args.llama_args.split())
    server = LlamaServer(args.gguf, device, ctx=args.ctx, n_slots=n_slots, extra_args=extra)
    fname = os.path.basename(server.model_path)
    if not fname.startswith(tok_id.split("/")[-1]):
        print(f"Warning: tokenizer {tok_id} may not match {fname}", file=sys.stderr)
    quant = fname.removesuffix(".gguf").rsplit("-", 1)[-1]          # e.g. Q4_K_M
    return tok, server, f"llama-{device}", quant
