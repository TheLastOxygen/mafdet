"""
ChatService: the one object both interfaces talk to.

It owns the chat Session (mafdet/watch.py, unchanged in behaviour), the
settings, the catalog and the background jobs. The window reaches it through
the local HTTP API (mafdet/gui.py), the terminal app calls it directly.

States, in the order a first start goes through them:
    no-runtime   llama.cpp is not installed
    no-model     no model chosen, or the chosen one is not downloaded
    loading      llama-server is starting and loading the model
    ready        chatting
    error        loading failed: error_code says why, error the details
"""
from __future__ import annotations

import dataclasses
import socket
import threading
import time

from . import bench, catalog, config, downloads, hardware, llama_setup
from ..llama_backend import find_llama_server
from .tasks import TaskManager

# Settings that need the model loaded again when they change.
RELOAD_KEYS = {"think", "ahead", "think_ahead", "max_tokens", "ctx", "device", "recycle", "gatherer"}

ERROR_CODES = [                                   # (text in the message, code the interfaces translate)
    ("Cannot find llama-server", "llama_missing"),
    ("already listening on port", "port_busy"),
    ("exited right away", "server_exited"),
    ("not responding after", "server_timeout"),
    ("no NVIDIA card", "no_gpu"),
    ("stopped answering", "server_stopped"),
]


def error_code(message: str) -> str:
    return next((code for text, code in ERROR_CODES if text in message), "load_failed")


def free_port() -> int:
    """A port nobody listens on: a llama-server left over from a crash cannot block the chat."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ChatService:
    def __init__(self, overrides=None):
        """`overrides`: the parsed options of `mafdet watch/strike` (legacy commands), used as they are."""
        self.settings = config.load()
        self.tasks = TaskManager()
        self.session = None
        self.variant_id = ""
        self.overrides = overrides
        self._lock = threading.RLock()
        # The VRAM other programs use, measured before any model of ours is loaded: the catalog
        # estimates what fits in the card, and our own model must not count against itself.
        gpu = hardware.detect(refresh=True).gpu
        self._others_vram_mb = gpu.vram_used_mb if gpu else 0

    # ---- what the interfaces show ------------------------------------------------

    def hardware(self, refresh: bool = False) -> dict:
        hw = hardware.detect(refresh)
        if refresh and self.session is None and hw.gpu:
            self._others_vram_mb = hw.gpu.vram_used_mb
        return hw.to_dict()

    def machine(self) -> hardware.Hardware:
        """The machine as the estimates see it: the VRAM taken by our own model counts as free."""
        hw = hardware.detect()
        if hw.gpu is None:
            return hw
        gpus = [dataclasses.replace(g, vram_used_mb=self._others_vram_mb) if g is hw.gpu else g for g in hw.gpus]
        return dataclasses.replace(hw, gpus=gpus)

    def catalog(self) -> dict:
        hw = self.machine()
        out = catalog.listing(hw, self.settings["device"], self.settings["ctx"])
        out["vram_others_mb"] = hw.gpu.vram_used_mb if hw.gpu else 0
        out["loaded"] = self.variant_id if self.session is not None else ""
        return out

    def runtime(self) -> dict:
        return llama_setup.status()

    def status(self) -> dict:
        with self._lock:
            s = self.session
            if s is None:
                # Only a file check here: this runs several times a second (llama_setup.status() starts the server).
                if find_llama_server(required=False) is None:
                    return {"state": "no-runtime"}
                return {"state": "no-model", "model_id": self.settings["model"]}
            st = s.status()
        st["model_id"] = self.variant_id
        if st["state"] == "error":
            st["error_code"] = error_code(st.get("error", ""))
        return st

    # ---- model -------------------------------------------------------------------

    def _args(self, v: catalog.Variant, path: str):
        """The options of `mafdet strike`, filled from the settings for this model."""
        from ..cli import build_parser
        st = self.settings
        argv = ["strike", "--gguf", path, "--model", v.model["tokenizer"], "--ctx", str(st["ctx"]),
                "--device", st["device"], "--ahead", str(st["ahead"]), "--think-ahead", str(st["think_ahead"]),
                "--max-tokens", str(st["max_tokens"])]
        if not st["think"] or not v.model["reason_ahead"]:
            argv.append("--no-think")              # only a measured frame reasons ahead (Qwen3 today)
        if v.model.get("llama_args"):
            argv.append("--llama-args=" + " ".join(v.model["llama_args"]))
        if st["recycle"]:
            argv.append("--recycle")
        if st["gatherer"] and v.model["reason_ahead"] and st["think"]:   # material for a model that reasons
            g = catalog.find(st["gatherer"])
            gpath = catalog.local_path(g) if g else st["gatherer"]
            if gpath:
                argv += ["--gatherer", gpath]
        args = build_parser().parse_args(argv)
        args.port = free_port()
        args.tokenizer_kwargs = v.model.get("tokenizer_kwargs", {})
        return args

    def load(self, variant_id: str = "") -> dict:
        """Load a model (default: the one in the settings). The previous one is closed first."""
        from ..watch import Session
        with self._lock:
            self.unload()
            if self.overrides is not None and not variant_id:
                args = self.overrides
                args.port = getattr(args, "port", 0) or free_port()
                self.variant_id = ""
                self.session = Session(args)
                return self.status()
            variant_id = variant_id or self.settings["model"]
            v = catalog.find(variant_id) if variant_id else None
            if v is None:
                return self.status()
            path = catalog.local_path(v)
            if path is None:
                self.settings = config.save({"model": v.id})
                return {"state": "no-model", "model_id": v.id, "missing": True}
            self.settings = config.save({"model": v.id})
            self.variant_id = v.id
            self.session = Session(self._args(v, path))
            return self.status()

    def unload(self):
        with self._lock:
            if self.session is not None:
                self.session.close()
            self.session = None

    def autoload(self):
        """At start: load the model of the settings, if llama.cpp and the file are there."""
        if self.overrides is not None or (self.settings["model"] and find_llama_server(required=False)):
            threading.Thread(target=self.load, daemon=True).start()

    # ---- chat --------------------------------------------------------------------

    def _ready(self):
        s = self.session
        return s if s is not None and s.state == "ready" else None

    def draft(self, text: str) -> dict:
        s = self._ready()
        if s is not None:
            s.draft(text)
        return self.status()

    def send(self, text: str):
        """The answer as events: think, token, done, error (see Session.send)."""
        s = self._ready()
        if s is None:
            yield {"type": "error", "message": "The model is not ready yet.", "code": "not_ready"}
            return
        yield from s.send(text)

    def stop(self):
        s = self.session
        if s is not None:
            s.stop.set()

    def new_chat(self):
        s = self._ready()
        if s is not None:
            s.new_chat()

    # ---- settings ----------------------------------------------------------------

    def update_settings(self, changes: dict) -> dict:
        before = dict(self.settings)
        self.settings = config.save(changes)
        changed = {k for k in RELOAD_KEYS if before.get(k) != self.settings.get(k)}
        if changed and self.session is not None:
            threading.Thread(target=self.load, args=(self.variant_id,), daemon=True).start()
        return self.settings

    # ---- background jobs -----------------------------------------------------------

    def download(self, variant_id: str) -> dict:
        task = self.tasks.start("download", variant_id, lambda t: downloads.download_model(variant_id, t))
        return task.to_dict()

    def install_runtime(self) -> dict:
        return self.tasks.start("llama", llama_setup.LLAMA_TAG, llama_setup.install).to_dict()

    def benchmark(self, variant_id: str) -> dict:
        """Measure a model. The chat's model is set aside meanwhile: two models on one GPU would
        measure the sharing, not the model."""
        def job(task):
            loaded = self.variant_id if self.session is not None else ""
            self.unload()
            time.sleep(0.5)                        # let the driver give the VRAM back
            try:
                return bench.run(variant_id, self.settings["device"], self.settings["ctx"],
                                 progress=lambda stage, done: setattr(task, "stage", stage) or
                                 setattr(task, "progress", done))
            finally:
                if loaded:
                    self.load(loaded)
        return self.tasks.start("bench", variant_id, job).to_dict()

    def cancel(self, task_id: str):
        self.tasks.cancel(task_id)

    def jobs(self) -> list[dict]:
        return self.tasks.snapshot()

    def close(self):
        self.unload()
