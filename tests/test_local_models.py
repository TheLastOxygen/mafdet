"""Models of your own: GGUF headers, files on disk, the Hugging Face search, downloads that keep a token home."""
import http.server
import struct
import threading
from types import SimpleNamespace

import pytest

from mafdet.core import catalog, config, downloads, gguf, hub
from mafdet.core.hardware import GPU, Hardware
from mafdet.core.tasks import Task
from mafdet.llama_backend import SERVER_TOKENIZER, tokenizer_id

QWEN3_TEMPLATE = "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>{%- endif %}"


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    if isinstance(value, str):
        return _str(key) + struct.pack("<I", 8) + _str(value)
    if isinstance(value, list) and value and isinstance(value[0], str):     # a vocabulary
        return _str(key) + struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_str(v) for v in value)
    if isinstance(value, list):
        return _str(key) + struct.pack("<IIQ", 9, 4, len(value)) + struct.pack(f"<{len(value)}I", *value)
    return _str(key) + struct.pack("<II", 4, value)


def write_gguf(path, arch="llama", meta=None, tensors=None):
    """A GGUF header (no tensor data: the reader never looks at it)."""
    meta = {"general.architecture": arch, f"{arch}.block_count": 2, f"{arch}.attention.head_count": 4,
            f"{arch}.attention.head_count_kv": 2, f"{arch}.embedding_length": 64,
            "tokenizer.ggml.tokens": ["<s>", "a", "b"], **(meta or {})}
    tensors = tensors or {"token_embd.weight": (64, 3), "blk.0.attn_q.weight": (64, 64)}
    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    body += b"".join(_kv(k, v) for k, v in meta.items())
    for name, dims in tensors.items():
        body += _str(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body + b"\0" * 64)
    return path


def machine():
    return Hardware(os="windows", system="windows", arch="amd64", cpu="cpu", cores=8, threads=16, ram_mb=32000,
                    ram_free_mb=20000, ram_bandwidth_gbs=40, gpus=[GPU("NVIDIA GeForce RTX 2060", 6144, 0, "1", 336, True)])


# ---- the header -----------------------------------------------------------------------

def test_read_header(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", meta={"general.name": "Tiny", "tokenizer.chat_template": "{{ x }}"})
    info = gguf.read(p)
    assert info["meta"]["general.name"] == "Tiny" and info["meta"]["llama.block_count"] == 2
    assert "tokenizer.ggml.tokens" not in info["meta"]                   # the vocabulary is skipped
    assert info["params"] == 64 * 3 + 64 * 64 and info["expert_params"] == 0


def test_embeddings_on_the_cpu_unless_shared(tmp_path):
    """llama.cpp keeps token_embd in RAM when the model has an output layer of its own."""
    shared = gguf.read(write_gguf(tmp_path / "tied.gguf"))
    assert shared["embd_params"] == 64 * 3 and not shared["output"]
    own = gguf.read(write_gguf(tmp_path / "own.gguf", tensors={"token_embd.weight": (64, 3), "output.weight": (64, 3),
                                                                "blk.0.attn_q.weight": (64, 64)}))
    assert own["embd_params"] == 64 * 3 and own["output"]
    assert catalog.local_model(tmp_path / "own.gguf", {**own, "bytes": 1})["cpu_embd_b"] == 64 * 3 / 1e9
    assert catalog.local_model(tmp_path / "tied.gguf", {**shared, "bytes": 1})["cpu_embd_b"] == 0


def test_read_mixture_of_experts(tmp_path):
    p = write_gguf(tmp_path / "moe.gguf", "qwen3moe", tensors={"blk.0.ffn_up_exps.weight": (64, 32, 8),
                                                                 "blk.0.attn_q.weight": (64, 64)})
    info = gguf.read(p)
    assert info["expert_params"] == 64 * 32 * 8 and info["params"] == 64 * 32 * 8 + 64 * 64


def test_not_a_gguf(tmp_path):
    (tmp_path / "x.gguf").write_bytes(b"nope" * 10)
    with pytest.raises(ValueError):
        gguf.read(tmp_path / "x.gguf")


@pytest.mark.parametrize("name,quant,model", [
    ("Qwen3-4B-Q4_K_M.gguf", "Q4_K_M", "Qwen3-4B"),
    ("Qwen3-4B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL", "Qwen3-4B"),
    ("Qwen3-4B-BF16.gguf", "BF16", "Qwen3-4B"),
    ("gpt-oss-20b-MXFP4.gguf", "MXFP4", "gpt-oss-20b"),
    ("Phi-4-mini-instruct.Q8_0.gguf", "Q8_0", "Phi-4-mini-instruct"),
    ("qwen3-4b-q4_k_m.gguf", "Q4_K_M", "qwen3-4b"),
    ("Qwen3-30B-A3B-IQ4_XS.gguf", "IQ4_XS", "Qwen3-30B-A3B"),
    ("Llama-3.3-70B-Instruct-Q4_K_M-00001-of-00002.gguf", "Q4_K_M", "Llama-3.3-70B-Instruct"),
    ("model.gguf", "GGUF", "model"),
])
def test_names(name, quant, model):
    assert gguf.quant_label(name) == quant and gguf.model_name(name) == model


def test_parts(tmp_path):
    first = tmp_path / "M-Q4_K_M-00001-of-00003.gguf"
    assert [p.name for p in gguf.shards(first)][-1] == "M-Q4_K_M-00003-of-00003.gguf"
    assert gguf.is_first_part(first.name) and not gguf.is_first_part("M-Q4_K_M-00002-of-00003.gguf")
    assert gguf.is_projector("mmproj-F16.gguf") and not gguf.is_projector(first.name)


# ---- your models in the catalog -------------------------------------------------------------

def test_files_of_your_own(tmp_path):
    root = config.models_dir()
    write_gguf(root / "Mine-Q4_K_M.gguf")
    write_gguf(root / "Mine-Q8_0.gguf")                                    # the same model: one card, two buttons
    write_gguf(root / "hf" / "someone" / "Thinker-GGUF" / "Thinker-Q4_K_M.gguf", "qwen3",
               meta={"tokenizer.chat_template": QWEN3_TEMPLATE})
    write_gguf(root / "mmproj-F16.gguf")                                   # a vision projector: not a model
    write_gguf(root / "Split-Q4_K_M-00001-of-00002.gguf")                  # its second part is missing
    write_gguf(root / "Qwen3-4B-Q4_K_M.gguf")                              # a catalog file: the catalog's card
    own = {v.file.rsplit("/", 1)[-1]: v for v in catalog.local_variants()}
    assert set(own) == {"Mine-Q4_K_M.gguf", "Mine-Q8_0.gguf", "Thinker-Q4_K_M.gguf"}
    assert own["Mine-Q4_K_M.gguf"].model["id"] == own["Mine-Q8_0.gguf"].model["id"]
    thinker = own["Thinker-Q4_K_M.gguf"].model
    assert thinker["reason_ahead"] and thinker["repo"] == "someone/Thinker-GGUF" and thinker["vendor"] == "someone/Thinker-GGUF"
    assert not own["Mine-Q4_K_M.gguf"].model["reason_ahead"]                # a plain chat model
    assert own["Mine-Q4_K_M.gguf"].model["tokenizer"] == SERVER_TOKENIZER
    v = own["Mine-Q8_0.gguf"]
    assert catalog.find(v.id).file == v.file and catalog.local_path(v) == v.file

    listing = catalog.listing(machine())
    assert listing["families"][-1]["id"] == "local"
    mine = [e for e in listing["models"] if e["local"]]
    assert len(mine) == 3 and all(e["installed"] and e["tier"] == "custom" for e in mine)
    assert not listing["recommended"].startswith(catalog.LOCAL_PREFIX)    # the pick stays in the catalog


def test_split_model_with_all_parts(tmp_path):
    root = config.models_dir()
    for i in (1, 2):
        write_gguf(root / f"Big-Q4_K_M-0000{i}-of-00002.gguf")
    (v,) = catalog.local_variants()
    assert v.bytes == sum((root / f"Big-Q4_K_M-0000{i}-of-00002.gguf").stat().st_size for i in (1, 2))
    assert v.model["params_b"] == pytest.approx(2 * (64 * 3 + 64 * 64) / 1e9)


def test_add_a_file_from_elsewhere(tmp_path):
    p = write_gguf(tmp_path / "elsewhere" / "Far-Q5_K_M.gguf")
    v = catalog.add_local(str(p))
    assert config.load()["local_models"] == [str(p)] and catalog.find(v.id).file == str(p)
    assert catalog.add_local(str(p)).id == v.id and config.load()["local_models"] == [str(p)]   # not twice
    catalog.remove_local(v.id)
    assert config.load()["local_models"] == [] and p.exists()             # forgotten, not deleted
    with pytest.raises(ValueError):
        catalog.add_local(str(tmp_path / "missing.gguf"))


def test_a_local_file_brings_its_own_tokenizer(tmp_path):
    p = write_gguf(tmp_path / "m.gguf")
    assert tokenizer_id(str(p)) == SERVER_TOKENIZER
    assert tokenizer_id("Qwen/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf") == "Qwen/Qwen3-4B"


# ---- Hugging Face ---------------------------------------------------------------------------

def test_files_of_a_repository(monkeypatch):
    sibling = lambda name, size: SimpleNamespace(rfilename=name, size=size)
    info = SimpleNamespace(
        gguf={"total": 30_500_000_000, "architecture": "qwen3moe", "chat_template": QWEN3_TEMPLATE},
        siblings=[sibling("Qwen3-30B-A3B-Q4_K_M.gguf", 18 * 2**30), sibling("mmproj-F16.gguf", 2**30),
                  sibling("BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf", 30 * 2**30),
                  sibling("BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf", 30 * 2**30),
                  sibling("Q8_0/Qwen3-30B-A3B-Q8_0-00001-of-00002.gguf", 16 * 2**30),     # a part is missing
                  sibling("README.md", 100)],
        gated=False, card_data=SimpleNamespace(license="apache-2.0"))
    monkeypatch.setattr(hub, "_api", lambda: SimpleNamespace(model_info=lambda repo, files_metadata: info))
    r = hub.files("unsloth/Qwen3-30B-A3B-GGUF", machine())
    assert [f["quant"] for f in r["files"]] == ["Q4_K_M", "BF16"]       # the suggested one first, then by size
    assert r["files"][0]["suggested"] and not r["files"][1]["suggested"]
    assert r["files"][1]["parts"] == ["BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf",
                                      "BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf"]
    assert r["active_b"] == 3.0 and r["reason_ahead"] and r["license"] == "apache-2.0"
    assert r["files"][1]["placement"] == "too_big" and not r["files"][0]["installed"]


class RedirectHandler(http.server.BaseHTTPRequestHandler):
    seen: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        host = self.headers.get("Host", "").split(":")[0]
        RedirectHandler.seen[host] = self.headers.get("Authorization")
        if host == "127.0.0.1":                    # the Hub sends the file elsewhere
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.server.server_address[1]}/blob")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", "5")
        self.end_headers()
        self.wfile.write(b"hello")


def test_token_stays_home(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/file"
        downloads.fetch(url, tmp_path / "f.gguf", Task("1", "download", "x"), size=5,
                        headers={"Authorization": "Bearer secret"})
    finally:
        srv.shutdown()
    assert (tmp_path / "f.gguf").read_bytes() == b"hello"
    assert RedirectHandler.seen == {"127.0.0.1": "Bearer secret", "localhost": None}
