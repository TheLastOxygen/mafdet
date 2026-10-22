"""Which llama.cpp build each machine gets, and downloads that resume where they stopped."""
import http.server
import threading

import pytest

from mafdet.core import downloads, llama_setup
from mafdet.core.hardware import GPU, Hardware
from mafdet.core.tasks import Cancelled, Task


def machine(system="windows", arch="amd64", gpu_name=None, driver="617.42"):
    gpus = [GPU(gpu_name, 8192, 0, driver, 400, True)] if gpu_name else []
    return Hardware(os=system, system=system, arch=arch, cpu="cpu", cores=8, threads=16, ram_mb=32000,
                    ram_free_mb=20000, ram_bandwidth_gbs=40, gpus=gpus)


@pytest.mark.parametrize("hw,flavor,first", [
    (machine(gpu_name="NVIDIA GeForce RTX 2060"), "cuda-13.4", "llama-{tag}-bin-win-cuda-13.4-x64.zip"),
    (machine(gpu_name="NVIDIA GeForce RTX 2060", driver="552.22"), "cuda-12.4", "llama-{tag}-bin-win-cuda-12.4-x64.zip"),
    (machine(gpu_name="NVIDIA GeForce GTX 1080"), "cuda-12.4", "llama-{tag}-bin-win-cuda-12.4-x64.zip"),
    (machine(), "cpu", "llama-{tag}-bin-win-cpu-x64.zip"),
    (machine("linux", "x86_64", "NVIDIA GeForce RTX 4090"), "cuda-13.4", "llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz"),
    (machine("linux", "x86_64", "NVIDIA GeForce RTX 3060", driver="550.1"), "cuda-12.8",
     "llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"),
    (machine("linux", "x86_64"), "cpu", "llama-{tag}-bin-ubuntu-x64.tar.gz"),
    (machine("darwin", "arm64"), "metal", "llama-{tag}-bin-macos-arm64.tar.gz"),
])
def test_build_for(hw, flavor, first):
    b = llama_setup.build_for(hw)
    assert b["flavor"] == flavor and b["assets"][0] == first
    assert len(b["assets"]) == (2 if flavor.startswith("cuda") else 1)      # CUDA brings its runtime


DATA = bytes(range(256)) * 4096          # 1 MB


class RangeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = DATA[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/file"
    srv.shutdown()


def test_fetch_resumes(server, tmp_path):
    dest = tmp_path / "model.gguf"
    part = tmp_path / "model.gguf.part"
    part.write_bytes(DATA[:300_000])                     # an earlier, interrupted download
    task = Task("1", "download", "x")
    downloads.fetch(server, dest, task, size=len(DATA))
    assert dest.read_bytes() == DATA and not part.exists()
    assert task.done_bytes == len(DATA) and task.progress == 1.0


def test_fetch_cancel_keeps_part(server, tmp_path):
    dest = tmp_path / "model.gguf"
    task = Task("1", "download", "x")
    task._cancel.set()
    with pytest.raises(Cancelled):
        downloads.fetch(server, dest, task, size=len(DATA))
    assert not dest.exists() and (tmp_path / "model.gguf.part").exists()
