"""
What a GGUF file says about itself, read from its header: no download, no dependency.

A GGUF starts with key-value metadata (architecture, layers, chat template,
the tokenizer's vocabulary) and the list of its tensors (name, shape, type).
The catalog needs only a few of the values and the parameter count; the
vocabulary arrays are skipped, not decoded.

Format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
from __future__ import annotations

import mmap
import re
import struct
from math import prod
from pathlib import Path

MAGIC = b"GGUF"
# Value types: the struct format of each fixed-size one; 8 is a string, 9 an array.
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9
SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


class Reader:
    def __init__(self, buf):
        self.buf, self.at = buf, 0

    def scalar(self, fmt: str):
        v = struct.unpack_from("<" + fmt, self.buf, self.at)[0]
        self.at += struct.calcsize(fmt)
        return v

    def string(self) -> str:
        n = self.scalar("Q")
        self.at += n
        return bytes(self.buf[self.at - n:self.at]).decode("utf-8", "replace")

    def value(self, kind: int, keep_arrays: bool):
        if kind == STRING:
            return self.string()
        if kind != ARRAY:
            return self.scalar(SCALARS[kind])
        item, n = self.scalar("I"), self.scalar("Q")
        if keep_arrays and item != STRING:                     # small numeric arrays: per-layer head counts
            return [self.value(item, False) for _ in range(n)]
        if item == STRING:                                     # the vocabulary: skipped, only lengths read
            for _ in range(n):
                self.at += 8 + struct.unpack_from("<Q", self.buf, self.at)[0]
        elif item == ARRAY:
            for _ in range(n):
                self.value(ARRAY, False)
        else:
            self.at += n * struct.calcsize(SCALARS[item])
        return None


def read(path: str | Path) -> dict:
    """The metadata (strings and numbers; the vocabulary left out) and the tensors' parameter counts:
    {"meta": {...}, "params": n, "expert_params": n, "embd_params": n, "output": bool}. Expert tensors (mixture of
    experts) are counted apart: only some of them work for each token. The token embeddings stay on the
    CPU in llama.cpp, unless there is no output layer of its own (it shares them, and takes them to the GPU)."""
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        r = Reader(buf)
        if bytes(buf[:4]) != MAGIC:
            raise ValueError(f"{Path(path).name} is not a GGUF file")
        r.at = 4
        version = r.scalar("I")
        if version < 2:
            raise ValueError(f"{Path(path).name}: GGUF version {version} is too old")
        n_tensors, n_kv = r.scalar("Q"), r.scalar("Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            v = r.value(r.scalar("I"), keep_arrays=key.startswith(meta.get("general.architecture", "\0")))
            if v is not None:
                meta[key] = v
        params = experts = embd = 0
        output = False
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.scalar("Q") for _ in range(r.scalar("I"))]
            r.at += 4 + 8                                      # type, offset
            n = prod(dims)
            params += n
            if "_exps" in name:
                experts += n
            if name == "token_embd.weight":
                embd = n
            output = output or name == "output.weight"
    return {"meta": meta, "params": params, "expert_params": experts, "embd_params": embd, "output": output}


def shards(path: str | Path) -> list[Path]:
    """All the files of a model split in parts ('...-00001-of-00003.gguf'), or the file alone."""
    path = Path(path)
    m = SHARD.search(path.name)
    if not m:
        return [path]
    stem, total = path.name[:m.start()], int(m.group(2))
    return [path.with_name(f"{stem}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]


def is_first_part(name: str) -> bool:
    m = SHARD.search(name)
    return not m or int(m.group(1)) == 1


def is_projector(name: str) -> bool:
    """The vision encoder of a multimodal model ('mmproj-...gguf'): not a model to chat with."""
    return "mmproj" in name.lower()


QUANT = re.compile(r"(?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|[BM]?F(?:16|32)|MXFP4(?:_MOE)?|TQ\d_\d)", re.IGNORECASE)


def quant_label(name: str) -> str:
    """The quantization written in a file name: 'Qwen3-4B-UD-Q4_K_XL.gguf' → 'UD-Q4_K_XL'."""
    stem = SHARD.sub("", name).removesuffix(".gguf")
    found = [m for m in QUANT.finditer(stem)
             if (m.start() == 0 or stem[m.start() - 1] in "-_.") and (m.end() == len(stem) or stem[m.end()] in "-_.")]
    return found[-1].group(0).upper() if found else "GGUF"


def model_name(name: str) -> str:
    """A file name without its quantization and extension: 'Qwen3-4B-Q4_K_M.gguf' → 'Qwen3-4B'."""
    stem = SHARD.sub("", name).removesuffix(".gguf")
    q = quant_label(name)
    if q != "GGUF":
        i = stem.upper().rfind(q)
        stem = (stem[:i] + stem[i + len(q):]).strip("-_. ")
    return stem or name
