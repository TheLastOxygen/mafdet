"""The persisted record of what one engine, at one address, did when probed.

A profile is data, not a code branch: the proxy reads the spelling and the reuse
channel out of it and behaves accordingly.  Known engines just ship with a
pre-measured profile; an unknown engine gets probed once and produces the same
kind of record.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .engine import EngineCaps

# Bump when the meaning of a field changes; older files are then re-probed instead
# of silently misread.
FORMAT = 1


def _home() -> Path:
    """Same rule as the app: $MAFDET_HOME or ~/.mafdet (see mafdet/core/config.py)."""
    path = Path(os.environ.get("MAFDET_HOME") or Path.home() / ".mafdet")
    path.mkdir(parents=True, exist_ok=True)
    return path


def profiles_dir() -> Path:
    path = _home() / "proxy"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class EngineProfile:
    name: str
    url: str
    model: str
    caps: EngineCaps
    probed_at: float = field(default_factory=time.time)
    latencies: dict = field(default_factory=dict)   # {"warm_ms": .., "probe_ms": ..}
    version: str = ""                                # label only, never gates behavior
    format: int = FORMAT

    # ---- tier shortcuts ---------------------------------------------------------
    @property
    def tier(self) -> str:
        return self.caps.tier

    @property
    def spelling(self) -> str:
        return self.caps.spelling

    @property
    def channel(self) -> str:
        return self.caps.reuse_channel

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tier"] = self.tier
        return data

    # ---- storage ----------------------------------------------------------------
    def path(self) -> Path:
        safe = f"{self.url.replace('://', '_').replace(':', '_').replace('/', '_')}__{self.model}"
        return profiles_dir() / (safe + ".json")

    def save(self) -> Path:
        path = self.path()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, ensure_ascii=False)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, url: str, model: str) -> "EngineProfile | None":
        path = cls(name="", url=url, model=model, caps=EngineCaps()).path()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get("format") != FORMAT:
            return None
        caps = EngineCaps(**data.get("caps", {}))
        return cls(name=data.get("name", ""), url=data.get("url", url), model=data.get("model", model),
                   caps=caps, probed_at=data.get("probed_at", 0.0),
                   latencies=data.get("latencies", {}), version=data.get("version", ""))
