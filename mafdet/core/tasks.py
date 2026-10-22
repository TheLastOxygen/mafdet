"""
Long jobs in the background (downloads, installing llama.cpp, benchmarks), with progress the
interfaces poll: the window every few hundred milliseconds, the terminal app on a timer.
"""
from __future__ import annotations

import itertools
import threading
import time
import traceback
from dataclasses import dataclass, field, fields
from typing import Callable


class Cancelled(Exception):
    pass


@dataclass
class Task:
    id: str
    kind: str                         # "download", "llama", "bench"
    target: str                       # what it works on: a model id, a llama.cpp build
    state: str = "running"            # running, done, error, cancelled
    stage: str = ""                   # a step within the job, for the interface to name
    progress: float = 0.0             # 0..1
    done_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0
    eta_s: float | None = None
    error: str = ""
    result: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    ended: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _rate: list = field(default_factory=list, repr=False)

    def check(self):
        """Called by the job between steps: stops it if the user cancelled."""
        if self._cancel.is_set():
            raise Cancelled()

    def bytes(self, done: int, total: int):
        """Byte progress, with a speed over the last few seconds and the time left."""
        now = time.perf_counter()
        self._rate.append((now, done))
        while len(self._rate) > 2 and now - self._rate[0][0] > 4:
            self._rate.pop(0)
        t0, d0 = self._rate[0]
        self.speed_bps = (done - d0) / (now - t0) if now > t0 else 0.0
        self.done_bytes, self.total_bytes = done, total
        self.progress = done / total if total else 0.0
        self.eta_s = (total - done) / self.speed_bps if self.speed_bps > 0 and total else None
        self.check()

    def to_dict(self) -> dict:
        # Not asdict(): it would deep-copy the cancel event, and a lock cannot be copied.
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}


class TaskManager:
    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def start(self, kind: str, target: str, job: Callable[[Task], dict | None]) -> Task:
        """Run job(task) in a thread; one job per kind and target at a time."""
        with self._lock:
            for t in self._tasks.values():
                if t.kind == kind and t.target == target and t.state == "running":
                    return t
            task = Task(id=str(next(self._ids)), kind=kind, target=target)
            self._tasks[task.id] = task

        def body():
            try:
                task.result = job(task) or {}
                task.progress, task.state = 1.0, "done"
            except Cancelled:
                task.state = "cancelled"
            except BaseException as e:            # SystemExit carries messages meant for the user
                task.error = str(e) or type(e).__name__
                task.state = "error"
                traceback.print_exc()
            finally:
                task.ended = time.time()
        threading.Thread(target=body, daemon=True, name=f"mafdet-{kind}-{target}").start()
        return task

    def cancel(self, task_id: str):
        task = self._tasks.get(task_id)
        if task:
            task._cancel.set()

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def running(self, kind: str | None = None) -> list[Task]:
        return [t for t in self._tasks.values() if t.state == "running" and (kind is None or t.kind == kind)]

    def snapshot(self, recent_s: float = 30) -> list[dict]:
        """Running jobs, and the ones that ended in the last `recent_s` seconds (for the final state)."""
        now = time.time()
        with self._lock:
            tasks = list(self._tasks.values())
        return [t.to_dict() for t in tasks if t.ended is None or now - t.ended < recent_s]
