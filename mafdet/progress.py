"""
The long setup jobs in the terminal: downloading a model, installing llama.cpp. A line for each
step (which file, how big, from where) and a bar with size, speed and time left.

The commands follow their one job until it ends (follow); the window, whose terminal stays
free behind it, shows there every download and install started from it (watch).
"""
from __future__ import annotations

import threading
import time

from rich.console import Console
from rich.progress import (BarColumn, DownloadColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeRemainingColumn, TransferSpeedColumn)

from .core.tasks import Task, TaskManager

KINDS = ("download", "llama")
STAGES = {"release": "looking for the release on GitHub", "downloading": "downloading",
          "extracting": "unpacking", "checking": "checking that it starts"}


def title(task: Task) -> str:
    return f"llama.cpp {task.target}" if task.kind == "llama" else task.target


def outcome(task: Task) -> str:
    if task.state == "done":
        r = task.result
        if task.kind == "llama":
            return f"[green]✓[/] llama.cpp {r.get('build', task.target)} installed ({r.get('label', '')})"
        return f"[green]✓[/] {task.target} on disk: {r.get('path', '')}"
    if task.state == "cancelled":
        return f"[#D4892A]![/] {title(task)}: cancelled"
    return f"[red]✗ {title(task)}: {task.error or task.state}[/]"


class Follower:
    """Rows of a rich Progress for some tasks; step() brings them up to date."""

    def __init__(self, console: Console):
        self.bar = Progress(TextColumn("[bold]{task.description}"), BarColumn(), TaskProgressColumn(),
                            DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn(), console=console)
        self.rows: dict[str, list] = {}               # task id -> [task, row, last (stage, detail)]

    def add(self, task: Task):
        if task.id not in self.rows:
            self.rows[task.id] = [task, self.bar.add_task(title(task), total=None), None]

    def step(self):
        for key, entry in list(self.rows.items()):
            task, row, last = entry
            now = (task.stage, task.detail)
            if task.stage and now != last:
                entry[2] = now
                self.bar.console.print(f"[bold]{title(task)}[/] · {STAGES.get(task.stage, task.stage)}"
                                       + (f" [dim]· {task.detail}[/]" if task.detail else ""))
            self.bar.update(row, completed=task.done_bytes, total=task.total_bytes or None)
            if task.state != "running":
                self.bar.remove_task(row)
                self.bar.console.print(outcome(task))
                del self.rows[key]


def follow(task: Task, console: Console) -> bool:
    """Show one job until it ends; True if it went well."""
    f = Follower(console)
    f.add(task)
    with f.bar:
        while f.rows:
            f.step()
            time.sleep(0.2)
    return task.state == "done"


def watch(tasks: TaskManager, console: Console | None = None) -> threading.Event:
    """Show in this terminal, from a thread, the downloads and installs `tasks` runs; set the event to stop."""
    console = console or Console()
    stop = threading.Event()

    def loop():
        f = None
        while not stop.wait(0.25):
            running = [t for k in KINDS for t in tasks.running(k)]
            if running and f is None:
                f = Follower(console)
                f.bar.start()
            for t in running:
                f.add(t)
            if f is not None:
                f.step()
                if not f.rows:
                    f.bar.stop()
                    f = None
        if f is not None:
            f.bar.stop()

    threading.Thread(target=loop, daemon=True, name="mafdet-progress").start()
    return stop
