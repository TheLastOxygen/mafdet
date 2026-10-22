"""
mafdet doctor: is everything in place? GPU, memory, llama.cpp, models; with --install it puts
llama.cpp in place by itself.

    mafdet doctor
    mafdet doctor --install
"""
from __future__ import annotations

import time

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn

from .core import catalog, config, hardware, llama_setup
from .core.tasks import TaskManager


def run(args) -> int:
    console = Console()
    ok = "[green]✓[/]"
    warn = "[#D4892A]![/]"
    hw = hardware.detect()
    console.print(f"[bold]MAFDET doctor[/]   [dim]{config.home()}[/]\n")
    console.print(f"{ok} System     {hw.os} · {hw.arch}")
    console.print(f"{ok} Processor  {hw.cpu} · {hw.cores} cores, {hw.threads} threads")
    console.print(f"{ok} Memory     {hw.ram_mb / 1024:.1f} GB, {hw.ram_free_mb / 1024:.1f} GB free · "
                  f"~{hw.ram_bandwidth_gbs:.0f} GB/s")
    if hw.gpu:
        g = hw.gpu
        console.print(f"{ok} GPU        {g.name} · {g.vram_mb / 1024:.1f} GB VRAM ({g.vram_free_mb / 1024:.1f} GB free) · "
                      f"driver {g.driver} · {g.bandwidth_gbs:.0f} GB/s")
    else:
        console.print(f"{warn} GPU        no NVIDIA card found (nvidia-smi): models run on the CPU")

    if args.install:
        install(console)
    rt = llama_setup.status()
    if rt["installed"]:
        same = rt["build"] == rt["tested_build"]
        console.print(f"{ok if same else warn} llama.cpp  {rt['build'] or 'unknown build'} at {rt['path']}"
                      + ("" if same else f" [dim](MAFDET is tested with {rt['tested_build']})[/]"))
    else:
        console.print(f"{warn} llama.cpp  not installed: `mafdet doctor --install` installs {rt['recommended']}")

    listing = catalog.listing(hw)
    have = [e["id"] for e in listing["models"] if e["installed"]]
    if have:
        console.print(f"{ok} Models     {', '.join(have)}")
    else:
        console.print(f"{warn} Models     none downloaded yet")
    console.print(f"\n  Recommended for this machine: [bold #D4892A]{listing['recommended']}[/] · open the app with `mafdet`")
    return 0 if rt["installed"] else 1


def install(console: Console):
    task = TaskManager().start("llama", llama_setup.LLAMA_TAG, llama_setup.install)
    with Progress("{task.description}", BarColumn(), TaskProgressColumn(), console=console) as bar:
        row = bar.add_task("llama.cpp", total=1.0)
        while task.state == "running":
            bar.update(row, completed=task.progress, description=f"llama.cpp · {task.stage}")
            time.sleep(0.2)
        bar.update(row, completed=1.0)
    if task.state == "done":
        console.print(f"[green]✓[/] Installed llama.cpp {task.result['build']} ({task.result['label']})")
    else:
        console.print(f"[red]llama.cpp: {task.error or task.state}[/]")
