"""
mafdet models: the catalog in the terminal, with what each model does on this machine.

    mafdet models                        # the recommended quantization of each model
    mafdet models --all                  # every quantization
    mafdet models --download qwen3-4b    # download one (its default quantization, or qwen3-4b:Q8_0)
"""
from __future__ import annotations

import time

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table

from .core import catalog, hardware
from .core.downloads import download_model
from .core.tasks import TaskManager
from .identity import AMBER

WHERE = {"gpu": ("All on GPU", "green"), "partial": ("{p}% on GPU", AMBER), "cpu": ("On CPU", "white"),
         "too_big": ("Too large", "red")}


def dur(s):
    return "–" if s is None else f"{s:.1f} s" if s < 10 else f"{s:.0f} s" if s < 60 else f"{s / 60:.1f} min"


def run(args) -> int:
    console = Console()
    if args.download:
        return download(console, args.download)
    hw = hardware.detect()
    listing = catalog.listing(hw, args.device)
    best = listing["recommended"]
    where = (f"{hw.gpu.name} ({hw.gpu.vram_mb / 1024:.0f} GB)" if hw.gpu else "CPU only") + \
        f", {hw.ram_mb / 1024:.0f} GB RAM"
    table = Table(title=f"MAFDET models for {where}", title_style="bold", caption="estimated, or ✓ measured on this "
                  "machine · download with `mafdet models --download <id>`", caption_style="dim", header_style="dim")
    for col, kw in (("", {}), ("id", {}), ("runs", {}), ("writes", {"justify": "right"}),
                    ("answer after Enter", {"justify": "right"}), ("without MAFDET", {"justify": "right"}),
                    ("size", {"justify": "right"}), ("", {})):
        table.add_column(col, **kw)
    for e in listing["models"]:
        if not args.all and not e["default"]:
            continue
        label, colour = WHERE[e["placement"]]
        table.add_row("[bold #D4892A]★[/]" if e["id"] == best else "", f"[bold]{e['id']}[/]" if e["id"] == best else e["id"],
                      f"[{colour}]{label.format(p=round(e['gpu_share'] * 100))}[/]",
                      f"{e['decode_tps']:.0f} tok/s" + (" [green]✓[/]" if e["measured"] else ""),
                      f"[#D4892A]≈ {dur(e['with_mafdet_s'])}[/]", f"[dim]{dur(e['without_s'])}[/]",
                      f"{e['bytes'] / 2**30:.1f} GB", "[green]downloaded[/]" if e["installed"] else "")
    console.print(table)
    if best:
        console.print(f"\nRecommended here: [bold #D4892A]{best}[/]. Open the app with `mafdet` to download and use it.")
    return 0


def download(console: Console, variant_id: str) -> int:
    v = catalog.find(variant_id)
    if v is None:
        console.print(f"[red]Unknown model {variant_id}[/]: see `mafdet models --all`.")
        return 1
    task = TaskManager().start("download", v.id, lambda t: download_model(v.id, t))
    columns = ("{task.description}", BarColumn(), DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn())
    with Progress(*columns, console=console) as bar:
        row = bar.add_task(v.id, total=v.bytes)
        while task.state == "running":
            bar.update(row, completed=task.done_bytes)
            time.sleep(0.2)
        bar.update(row, completed=task.done_bytes if task.state != "done" else v.bytes)
    if task.state != "done":
        console.print(f"[red]{task.error or task.state}[/]")
        return 1
    console.print(f"[green]Downloaded[/] {v.id} to {task.result['path']}")
    return 0
