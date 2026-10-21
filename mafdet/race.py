"""
mafdet race: how much reading the prompt (prefill) and generating (decode)
cost on your machine.

Produces two tables:
  1. prefill from scratch for different prompt lengths, plus the cost of one
     decode token at that length;
  2. the cost of adding k new tokens to a cache that is already full: it shows
     that a forward pass has a fixed cost that barely depends on how many
     tokens it processes. That is why debouncing pays off.

    mafdet race
    mafdet race --device cpu --lengths 50 250 1000 2000
"""
import csv
import os
import statistics

import torch

from .engine import (Template, argmax_id, device_label, encode, make_engine, model_label, peak_mem_mb,
                     read_sample, setup, timed)


def filler_ids(tok, n: int) -> list[int]:
    base = encode(tok, "\n\n" + read_sample("documento.txt"))
    reps = n // len(base) + 1
    return (base * reps)[:n]


def run(args) -> int:
    tok, model, device, dtype = setup(args)
    gpu = device.endswith("cuda")
    lengths = args.lengths or ([50, 100, 250, 500, 1000, 2000, 4000, 8000] if gpu
                               else [50, 100, 250, 500, 1000, 2000])
    repeats = args.repeats or (3 if gpu else 2)
    tpl = Template.from_tokenizer(tok, thinking=False)
    pre, suf = encode(tok, tpl.prefix), encode(tok, tpl.suffix)
    eng = make_engine(model, device)
    dev_name, dt_name = device_label(device), str(dtype).replace("torch.", "")
    os.makedirs(args.out, exist_ok=True)
    print(f"{model_label(args)} on {dev_name} ({dt_name})\n")

    # Warm-up: the first runs pay for allocations and kernel selection.
    for _ in range(2):
        eng.reset()
        eng.sync(pre + filler_ids(tok, 64) + suf, need_logits=True)

    # ---- Table 1: prefill from scratch and decode ----------------------------
    rows = []
    print(f"{'tokens':>7} {'prefill ms':>11} {'tokens/s':>9} {'decode ms/tok':>14} {'peak MB':>9}")
    for n in lengths:
        ids = pre + filler_ids(tok, n) + suf
        try:
            times = []
            for _ in range(repeats):
                eng.reset()
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                logits, ms = timed(device, eng.sync, ids, need_logits=True)
                times.append(ms)
            # decode: one token at a time on top of the cache just filled
            steps = []
            for _ in range(8):
                t = argmax_id(logits)
                logits, ms = timed(device, eng.sync, eng.ids + [t], need_logits=True)
                steps.append(ms)
        except torch.cuda.OutOfMemoryError:
            print(f"{len(ids):>7}  out of GPU memory: stopping here")
            torch.cuda.empty_cache()
            break
        pf = statistics.median(times)
        dec = statistics.median(steps)
        row = dict(model=model_label(args), device=dev_name, dtype=dt_name, prompt_tokens=len(ids),
                   prefill_ms=round(pf, 2), prefill_tok_s=round(len(ids) / pf * 1000, 1),
                   decode_ms_per_token=round(dec, 2), peak_mem_mb=round(peak_mem_mb(device), 0))
        rows.append(row)
        mem = "-" if device != "cuda" else f"{row['peak_mem_mb']:.0f}"
        print(f"{len(ids):>7} {pf:>11.1f} {row['prefill_tok_s']:>9.0f} {dec:>14.1f} {mem:>9}")

    f1 = os.path.join(args.out, f"prefill_{device}.csv")
    with open(f1, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- Table 2: cost of k new tokens on a full cache -----------------------
    base = pre + filler_ids(tok, args.base_context)
    eng.reset()
    eng.sync(base)
    extra = filler_ids(tok, args.base_context + 256)[args.base_context:]
    rows2 = []
    print(f"\nAdding k tokens to a cache of {len(base)} tokens")
    print(f"{'k':>5} {'ms':>9} {'ms per token':>13}")
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        times = []
        for _ in range(max(repeats, 3)):
            # Bring the cache back to the base context. With need_logits the crop
            # really happens on llama-server too, which only crops when it gets a
            # request (and recomputes the last base token, outside the measurement).
            eng.sync(base, need_logits=True)
            _, ms = timed(device, eng.sync, base + extra[:k])
            times.append(ms)
        ms = statistics.median(times)
        rows2.append(dict(model=model_label(args), device=dev_name, dtype=dt_name, base_context=len(base),
                          new_tokens=k, ms=round(ms, 2), ms_per_token=round(ms / k, 3)))
        print(f"{k:>5} {ms:>9.1f} {ms / k:>13.2f}")

    f2 = os.path.join(args.out, f"incremental_cost_{device}.csv")
    with open(f2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows2[0]))
        w.writeheader()
        w.writerows(rows2)

    print(f"\nResults saved to:\n  {f1}\n  {f2}")
    print("\nHow to read them: if 'ms' in the second table barely grows from k=1 to k=16, a\n"
          "forward pass is dominated by a fixed cost. Prefilling on every keystroke wastes\n"
          "it; grouping tokens with a debounce does not.")
    return 0
