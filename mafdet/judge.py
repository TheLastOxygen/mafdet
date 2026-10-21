"""
mafdet judge: verify that incremental prefill gives the same result as a
from-scratch computation, even after random edits to the text (appends,
backspaces, edits in the middle, replacements, pastes, words without spaces).

Each trial compares:
  - the logits of the first token (same argmax, maximum difference, no NaN/inf)
  - a greedy continuation of a few tokens

    mafdet judge
    mafdet judge --trials 50 --holdback 0
"""
import random

from .engine import LLAMA_TOL, Template, compare_logits, encode, make_engine, read_sample, setup, stable_ids


def random_edits(rng: random.Random, base: str, n_edits: int):
    """Return the sequence of text states after each edit."""
    text, states = "", []
    pool = base.split(" ")
    for _ in range(n_edits):
        op = rng.choices(["append", "backspace", "mid_insert", "mid_delete", "replace", "paste", "nospace"],
                         weights=[45, 15, 10, 8, 7, 5, 10])[0]
        if op == "append" or not text:
            text += (" " if text else "") + rng.choice(pool)
        elif op == "backspace":
            text = text[:-rng.randint(1, 6)]
        elif op == "mid_insert":
            i = rng.randint(0, len(text))
            text = text[:i] + rng.choice(pool) + " " + text[i:]
        elif op == "mid_delete":
            i = rng.randint(0, len(text))
            text = text[:i] + text[i + rng.randint(1, 10):]
        elif op == "replace":
            i = rng.randint(0, max(0, len(text) - 1))
            text = text[:i] + rng.choice("abcdefgh,.;!? ") + text[i + 1:]
        elif op == "paste":
            k = rng.randint(0, max(0, len(pool) - 20))
            text += " " + " ".join(pool[k:k + 20])
        elif op == "nospace":
            text += "".join(rng.choice(pool) for _ in range(3))      # words glued together
        # intermediate state: sometimes in the middle of a word
        if rng.random() < 0.3 and len(text) > 3:
            states.append(text[:-rng.randint(1, 3)])
        states.append(text)
    return states


def run(args) -> int:
    llama = args.backend == "llama"
    tol = args.tol if args.tol is not None else (LLAMA_TOL if llama else 0.1)
    metric = "shifted probability" if llama else "relative difference"

    tok, model, device, dtype = setup(args, n_slots=2)     # two engines, two caches
    tpl = Template.from_tokenizer(tok, thinking=False)
    base = " ".join(read_sample("documento.txt").split())
    rng = random.Random(args.seed)
    inc, ref = make_engine(model, device), make_engine(model, device)

    counts = {"ok": 0, "tie": 0, "ERROR": 0}
    worst_diff, greedy_div = 0.0, 0
    for trial in range(args.trials):
        states = random_edits(rng, base, args.edits)
        mismatches_before = getattr(inc, "reuse_mismatches", 0)
        inc.reset()
        for s in states:
            if rng.random() < 0.6:                 # not every state triggers a prefill
                inc.sync(stable_ids(encode(tok, tpl.prefix + s), args.holdback))
        final_ids = encode(tok, tpl.prefix + states[-1] + tpl.suffix)
        before = inc.stats.tokens_computed
        a = inc.sync(final_ids, need_logits=True)
        at_enter = inc.stats.tokens_computed - before
        ref.reset()
        b = ref.sync(final_ids, need_logits=True)
        cmp = compare_logits(a, b, tol=tol)
        verdict = "ERROR" if cmp["verdict"] == "NaN" else cmp["verdict"]
        # llama-server: did it reuse exactly the common prefix on every request?
        bad_reuse = getattr(inc, "reuse_mismatches", 0) - mismatches_before
        if bad_reuse:
            verdict = "ERROR"
        counts[verdict] += 1
        worst_diff = max(worst_diff, cmp["rel_diff"]) if cmp["finite"] else worst_diff

        ga, gb = inc.greedy(a, args.greedy), ref.greedy(b, args.greedy)
        div = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
        if div is not None:
            greedy_div += 1
        print(f"trial {trial + 1:>3}: {len(final_ids):>5} tokens, {len(states):>3} states, "
              f"recomputed on send {at_enter:>3}, "
              f"max diff {cmp['max_abs_diff']:.4f} ({metric} {cmp['rel_diff']:.4f}) → {verdict}"
              + (f" (cache reuse differed from expected {bad_reuse} times)" if bad_reuse else "")
              + ("" if div is None else f" (greedy diverges at token {div})"))

    print(f"\nResult: {counts['ok']} ok, {counts['tie']} ties, {counts['ERROR']} errors "
          f"in {args.trials} trials. Largest {metric}: {worst_diff:.4f} (threshold {tol})")
    print(f"Different greedy continuations: {greedy_div} of {args.trials}")
    if llama:
        print(f"Cache reuse checked on {inc.reuse_checks + ref.reuse_checks} requests: "
              f"{inc.reuse_mismatches + ref.reuse_mismatches} differed from the expected common prefix")
        print("\nWith llama-server the decisive check is reuse: on every request the server must reuse\n"
              "exactly the common prefix between its cache and the prompt. The shifted probability\n"
              "only catches gross errors: with 4-bit weights, computing the same text in different\n"
              "chunks changes the rounding, and the 'max diff' (in nats, over the top 10 candidates)\n"
              "can reach 0.5 without any error.")
    else:
        print("\nThe relative difference is the largest difference between the logits divided by their\n"
              "standard deviation. Small values are normal: the order of operations changes and fp16\n"
              "rounding adds up. A 'tie' is a different argmax with nearly identical logits. An ERROR\n"
              "means a bug in the cache, or NaN/inf in fp16.")
    return 1 if counts["ERROR"] else 0
