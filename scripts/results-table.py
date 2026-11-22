#!/usr/bin/env python3
"""The README results table, every row measured with the same server arguments, the same 8 prompts and the same 6k-token prompt.

    results-table.py

Decode: mean tok/s over the 8 greedy 256-token answers. Prompt: tok/s on a fresh 6k-token prompt. Percentages are against the CUDA row with the fusions off. Common arguments: -c 32768 -ub 32 -b 32, 4-bit KV cache.
"""
import lib

COMMON = "-c 32768 -ub 32 -b 32 -ctk q4_0 -ctv q4_0"
ROWS = [("CUDA, stock behaviour (fusions off)", COMMON, {"OOT2_NOFUSE": "1"}, {}),
        ("CUDA + patches", COMMON, {}, {}),
        ("CUDA + patches + DSpark", f"-c 32768 -ub 32 -b 32 {lib.SPEC}", {}, {})]

print("| row | decode tok/s (8-prompt mean) | prompt tok/s (6k prompt) |\n|---|---|---|")
res = []
for name, args, env, kw in ROWS:
    try:
        with lib.Server(args, env, **kw):
            lib.chat([{"role": "user", "content": "Hi"}], 8)
            _, t = lib.chat([{"role": "user", "content": lib.LONG + "\nHow many times does the word 'paragraph' appear? Answer with a number."}], 8)
            pp = t["prompt_per_second"]
            rows = [lib.chat([{"role": "user", "content": q}], 256)[1] for _, q in lib.PROMPTS]
        toks = sum(t["predicted_n"] for t in rows)
        res.append((name, toks / sum(t["predicted_n"] / t["predicted_per_second"] for t in rows), pp))
    except RuntimeError as e:
        print(f"| {name} | FAILED: {e} | |", flush=True)
base = next(r for r in res if "fusions off" in r[0])   # the percentages are against the unpatched CUDA row
for name, d, p in res:
    pct = lambda x, b: "" if x == b else f" ({x / b * 100 - 100:+.1f} %)"
    print(f"| {name} | {d:.1f}{pct(d, base[1])} | {p:.0f}{pct(p, base[2])} |")
