#!/usr/bin/env python3
"""Decode speed and draft accept rate over 8 prompts, one throw-away server per configuration.

    spec-bench.py '[["name", "extra llama-server args", {"ENV": "value"}], ...]'

Prints the mean tok/s over the prompts, the accepted/drafted counts and the per-prompt speeds. Example args: lib.SPEC (DSpark with 2 drafts, 4-bit KV).
"""
import json, sys
import lib

PROMPTS = lib.PROMPTS

for name, args, env in json.loads(sys.argv[1]):
    try:
        with lib.Server(args, env):
            lib.chat([{"role": "user", "content": "Hi"}], 8)   # warm up
            rows = [(k, *lib.chat([{"role": "user", "content": q}])) for k, q in PROMPTS]
    except RuntimeError as e:
        print(name, "FAILED:", e)
        continue
    toks = sum(t["predicted_n"] for _, _, t in rows)
    secs = sum(t["predicted_n"] / t["predicted_per_second"] for _, _, t in rows)
    acc = sum(t.get("draft_n_accepted") or 0 for _, _, t in rows)
    drafted = sum(t.get("draft_n") or 0 for _, _, t in rows)
    print(name, "| mean", round(toks / secs, 1), "tok/s |", "accept", f"{acc}/{drafted}" if drafted else "-", "|",
          " ".join(f"{k}:{t['predicted_per_second']:.1f}" for k, _, t in rows), flush=True)
