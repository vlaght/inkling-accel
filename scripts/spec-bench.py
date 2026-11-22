#!/usr/bin/env python3
"""Decode speed and draft accept rate over 8 prompts, one throw-away server per configuration.

    spec-bench.py '[["name", "extra llama-server args", {"ENV": "value"}], ...]'

Prints the mean tok/s over the prompts, the accepted/drafted counts and the per-prompt speeds. Example args: lib.SPEC (DSpark with 2 drafts, 4-bit KV).
"""
import json, sys
import lib

PROMPTS = [("code", "Write a Python implementation of an LRU cache class with get and put methods, with docstrings and type hints."),
           ("prose", "Tell me in detail about the history of the Roman Empire, starting with the founding of Rome."),
           ("json", "Generate a JSON array of 15 objects with fields id, name, email, city, age. Output only JSON."),
           ("sql", "Write a PostgreSQL schema for a blog with users, posts, comments and tags, with indexes and foreign keys."),
           ("refactor", "Refactor this into clean TypeScript with types: function f(a,b){var r=[];for(var i=0;i<a.length;i++){if(a[i]>b){r.push(a[i]*2)}}return r}"),
           ("docs", "Write a README section explaining how to install, configure and run a small command line tool called todo-cli."),
           ("math", "Explain step by step how to solve the quadratic equation 3x^2 - 7x + 2 = 0 and verify the roots."),
           ("rust", "Write a Rust function that parses a CSV line into a Vec<String>, handling quoted fields, with unit tests.")]

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
