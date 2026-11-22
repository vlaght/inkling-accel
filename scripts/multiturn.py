#!/usr/bin/env python3
"""A coding session of 4 turns on one server (the history grows, llama.cpp reuses the cached prefix): decode speed, draft accept rate and the number of
prompt tokens actually processed per turn. A drafter that is fine on the first turn can break on a later one.

    multiturn.py '[["name", "extra llama-server args", {"ENV": "value"}], ...]'
"""
import json, sys
import lib

TURNS = ["Write a Python class RateLimiter with a token bucket, thread-safe, with type hints and docstrings.",
         "Now add a method that returns how many seconds until the next token is available, and update the docstring.",
         "Write pytest tests for the whole class, including the new method and a thread-safety test.",
         "Summarize in 5 bullet points what could still go wrong in production with this limiter."]

for name, args, env in json.loads(sys.argv[1]):
    rows, msgs = [], []
    try:
        with lib.Server(args, env):
            lib.chat([{"role": "user", "content": "Hi"}], 8)
            for i, q in enumerate(TURNS, 1):
                msgs.append({"role": "user", "content": q})
                m, t = lib.chat(msgs, 300)
                msgs.append({"role": "assistant", "content": m.get("content") or ""})
                rows.append(f"turn {i}: {t['predicted_per_second']:.1f} tok/s decode, {t['prompt_n']} new prompt tokens, accept {t.get('draft_n_accepted')}/{t.get('draft_n')}")
    except Exception as e:
        rows.append(f"ERROR {e}")
    print(f"{name}\n  " + "\n  ".join(rows), flush=True)
