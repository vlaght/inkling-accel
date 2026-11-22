#!/usr/bin/env python3
"""Context and micro-batch sweep: does a configuration start, how fast does it decode and how fast does it read a 6k-token prompt.

    ctx-bench.py '[["name", "extra llama-server args"], ...]'     e.g. "-c 131072 -ub 8 -ctk q4_0 -ctv q4_0"
"""
import json, subprocess, sys
import lib

LONG = "This is a test paragraph about the history of computing. " * 400

for name, args in json.loads(sys.argv[1]):
    try:
        with lib.Server(args):
            _, short = lib.chat([{"role": "user", "content": "Write a 200 word story about cats."}], 200)
            _, long_ = lib.chat([{"role": "user", "content": LONG + "\nHow many times does the word 'paragraph' appear? Answer with a number."}], 30)
            vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print(f"{name} | decode {short['predicted_per_second']:.1f} tok/s | 6k prompt {long_['prompt_per_second']:.0f} tok/s, then {long_['predicted_per_second']:.1f} tok/s | VRAM {vram}", flush=True)
    except RuntimeError as e:
        print(name, "FAILED:", e)
