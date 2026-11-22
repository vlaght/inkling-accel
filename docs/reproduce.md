# Reference: settings, measuring, serving

The build steps are in the README ("Apply"). This page covers what comes after.

## Toolchain notes

Tested on Fedora 44 with NVIDIA driver 615 inside a build container, CUDA 13.4 from NVIDIA's repository, gcc 15 as the host compiler (CUDA 13 does not accept gcc 16).
Pick a toolkit not newer than the CUDA version your driver reports (`nvidia-smi`). On Fedora:

```
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo
sudo dnf install cuda-toolkit-13-4 gcc15 gcc15-c++
```

## Script settings

All scripts read the environment (defaults in `scripts/lib.py`):

| Variable | Meaning | Default |
|---|---|---|
| `ROOT` | patched llama.cpp tree | `~/llama.cpp` |
| `BUILD` | build directory inside it | `build-cuda` |
| `MODEL`, `DRAFT` | target and drafter GGUF | `~/models/inkling-2-mini/Inkling-2-Mini-UD-IQ3_XXS.gguf`, `.../dspark-Inkling-2-Mini-Q8_0.gguf` |
| `CUDA` | CUDA toolkit | `/usr/local/cuda` |
| `DBX` | container to run the binaries in (empty runs on the host) | empty |
| `WORK` | logs, KL baselines, profiles | `~/.cache/inkling-accel` |
| `PORT` | port of the throw-away test servers | `8082` |
| `NSYS` | Nsight Systems binary (the standalone one) | `nsys` |

## Scripts

| Script | What it does |
|---|---|
| `validate.py all` or one step: `bench`, `kld`, `launches`, `spec`, `multiturn` | the full check as a markdown report |
| `spec-bench.py '[["name","server args",{"ENV":"v"}],...]'` | decode speed and draft accept rate over 8 prompts |
| `ctx-bench.py '[["name","server args"],...]'` | does a context / micro-batch setting start, and its decode and 6k-prompt speed |
| `multiturn.py '[["name","server args",{}],...]'` | a 4-turn session: speed and accept rate per turn |
| `serve.sh` | the server launcher (README step 6) |

`lib.SPEC` in `scripts/lib.py` holds the DSpark flags used in the sweeps (2 drafted tokens, 4-bit KV cache for target and drafter).

## Measuring by hand

- Speed: `llama-bench -m <model> -ngl 99 -fa 1 -p 512 -n 128 -r 3`, with `OOT2_NOFUSE=1` for the baseline.
- Quality (KL divergence): save the unfused distribution once, then compare a fusion mask against it:
  ```
  OOT2_NOFUSE=1 llama-perplexity -m <model> -f <text> -ngl 99 -c 512 -b 512 -ub 512 --chunks 12 --kl-divergence-base base.kld
  OOT2_NOFUSE_MASK=<bits> llama-perplexity ... --kl-divergence-base base.kld --kl-divergence
  ```
  Judge the result against the engine's own noise floor: the same unfused model with `-b 256 -ub 256` gives KLD 0.0044 against the 512 run.
  Use `-b 1 -ub 1` for the decode path, which runs fusions a 512-token batch never reaches.
- Launches per token: `nsys profile --cuda-graph-trace=node` on `llama-bench -p 0 -n 24` and on `-n 72`, then `nsys stats --report cuda_gpu_kern_sum`;
  launches per token = (instances of the long run minus the short run) / 48. Use the standalone Nsight install, not the wrapper inside the CUDA toolkit.

## Serving options

`serve.sh` reads `ROOT`, `BUILD`, `MODEL`, `DRAFT`, `CUDA` as above, plus `HOST` (default `0.0.0.0`), `PORT` (default `8080`), `API_KEY_FILE` (adds `--api-key`),
`CTX` (default 65536), `UB` (default 32, the micro-batch) and `NSPEC` (default 2, drafted tokens per step). About 15 GB of VRAM at the defaults.
The compute buffer grows with `CTX` x `UB`: `CTX=131072 UB=8` fits, `CTX=163840` does not.
