# inkling-accel

> **Vibecoded.** Written and tested by an AI coding agent while a human steered it, ran the hardware and reviewed the results. All numbers come from one machine
> (below). Read the patches and `docs/experiments.md` before you rely on any of it.

Faster single-stream decoding of **Inkling 2 Mini** (Thinking Machines, 26B MoE, 3.4B active) on one 16 GB NVIDIA GPU with
[llama.cpp](https://github.com/ggml-org/llama.cpp): CUDA kernel fusions (`patches/`), DSpark speculative decoding, and scripts that validate both.
It works because decoding is bound by kernel launch count (about 7,300 small kernels per token), and the patches remove 46 % of them.

## Result

RTX 4070 Ti SUPER 16 GB (Ada), CUDA 13.4, model `Inkling-2-Mini-UD-IQ3_XXS`, one stream, greedy. Every row was measured the same way: `llama-server` with
`-c 32768 -ub 32 -b 32 -ctk q4_0 -ctv q4_0` (the last row adds the DSpark drafter, 2 drafted tokens), the same 8 prompts of 256 tokens, and the same fresh 6k-token prompt.
Reproduce with `scripts/results-table.py`.

| | decode tok/s (8-prompt mean) | prompt tok/s (6k prompt) |
|---|---|---|
| llama.cpp Vulkan | 50.5 (-22.7 %) | 361 (-30.3 %) |
| llama.cpp CUDA (patches switched off), the baseline | 65.3 | 518 |
| CUDA + patches | **82.5** (+26.3 %) | **568** (+9.7 %) |
| CUDA + patches + DSpark | **101.7** (+55.7 %) | 545 (+5.2 %) |

Percentages are against the CUDA baseline row.

Peak decode with `llama-bench` (`-ub 512`, f16 KV, no server) is 88.0 tok/s patched against 67.4 unpatched; with a larger micro-batch the server reads prompts faster too
(688 tok/s at `-ub 128`, `docs/experiments.md` 5.7). The small micro-batch is what lets a 65k to 131k context fit.

Quality: KL divergence against the unfused run is 0.0049, the same as the engine's own noise (0.0044 to 0.0047 when only the micro-batch changes).

## Apply

Applies to **Inkling 2 Mini only**, through the CUDA backend of llama.cpp. Needs: an NVIDIA GPU with at least 16 GB, a driver, the CUDA toolkit (13.x),
a host compiler the toolkit accepts (for CUDA 13 use gcc 15 or older), git, cmake, Python 3.9+.

1. **Get llama.cpp at the base commit** (the DSpark branch of PR #32773, head `61031e7`):
   ```
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   git fetch origin pull/32773/head:pr32773 && git checkout pr32773
   ```
2. **Apply the patches** from this repository, in order:
   ```
   git apply /path/to/inkling-accel/patches/0001-*.patch /path/to/inkling-accel/patches/0002-*.patch
   ```
3. **Build**, setting the architecture to your GPU's compute capability (Ada 4070/4080/4090: `89`, Ampere 3090: `86`; see `nvidia-smi --query-gpu=compute_cap --format=csv`):
   ```
   cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build-cuda -j --target llama-server llama-bench llama-perplexity
   ```
   If the compiler is rejected, add `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15` (path to a supported g++).
4. **Download the models** from Unsloth's `Inkling-2-Mini-GGUF`. Use the files under `Mainline_Support/` (suffix `.gguf_file`), not the files in the repository root:
   ```
   B=https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main
   curl -L -o Inkling-2-Mini-UD-IQ3_XXS.gguf $B/Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file
   curl -L -o dspark-Inkling-2-Mini-Q8_0.gguf $B/dspark-Inkling-2-Mini-Q8_0.gguf
   ```
5. **Verify** with the GPU otherwise idle. Set the variables to your paths, then run:
   ```
   export ROOT=/path/to/llama.cpp MODEL=/path/to/Inkling-2-Mini-UD-IQ3_XXS.gguf DRAFT=/path/to/dspark-Inkling-2-Mini-Q8_0.gguf CUDA=/usr/local/cuda
   python3 /path/to/inkling-accel/scripts/validate.py bench
   ```
   Expected: a table where "all fusions on" shows decode about +25 % and prompt about +15 % over "all fusions off". If it does, the patches work.
   `validate.py all` also checks quality (KL divergence), kernel launches per token, DSpark and a multi-turn session (about 30 minutes).
6. **Serve** (OpenAI-compatible API with tool calling, DSpark on, 65,536-token context):
   ```
   HOST=127.0.0.1 PORT=8080 sh /path/to/inkling-accel/scripts/serve.sh
   ```
   Use `CTX=131072 UB=8` for the long-context mode, `API_KEY_FILE=<file>` to require a key.

To turn the fusions off (for comparison or bit-identical greedy output): `OOT2_NOFUSE=1`. Single fusions: `OOT2_NOFUSE_MASK=<bits>` (names and bits in `scripts/validate.py`).

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `git apply` fails | llama.cpp is not at the base commit | do step 1 exactly; if the PR has moved, `git apply --3way` and fix conflicts by hand |
| `unknown model architecture 'inkling2'` or `tensor ... not found` | wrong GGUF files | use the `Mainline_Support/` files (step 4) |
| draft model fails with `markov_w1 ... wrong shape` | llama.cpp is not the PR #32773 branch | redo step 1 |
| out of memory at start | not enough free VRAM | stop other GPU users; lower `CTX`, keep `UB` small (the compute buffer grows with `CTX` x `UB`) |
| no speedup in `validate.py bench` | the build is not the patched tree | rebuild step 3 from the patched checkout; `validate.py launches` lists fusions that never launched |
| `nvcc` / gcc errors | host compiler too new for the toolkit | pass `-DCMAKE_CUDA_HOST_COMPILER` (step 3) |

## Limits

- Tested only on Ada (sm_89), 16 GB, the IQ3_XXS quant. Other CUDA GPUs and other weight quants should work and are untested; Inkling 2 Small is untested. Not for Vulkan, ROCm, Metal or CPU.
- One stream (batch 1): not for servers with many parallel requests. Greedy text matches the unpatched build for the first 80 to 250 characters, then may differ at near-tie tokens.
- The patches target an unmerged PR and may need rebasing as llama.cpp moves.

## More

- `docs/experiments.md`: every experiment with numbers and verdicts, including what failed.
- `docs/reproduce.md`: script settings, measuring by hand, serving options.
- `patches/`: `0001` adds the fused CUDA kernels (`ggml-cuda.cu`), `0002` skips a redundant `REPEAT` in the model graph (`inkling2.cpp`).

MIT license. The patches build on llama.cpp (MIT) and the DSpark work in PR #32773.
