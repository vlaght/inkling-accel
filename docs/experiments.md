# Experiment log

Everything that was tried while speeding up Inkling 2 Mini on one RTX 4070 Ti SUPER (16 GB, Ada, sm_89), including what did not work.
Verdict: **kept**, **dropped**, or **info**. Unless noted: model `UD-IQ3_XXS` fully on the GPU, single stream, greedy.

## Method

- **Speed:** `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3` (decode `tg128`, prompt `pp512`) with the GPU otherwise idle; end to end through `llama-server`
  with 8 prompts (code, SQL, JSON, prose, math, ...) of 256 generated tokens. Single prompts vary by about 5 % between runs: compare means over the 8.
- **Launches:** Nsight Systems (`nsys profile --cuda-graph-trace=node`), kernels per token = (instances of a 72-token run minus a 24-token run) / 48.
- **Quality:** KL divergence of the next-token distribution against the unfused run (`llama-perplexity --kl-divergence`), on the prompt path (micro-batch 512)
  and the decode path (micro-batch 1), judged against the engine's own noise floor (below). Perplexity and greedy text are weaker, secondary checks.
- All of it is automated in `scripts/validate.py`.

## 1. Backend and baseline

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 1.1 | llama.cpp Vulkan, IQ3_XXS all on GPU | decode 52.8, prompt 1439 tok/s | info (baseline) |
| 1.2 | Vulkan op profile (`GGML_VK_PERF_LOGGER`) | about 7,500 dispatches per token at about 3 µs each; elementwise ops (`MUL`, `SCALE`, `CONCAT`, `ADD`, `CONT`) dominate, weight reads do not | info |
| 1.3 | Flash attention on/off | no change (52.4 vs 52.3) | dropped |
| 1.4 | CUDA 13.4 backend instead of Vulkan | decode 67.4 (+28 %), prompt 1901 (+32 %) | kept |
| 1.5 | CUDA graphs off (`GGML_CUDA_DISABLE_GRAPHS`) | decode 51.5: graphs are worth +31 %, so launch overhead matters even with graphs | info |
| 1.6 | `GGML_CUDA_GRAPH_OPT=1`, `GGML_CUDA_GRAPH_CHUNK_SIZE` 64 to 1024 | no real change (66.8 to 67.4) | dropped |
| 1.7 | Experts offloaded to RAM (`--n-cpu-moe`), IQ4_XS on Vulkan | decode 42.9 / 32.9 / 23.0 / 16.7 at 4 / 8 / 12 / 16 offloaded layers: the 3x repeated middle layers read the offloaded experts three times | dropped; use a quant that fits in VRAM |
| 1.8 | Nsight profile of the CUDA decode step | about 7,000 kernels per token averaging 2 µs; quantized mat-vecs about 42 % of kernel time and already near the memory-bandwidth floor, elementwise ops about 33 %; even removing all small ops would give at most 1.4x | info; fusion is the lever |

## 2. Kernel fusions (patch 0001) and graph change (patch 0002)

Gain is decode tok/s measured with only the named fusion added to the previous state (`llama-bench`).

| # | Fusion | Replaces | Gain | Bitwise | Verdict |
|---|---|---|---|---|---|
| 2.1 | `clip` | `SQR, SUM_ROWS, SQRT, SCALE, FILL, DIV, CLAMP, MUL` (144 calls per token) | 67.3 to 71.4 (+6 %) | no (sum order) | kept |
| 2.2 | `tanh_fma` | `TANH, MUL, SUB/ADD` of the loop controllers | +2.6 % | yes | kept (decode now uses `control_tail`; still serves larger batches) |
| 2.3 | `situ` | 9 elementwise kernels of the SiTU activation | 67.9 to 69.3 (+2 %) | yes | kept |
| 2.4 | no `REPEAT` of the mix queries for one token (patch 0002) | `REPEAT` (120 per token) | about +0.7 % | yes | kept |
| 2.5 | `conv_add_scale` | `SSM_CONV, RESHAPE, ADD, SCALE` | 76.0 to 77.5 (+1.5 %) | yes | kept |
| 2.6 | `concat_cpy` | conv-state `CONCAT, CPY` | 76.0 to 77.4 (+1.4 %) | yes | kept |
| 2.7 | `gather_rms` | `CONCAT` chain + `RMS_NORM` of `mix()` | 79.0 to 79.6 (+0.8 %) | yes | kept (fallback of 2.8) |
| 2.8 | `mix_attn` | `CONCAT` chain, `RMS_NORM`, `TRANSPOSE`, `CONT`, `REPEAT`, `MUL_MAT`, `SOFT_MAX`, `MUL_MAT` as 2 kernels | 79.75 to 84.07 (+5.4 %); also prompt 1896 to 2163 (+14 %) once extended to several tokens | no (sum order) | kept |
| 2.9 | `control_tail` | 2 mat-vecs, 2 `TANH`, 2 `MUL`, `SUB`, `ADD` of a loop controller | 84.15 to 86.35 (+2.6 %) | no (sum order) | kept, only for up to 8 tokens |
| 2.10 | `gate_build` | 2 norms + concat that build the controller gate | 86.3 to 88.0 (+2 %) | no (sum order) | kept |

Cumulative decode (`llama-bench`): 67.4, 73.7 (2.1 to 2.2), 75.9 (2.3, 2.4), 79.0 (2.5, 2.6), 79.6 (2.7), 84.1 (2.8), 86.4 (2.9), 88.0 (2.10).
Launches per token: 7,258 without the patches, 3,911 with them (-46 %); kernel time 14.7 to 11.4 ms.

Dropped or fixed along the way:

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 2.11 | `mix()` tail as a single block per token | wrong results and no faster (72.9 vs 73.6) | dropped; replaced by the two-kernel 2.8 |
| 2.12 | Use ggml's own overlap check (`ggml_cuda_check_fusion_memory_ranges`) for the elementwise fusions | it rejects the case where the output lies exactly on a dead input, which the allocator produces constantly; gain fell from 73.6 to 68.0 | replaced by a rule: allowed only if the ranges are identical or disjoint |
| 2.13 | Fuse the MoE weighted reduction (`MUL, MUL, VIEW x8, ADD x7`) | llama.cpp already fuses it (`ggml_cuda_match_moe_weighted_reduction`); the version here never ran | dropped (duplicate) |
| 2.14 | `control_tail` for any number of tokens | prompt speed fell 11 % (each token re-reads both matrices) | limited to 8 tokens |
| 2.15 | First `cat_rms` (norm + concat of the controller gate) | **never fired**: the message copy is read a second time by the controller tail, and a fused chain may not swallow a node that is read outside it. The kernel list of the profile had no such kernel, and the gain credited to it had come from other fusions | replaced by `gate_build`, which writes that copy itself |

Debugging note: when checking a fused kernel against a host reference, copy the inputs to the host before launching it. The allocator may place a result on dead
inputs, and reading them afterwards shows false errors (this produced a bogus relative error of 0.4 once). The fused `mix_attn` matches a double-precision
reference to 4.5e-7 over 2,400 calls.

## 3. Quality

Mean KL divergence against the unfused run, 12 chunks of 512 tokens, prompt path:

| Run | mean KLD | same top token |
|---|---|---|
| noise floor: unfused, micro-batch 256 vs 512 | 0.0044 | 97.2 % |
| noise floor: unfused, micro-batch 128 vs 512 | 0.0047 | 97.4 % |
| `tanh_fma`, `situ`, `conv_add_scale`, `concat_cpy`, `gather_rms` alone | 0 | 100 % |
| `gate_build` alone | 0.0011 | 98.6 % |
| `mix_attn` alone | 0.0046 | 97.2 % |
| `clip` alone | 0.0047 | 97.2 % |
| `control_tail` alone, micro-batch 1 and 3 | 0.0005 | 99.3 to 99.6 % |
| all fusions | 0.0049 | 96.7 % |

The model amplifies float differences of about 1e-7 to a KLD of a few 1e-3, which the unfused engine also shows when only the batch size changes. The fusions
therefore sit inside the engine's own numeric noise. Perplexity, fused vs unfused: 4.0842 vs 4.0802 (prompt path, 40 chunks, error +-0.09) and 3.9608 vs 3.9554
(decode path, 16 chunks, error +-0.19).

## 4. Speculative decoding (DSpark)

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 4.1 | DSpark drafter on the older branch (PR #32218) | drafter fails to load: vocabulary 201024 vs 200058 | dropped; use PR #32773 |
| 4.2 | PR #32773 branch, drafted tokens per step (ctx 8k, 3 prompts, tok/s code / prose / JSON) | none 71.1 / 70.5 / 70.6; 1: 91.1 / 76.2 / 87.6; **2: 98.7 / 75.2 / 93.1**; 3: 92.2 / 61.7 / 85.0; 4: 92.0 / 58.9 / 85.2 | 2 kept; more drafts hurt on prose |
| 4.3 | Minimum draft probability 0.5 | 79.9 / 50.7 / 71.6 | dropped |
| 4.4 | `LLAMA_SPEC_ADAPT=1` (adaptive pausing) with 4 drafts | 88.0 / 67.6 / 82.3 against 92.0 / 58.9 / 85.2 without: better on prose, worse on code and JSON, and below a fixed 2 | dropped |
| 4.5 | 8-prompt mean with 2 drafts, patches on vs off | 100.5 vs 90.6 tok/s (+11 %); with the final English prompt set of `scripts/spec-bench.py` 102.0 vs 91.9 (+11 %) | info |
| 4.6 | Patches on vs off for `control_tail` and `gate_build` under DSpark | 100.5 / 100.3 / 101.9 / 101.1 tok/s with both on / `gate_build` off / `control_tail` off / both off: inside noise | info: neutral with speculation, +4.6 % without |
| 4.7 | Verify the drafter on later turns of a session | 4 turns: accept rate 50 to 66 % on each; decode 85 to 110 tok/s with the patches (80 to 93 without) | info |
| 4.8 | Output equality | not byte-identical to no speculation: differs after 140 to 250 characters at synonymous near-tie tokens (the verification batch uses other kernels) | info |
| 4.9 | Results table, one server configuration for every row (`scripts/results-table.py`: `-ub 32`, 4-bit KV, 8 prompts, 6k prompt) | CUDA unpatched 65.3 / 518, patched 82.5 / 568, patched + DSpark 101.7 / 545 (decode / prompt tok/s) | info |

## 5. Context length

| # | Setup | Result |
|---|---|---|
| 5.1 | Vulkan, f16 KV, all on GPU | 16k fits; 32k and above fail (compute buffer about 3.9 GB plus KV) |
| 5.2 | The compute buffer scales with context x micro-batch | lowering `-ub` is what makes long contexts fit |
| 5.3 | Vulkan, 4-bit KV, `-ub 16` | 262,144 tokens at 49 tok/s (44 with 6k in context) |
| 5.4 | CUDA, 4-bit KV, `-ub 16`, no speculation, before the fusions | 196,608 at 61.6 tok/s; 229k fails (CUDA context needs about 1 GB more VRAM than Vulkan) |
| 5.5 | KV cache on the CPU (`-nkvo`) | up to 1,048,576 tokens at `-ub 8`, but 17 tok/s: attention runs on the CPU |
| 5.6 | CUDA + DSpark, 4-bit KV for target and drafter | 131,072 at `-ub 8` (94 tok/s code); 163,840 does not start |
| 5.7 | Prompt speed by micro-batch (6k prompt, 65k context, DSpark) | 221 tok/s at `-ub 8`, 482 at 32, 688 at 128 |

## 6. Other engines considered

Not run; judged from their documentation.

| Project | Why it does not apply here |
|---|---|
| `maya-ik2` (custom Inkling 2 Mini engine, one CUDA graph of about 780 launches per token, custom FP8-KV attention and NVFP4 expert kernels) | needs 24 GB and Ampere (sm_86) and reads only the Q4_K_XL file; its claimed 129 tok/s is on an RTX 3090. The idea (a whole token in one graph) is the next step beyond fusing chain by chain |
| `Basalt` | Blackwell (sm_120) with FP4 hardware |
| `resin` (Inkling 2 CUDA runner) | Blackwell and the original BF16/NVFP4 checkpoints; it is a library of serving blocks, the forward pass is model-specific |

## 7. Open ideas

Each is estimated at 1 to 3 %: sharing the `q8_1` quantization of an activation between mat-vecs that read it (about 385 quantize launches per token), merging the
gain and forget matrices of the controllers into one mat-vec, fusing the small F32 mat-vecs (about 330 launches per token at about 2.8 µs), and a persistent
kernel for a whole layer visit.
