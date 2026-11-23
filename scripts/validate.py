#!/usr/bin/env python3
"""One-command validation of a build of the Inkling fusion patches (the GPU must be free for every step).

    python3 scripts/validate.py all          # everything below, as a compact markdown report
    python3 scripts/validate.py bench        # decode and prompt speed, fusions off vs on
    python3 scripts/validate.py kld          # KL divergence per fusion on the prompt and the decode path, against the engine's own noise floor
    python3 scripts/validate.py launches     # kernel launches per token (Nsight), off vs on, with the top kernels
    python3 scripts/validate.py spec         # DSpark sweep over 8 prompts, fusions off vs on
    python3 scripts/validate.py multiturn    # a 4-turn session: speed and draft accept rate per turn

Settings come from the environment (scripts/lib.py, docs/reproduce.md); CHUNKS sets the KL chunks (default 8), NSYS the Nsight binary.

Fusion bits live in FUSIONS below; the script warns when ggml-cuda.cu uses a bit it does not know, so keep it in step with the patch. A fusion is "bitwise" when its
KLD against the unfused run is about 0; the others are compared with the noise floor (the unfused model run with another micro-batch): a KLD at or below the
floor cannot be told from the engine's own numeric noise.
"""
import csv, json, os, re, subprocess, sys
import lib
from lib import bin_, sh

CHUNKS = int(os.environ.get("CHUNKS", "8"))
NSYS = os.environ.get("NSYS", "nsys")   # the standalone Nsight Systems: the wrapper that ships in the CUDA toolkit refuses to run
KLD_DIR = os.path.join(lib.WORK, "kld")

# name -> bit of OOT2_NOFUSE_MASK (the bit DISABLES the fusion)
FUSIONS = {"clip": 1, "tanh_fma": 2, "gate_build": 4, "situ": 16, "conv_add_scale": 64, "concat_cpy": 128,
           "gather_rms": 256, "mix_attn": 512, "control_tail": 1024, "rel_bias_mask": 2048,
           "conv_dyn_add_scale": 4096, "q8_1_cache": 8192, "mmvq_scale_only": 16384, "topk_norm_eps": 32768, "clip_mul": 65536, "mix_single": 131072, "mix_regq": 262144}
ALL_OFF = sum(FUSIONS.values())
# the kernel each fusion launches, to spot a fusion that never fires
KERNELS = {"clip": "k_inkling_clip", "tanh_fma": "k_inkling_tanh_fma", "gate_build": "k_inkling_gate_build", "situ": "k_inkling_situ",
           "conv_add_scale": "k_inkling_conv_add_scale", "concat_cpy": "k_inkling_concat_cpy", "gather_rms": "k_inkling_gather_rms",
           "mix_attn": "k_inkling_mix_regq", "control_tail": "k_inkling_control_tail",
           "rel_bias_mask": "k_inkling_rel_bias_mask", "conv_dyn_add_scale": "k_inkling_conv_dyn_add_scale",
           "mmvq_scale_only": "mul_mat_vec_q<(ggml_type)40, (int)1, (int)1", "topk_norm_eps": "topk_moe_cuda", "clip_mul": "k_inkling_clip_mul", "mix_single": "k_inkling_mix_regq", "mix_regq": "k_inkling_mix_regq"}


def check_bits():
    src = open(os.path.join(lib.ROOT, "ggml/src/ggml-cuda/ggml-cuda.cu"), errors="replace").read()
    unknown = {int(x) for x in re.findall(r"ggml_cuda_inkling_nofuse_mask\(\) & (\d+)", src)} - set(FUSIONS.values())
    if unknown:
        print(f"WARNING: bits {sorted(unknown)} are used in ggml-cuda.cu but are not in FUSIONS; update scripts/validate.py", file=sys.stderr)


def bench_one(mask):
    out = sh(f"{bin_('llama-bench')} -m {lib.MODEL} -ngl 99 -fa 1 -p 512 -n 128 -r 3", {"OOT2_NOFUSE_MASK": mask})
    pp, tg = re.search(r"pp512 \|\s+([\d.]+)", out), re.search(r"tg128 \|\s+([\d.]+)", out)
    if not (pp and tg):
        raise SystemExit("llama-bench failed (is the GPU free?):\n" + out[-600:])
    return float(pp.group(1)), float(tg.group(1))


def cmd_bench():
    off, on = bench_one(ALL_OFF), bench_one(0)
    print("### Speed (llama-bench, fa=1, 3 repeats)\n")
    print("| | prompt pp512 | decode tg128 |\n|---|---|---|")
    print(f"| all fusions off | {off[0]:.0f} | {off[1]:.1f} |")
    print(f"| all fusions on | {on[0]:.0f} ({on[0]/off[0]*100-100:+.1f} %) | {on[1]:.1f} ({on[1]/off[1]*100-100:+.1f} %) |\n")


def text_file():
    path = os.path.join(KLD_DIR, "text.txt")
    if not os.path.exists(path):   # 400 KB of the llama.cpp docs: any text will do
        os.makedirs(KLD_DIR, exist_ok=True)
        docs = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(lib.ROOT, "docs")) for f in fs if f.endswith(".md"))
        open(path, "wb").write(b"".join(open(f, "rb").read() + b"\n" for f in docs)[:400000])
    return path


def kld_run(extra, env, ub=512, ctx=512, chunks=None):
    out = sh(f"{bin_('llama-perplexity')} -m {lib.MODEL} -f {text_file()} -ngl 99 -c {ctx} -b {ub} -ub {ub} --chunks {chunks or CHUNKS} {extra}", env)
    m = re.search(r"Mean\s+KLD:\s+(-?[\d.]+)\s+±\s+([\d.]+)", out)
    t = re.search(r"Same top p:\s+([\d.]+)", out)
    return (float(m.group(1)), float(m.group(2)), float(t.group(1))) if m and t else None


def report(rows, floor):
    for name, r in rows:
        if r is None:
            print(f"| {name} | (no result) | | |")
            continue
        if "noise floor)" in name:
            verdict = "reference"
        elif abs(r[0]) < 2e-6 and r[2] > 99.99:
            verdict = "bitwise"
        else:
            verdict = "within the noise floor" if r[0] <= floor * 1.1 else "ABOVE the noise floor: investigate"
        print(f"| {name} | {r[0]:.6f} ± {r[1]:.6f} | {r[2]:.2f} % | {verdict} |")


def cmd_kld():
    os.makedirs(KLD_DIR, exist_ok=True)
    base = os.path.join(KLD_DIR, f"base_ub512_c{CHUNKS}.kld")
    if not os.path.exists(base):
        kld_run(f"--kl-divergence-base {base}", {"OOT2_NOFUSE": "1"})
    cmp_ = f"--kl-divergence-base {base} --kl-divergence"
    rows = [(f"unfused, micro-batch {ub} (noise floor)", kld_run(cmp_, {"OOT2_NOFUSE": "1"}, ub)) for ub in (256, 128)]
    floor = max(r[0] for _, r in rows if r)
    rows += [(f"{n} only", kld_run(cmp_, {"OOT2_NOFUSE_MASK": ALL_OFF - b})) for n, b in FUSIONS.items()]
    rows.append(("**all fusions**", kld_run(cmp_, {"OOT2_NOFUSE_MASK": 0})))
    # the decode path (one token per step) runs fusions that a 512-token batch never reaches (control_tail, the one-token mix_attn)
    dbase = os.path.join(KLD_DIR, f"base_ub1_c{CHUNKS}.kld")
    dec = lambda extra, env: kld_run(extra, env, ub=1, ctx=192, chunks=max(3, CHUNKS // 2))
    if not os.path.exists(dbase):
        dec(f"--kl-divergence-base {dbase}", {"OOT2_NOFUSE": "1"})
    dcmp = f"--kl-divergence-base {dbase} --kl-divergence"
    drows = [(f"{n} only", dec(dcmp, {"OOT2_NOFUSE_MASK": ALL_OFF - b})) for n, b in FUSIONS.items()]
    drows.append(("**all fusions**", dec(dcmp, {"OOT2_NOFUSE_MASK": 0})))
    head = "| run | mean KLD | same top token | verdict |\n|---|---|---|---|"
    print(f"### Quality: KL divergence vs the unfused run ({CHUNKS} chunks of 512 tokens, micro-batch 512: the prompt path)\n\n{head}")
    report(rows, floor)
    print(f"\n### Same, one token per step (the decode path; judged against the same noise floor)\n\n{head}")
    report(drows, floor)
    print()


def profile(mask, n):
    tmp = os.path.join(lib.WORK, "nsys")
    os.makedirs(tmp, exist_ok=True)
    sh(f"rm -f {tmp}/p.* {tmp}/p_*; {NSYS} profile --cuda-graph-trace=node -o {tmp}/p --force-overwrite true --stats=false "
       f"{bin_('llama-bench')} -m {lib.MODEL} -ngl 99 -fa 1 -p 0 -n {n} -r 1 >/dev/null 2>&1; "
       f"{NSYS} stats --report cuda_gpu_kern_sum --format csv -o {tmp}/p {tmp}/p.nsys-rep >/dev/null 2>&1", {"OOT2_NOFUSE_MASK": mask})
    rows = csv.DictReader(open(f"{tmp}/p_cuda_gpu_kern_sum.csv"))
    return {r["Name"]: (int(r["Instances"]), float(r["Total Time (ns)"])) for r in rows}


def cmd_launches(top=12):
    res = {}
    for label, mask in (("off", ALL_OFF), ("on", 0)):
        a, b = profile(mask, 24), profile(mask, 72)
        res[label] = {k: ((b[k][0] - a.get(k, (0, 0))[0]) / 48, (b[k][1] - a.get(k, (0, 0))[1]) / 48e3) for k in b}
    tot = {l: (sum(v[0] for v in d.values()), sum(v[1] for v in d.values())) for l, d in res.items()}
    print("### Kernel launches per token (Nsight, difference of a 24- and a 72-token decode run)\n")
    print(f"- fusions off: {tot['off'][0]:.0f} launches, {tot['off'][1]:.0f} us of kernel time")
    print(f"- fusions on:  {tot['on'][0]:.0f} launches ({tot['on'][0]/tot['off'][0]*100-100:+.0f} %), {tot['on'][1]:.0f} us\n")
    print(f"Top {top} kernels with fusions on (per token):\n\n| launches | us | share | kernel |\n|---|---|---|---|")
    for k, (n_, t_) in sorted(res["on"].items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"| {n_:.1f} | {t_:.0f} | {t_/tot['on'][1]*100:.1f} % | `{k[:80]}` |")
    # a fusion that is in the binary but never launched is dead code: such a fusion once passed every other check
    fired = " ".join(res["on"])
    dead = [n for n, k in KERNELS.items() if k not in fired]
    print("\nFusions whose kernel never launched in this decode run: " + (", ".join(dead) if dead else "none") +
          " (expected: gather_rms, which mix_attn supersedes, and tanh_fma, which control_tail replaces in decode)\n")


def run_script(name, title, configs):
    out = subprocess.run([sys.executable, os.path.join(lib.HERE, name), json.dumps(configs)], capture_output=True, text=True).stdout
    print(f"### {title}\n\n```\n{out.strip()}\n```\n")


def cmd_spec():
    base = "-c 32768 -ub 32 -b 32 " + lib.SPEC
    run_script("spec-bench.py", "DSpark (2 drafts) over 8 prompts, greedy, 256 tokens each", [["fusions on", base, {}], ["all fusions off", base, {"OOT2_NOFUSE": "1"}]])


def cmd_multiturn():
    base = "-c 32768 -ub 32 -b 32 " + lib.SPEC
    run_script("multiturn.py", "Multi-turn session (4 turns, the history grows, prefix cache on)", [["fusions on", base, {}], ["all fusions off", base, {"OOT2_NOFUSE": "1"}]])


STEPS = {"bench": cmd_bench, "kld": cmd_kld, "launches": cmd_launches, "spec": cmd_spec, "multiturn": cmd_multiturn}

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what != "all" and what not in STEPS:
        raise SystemExit(__doc__)
    check_bits()
    for name in (STEPS if what == "all" else [what]):
        STEPS[name]()
