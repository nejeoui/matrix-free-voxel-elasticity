"""Session J analysis: Nsight Compute counters of the FP64 product kernels (RTX 4090 VM, attempt 07).

    python3 analysis/session_j_counters.py -> paper/session_j_numbers.json

H15 as declared: pipe utilization >= 70 % for dense8, modal8 and modal8_muladd, AND issued FP64 thread
instructions (DADD+DFMA+DMUL) with ratios dense8/muladd and modal8/muladd within 5 % of the static-site
ratios 78/34 and 52/34. The time model (time ~ instructions / utilization) is post-hoc and descriptive.
"""
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/session-j-rtx4090vm-20260926-attempt07/remote/ncu"
KERNELS = ["dense8", "modal8", "modal8_muladd", "fused_matvec_ai_fp64", "node_matvec_fp64"]
OPS = ["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
       "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"]
PIPE = "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active"
DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
TIME = "gpu__time_duration.sum"
STATIC = {"dense8": 78, "modal8": 52, "modal8_muladd": 34}
THREADS = 128 * 64 * 64 * 8


def row(kernel):
    lines = [l for l in (RUN / f"ncu-{kernel}.csv").read_text().splitlines() if l.startswith('"')]
    head, _units, data = list(csv.reader(io.StringIO("\n".join(lines))))[:3]
    return {h: v for h, v in zip(head, data)}


def num(v):
    return float(v.replace(",", ""))


def main():
    k = {}
    for name in KERNELS:
        r = row(name)
        fp64 = sum(num(r[o]) for o in OPS)
        k[name] = {"fp64_thread_instructions": fp64, "per_thread": fp64 / THREADS, "pipe_pct": num(r[PIPE]),
                   "dram_pct": num(r[DRAM]), "time_ns": num(r[TIME])}
        k[name]["model"] = k[name]["fp64_thread_instructions"] / (k[name]["pipe_pct"] / 100)
    ref = "modal8_muladd"
    out = {"kernels": k, "ratios_vs_muladd": {}}
    for name in KERNELS:
        out["ratios_vs_muladd"][name] = {
            "time": k[name]["time_ns"] / k[ref]["time_ns"],
            "instructions": k[name]["fp64_thread_instructions"] / k[ref]["fp64_thread_instructions"],
            "model": k[name]["model"] / k[ref]["model"],
            "static": STATIC[name] / STATIC[ref] if name in STATIC else None}
    util_ok = all(k[n]["pipe_pct"] >= 70 for n in STATIC)
    count_dev = {n: abs(out["ratios_vs_muladd"][n]["instructions"] / out["ratios_vs_muladd"][n]["static"] - 1)
                 for n in ("dense8", "modal8")}
    out["H15"] = {"utilization_part": util_ok, "count_deviation": count_dev,
                  "count_part": all(v <= 0.05 for v in count_dev.values()),
                  "passed": util_ok and all(v <= 0.05 for v in count_dev.values())}
    out["model_max_error"] = max(abs(v["model"] / v["time"] - 1) for v in out["ratios_vs_muladd"].values())
    (ROOT / "paper/session_j_numbers.json").write_text(json.dumps(out, indent=1) + "\n")
    for n in KERNELS:
        r = out["ratios_vs_muladd"][n]
        print(f"{n:22s} pipe {k[n]['pipe_pct']:5.1f}% dram {k[n]['dram_pct']:5.1f}% instr/thread {k[n]['per_thread']:5.1f} "
              f"time/muladd {r['time']:.3f} model {r['model']:.3f}")
    print("H15:", out["H15"], "model max error", round(out["model_max_error"], 4))


if __name__ == "__main__":
    main()
