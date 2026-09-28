"""Session J analysis: Nsight Compute counters of the FP64 product kernels (RTX 4090 VM, attempt 07).

    python3 analysis/session_j_counters.py -> paper/session_j_numbers.json

H15 as declared: pipe utilization >= 70 % for dense8, modal8 and modal8_muladd, AND executed FP64 thread
instructions (DADD+DFMA+DMUL) with ratios dense8/muladd and modal8/muladd within 5 % of the static-site
ratios 78/34 and 52/34. The time model (time ~ instructions / utilization) is post-hoc and descriptive.
"""
import csv
import io
import json
import math
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
PROTOCOL = json.loads((ROOT / "experiments/session_j/protocol.json").read_text())
ELEMENTS = math.prod(PROTOCOL["ncu"]["grid"])


def row(kernel):
    lines = [l for l in (RUN / f"ncu-{kernel}.csv").read_text().splitlines() if l.startswith('"')]
    records = list(csv.reader(io.StringIO("\n".join(lines))))
    if len(records) != 3:
        raise ValueError(f"Expected header, units and one profiled launch for {kernel}")
    head, units, data = records
    unit = dict(zip(head, units))
    for metric, expected in {**dict.fromkeys(OPS, "inst"), PIPE: "%", DRAM: "%", TIME: "nsecond"}.items():
        if unit.get(metric) != expected:
            raise ValueError(f"Unexpected {metric} unit for {kernel}: {unit.get(metric)!r}")
    result = dict(zip(head, data))
    if result["Kernel Name"] != kernel:
        raise ValueError(f"Wrong kernel in {kernel} record: {result['Kernel Name']}")
    return result


def num(v):
    return float(v.replace(",", ""))


def main():
    k = {}
    for name in KERNELS:
        r = row(name)
        fp64 = sum(num(r[o]) for o in OPS)
        k[name] = {"executed_fp64_thread_instructions_total": int(fp64),
                   "executed_fp64_thread_instructions_per_element": fp64 / ELEMENTS,
                   "executed_fp64_thread_instructions_by_metric": {o: int(num(r[o])) for o in OPS},
                   "pipe_pct": num(r[PIPE]), "dram_pct": num(r[DRAM]), "time_ns": num(r[TIME]),
                   "replay_passes": int(num(r["profiler__replayer_passes"])),
                   "launch_thread_count": int(num(r["launch__thread_count"]))}
        k[name]["model"] = fp64 / (k[name]["pipe_pct"] / 100)
    ref = "modal8_muladd"
    job = json.loads((RUN.parent / "job.json").read_text())
    out = {"schema_version": 2,
           "workload": {"grid_elements": PROTOCOL["ncu"]["grid"], "elements": ELEMENTS,
                        "gpu": "RTX 4090 VM", "launch_skip": 2, "launch_count_per_kernel": 1,
                        "nsight_compute_version": job["version"]},
           "units_and_metrics": {
               "executed_fp64_thread_instructions_total": "sum of predicate-enabled DADD, DFMA and DMUL thread instructions; DFMA counts once",
               "executed_fp64_thread_instructions_per_element": "total divided by grid element count, independent of thread layout",
               "fp64_thread_instruction_metrics": OPS,
               "pipe_pct": PIPE, "dram_pct": DRAM, "time_ns": TIME,
               "model": "post-hoc total instructions / (active-cycle FP64 pipe percentage / 100); not an independent time prediction"},
           "profiling_policy": {
               "explicit_clock_cache_replay_overrides": False,
               "version_documented_defaults": {"clock_control": "base", "cache_control": "all", "replay_mode": "kernel"},
               "default_source": "https://archive.docs.nvidia.com/nsight-compute/2023.2/NsightComputeCli/index.html",
               "actual_clock_lock_verified": False,
               "timing_boundary": "selected kernel duration under profiler replay; excludes wrapper zeroing/projection and host overhead"},
           "kernels": k, "ratios_vs_muladd": {}}
    for name in KERNELS:
        out["ratios_vs_muladd"][name] = {
            "time": k[name]["time_ns"] / k[ref]["time_ns"],
            "instructions": k[name]["executed_fp64_thread_instructions_total"] / k[ref]["executed_fp64_thread_instructions_total"],
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
        print(f"{n:22s} pipe(active) {k[n]['pipe_pct']:5.1f}% dram(elapsed) {k[n]['dram_pct']:5.1f}% "
              f"executed FP64 thread instr/element {k[n]['executed_fp64_thread_instructions_per_element']:5.1f} "
              f"time/muladd {r['time']:.3f} post-hoc ratio {r['model']:.3f}")
    print("H15:", out["H15"], "post-hoc consistency max relative error", round(out["model_max_error"], 4))


if __name__ == "__main__":
    main()
