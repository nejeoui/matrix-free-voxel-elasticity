"""Derive every number the short paper cites from the imported evidence files.

Writes paper/numbers.json and paper/tables.md. Reads only files under evidence/ plus the declared
device specification constants below. Nothing here is a new measurement.

Achieved-rate figures use STATIC operation counts from the implementation README
(code/hex_modal/README.md): dense local product 576 FMA = 1,152 flops per element; modal product
2 x 72 butterfly additions + 72 FMA = 288 flops per element. They exclude material scaling,
indexing, gathers and atomic scatter, so they are lower bounds on work performed and upper bounds
on how "compute-bound" a kernel is not. Device peaks are vendor specification values.
"""
import json, statistics
from pathlib import Path

V = Path(__file__).resolve().parents[1]
E = V / "evidence"
FLOPS = {"dense8": 1152, "modal8": 288}           # per element, static count (see docstring)
SPEC = {  # vendor specification sheets; FP64 non-tensor peak and DRAM bandwidth
    "RTX 4090": {"fp64_tflops": 1.290, "fp32_tflops": 82.58, "dram_gbs": 1008},
    "RTX 3090": {"fp64_tflops": 0.556, "fp32_tflops": 35.58, "dram_gbs": 936},
    "A100 80GB SXM": {"fp64_tflops": 9.7, "fp32_tflops": 19.5, "dram_gbs": 2039},
    "H100 SXM": {"fp64_tflops": 34.0, "fp32_tflops": 67.0, "dram_gbs": 3350},
}


def load(p):
    return json.loads((E / p).read_text())


def kernel_table(path, gpu):
    d = load(path)
    out = []
    for c in d["comparisons"]:
        out.append({"gpu": gpu, "shape": "x".join(map(str, c["shape"])),
                    "elements": c["shape"][0] * c["shape"][1] * c["shape"][2],
                    "baseline": c["baseline"], "primary": c["primary"],
                    "modal8_ms": c["candidate_median_ms"], "baseline_ms": c["baseline_median_ms"],
                    "ratio": c["paired_median_ratio"], "lower95": c["paired_bootstrap_lower_95"]})
    return out


def achieved(rows, gpu):
    """Achieved FP64 rate of the local-product arithmetic for dense8 and modal8."""
    res = []
    by_shape = {}
    for r in rows:
        by_shape.setdefault(r["shape"], {"elements": r["elements"], "modal8_ms": r["modal8_ms"]})
        if r["baseline"] == "dense8":
            by_shape[r["shape"]]["dense8_ms"] = r["baseline_ms"]
    peak = SPEC[gpu]["fp64_tflops"] * 1e12
    for shape, s in by_shape.items():
        if "dense8_ms" not in s:
            continue
        row = {"gpu": gpu, "shape": shape, "elements": s["elements"]}
        for k in ("dense8", "modal8"):
            rate = s["elements"] * FLOPS[k] / (s[f"{k}_ms"] * 1e-3)
            row[f"{k}_gflops"] = rate / 1e9
            row[f"{k}_fraction_of_fp64_peak"] = rate / peak
        res.append(row)
    return res


def main():
    k4090 = kernel_table("kernel-rtx4090/local-verification.json", "RTX 4090")
    k3090 = kernel_table("kernel-rtx3090-replication/local-verification.json", "RTX 3090")
    ach = achieved(k4090, "RTX 4090") + achieved(k3090, "RTX 3090")

    prof = load("solve-profile-rtx4090/local-verification.json")["diagnostic_decision"]
    solves = [{k: s[k] for k in ("case", "path", "primary", "median_plain_seconds", "plain_iterations",
                                 "profile_iterations", "profile_product_fraction",
                                 "profile_wall_over_plain_median_minus_one",
                                 "profile_and_plain_work_identical", "profile_qualified")}
              for s in prof["summaries"]]
    comps = [{k: c[k] for k in ("case", "primary", "baseline", "median_plain_solver_ratio",
                                "profiled_product_per_call_ratio", "baseline_profile_product_fraction",
                                "conditional_fixed_work_solver_ratio",
                                "same_profile_iterations_across_paths", "profiles_qualified",
                                "passes_planning_ratio")}
             for c in prof["comparisons"]]

    lay = load("product-layout-rtx4090/local-verification.json")["decision"]
    layout = [{k: c[k] for k in ("case", "primary", "candidate", "baseline", "candidate_median_ms",
                                 "baseline_median_ms", "paired_median_ratio", "one_sided_95_lower",
                                 "passed")} for c in lay["comparisons"]]

    # Decision rule declared by the researcher before the profile result: share of the FASTEST
    # baseline's solve spent in products must be >= 44% on the RTX 4090 primary cases.
    fastest = "fused_ai_fp64"
    share = {s["case"]: s["profile_product_fraction"] for s in solves
             if s["path"] == fastest and s["primary"]}
    rule = {"threshold": 0.44, "fastest_baseline": fastest, "shares": share,
            "shares_qualified": {s["case"]: s["profile_qualified"] for s in solves
                                 if s["path"] == fastest and s["primary"]},
            "met": all(v >= 0.44 for v in share.values())}

    numbers = {"kernel_products": k4090 + k3090, "achieved_fp64_rates": ach,
               "solve_profile": {"summaries": solves, "comparisons": comps,
                                 "protocol_decision": prof["decision"],
                                 "advance_to_charged_campaign": prof["advance_to_charged_campaign"]},
               "product_layout": {"comparisons": layout, "gates": lay["development_gates"]},
               "decision_rule_44pct": rule, "device_spec": SPEC, "static_flops": FLOPS}
    (V / "paper/numbers.json").write_text(json.dumps(numbers, indent=1))

    L = ["# Paper tables (generated by analysis/paper_numbers.py — do not edit)", ""]
    L += ["## T1. Element-product time, modal8 vs published and control baselines", "",
          "| GPU | shape | elements | baseline | modal8 ms | baseline ms | ratio | 95% lower |",
          "|---|---|---:|---|---:|---:|---:|---:|"]
    for r in k4090 + k3090:
        if r["primary"]:
            L.append(f"| {r['gpu']} | {r['shape']} | {r['elements']:,} | `{r['baseline']}` | "
                     f"{r['modal8_ms']:.4f} | {r['baseline_ms']:.4f} | {r['ratio']:.4f} | {r['lower95']:.4f} |")
    L += ["", "## T2. Achieved FP64 rate of the local-product arithmetic (static counts)", "",
          "| GPU | shape | dense8 GFLOP/s | dense8 / FP64 peak | modal8 GFLOP/s | modal8 / FP64 peak |",
          "|---|---|---:|---:|---:|---:|"]
    for a in ach:
        L.append(f"| {a['gpu']} | {a['shape']} | {a['dense8_gflops']:.0f} | {a['dense8_fraction_of_fp64_peak']:.1%} | "
                 f"{a['modal8_gflops']:.0f} | {a['modal8_fraction_of_fp64_peak']:.1%} |")
    L += ["", "## T3. Warm PCG solve on physical states, RTX 4090 (3 plain repetitions)", "",
          "| case | baseline | solver ratio | in-solve product ratio | baseline product share | Amdahl reconstruction | profiles qualified |",
          "|---|---|---:|---:|---:|---:|---|"]
    for c in comps:
        if c["case"] != "smoke-q4":
            L.append(f"| {c['case']} | `{c['baseline']}` | {c['median_plain_solver_ratio']:.4f} | "
                     f"{c['profiled_product_per_call_ratio']:.4f} | {c['baseline_profile_product_fraction']:.1%} | "
                     f"{c['conditional_fixed_work_solver_ratio']:.4f} | {c['profiles_qualified']} |")
    L += ["", "## T4. Product layout study on physical inputs, RTX 4090", "",
          "| case | candidate | baseline | ratio | 95% lower |", "|---|---|---|---:|---:|"]
    for c in layout:
        if c["primary"] and c["baseline"] in ("fused_ai_fp64", "sym8", "dense8", "modal8"):
            L.append(f"| {c['case']} | {c['candidate']} | `{c['baseline']}` | "
                     f"{c['paired_median_ratio']:.4f} | {c['one_sided_95_lower']:.4f} |")
    L += ["", f"## Decision rule (share ≥ 44% of `{fastest}` solve in products)", "",
          f"shares: {', '.join(f'{k} {v:.1%}' for k, v in share.items())}; qualified: "
          f"{rule['shares_qualified']}; **met: {rule['met']}**."]
    (V / "paper/tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
