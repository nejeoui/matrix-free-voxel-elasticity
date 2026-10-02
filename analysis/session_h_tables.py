"""Session H analysis: declared equivalence, work-match, H11/H12 and breakdown per host.

    python3 analysis/session_g_tables.py results/session-g-host1-20260926 [results/session-g-host2-...]

Rules are read from the frozen protocol; nothing here is tuned after the results.
"""
import json
import statistics
import sys
from pathlib import Path
from preset_metadata import preset_volume_fractions
from supplement_outcomes import outcome_run

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "experiments/session_h/protocol.json").read_text())
WORK_TOL = 0.03
EQ_TOL = 1e-3
H_THRESHOLD = 1.10


def load(host_dir):
    return json.loads((Path(host_dir) / "remote/run-01/result.json").read_text())


def analyze(result):
    runs = result["runs"]
    by = {(r["repetition"], r["case"], r["schedule"], r["arm"]): r for r in runs}
    reps = sorted({r["repetition"] for r in runs})
    cases = [c["id"] for c in PROTOCOL["cases"]]
    primary = {c["id"] for c in PROTOCOL["cases"] if c["primary"]}
    out = {"repetitions_completed": [], "cases": {}}
    for rep in reps:
        if all((rep, c, s, a) in by for c in cases for s in PROTOCOL["schedules"] for a in PROTOCOL["arms"]):
            out["repetitions_completed"].append(rep)
    for case in cases:
        for schedule in PROTOCOL["schedules"]:
            key = f"{case}/{schedule}"
            row = {"primary": case in primary}
            row["agreement_coverage"] = {"compliance_repetitions": [1], "volume_and_linear_iteration_cap": "all fixed-length repetitions", "density_fields_compared": False}
            first = {a: by.get((1, case, schedule, a)) for a in PROTOCOL["arms"]}
            stock = first["stock"]
            if stock:
                eq = {}
                for arm in ("fused_ai_fp64", "modal8_muladd"):
                    r = first[arm]
                    if r:
                        eq[arm] = abs(r["final_compliance"] - stock["final_compliance"]) / abs(stock["final_compliance"])
                row["final_compliance_relative_vs_stock"] = eq
                row["final_compliance_stock"] = stock["final_compliance"]
                row["final_volume"] = {a: first[a]["history"][-1]["volume"] for a in first if first[a]}
            all_runs = [by[k] for k in by if k[1] == case and k[2] == schedule]
            row["not_converged_solves"] = sum(r["not_converged_solves"] for r in all_runs)
            vol_ok = all(abs(r["history"][-1]["volume"] - PROTOCOL_VOLFRAC.get(case, r["history"][-1]["volume"])) <= EQ_TOL
                         for r in all_runs)
            row["equivalence"] = bool(stock) and all(v <= EQ_TOL for v in row["final_compliance_relative_vs_stock"].values()) \
                and row["not_converged_solves"] == 0 and vol_ok
            ratios, work_ok, stock_ratios, iters = [], True, [], []
            for rep in out["repetitions_completed"]:
                f, m, s = (by.get((rep, case, schedule, a)) for a in ("fused_ai_fp64", "modal8_muladd", "stock"))
                if f and m:
                    ratios.append(f["wall_seconds"] / m["wall_seconds"])
                    iters.append((f["total_outer_iterations"], m["total_outer_iterations"]))
                    if abs(f["total_outer_iterations"] - m["total_outer_iterations"]) > WORK_TOL * f["total_outer_iterations"]:
                        work_ok = False
                if s and m:
                    stock_ratios.append(s["wall_seconds"] / m["wall_seconds"])
            row.update(ratios_fused_over_modal=ratios, median_ratio=statistics.median(ratios) if ratios else None,
                       outer_iterations_fused_modal=iters, work_matched=work_ok,
                       median_stock_over_modal=statistics.median(stock_ratios) if stock_ratios else None)
            if first["modal8_muladd"]:
                h = first["modal8_muladd"]["history"]
                tot = lambda k: sum((x.get(k) or 0) for x in h) / 1e3
                row["modal_breakdown_seconds"] = {"gmg_setup": tot("gmg_setup_ms"), "cg": tot("cg_ms"),
                                                  "sensitivity": tot("sensitivity_ms"),
                                                  "wall": first["modal8_muladd"]["wall_seconds"]}
            out["cases"][key] = row
    fp64_primary = [out["cases"][f"{c}/fp64"] for c in primary]
    out["H11"] = len(out["repetitions_completed"]) >= 2 and all(
        r["median_ratio"] is not None and r["median_ratio"] >= H_THRESHOLD and r["work_matched"] and r["equivalence"]
        for r in fp64_primary)
    out["H12_median_ratios"] = {c: out["cases"][f"{c}/mixed"]["median_ratio"] for c in cases}
    out["fixed_length_outcomes"] = [outcome_run(r, "fixed_length") for r in runs]
    out["complete"] = [outcome_run(c, "capped", PROTOCOL["complete_optimization"]["stop_change"])
                       for c in result.get("complete", [])]
    out["complete_key_semantics"] = "historical raw key: one capped trajectory per schedule/arm, not a design-convergence claim"
    out["population_ranges"] = {schedule: {"all_cases": [min(out["cases"][f"{c}/{schedule}"]["median_ratio"] for c in cases), max(out["cases"][f"{c}/{schedule}"]["median_ratio"] for c in cases)],
                                          "primary_cases": [min(out["cases"][f"{c}/{schedule}"]["median_ratio"] for c in primary), max(out["cases"][f"{c}/{schedule}"]["median_ratio"] for c in primary)]}
                                for schedule in PROTOCOL["schedules"]}
    out["errors"] = result.get("errors", [])
    return out


def _volfracs():
    presets = preset_volume_fractions(
        ROOT / "experiments/combined/donor_yang_gmg/src/gpu_fem/presets.py"
    )
    return {c["id"]: presets[c["preset"]] for c in PROTOCOL["cases"]}


PROTOCOL_VOLFRAC = _volfracs()


def main():
    hosts = {Path(d).name: analyze(load(d)) for d in sys.argv[1:]}
    summary = {"hosts": hosts, "H11b_replicated": len(hosts) >= 2 and all(h["H11"] for h in hosts.values())}
    path = ROOT / "paper/session_h_numbers.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    for name, h in hosts.items():
        print(f"== {name}: reps {h['repetitions_completed']}  H11={h['H11']}")
        for key, r in h["cases"].items():
            print(f"  {key:22s} eq={r['equivalence']!s:5} work={r['work_matched']!s:5} "
                  f"median fused/modal={r['median_ratio'] and round(r['median_ratio'], 3)} "
                  f"stock/modal={r['median_stock_over_modal'] and round(r['median_stock_over_modal'], 3)} "
                  f"dC={ {a: f'{v:.1e}' for a, v in r.get('final_compliance_relative_vs_stock', {}).items()} } "
                  f"nonconv={r['not_converged_solves']}")
        print("  complete:", h["complete"])
        if h["errors"]:
            print("  ERRORS:", [e.get("error", e.get("skipped", e.get("fatal")))[:120] for e in h["errors"]])
    print("H11b replicated:", summary["H11b_replicated"])


if __name__ == "__main__":
    main()
