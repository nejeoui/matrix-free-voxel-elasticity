"""Generate combined_numbers.tex (LaTeX macros) for the combined paper from verified session summaries.

Inputs: paper/session_f_summary.json (analysis/session_f_summary.py), paper/session_g_numbers.json
(analysis/session_g_tables.py) and the raw session G results (for the E3 trajectory and breakdown). Later
sessions (H, I) are added when their analyses exist. The manuscript must not contain a hand-typed result;
every number it quotes is one of these macros or a macro of paper/manuscript/numbers.tex.
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parents[1]


def load(p):
    return json.loads(Path(p).read_text())


def fmt(x, digits=2):
    return f"{x:.{digits}f}"


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def session_f(m):
    s = load(V / "paper/session_f_summary.json")
    m["FProductMaxErr"] = sci(max(v["max_product_relative_l2"] for v in s.values()))
    m["FCountersHosts"] = str(sum(1 for v in s.values() if not v["counters_available"]))
    ok = {g: next(r for r in v["sizing"] if r["size"] == "2M" and r["schedule"] == "fp64" and r["arm"] == "stock")
          for g, v in s.items() if g != "RTX 4090"}
    m["FTwoMGiB"] = fmt(statistics.mean(r["gpu_used_gib"] for r in ok.values()), 0)
    m["FTwoMNdofMillions"] = fmt(next(iter(ok.values()))["ndof"] / 1e6, 1)
    m["FGiBPerMillionElements"] = fmt(statistics.mean(r["gpu_used_gib"] for r in ok.values()) / 2.0, 0)


def session_g(m):
    s = load(V / "paper/session_g_numbers.json")["hosts"]
    primary = ["cantilever-512k", "mbb-514k", "torsion-499k"]
    fp64 = [h["cases"][f"{c}/fp64"]["median_ratio"] for h in s.values() for c in primary]
    m["GEndToEndMin"], m["GEndToEndMax"] = fmt(min(fp64)), fmt(max(fp64))
    m["GHElevenPasses"] = "no" if not any(h["H11"] for h in s.values()) else "yes"
    cg, share, stock = [], [], []
    for host in ("host1", "host2"):
        res = load(V / f"results/session-g-{host}-20260926/remote/run-01/result.json")
        by = {(r["repetition"], r["case"], r["schedule"], r["arm"]): r for r in res["runs"]}
        t = lambda r: sum(x["cg_ms"] or 0 for x in r["history"]) / 1e3
        for c in primary:
            cg.append(statistics.median(t(by[(k, c, "fp64", "fused_ai_fp64")]) / t(by[(k, c, "fp64", "modal8_muladd")])
                                        for k in (1, 2, 3)))
            stock.append(statistics.median(t(by[(k, c, "fp64", "stock")]) / t(by[(k, c, "fp64", "modal8_muladd")])
                                           for k in (1, 2, 3)))
            mm = by[(1, c, "fp64", "modal8_muladd")]
            share.append(100 * t(mm) / mm["wall_seconds"])
        if host == "host1":
            diag = res["their_e3_diagnostic"]["history"]
            m["GEThreeVolumeOne"] = fmt(diag[0]["volume"], 2)
            m["GEThreeVolumeTwo"] = fmt(diag[1]["volume"], 3)
            m["GEThreeComplianceZero"] = fmt(diag[0]["compliance"], 1)
            m["GEThreeCompliancePeak"] = sci(max(h["compliance"] for h in diag))
            comp = [c for c in res["complete"] if "final_compliance" in c]
            m["GCompleteCompliance"] = f"{comp[0]['final_compliance']:.6f}"
    m["GCgRatioMin"], m["GCgRatioMax"] = fmt(min(cg)), fmt(max(cg))
    m["GCgStockMin"], m["GCgStockMax"] = fmt(min(stock), 1), fmt(max(stock), 1)
    m["GCgShareMin"], m["GCgShareMax"] = fmt(min(share), 0), fmt(max(share), 0)


def session_h(m):
    s = load(V / "paper/session_h_numbers.json")["hosts"]
    for case, key in (("cantilever-216k", "SmallCant"), ("cantilever-512k", "Cant"), ("mbb-514k", "Mbb"), ("torsion-499k", "Tors")):
        r = [h["cases"][f"{case}/fp64"]["median_ratio"] for h in s.values()]
        m[f"HEndToEnd{key}Min"], m[f"HEndToEnd{key}Max"] = fmt(min(r)), fmt(max(r))
    all_cases = [h["cases"][f"{c}/fp64"]["median_ratio"] for h in s.values()
                 for c in ("cantilever-216k", "cantilever-512k", "mbb-514k", "torsion-499k")]
    m["HEndToEndAllMin"], m["HEndToEndAllMax"] = fmt(min(all_cases), 3), fmt(max(all_cases), 3)
    stock = [h["cases"][f"{c}/fp64"]["median_stock_over_modal"] for h in s.values()
             for c in ("cantilever-512k", "mbb-514k", "torsion-499k")]
    m["HStockMin"], m["HStockMax"] = fmt(min(stock)), fmt(max(stock))
    m["HHElevenBPasses"] = "no" if not any(h["H11"] for h in s.values()) else "yes"
    res = load(V / "results/session-h-host2-20260926/remote/run-01/result.json")
    comp = [c for c in res["complete"] if "final_compliance" in c]
    m["HCompleteCompliance"] = f"{comp[0]['final_compliance']:.6f}"
    m["HCompleteSeconds"] = fmt(statistics.median(c["wall_seconds"] for c in comp), 0)
    parity = next(c for c in comp if c["arm"] == "modal8_muladd" and c["schedule"] == "fp64")
    m["HCappedParityChange"] = fmt(parity["history"][-1]["change"], 5)
    g = load(V / "results/session-g-host1-20260926/remote/run-01/result.json")
    m["GCompleteSeconds"] = fmt(statistics.median(c["wall_seconds"] for c in g["complete"] if "wall_seconds" in c), 0)


def session_i(m):
    s = load(V / "paper/session_i_numbers.json")
    g4090, a100, h100 = (s[f"session-i-{k}-20260926"] for k in ("rtx4090", "a100", "h100"))
    m["IHFourteenRatio"] = fmt(g4090["H14"]["median_fused_over_modal"])
    m["IHFourteenPasses"] = "yes" if g4090["H14"]["passed"] else "no"
    m["IFourNinetyNodeOverModal"] = fmt(g4090["cases"]["q96x2"]["node_over_modal_mg64"])
    m["IFourNinetyMgThirtyTwo"] = fmt(g4090["cases"]["q96x2"]["fused_over_modal_mg32"])
    ndof = lambda case: next(r for r in load(V / "results/session-i-a100-20260926/remote/generated.json")["rows"]
                             if r["id"] == case)["ndof"]
    m["IMaxDofsFourNinety"] = fmt(ndof("q96x2") / 1e6, 1)
    m["IMaxDofsEighty"] = fmt(ndof("q96x3") / 1e6, 1)
    m["IMaxElementsEighty"] = fmt(next(r for r in load(V / "results/session-i-a100-20260926/remote/generated.json")["rows"]
                                       if r["id"] == "q96x3")["elements"] / 1e6, 1)
    for name, h in (("Ahundred", a100), ("Hhundred", h100)):
        r = list(h["H13"]["median_modal_over_node"].values())
        m[f"I{name}ModalOverNodeMin"], m[f"I{name}ModalOverNodeMax"] = fmt(min(r)), fmt(max(r))
        m[f"I{name}InitialModalOverNode"] = fmt(1 / h["cases"]["q96x1"]["node_over_modal_mg64"], 4)
        m[f"I{name}BigFastest"] = fmt(min(h["cases"]["q96x3"]["tol_seconds_median"].values()), 2)
        m[f"I{name}BigModal"] = fmt(h["cases"]["q96x3"]["tol_seconds_median"]["mg64-modal8_muladd"], 2)
    m["IFourNinetyBigFastest"] = fmt(min(g4090["cases"]["q96x2"]["tol_seconds_median"].values()), 2)
    m["IFourNinetyBigModal"] = fmt(g4090["cases"]["q96x2"]["tol_seconds_median"]["mg64-modal8_muladd"], 2)
    m["IHThirteenPasses"] = "yes" if a100["H13"]["passed"] and h100["H13"]["passed"] else "no"
    for arm, key in (("mg32-fused_ai64-node32", "Fused"), ("mg32-node64-node32", "Node")):
        m[f"IAhundredInitialMixed{key}Seconds"] = fmt(a100["cases"]["q96x1"]["tol_seconds_median"][arm], 5)


def session_j(m):
    s = load(V / "paper/session_j_numbers.json")
    k, r = s["kernels"], s["ratios_vs_muladd"]
    declared = ("dense8", "modal8", "modal8_muladd")
    m["JPipeMin"] = fmt(min(k[n]["pipe_pct"] for n in k), 0)
    m["JPipeMax"] = fmt(max(k[n]["pipe_pct"] for n in k), 0)
    m["JDramMin"] = fmt(min(k[n]["dram_pct"] for n in k), 0)
    m["JDramMax"] = fmt(max(k[n]["dram_pct"] for n in k), 0)
    for n, key in (("dense8", "Dense"), ("modal8", "Plain"), ("modal8_muladd", "Muladd"),
                   ("fused_matvec_ai_fp64", "Fused"), ("node_matvec_fp64", "Node")):
        m[f"JInstr{key}"] = fmt(k[n]["executed_fp64_thread_instructions_per_element"], 0)
        m[f"JTotal{key}"] = str(k[n]["executed_fp64_thread_instructions_total"])
    m["JDenseOverMuladdExecuted"] = fmt(r["dense8"]["instructions"])
    m["JDenseOverMuladdStatic"] = fmt(r["dense8"]["static"])
    m["JCountDeviationPct"] = fmt(100 * s["H15"]["count_deviation"]["dense8"], 0)
    m["JModelMaxErrPct"] = fmt(100 * s["model_max_error"], 1)
    m["JHFifteenPasses"] = "yes" if s["H15"]["passed"] else "no"
    m["JDeclaredPipeMin"] = fmt(min(k[n]["pipe_pct"] for n in declared), 0)


def main():
    m = {}
    session_f(m)
    session_g(m)
    session_h(m)
    session_i(m)
    session_j(m)
    algebra = load(V / "paper/revision/algebra-checks.json")
    for r, key in zip(algebra["quadrature"], ("Two", "Three", "FourFive")):
        m[f"CompositeDiscrepancy{key}"] = fmt(100 * r["relative_frobenius_difference_from_exact_gauss"], 6)
    out = HERE / "combined_numbers.tex"
    out.write_text("% generated by make_numbers.py -- do not edit\n" +
                   "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items())))
    print(out.read_text())


if __name__ == "__main__":
    main()
