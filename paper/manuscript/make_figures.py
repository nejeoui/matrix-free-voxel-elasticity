"""Generate every figure, table and quoted number of the manuscript from verified evidence.

Inputs: results/session-{a,b,c,d}-20260923/local-verification.json (independent CPU replays and the
pre-declared decisions), session D's result.json (profiles, hierarchy), paper/numbers.json (companion
evidence re-derived by analysis/paper_numbers.py) and paper/session_e_numbers.json (session E, two
RTX 4090 hosts, plus companion instruction counts and libCEED ratios; analysis/session_e_tables.py). Outputs in this directory: fig_*.pdf, tab_*.tex
and numbers.tex (LaTeX macros). The manuscript must not contain a hand-typed result.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
V = HERE.parents[1]
R = V / "results"


def load(p):
    return json.loads(Path(p).read_text())


A = load(R / "session-a-20260923/local-verification.json")["decision"]
B = load(R / "session-b-20260923/local-verification.json")["decision"]
C = load(R / "session-c-20260923/local-verification.json")["decision"]
D = load(R / "session-d-20260923/local-verification.json")["decision"]
Dres = load(R / "session-d-20260923/remote/run-01/result.json")
J = load(V / "paper/numbers.json")
EN = load(V / "paper/session_e_numbers.json")
E1, E2 = EN["hosts"]["host1"], EN["hosts"]["host2"]
EREAL = ["optimized-q64", "optimized-q96", "uniform-q64"]

# ------------------------------------------------------------------ style (validated palette)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHTGRAY, INK, INK2 = "#9a9993", "#cfcec8", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
COL = 3.45   # IEEE single-column width, inches

macros = {}


def m(name, value, fmt="{:.2f}"):
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


# ------------------------------------------------------------------ helpers
def product_times(dec, case, precision):
    """{kernel: median ms per complete constrained product} for one case/precision block."""
    rows = [e for e in dec["products"] if e["case"] == case and e["precision"] == precision]
    t = {e["baseline"]: e["baseline_median_ms"] for e in rows}
    cand = "modal8" if precision == "fp64" else "modal8_fp32"
    t[cand] = rows[0]["candidate_median_ms"]
    return t


def ratio(dec_list, case, baseline, key="baseline"):
    return next(e for e in dec_list if e["case"] == case and e[key] == baseline)


PUBLISHED64 = ["fused_ai_fp64", "fused_fp64", "node_fp64"]
PUBLISHED32 = ["fused_ai_fp32", "fused_fp32", "node_fp32"]
LABEL = {"fused_ai_fp64": "fused-ai", "fused_fp64": "fused", "node_fp64": "node", "dense8": "dense8",
         "modal8": "parity, plain", "fused_ai_fp32": "fused-ai", "fused_fp32": "fused", "node_fp32": "node",
         "dense8_fp32": "dense8", "modal8_fp32": "parity, plain"}

# ------------------------------------------------------------------ Figure 1: product speed map
panels = [("RTX 4090, FP64", product_times(A, "optimized-q96", "fp64"), PUBLISHED64),
          ("A100, FP64", product_times(B, "optimized-q96", "fp64"), PUBLISHED64),
          ("RTX 4090, FP32", product_times(A, "optimized-q96", "fp32"), PUBLISHED32),
          ("A100, FP32", product_times(B, "optimized-q96", "fp32"), PUBLISHED32)]
fig, axes = plt.subplots(2, 2, figsize=(COL, 2.55), sharex=True)
for ax, (title, t, published) in zip(axes.ravel(), panels):
    ref = min(t[k] for k in published)                 # fastest published kernel = 1.0
    suffix = "" if published is PUBLISHED64 else "_fp32"
    order = list(published) + ["dense8" + suffix, "modal8" + suffix]      # same order in every panel
    speeds = [ref / t[k] for k in order]
    colors = [BLUE if k.startswith("modal8") else GRAY for k in order]
    y = range(len(order))
    ax.barh(y, speeds, color=colors, height=0.62, edgecolor="white", linewidth=0.8)
    ax.axvline(1.0, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)))
    for yi, s, k in zip(y, speeds, order):
        ax.text(s + 0.03, yi, f"{s:.2f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(list(y), [LABEL[k] for k in order])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", color=INK, pad=2)
    ax.set_xlim(0, 1.75)
    ax.tick_params(length=2, pad=1.5)
fig.tight_layout(h_pad=0.6, w_pad=0.8)
fig.supxlabel("speed relative to the fastest published kernel (dashed = 1)", fontsize=8, y=-0.03)
fig.savefig(HERE / "fig_products.pdf", metadata={"CreationDate": None})  # no timestamp: rebuilds are byte-identical
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: multigrid time to 1e-8
case = "optimized-q96"
tol = [{"arm": a, "median_seconds": v["seconds"]} for a, v in E1["to_tolerance"][case].items()]
ARM = {"mg64-fused_ai": "FP64 V: fused-ai", "mg64-fused": "FP64 V: fused", "mg64-node": "FP64 V: node",
       "mg64-dense8": "FP64 V: dense8", "mg64-modal8": "FP64 V: parity, plain",
       "mg64-modal8_muladd": "FP64 V: parity, mul-add", "mg64-modal8_signbit": "FP64 V: parity, sign-bit",
       "mg32-node64-node32": "FP32 V | outer node", "mg32-fused_ai64-node32": "FP32 V | outer fused-ai",
       "mg32-modal8-node32": "FP32 V | outer parity, plain", "mg32-modal8_muladd-node32": "FP32 V | outer parity, mul-add"}
tol.sort(key=lambda e: e["median_seconds"])
fig, ax = plt.subplots(figsize=(COL, 2.5))
y = range(len(tol))
ax.barh(y, [e["median_seconds"] for e in tol], height=0.62, edgecolor="white", linewidth=0.8,
        color=[BLUE if "modal8" in e["arm"] else GRAY for e in tol])
for yi, e in zip(y, tol):
    ax.text(e["median_seconds"] + 0.012, yi, f"{e['median_seconds']:.3f} s", va="center", fontsize=6.5, color=INK)
ax.set_yticks(list(y), [ARM[e["arm"]] for e in tol])
ax.invert_yaxis()
ax.set_xlabel("time to a true relative residual of $10^{-8}$ (s)")
ax.set_xlim(0, max(e["median_seconds"] for e in tol) * 1.22)
ax.tick_params(length=2, pad=1.5)
fig.tight_layout()
fig.savefig(HERE / "fig_mg_time.pdf", metadata={"CreationDate": None})  # no timestamp: rebuilds are byte-identical
plt.close(fig)

# ------------------------------------------------------------------ Figure 3: multigrid time breakdown
KE = int(next(r["fixed_iterations"] for r in load(R / "session-e-host1-20260925/remote/run-01/result.json")["profiles"]
                if r["case"] == case))
prof = [dict(v, arm=a, fixed_iterations=KE) for a, v in E1["profiles"][case].items()]
PL = {"mg64-fused_ai": "FP64 V, fused-ai", "mg64-modal8": "FP64 V, parity plain",
      "mg64-modal8_muladd": "FP64 V, parity mul-add",
      "mg32-fused_ai64-node32": "FP32 V (node),\nouter fused-ai", "mg32-modal8_muladd-node32": "FP32 V (node),\nouter mul-add"}
fig, ax = plt.subplots(figsize=(COL, 2.2))
for i, p in enumerate(prof):
    v, o, tot = p["vcycle_finest_product_seconds"], p["outer_product_seconds"], p["solver_seconds"]
    parts = [(v, BLUE, "V-cycle finest products"), (o, ORANGE, "outer FP64 products"),
             (tot - v - o, LIGHTGRAY, "everything else")]
    left = 0.0
    for val, colr, lab in parts:
        ax.barh(i, val * 1000, left=left * 1000, color=colr, height=0.6, edgecolor="white", linewidth=1.0,
                label=lab if i == 0 else None)
        if val * 1000 > 90:                              # inline only where it fits
            ax.text((left + val / 2) * 1000, i, f"{val / tot:.0%}", ha="center", va="center", fontsize=6.3,
                    color="white" if colr != LIGHTGRAY else INK)
        left += val
    small = f"\nV {v / tot:.0%} · outer {o / tot:.0%}" if v * 1000 <= 90 else ""
    ax.text(tot * 1000 + 8, i, f"{tot * 1000:.0f} ms{small}", va="center", fontsize=6.3, color=INK,
            linespacing=1.1)
ax.set_yticks(range(len(prof)), [PL[p["arm"]] for p in prof])
ax.invert_yaxis()
ax.set_xlabel(f"one fixed-work solve ({prof[0]['fixed_iterations']} iterations), ms")
ax.set_xlim(0, max(p["solver_seconds"] for p in prof) * 1000 * 1.22)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=6.3, frameon=False, ncol=3,
          handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
ax.tick_params(length=2, pad=1.5)
fig.tight_layout()
fig.savefig(HERE / "fig_mg_breakdown.pdf", metadata={"CreationDate": None})  # no timestamp: rebuilds are byte-identical
plt.close(fig)

# ------------------------------------------------------------------ macros used in the text
kp = {(r["gpu"], r["elements"], r["baseline"]): r for r in J["kernel_products"]}
m("ProdQnineSixFourNineOhFusedAi", kp[("RTX 4090", 1769472, "fused_ai_fp64")]["ratio"])
m("ProdQsixFourFourNineOhFusedAi", kp[("RTX 4090", 524288, "fused_ai_fp64")]["ratio"])
m("ProdThreeNineOhQnineSix", kp[("RTX 3090", 1769472, "fused_ai_fp64")]["ratio"])
m("ProdThreeNineOhQsixFour", kp[("RTX 3090", 524288, "fused_ai_fp64")]["ratio"])
ach = J["achieved_fp64_rates"]
m("DenseFracMin", min(a["dense8_fraction_of_fp64_peak"] for a in ach) * 100, "{:.0f}")
m("DenseFracMax", max(a["dense8_fraction_of_fp64_peak"] for a in ach) * 100, "{:.0f}")
m("ModalFracMin", min(a["modal8_fraction_of_fp64_peak"] for a in ach) * 100, "{:.0f}")
m("ModalFracMax", max(a["modal8_fraction_of_fp64_peak"] for a in ach) * 100, "{:.0f}")
sp = J["solve_profile"]["summaries"]
share = {s["case"]: s["profile_product_fraction"] for s in sp if s["path"] == "fused_ai_fp64"}
m("JacobiShareQsixFour", share["optimized-q64"] * 100, "{:.0f}")
m("JacobiShareQnineSix", share["optimized-q96"] * 100, "{:.0f}")
# Session A fixed-work Jacobi-PCG and products on physical states
fa = {e["case"]: e for e in A["fixed_work"] if e["baseline"] == "fused_ai_fp64"}
m("JacobiFixedQsixFour", fa["optimized-q64"]["median_ratio"], "{:.3f}")
m("JacobiFixedQnineSix", fa["optimized-q96"]["median_ratio"], "{:.3f}")
m("JacobiFixedLoQsixFour", fa["optimized-q64"]["lower95"], "{:.3f}")
m("JacobiFixedLoQnineSix", fa["optimized-q96"]["lower95"], "{:.3f}")
node32 = ratio(A["products"], "optimized-q96", "node_fp32")
m("NodeThirtyTwoOverParity", 1 / node32["median_ratio"])
dense32 = ratio(A["products"], "optimized-q96", "dense8_fp32")
m("ParityThirtyTwoOverDense", dense32["median_ratio"])
# A100
bp = {e["baseline"]: e for e in B["products"] if e["case"] == "optimized-q96" and e["precision"] == "fp64"}
m("AhundredVsFusedAi", bp["fused_ai_fp64"]["median_ratio"])
m("AhundredVsNode", bp["node_fp64"]["median_ratio"])
m("AhundredDenseOverParity", bp["dense8"]["median_ratio"])
bf = {e["case"]: e for e in B["fixed_work"] if e["baseline"] == "fused_ai_fp64"}
m("AhundredFixedQnineSix", bf["optimized-q96"]["median_ratio"])
# time to tolerance, Jacobi-PCG (A, B, C)
def tt(dec, case, method):
    return next(e["median_seconds"] for e in dec["to_tolerance"] if e["case"] == case and e["method"] == method)
m("ATolPcgModalQnineSix", tt(A, "optimized-q96", "pcg64-modal8"))
m("ATolPcgFusedQnineSix", tt(A, "optimized-q96", "pcg64-fused_ai"))
m("ATolIrQnineSix", tt(A, "optimized-q96", "ir-modal8-fused_ai32"))
m("CTolIrNodeQnineSix", tt(C, "optimized-q96-E1e-6", "ir-fused_ai64-node32"))
m("CTolPcgModalQnineSix", tt(C, "optimized-q96-E1e-6", "pcg64-modal8"))
m("CTolPcgModalQsixFour", tt(C, "optimized-q64-E1e-6", "pcg64-modal8"))
m("CTolIrNodeQsixFour", tt(C, "optimized-q64-E1e-6", "ir-fused_ai64-node32"))
m("ATolIrFusedQsixFour", tt(A, "optimized-q64", "ir-fused_ai64-fused_ai32"))
m("CTolIrFusedQsixFour", tt(C, "optimized-q64-E1e-6", "ir-fused_ai64-fused_ai32"))
m("BTolPcgFusedQnineSix", tt(B, "optimized-q96", "pcg64-fused_ai"))
m("BTolBestIrQnineSix", min(e["median_seconds"] for e in B["to_tolerance"]
                            if e["case"] == "optimized-q96" and e["method"].startswith("ir")))
# multigrid (D)
its = {e["case"]: e["iterations"][0] for e in D["to_tolerance"] if e["arm"] == "mg64-fused_ai"}
m("MgItQsixFour", its["optimized-q64"], "{:d}")
m("MgItQnineSix", its["optimized-q96"], "{:d}")
m("MgItUniform", its["uniform-q64"], "{:d}")
dfw = {(e["case"], e["numerator"]): e for e in D["fixed_work"]}
m("MgFixedQsixFour", dfw[("optimized-q64", "mg64-fused_ai")]["median_ratio"], "{:.3f}")
m("MgFixedQnineSix", dfw[("optimized-q96", "mg64-fused_ai")]["median_ratio"], "{:.3f}")
m("MgFixedLoQsixFour", dfw[("optimized-q64", "mg64-fused_ai")]["lower95"], "{:.3f}")
m("MgFixedLoQnineSix", dfw[("optimized-q96", "mg64-fused_ai")]["lower95"], "{:.3f}")
m("MgFixedNodeQnineSix", dfw[("optimized-q96", "mg64-node")]["median_ratio"])
oe = {c: e for c, e in D["outer_effect"].items() if c != "smoke-q4"}   # smoke case is latency noise
m("OuterEffectQsixFour", (oe["optimized-q64"]["median_ratio"] - 1) * 100, "{:.1f}")
m("OuterEffectQnineSix", (oe["optimized-q96"]["median_ratio"] - 1) * 100, "{:.1f}")
m("OuterEffectLoMin", (min(e["lower95"] for e in oe.values()) - 1) * 100, "{:.1f}")
dt = {(e["case"], e["arm"]): e["median_seconds"] for e in D["to_tolerance"]}
m("MgBestQnineSix", dt[("optimized-q96", "mg32-modal8-node32")], "{:.3f}")
m("MgBestQsixFour", dt[("optimized-q64", "mg32-modal8-node32")], "{:.3f}")
m("MgFpSixFourFusedQnineSix", dt[("optimized-q96", "mg64-fused_ai")], "{:.3f}")
m("MgMixedGainQnineSix", dt[("optimized-q96", "mg64-fused_ai")] / dt[("optimized-q96", "mg32-modal8-node32")])
m("MgMixedGainQsixFour", dt[("optimized-q64", "mg64-fused_ai")] / dt[("optimized-q64", "mg32-modal8-node32")])
pr = {p["arm"]: p for p in D["profiles"] if p["case"] == "optimized-q96"}
m("MgVShareFp", pr["mg64-fused_ai"]["vcycle_finest_product_seconds"] / pr["mg64-fused_ai"]["solver_seconds"] * 100, "{:.0f}")
m("MgOuterShareMixed", pr["mg32-fused_ai64-node32"]["outer_product_seconds"] / pr["mg32-fused_ai64-node32"]["solver_seconds"] * 100, "{:.0f}")
m("JacobiOverMgBest", tt(A, "optimized-q96", "pcg64-fused_ai") / dt[("optimized-q96", "mg32-modal8-node32")], "{:.0f}")
# roofline, scaling and structural numbers (paper/roofline.json, paper/numbers.json, evidence/algebra)
roof = {(r["device"], r["precision"]): r for r in load(V / "paper/roofline.json")}
m("RidgeThreeNinetyFpSixFour", roof[("RTX 3090", "FP64")]["ridge_flop_per_byte"])
m("RidgeFourNinetyFpSixFour", roof[("RTX 4090", "FP64")]["ridge_flop_per_byte"])
m("RidgeAhundredFpSixFour", roof[("A100 80GB SXM", "FP64")]["ridge_flop_per_byte"], "{:.1f}")
m("RidgeAhundredFpThirtyTwo", roof[("A100 80GB SXM", "FP32")]["ridge_flop_per_byte"], "{:.1f}")
m("RidgeThreeNinetyFpThirtyTwo", roof[("RTX 3090", "FP32")]["ridge_flop_per_byte"], "{:.0f}")
m("RidgeFourNinetyFpThirtyTwo", roof[("RTX 4090", "FP32")]["ridge_flop_per_byte"], "{:.0f}")
m("DenseIntensityFpSixFour", roof[("RTX 4090", "FP64")]["dense_intensity"], "{:.1f}")
m("DenseIntensityFpThirtyTwo", roof[("RTX 4090", "FP32")]["dense_intensity"], "{:.1f}")
spec = J["device_spec"]
m("ComputeRatio", spec["RTX 4090"]["fp64_tflops"] / spec["RTX 3090"]["fp64_tflops"])
m("BandwidthRatio", spec["RTX 4090"]["dram_gbs"] / spec["RTX 3090"]["dram_gbs"])
kt = {(r["gpu"], r["elements"], k): r[f] for r in J["kernel_products"] for k, f in
      (("modal8", "modal8_ms"), (r["baseline"], "baseline_ms"))}
fused_like = ["modal8", "dense8", "fused_ai_fp64", "fused_fp64", "node_fp64"]
sc = [kt[("RTX 3090", el, k)] / kt[("RTX 4090", el, k)] for el in (524288, 1769472) for k in fused_like]
m("ScalingMin", min(sc)); m("ScalingMax", max(sc))
m("AhundredDenseOverFusedAi", bp["dense8"]["baseline_median_ms"] / bp["fused_ai_fp64"]["baseline_median_ms"], "{:.1f}")
basis = load(V / "evidence/algebra/results.json")
m("OffBlockShare", "{:.1f}".format(basis["off_block_share_rectangular_brick"] * 1e16) + r"\times10^{-16}")
# ---- session E (two RTX 4090 hosts) and companion evidence
def span(get):
    vals = [get(h, c) for h in (E1, E2) for c in EREAL]
    return min(vals), max(vals)
lo, hi = span(lambda h, c: h["products"][c]["fused_ai_fp64"]["ratio"]); m("MuladdProdFusedMin", lo); m("MuladdProdFusedMax", hi)
lo, hi = span(lambda h, c: h["products"][c]["modal8"]["ratio"]); m("MuladdProdModalMin", lo); m("MuladdProdModalMax", hi)
lo, hi = span(lambda h, c: h["products"][c]["dense8"]["ratio"]); m("MuladdProdDenseMin", lo); m("MuladdProdDenseMax", hi)
lo, hi = span(lambda h, c: h["products"][c]["node_fp64"]["ratio"]); m("MuladdProdNodeMin", lo); m("MuladdProdNodeMax", hi)
lo, hi = span(lambda h, c: h["products"][c]["modal8_signbit"]["ratio"])
m("SignbitSlowerMin", (lo - 1) * 100, "{:.0f}"); m("SignbitSlowerMax", (hi - 1) * 100, "{:.0f}")
m("MuladdLbWithin", max(1 - h["products"][c][k]["lower95"] / h["products"][c][k]["ratio"]
                        for h in (E1, E2) for c in EREAL for k in h["products"][c]) * 100, "{:.1f}")
for tag, c in (("QsixFour", "optimized-q64"), ("QnineSix", "optimized-q96")):
    a, b = sorted(h["mg_fixed"][c]["mg64-fused_ai"]["ratio"] for h in (E1, E2))
    m(f"MuladdMgFused{tag}Lo", a); m(f"MuladdMgFused{tag}Hi", b)
    a, b = sorted(h["mg_fixed"][c]["mg64-modal8"]["ratio"] for h in (E1, E2))
    m(f"MuladdMgModal{tag}Lo", a); m(f"MuladdMgModal{tag}Hi", b)
lo, hi = span(lambda h, c: h["mg_fixed"][c]["mg64-fused_ai"]["lower95"]); m("MuladdMgLbMin", lo)
lo, hi = span(lambda h, c: h["outer_effect"][c]["ratio"])
m("MuladdOuterMin", (lo - 1) * 100, "{:.1f}"); m("MuladdOuterMax", (hi - 1) * 100, "{:.1f}")
lo, hi = span(lambda h, c: h["outer_effect"][c]["lower95"]); m("MuladdOuterLoMin", (lo - 1) * 100, "{:.1f}")
m("MuladdBestQnineSixHone", E1["to_tolerance"]["optimized-q96"]["mg32-modal8_muladd-node32"]["seconds"], "{:.3f}")
m("MuladdBestQnineSixHtwo", E2["to_tolerance"]["optimized-q96"]["mg32-modal8_muladd-node32"]["seconds"], "{:.3f}")
m("MuladdFpSixFourQnineSixHone", E1["to_tolerance"]["optimized-q96"]["mg64-modal8_muladd"]["seconds"], "{:.3f}")
m("FusedFpSixFourQnineSixHone", E1["to_tolerance"]["optimized-q96"]["mg64-fused_ai"]["seconds"], "{:.3f}")
m("MuladdMixedGainQnineSix", E1["to_tolerance"]["optimized-q96"]["mg64-fused_ai"]["seconds"]
  / E1["to_tolerance"]["optimized-q96"]["mg32-modal8_muladd-node32"]["seconds"])
pe = [h["profiles"]["optimized-q96"] for h in (E1, E2)]
vs = [p["mg64-fused_ai"]["vcycle_finest_product_seconds"] / p["mg64-fused_ai"]["solver_seconds"] * 100 for p in pe]
m("EVShareFusedMin", min(vs), "{:.0f}"); m("EVShareFusedMax", max(vs), "{:.0f}")
vs = [p["mg64-modal8_muladd"]["vcycle_finest_product_seconds"] / p["mg64-modal8_muladd"]["solver_seconds"] * 100 for p in pe]
m("EVShareMuladdMin", min(vs), "{:.0f}"); m("EVShareMuladdMax", max(vs), "{:.0f}")
os_ = [p["mg32-fused_ai64-node32"]["outer_product_seconds"] / p["mg32-fused_ai64-node32"]["solver_seconds"] * 100 for p in pe]
m("EOuterShareMin", min(os_), "{:.0f}"); m("EOuterShareMax", max(os_), "{:.0f}")
m("MgItsSame", "yes" if all(v["iterations"] == [{"optimized-q64": 13, "optimized-q96": 24, "uniform-q64": 9}[c]]
                            for h in (E1, E2) for c in EREAL for v in h["to_tolerance"][c].values()) else "NO")
si = EN["companion_fp64_instruction_sites"]
m("SitesModal", si["modal"]["total"], "{:d}"); m("SitesDense", si["dense"]["total"], "{:d}")
m("SitesMuladd", si["modal_signed_muladd"]["total"], "{:d}"); m("SitesSignbit", si["modal_signbit"]["total"], "{:d}")
m("SitesSymHalf", si["symmetric_half"]["total"], "{:d}")
m("SitesDenseOverMuladd", si["dense"]["total"] / si["modal_signed_muladd"]["total"])
m("SitesModalOverMuladd", si["modal"]["total"] / si["modal_signed_muladd"]["total"])
cd = EN["companion_libceed"]
m("CeedGenMin", cd["ceed_gen"]["min"], "{:.1f}"); m("CeedGenMax", cd["ceed_gen"]["max"], "{:.1f}")
m("CeedRefMin", min(cd["ceed_ref"]["min"], cd["ceed_shared"]["min"]), "{:.0f}")
m("CeedRefMax", max(cd["ceed_ref"]["max"], cd["ceed_shared"]["max"]), "{:.0f}")
m("SymHalfMin", cd["symmetric_half"]["min"]); m("SymHalfMax", cd["symmetric_half"]["max"])
m("VariantMaxRel", "{:.0f}".format(max(E1["variant_max_rel"], E2["variant_max_rel"]) * 1e17) + r"\times10^{-17}")
m("EHostOneMachine", E1["machine_id"], "{:d}"); m("EHostTwoMachine", E2["machine_id"], "{:d}")

# ------------------------------------------------------------------ Table II: butterfly forms, session E
rowsE = []
for c, lab in (("optimized-q64", "0.52 M opt."), ("optimized-q96", "1.77 M opt."), ("uniform-q64", "0.52 M unif.")):
    f = lambda k, src: " / ".join(f"{h[src][c][k]['ratio']:.2f}" for h in (E1, E2))
    rowsE.append(f"{lab} & {f('fused_ai_fp64', 'products')} & {f('modal8', 'products')} & {f('mg64-fused_ai', 'mg_fixed')} \\\\")
(HERE / "tab_butterfly.tex").write_text("\n".join([
    "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
    "& \\multicolumn{2}{c}{product: X / mul-add} & multigrid, FP64 \\\\",
    "\\cmidrule(lr){2-3}\\cmidrule(l){4-4}",
    "design & X = fused-ai & X = parity plain & fused-ai / mul-add \\\\", "\\midrule"] + rowsE +
    ["\\bottomrule", "\\end{tabular}"]) + "\n")

(HERE / "numbers.tex").write_text("% generated by make_figures.py -- do not edit\n" +
                                  "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))

# ------------------------------------------------------------------ Table I: FP64 across hardware
rowsT = []
for gpu, el, src in [("RTX 4090", 524288, "synthetic field"), ("RTX 4090", 1769472, "synthetic field"),
                     ("RTX 3090", 524288, "synthetic field"), ("RTX 3090", 1769472, "synthetic field")]:
    r = kp[(gpu, el, "fused_ai_fp64")]
    rowsT.append((gpu, f"{el / 1e6:.2f} M", src, f"{r['ratio']:.3f}", f"{r['lower95']:.3f}"))
for dec, gpu in [(A, "RTX 4090"), (B, "A100")]:
    for c, el in [("optimized-q64", 524288), ("optimized-q96", 1769472)]:
        e = next(x for x in dec["products"] if x["case"] == c and x["precision"] == "fp64" and x["baseline"] == "fused_ai_fp64")
        rowsT.append((gpu, f"{el / 1e6:.2f} M", "optimized design", f"{e['median_ratio']:.3f}", f"{e['lower95']:.3f}"))
tab = ["\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
       "GPU & elements & input & ratio & 95\\% lower \\\\", "\\midrule"]
tab += [" & ".join(r) + " \\\\" for r in rowsT]
tab += ["\\bottomrule", "\\end{tabular}"]
(HERE / "tab_products.tex").write_text("\n".join(tab) + "\n")

# ------------------------------------------------------------------ Table II: fastest route per regime
reg = [("RTX 4090", "Jacobi-PCG", "FP64 PCG, parity", tt(A, "optimized-q96", "pcg64-modal8")),
       ("RTX 4090", "Jacobi-PCG", "refinement, node32 inner", tt(C, "optimized-q96-E1e-6", "ir-fused_ai64-node32")),
       ("A100", "Jacobi-PCG", "FP64 PCG, fused-ai", tt(B, "optimized-q96", "pcg64-fused_ai")),
       ("RTX 4090", "multigrid", "FP64 V-cycle, fused-ai", E1["to_tolerance"]["optimized-q96"]["mg64-fused_ai"]["seconds"]),
       ("RTX 4090", "multigrid", "FP64 V-cycle, mul-add", E1["to_tolerance"]["optimized-q96"]["mg64-modal8_muladd"]["seconds"]),
       ("RTX 4090", "multigrid", "FP32 V, outer fused-ai", E1["to_tolerance"]["optimized-q96"]["mg32-fused_ai64-node32"]["seconds"]),
       ("RTX 4090", "multigrid", "FP32 V, outer mul-add", E1["to_tolerance"]["optimized-q96"]["mg32-modal8_muladd-node32"]["seconds"])]
tab = ["\\begin{tabular}{@{}lllr@{}}", "\\toprule", "GPU & solver & configuration & time (s) \\\\", "\\midrule"]
tab += [f"{g} & {s} & {c} & {t:.3f} \\\\" for g, s, c, t in reg]
tab += ["\\bottomrule", "\\end{tabular}"]
(HERE / "tab_routes.tex").write_text("\n".join(tab) + "\n")
print(f"figures: 3, macros: {len(macros)}, tables: 3")
