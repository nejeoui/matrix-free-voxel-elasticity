"""Session I analysis: ladder, time to 1e-8, fixed-work ratios, H13/H14 and the small-mesh check, per GPU.

    python3 analysis/session_i_tables.py results/session-i-rtx4090-20260926 [results/session-i-a100-... ...]

Decision rules are read from the frozen protocol (experiments/session_i/protocol.json).
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "experiments/session_i/protocol.json").read_text())
PRIMARY = [c["id"] for c in PROTOCOL["cases"] if c["primary"]]
ORDER = [c["id"] for c in PROTOCOL["cases"]]


def fixed_ratio(fixed, case, num, den):
    by = {}
    for r in fixed:
        if r["case"] == case:
            by.setdefault(r["round"], {})[r["arm"]] = r["solver_seconds"]
    ratios = [row[num] / row[den] for row in by.values() if num in row and den in row]
    return statistics.median(ratios) if ratios else None, ratios


def analyze(d):
    remote = Path(d) / "remote"
    res = json.loads((remote / "run-01/result.json").read_text())
    dev = res.get("device", {}).get("name", "?")
    out = {"gpu": dev, "ladder": res["ladder"], "cases": {}}
    fixed = res["fixed_work"]
    for case in [c for c in ORDER if any(t["case"] == c for t in res["to_tolerance"])]:
        tol = [t for t in res["to_tolerance"] if t["case"] == case]
        row = {"physical_all_passed": all(t["physical_passed"] for t in tol),
               "iterations": sorted({t["iterations"] for t in tol}),
               "tol_seconds_median": {a: statistics.median(t["solver_seconds"] for t in tol if t["arm"] == a)
                                      for a in sorted({t["arm"] for t in tol})}}
        g = next((c for c in res["cases"] if c["case"] == case), {})
        row["shapes"] = g.get("shapes")
        for name, (num, den) in {"fused_over_modal_mg64": ("mg64-fused_ai", "mg64-modal8_muladd"),
                                 "node_over_modal_mg64": ("mg64-node", "mg64-modal8_muladd"),
                                 "modal_over_node_mg64": ("mg64-modal8_muladd", "mg64-node"),
                                 "fused_over_modal_mg32": ("mg32-fused_ai64-node32", "mg32-modal8_muladd-node32"),
                                 "mg64modal_over_mg32modal": ("mg64-modal8_muladd", "mg32-modal8_muladd-node32")}.items():
            row[name] = fixed_ratio(fixed, case, num, den)[0]
        out["cases"][case] = row
    fitting_primary = [c for c in PRIMARY if c in out["cases"]]
    if "4090" in dev:
        largest = fitting_primary[-1] if fitting_primary else None
        r = out["cases"][largest]["fused_over_modal_mg64"] if largest else None
        out["H14"] = {"case": largest, "median_fused_over_modal": r,
                      "passed": r is not None and r >= 1.18 and out["cases"][largest]["physical_all_passed"]}
    else:
        rows = {c: out["cases"][c]["modal_over_node_mg64"] for c in fitting_primary}
        out["H13"] = {"median_modal_over_node": rows,
                      "passed": bool(rows) and all(v is not None and v >= 1 / 1.05 for v in rows.values())}
    w2a = remote / "w2a-01/result.json"
    if w2a.exists():
        w = json.loads(w2a.read_text())
        out["w2a_status"] = w.get("status")
        recs = [r for r in w.get("records", []) if "fixed_work" in r]
        table = {}
        for r in recs:
            key = (r["size"], r["schedule"])
            t = statistics.median(x["seconds"] for x in r["fixed_work"])
            table.setdefault(key, {}).setdefault(r["arm"], []).append(t)
        out["w2a"] = {f"{s}/{sch}": {a: statistics.median(v) for a, v in arms.items()} for (s, sch), arms in table.items()}
        out["w2a_setup_memory"] = {f"{r['size']}/{r['schedule']}/{r['arm']}": (round(r["setup_seconds"], 1),
                                   round(r["gpu_used_bytes"] / 2**30, 1)) for r in recs if r["block"] == 0}
        out["w2a_errors"] = [r.get("error", "")[:160] for r in w.get("records", []) if "error" in r]
    return out


def main():
    hosts = {Path(d).name: analyze(d) for d in sys.argv[1:]}
    (ROOT / "paper/session_i_numbers.json").write_text(json.dumps(hosts, indent=1) + "\n")
    for name, h in hosts.items():
        print(f"== {name} ({h['gpu']})")
        print("  ladder:", [(l["case"], l["status"]) for l in h["ladder"]])
        for c, r in h["cases"].items():
            best = min(r["tol_seconds_median"].items(), key=lambda kv: kv[1])
            print(f"  {c}: its {r['iterations']} phys={r['physical_all_passed']} fastest {best[0]} {best[1]:.2f}s | "
                  f"fixed fused/modal(64) {r['fused_over_modal_mg64'] and round(r['fused_over_modal_mg64'], 3)} "
                  f"node/modal(64) {r['node_over_modal_mg64'] and round(r['node_over_modal_mg64'], 3)} "
                  f"fused/modal(32) {r['fused_over_modal_mg32'] and round(r['fused_over_modal_mg32'], 3)}")
        for k in ("H13", "H14"):
            if k in h:
                print(f"  {k}: {h[k]}")
        if "w2a" in h:
            print("  small-mesh check:", h["w2a"], "errors:", h["w2a_errors"])


if __name__ == "__main__":
    main()
