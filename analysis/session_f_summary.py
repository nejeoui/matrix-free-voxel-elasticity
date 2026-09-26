"""Session F summary: counters, smoke, integration and sizing per GPU, from the verified evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"RTX 4090": "results/session-f-rtx4090-20260926", "A100 80GB PCIe": "results/session-f-a100-20260926",
        "H100 NVL": "results/session-f-h100-20260926-attempt02"}


def main():
    out = {}
    for gpu, d in RUNS.items():
        remote = ROOT / d / "remote"
        job = json.loads((remote / "job.json").read_text())
        res = json.loads((remote / "run-01/result.json").read_text())
        integ = res["stages"]["integration"]["rows"]
        products = [r["relative_l2"] for r in integ if r["kind"] == "product"]
        solves = [r for r in integ if r["kind"] == "solve"]
        failed = [f"{r['grid']}/{r['field']}/{r['schedule']}/{r['arm']}: {r['iterations']} it, "
                  f"residual {r['true_residual']:.1e}" for r in solves
                  if not (r["converged"] and r["true_residual"] <= 1e-10 and r["solution_relative_l2_vs_stock"] <= 1e-8)]
        sizing = json.loads((remote / "run-01/sizing.json").read_text())["rows"]
        out[gpu] = {
            "counters_available": job["ncu"]["counters_available"],
            "ncu_permission_errors": sum(r.get("permission_error", False) for r in job["ncu"]["rows"]),
            "smoke": {r["variant"]: r["returncode"] for r in res["stages"]["smoke"]["rows"]},
            "max_product_relative_l2": max(products),
            "solves": len(solves), "solves_failing_rule": failed,
            "integration_ok": res["stages"]["integration"]["integration_ok"],
            "sizing": [{k: r.get(k) for k in ("size", "schedule", "arm", "status", "ndof", "setup_seconds",
                                              "seconds_per_iteration")} |
                       {"gpu_used_gib": round(r.get("gpu_used_after_bytes", 0) / 2**30, 1)} for r in sizing],
        }
    path = ROOT / "paper/session_f_summary.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
