"""Voxel session F probe: Yang et al.'s GMG stack, the three FP64-operator arms inside it, and sizing.

    run_probe.py --out DIR                        # stages: smoke (as released / adapted),
                                                  # integration, sizing (one subprocess per size)
    run_probe.py --size SIZE_ID --schedule S --arm A --result FILE   # one sizing case (internal)

Timings here are single, unrepeated observations for planning (protocol "not_claimed").
"""
import argparse
import dataclasses
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                  # <source>/experiments/session_f
SRC = HERE.parents[1]                                    # <source>
DONOR = SRC / "experiments/combined/donor_yang_gmg"
for p in (SRC / "experiments/combined", DONOR / "src", SRC / "experiments/session_e/dependencies"):
    sys.path.insert(0, str(p))
from common import now, read, write  # noqa: E402

PROTOCOL = read(HERE / "protocol.json")


def gpu_memory_used():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    return int(total - free), int(total)


def sync_time():
    import cupy as cp
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


class Problem:
    """Their cantilever preset (optionally resized), with their BC generator and DOF table."""

    def __init__(self, entry):
        import cupy as cp
        from gpu_fem.bc_generator import generate_bc
        from gpu_fem.presets import get_preset
        from gpu_fem.pub_simp_solver import KE_UNIT_3D, _edof_table_3d
        spec = get_preset(entry["preset"])
        if "nel" in entry:
            spec = dataclasses.replace(spec, nelx=entry["nel"][0], nely=entry["nel"][1], nelz=entry["nel"][2])
        self.dims = (spec.nelx, spec.nely, spec.nelz)
        self.n_elem = int(np.prod(self.dims))
        t = time.perf_counter()
        bc = generate_bc(spec)
        self.bc_seconds = time.perf_counter() - t
        self.ndof = int(bc.ndof)
        self.free = bc.free_dofs.astype(np.int32)
        self.free_gpu = cp.asarray(self.free)
        self.edof_gpu = cp.asarray(_edof_table_3d(*self.dims).astype(np.int32))
        self.ke = KE_UNIT_3D
        self.ke_gpu = cp.asarray(KE_UNIT_3D)
        self.F = cp.asarray(bc.F[self.free].astype(np.float64))

    def modulus(self, rho):
        import cupy as cp
        c = PROTOCOL["integration"]
        rho = cp.asarray(rho, dtype=cp.float64)
        return c["e_min"] + (1.0 - c["e_min"]) * rho ** c["penal"]

    def operator(self, arm):
        import kff_adapters
        ArmKff = kff_adapters.make_operator_class()
        core = kff_adapters.ArmCore(arm, self.dims, self.ke)
        return ArmKff(edof_gpu=self.edof_gpu, KE_unit_gpu=self.ke_gpu, free_gpu=self.free_gpu,
                      n_free=len(self.free), ndof=self.ndof, core=core)

    def gmg(self, op, schedule):
        from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
        s = PROTOCOL["schedules"][schedule]
        fused = None
        if s["fused_fine"]:
            from gpu_fem.cuda_fused_matvec import FusedMatvec
            fused = FusedMatvec(edof_gpu=self.edof_gpu, KE_unit_gpu=self.ke_gpu, ndof=self.ndof)
        nelx, nely, nelz = self.dims
        return GalerkinMatFreeGMG(mf_op=op, free=self.free, free_gpu=self.free_gpu, nelx=nelx, nely=nely,
                                  nelz=nelz, KE_UNIT=self.ke, n_levels=s["n_levels"],
                                  fine_smoother=s["fine_smoother"], smoother_type=s["smoother_type"],
                                  fused_op=fused)


def solve(problem, op, schedule, E, tol, maxiter):
    import cupy as cp
    from gpu_fem.solver_v2 import _cupy_pcg
    t0 = sync_time()
    gmg = problem.gmg(op, schedule)
    gmg.setup(E)
    t1 = sync_time()
    x, iters, converged = _cupy_pcg(lambda v: op.matvec(v, E), problem.F, gmg.apply, tol=tol, maxiter=maxiter)
    t2 = sync_time()
    return x, {"setup_seconds": t1 - t0, "solve_seconds": t2 - t1, "iterations": int(iters),
               "converged": bool(converged), "fp64_operator_calls": int(getattr(op, "fp64_calls", 0))}


def relative(a, b):
    import cupy as cp
    return float(cp.linalg.norm(a - b) / max(float(cp.linalg.norm(b)), 1e-300))


def stage_integration():
    import cupy as cp
    c = PROTOCOL["integration"]
    rows = []
    for grid in c["grids"]:
        problem = Problem(grid)
        stock = problem.operator("stock")
        ops = {arm: (stock if arm == "stock" else problem.operator(arm)) for arm in PROTOCOL["arms"]}
        rng = cp.random.default_rng(c["product_seed"])
        rng_rho = np.random.default_rng(c["density_fields"]["random"]["seed"])
        fields = {"uniform": np.full(problem.n_elem, c["density_fields"]["uniform"]),
                  "random": rng_rho.uniform(c["density_fields"]["random"]["low"],
                                            c["density_fields"]["random"]["high"], problem.n_elem)}
        for field, rho in fields.items():
            E = problem.modulus(rho)
            u = rng.standard_normal(len(problem.free))
            y_stock = stock.matvec(u, E)
            for arm, op in ops.items():
                if arm != "stock":
                    rows.append({"grid": grid["id"], "field": field, "kind": "product", "arm": arm,
                                 "relative_l2": relative(op.matvec(u, E), y_stock)})
            for schedule in PROTOCOL["schedules"]:
                solutions = {}
                for arm, op in ops.items():
                    x, info = solve(problem, op, schedule, E, c["solve"]["tol"], c["solve"]["maxiter"])
                    solutions[arm] = x
                    residual = relative(stock.matvec(x, E), problem.F)
                    rows.append({"grid": grid["id"], "field": field, "kind": "solve", "schedule": schedule,
                                 "arm": arm, "true_residual": residual, **info})
                for row in rows:
                    if row.get("kind") == "solve" and row["grid"] == grid["id"] and row["field"] == field \
                            and row["schedule"] == schedule:
                        row["solution_relative_l2_vs_stock"] = relative(solutions[row["arm"]], solutions["stock"])
        del problem
        cp.get_default_memory_pool().free_all_blocks()
    passed = all(r["relative_l2"] <= c["product_relative_l2_limit"] for r in rows if r["kind"] == "product") and \
        all(r["converged"] and r["true_residual"] <= c["true_residual_limit"]
            and r["solution_relative_l2_vs_stock"] <= c["solution_relative_l2_limit"]
            for r in rows if r["kind"] == "solve")
    return {"rows": rows, "integration_ok": passed}


def size_case(size_id, schedule, arm, result_path):
    """One sizing case in its own process, so an out-of-memory error cannot poison later cases."""
    import cupy as cp
    entry = next(e for e in PROTOCOL["sizing"]["ladder"] if e["id"] == size_id)
    out = {"size": size_id, "schedule": schedule, "arm": arm, "status": "started", "utc": now()}
    try:
        base_used, total = gpu_memory_used()
        problem = Problem(entry)
        out.update(dims=problem.dims, n_elem=problem.n_elem, ndof=problem.ndof, n_free=len(problem.free),
                   bc_seconds=problem.bc_seconds, gpu_total_bytes=total)
        op = problem.operator(arm)
        E = problem.modulus(np.full(problem.n_elem, PROTOCOL["sizing"]["density"]))
        s = PROTOCOL["sizing"]["solve"]
        _, info = solve(problem, op, schedule, E, s["tol"], s["maxiter"])
        used, _ = gpu_memory_used()
        out.update(info, status="ok", gpu_used_after_bytes=used - base_used,
                   pool_total_bytes=int(cp.get_default_memory_pool().total_bytes()),
                   seconds_per_iteration=info["solve_seconds"] / max(info["iterations"], 1))
    except Exception as exc:
        out.update(status="out_of_memory" if "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc)
                   else "error", error=repr(exc)[:2000], traceback=traceback.format_exc()[-4000:])
    write(result_path, out)


def stage_sizing(out_dir, deadline):
    rows = []
    timeout = PROTOCOL["limits"]["size_case_timeout_seconds"]
    for run in PROTOCOL["sizing"]["runs"]:
        for entry in PROTOCOL["sizing"]["ladder"]:
            if time.monotonic() + 60 > deadline:
                rows.append({**run, "size": entry["id"], "status": "skipped_job_deadline"})
                break
            path = out_dir / f"size-{entry['id']}-{run['schedule']}-{run['arm']}.json"
            command = [sys.executable, str(Path(__file__)), "--size", entry["id"], "--schedule", run["schedule"],
                       "--arm", run["arm"], "--result", str(path)]
            started = time.monotonic()
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      timeout=min(timeout, max(1, deadline - time.monotonic() - 30)))
                row = read(path) if path.exists() else {**run, "size": entry["id"], "status": "no_result",
                                                        "returncode": proc.returncode}
                row["stderr_tail"] = proc.stderr[-1500:]
            except subprocess.TimeoutExpired:
                row = {**run, "size": entry["id"], "status": "timeout"}
            row["wall_seconds"] = time.monotonic() - started
            rows.append(row)
            write(out_dir / "sizing.json", {"rows": rows})
            if row["status"] != "ok":
                break                                   # protocol stop rule
    return {"rows": rows}


def stage_smoke(out_dir):
    """Their ci/smoke_test.py as released, then with the empty scripts/ directory their resolver needs."""
    rows = []
    for label in ("as_released", "with_scripts_dir"):
        if label == "with_scripts_dir":
            (DONOR / "scripts").mkdir(exist_ok=True)
        proc = subprocess.run([sys.executable, "ci/smoke_test.py"], cwd=DONOR, capture_output=True, text=True,
                              timeout=900)
        (out_dir / f"smoke-{label}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        rows.append({"variant": label, "returncode": proc.returncode, "passed": proc.returncode == 0,
                     "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:]})
    return {"rows": rows, "stack_ok": rows[-1]["passed"],
            "note": "the only change for the second run is creating the empty scripts/ directory"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path)
    a.add_argument("--size")
    a.add_argument("--schedule")
    a.add_argument("--arm")
    a.add_argument("--result", type=Path)
    a.add_argument("--deadline-seconds", type=float, default=PROTOCOL["limits"]["maximum_job_seconds"] - 600)
    args = a.parse_args()
    if args.size:
        return size_case(args.size, args.schedule, args.arm, args.result)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.deadline_seconds
    result = {"started_utc": now(), "status": "running", "stages": {}}
    for name, fn in (("smoke", lambda: stage_smoke(out)), ("integration", stage_integration),
                     ("sizing", lambda: stage_sizing(out, deadline))):
        t = time.monotonic()
        print(f"{now()} stage {name}", flush=True)
        try:
            result["stages"][name] = {"status": "completed", **fn()}
        except Exception as exc:
            result["stages"][name] = {"status": "failed", "error": repr(exc)[:2000],
                                      "traceback": traceback.format_exc()[-4000:]}
        result["stages"][name]["seconds"] = time.monotonic() - t
        write(out / "result.json", result)
    result.update(status="completed", finished_utc=now())
    write(out / "result.json", result)
    print(f"{now()} done", flush=True)


if __name__ == "__main__":
    main()
