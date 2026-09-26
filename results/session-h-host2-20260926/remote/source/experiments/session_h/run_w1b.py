"""Voxel session H (combined paper W1b): session G's SIMP runs with the filter and OC on the GPU.

Session G (report section 23) found that 70-87 % of a 30-step run was outside the linear solver: per-run
solver construction (~9-10 s at 512k) and the CPU density filter / OC bisection. W1b keeps Yang et al.'s
solver, the three FP64 arms, the schedules and cases, and changes only (1) the filter and top88 OC now
run on the GPU with identical FP64 arithmetic (GpuDensityFilter, oc_top88_gpu) and (2) runs are 100
fixed steps so per-run construction is amortized. The CPU implementations remain for the checks.

    run_w1b.py --out DIR [--deadline-seconds S]
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                  # <source>/experiments/session_h
SRC = HERE.parents[1]
DONOR = SRC / "experiments/combined/donor_yang_gmg"
for p in (SRC / "experiments/combined", DONOR / "src", SRC / "experiments/session_e/dependencies"):
    sys.path.insert(0, str(p))
from common import now, read, write  # noqa: E402

PROTOCOL = read(HERE / "protocol.json")


def oc_update(rho, dc, volfrac, move):
    """Their E3 `_oc_update`, verbatim (run_experiments_e1_e10.py, commit fd9ddecd)."""
    lam_lo, lam_hi = 1e-40, 1e40
    for _ in range(100):
        lmid = 0.5 * (lam_lo + lam_hi)
        rho_new = np.clip(
            rho * np.sqrt(np.maximum(-dc / lmid, 0.0)),
            np.maximum(rho - move, 1e-3),
            np.minimum(rho + move, 1.0),
        )
        if rho_new.mean() > volfrac:
            lam_lo = lmid
        else:
            lam_hi = lmid
    return rho_new


def oc_top88(x, dc, dv, volfrac, move, density=None, l2=1e9, tol=1e-3):
    """Optimality-criteria update of Andreassen et al. (2011, top88) on filtered sensitivities.

    Bisection on the Lagrange multiplier until (l2 - l1) / (l1 + l2) <= tol. As in top88, the volume
    constraint is imposed on the physical (filtered) densities `density(x_new)` when a filter is given.
    """
    density = density or (lambda v: v)
    l1 = 0.0
    while (l2 - l1) / (l1 + l2) > tol:
        lmid = 0.5 * (l2 + l1)
        x_new = np.clip(x * np.sqrt(np.maximum(-dc / dv / lmid, 0.0)),
                        np.maximum(x - move, 0.0), np.minimum(x + move, 1.0))
        if density(x_new).mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return x_new


class GpuDensityFilter:
    """The same linear density filter as DensityFilter, stored and applied on the GPU (CuPy CSR)."""

    def __init__(self, cpu_filter):
        import cupy as cp
        import cupyx.scipy.sparse as csp
        self.cp = cp
        self.H = csp.csr_matrix(cpu_filter.H)
        self.Hs = cp.asarray(cpu_filter.Hs)

    def density(self, x):
        return self.H @ x / self.Hs

    def gradient(self, g):
        return self.H @ (g / self.Hs)


def oc_top88_gpu(x, dc, dv, volfrac, move, density, l2=1e9, tol=1e-3):
    """oc_top88 with CuPy arrays: identical arithmetic, the bisection scalar test syncs once per step."""
    import cupy as cp
    l1 = 0.0
    while (l2 - l1) / (l1 + l2) > tol:
        lmid = 0.5 * (l2 + l1)
        x_new = cp.clip(x * cp.sqrt(cp.maximum(-dc / dv / lmid, 0.0)),
                        cp.maximum(x - move, 0.0), cp.minimum(x + move, 1.0))
        if float(density(x_new).mean()) > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return x_new


class DensityFilter:
    """Linear density filter with weights max(0, rmin - distance) (Bruns & Tortorelli; top88/top3d)."""

    def __init__(self, dims, rmin):
        import scipy.sparse as sp
        nx, ny, nz = dims
        r = int(np.ceil(rmin)) - 1
        ex, ey, ez = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        ex, ey, ez = ex.ravel(), ey.ravel(), ez.ravel()
        rows, cols, vals = [], [], []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    w = rmin - np.sqrt(dx * dx + dy * dy + dz * dz)
                    if w <= 0:
                        continue
                    jx, jy, jz = ex + dx, ey + dy, ez + dz
                    ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny) & (jz >= 0) & (jz < nz)
                    i = (ex * ny + ey) * nz + ez          # their element order
                    j = (jx * ny + jy) * nz + jz
                    rows.append(i[ok]); cols.append(j[ok]); vals.append(np.full(ok.sum(), w))
        n = nx * ny * nz
        self.H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self.Hs = np.asarray(self.H.sum(axis=1)).ravel()

    def density(self, x):
        return self.H @ x / self.Hs

    def gradient(self, g):
        return self.H @ (g / self.Hs)


class Case:
    def __init__(self, preset):
        from gpu_fem.bc_generator import generate_bc
        from gpu_fem.presets import get_preset
        from gpu_fem.pub_simp_solver import KE_UNIT_3D, _build_sparse_indices, _edof_table_3d
        self.spec = get_preset(preset)
        s = self.spec
        self.dims = (s.nelx, s.nely, s.nelz)
        self.n_elem = int(np.prod(self.dims))
        bc = generate_bc(s)
        edof = _edof_table_3d(*self.dims)
        row_idx, col_idx = _build_sparse_indices(edof)
        self.filter = DensityFilter(self.dims, PROTOCOL["simp"]["filter_rmin"])
        self._gpu_filter = None
        self.common = dict(edof=edof, row_idx=row_idx, col_idx=col_idx, KE_UNIT=KE_UNIT_3D,
                           free=bc.free_dofs.astype(np.int32), F=bc.F, ndof=bc.ndof, backend="cupy",
                           enable_warm_start=True)

    @property
    def gpu_filter(self):
        if self._gpu_filter is None:
            self._gpu_filter = GpuDensityFilter(self.filter)
        return self._gpu_filter

    def solver(self, arm, schedule):
        import kff_adapters
        s = PROTOCOL["schedules"][schedule]
        ArmSolverV4 = kff_adapters.make_solver_class()
        return ArmSolverV4(arm=arm, enable_matrix_free=True, enable_fused_cuda=s["enable_fused_cuda"],
                           enable_matfree_gmg=True, matfree_gmg_levels=s["levels"],
                           gmg_fine_smoother=s["fine_smoother"], gmg_smoother_type=s["smoother_type"],
                           grid_dims=self.dims, enable_profiling=True, **self.common)


def simp_run(case, arm, schedule, steps, stop_change=None, update="top88"):
    """One SIMP optimization; wall time covers solver construction and every step (their E3 scope).

    update="top88": density-filtered SIMP with the top88 OC (the W1 optimization).
    update="their_e3": their E3 loop verbatim (no filter, their OC), used only as a diagnostic.
    """
    import cupy as cp
    simp = PROTOCOL["simp"]
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    solver = case.solver(arm, schedule)
    x = np.full(case.n_elem, case.spec.volfrac)
    rho = case.filter.density(x) if update == "top88" else x
    if update == "top88_gpu":
        f = case.gpu_filter
        x = cp.full(case.n_elem, case.spec.volfrac)
        dv_x = f.gradient(cp.ones(case.n_elem))
        rho = cp.asnumpy(f.density(x))
    history = []
    for k in range(steps):
        c, dc = solver.solve(rho, penal=simp["penal"])
        timing = dict(getattr(solver, "last_timing", {}))
        iters = int(getattr(solver, "last_cg_iters", -1))
        if update == "top88_gpu":
            dc_x = f.gradient(cp.asarray(dc))
            x_new = oc_top88_gpu(x, dc_x, dv_x, case.spec.volfrac, simp["move"], density=f.density)
            rho_gpu = f.density(x_new)
            change = float(cp.max(cp.abs(x_new - x)))
            x = x_new
            rho_new = cp.asnumpy(rho_gpu)
        elif update == "top88":
            dc_x = case.filter.gradient(dc)
            dv_x = case.filter.gradient(np.ones_like(dc))
            x_new = oc_top88(x, dc_x, dv_x, case.spec.volfrac, simp["move"], density=case.filter.density)
            rho_new = case.filter.density(x_new)
            change = float(np.max(np.abs(x_new - x)))
            x = x_new
        else:
            rho_new = oc_update(rho, dc, case.spec.volfrac, simp["move"])
            change = float(np.max(np.abs(rho_new - rho)))
        history.append({"step": k, "compliance": float(c), "outer_iterations": iters,
                        "not_converged": iters >= solver.cg_maxiter, "change": change,
                        "volume": float(rho_new.mean()),
                        **{key: timing.get(key) for key in ("gmg_setup_ms", "cg_ms", "sensitivity_ms", "diag_ms",
                                                            "total_ms", "outer_solver")}})
        rho = rho_new
        if stop_change is not None and change <= stop_change:
            break
    cp.cuda.Stream.null.synchronize()
    wall = time.perf_counter() - t0
    fp64_calls = int(getattr(solver._matfree_op, "fp64_calls", 0))
    del solver
    cp.get_default_memory_pool().free_all_blocks()
    return {"arm": arm, "schedule": schedule, "wall_seconds": wall, "steps": len(history),
            "final_compliance": history[-1]["compliance"],
            "total_outer_iterations": int(sum(h["outer_iterations"] for h in history)),
            "not_converged_solves": int(sum(h["not_converged"] for h in history)),
            "fp64_operator_calls": fp64_calls, "history": history}, rho


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--deadline-seconds", type=float, default=PROTOCOL["limits"]["maximum_job_seconds"] - 600)
    args = a.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "designs").mkdir()
    deadline = time.monotonic() + args.deadline_seconds
    simp = PROTOCOL["simp"]
    result = {"started_utc": now(), "status": "running", "warmup": [], "runs": [], "complete": [], "errors": []}
    rng = np.random.default_rng(PROTOCOL["order_seed"])

    def save():
        write(out / "result.json", result)

    cases = {c["id"]: Case(c["preset"]) for c in PROTOCOL["cases"]}
    try:
        for cid, case in cases.items():
            for schedule in PROTOCOL["schedules"]:
                for arm in PROTOCOL["arms"]:
                    r, _ = simp_run(case, arm, schedule, simp["warmup_steps"], update="top88_gpu")
                    result["warmup"].append({"case": cid, **{k: v for k, v in r.items() if k != "history"}})
        save()
        for rep in range(1, PROTOCOL["repetitions"] + 1):
            block = []
            for cid, case in cases.items():
                for schedule in PROTOCOL["schedules"]:
                    for arm in rng.permutation(PROTOCOL["arms"]):
                        block.append((cid, case, schedule, str(arm)))
            if time.monotonic() + PROTOCOL["limits"]["estimated_repetition_seconds"] > deadline:
                result["errors"].append({"repetition": rep, "skipped": "job deadline before a complete repetition"})
                break
            for cid, case, schedule, arm in block:
                print(f"{now()} rep {rep} {cid} {schedule} {arm}", flush=True)
                try:
                    r, rho = simp_run(case, arm, schedule, simp["steps"], update="top88_gpu")
                    if rep == 1:
                        np.savez_compressed(out / "designs" / f"{cid}-{schedule}-{arm}.npz", rho=np.asarray(rho.get() if hasattr(rho, "get") else rho, dtype=np.float32))
                    result["runs"].append({"repetition": rep, "case": cid, **r})
                except Exception as exc:
                    result["errors"].append({"repetition": rep, "case": cid, "schedule": schedule, "arm": arm,
                                             "error": repr(exc)[:2000], "traceback": traceback.format_exc()[-3000:]})
                save()
        complete = PROTOCOL["complete_optimization"]
        case = cases[complete["case"]]
        for schedule in PROTOCOL["schedules"]:
            for arm in PROTOCOL["arms"]:
                if time.monotonic() + complete["estimated_run_seconds"] > deadline:
                    result["complete"].append({"schedule": schedule, "arm": arm, "skipped": "job deadline"})
                    continue
                print(f"{now()} complete {complete['case']} {schedule} {arm}", flush=True)
                r, rho = simp_run(case, arm, schedule, complete["max_steps"], stop_change=complete["stop_change"],
                                  update="top88_gpu")
                np.savez_compressed(out / "designs" / f"complete-{schedule}-{arm}.npz", rho=np.asarray(rho.get() if hasattr(rho, "get") else rho, dtype=np.float32))
                result["complete"].append({"case": complete["case"], **r})
                save()
    except Exception as exc:
        result["errors"].append({"fatal": repr(exc)[:2000], "traceback": traceback.format_exc()[-4000:]})
    result.update(status="completed", finished_utc=now())
    save()


if __name__ == "__main__":
    main()
