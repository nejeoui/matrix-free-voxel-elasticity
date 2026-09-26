"""Session I stage W2a: Yang et al.'s Galerkin GMG at 1 M and 2 M elements on 80 GB GPUs, four FP64 arms.

For each (size, schedule, arm): build their GMG around the arm's FP64 operator (kff_adapters), set up
once on their uniform-density field (rho = 0.5, p = 3, their E2 convention), then time `rounds`
fixed-work PCG solves of exactly K outer iterations (tol = 0 so the cap binds). One arm is resident at a
time (a 2 M hierarchy takes ~46 GiB); arms run in a seeded order, in `blocks` blocks, so every arm is
timed in each block. A first converged solve per arm (tol 1e-8, cap 1000) records iteration counts.

    run_yang_scale.py --out DIR --deadline-seconds S
"""
import argparse
import dataclasses
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
DONOR = SRC / "experiments/combined/donor_yang_gmg"
for p in (SRC / "experiments/combined", DONOR / "src", HERE / "dependencies"):
    sys.path.insert(0, str(p))
from common import now, read, write  # noqa: E402

PROTOCOL = read(HERE / "protocol.json")
W2A = PROTOCOL["w2a"]


def sync():
    import cupy as cp
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def build(size_entry, schedule, arm):
    import cupy as cp
    import kff_adapters
    from gpu_fem.bc_generator import generate_bc
    from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
    from gpu_fem.presets import get_preset
    from gpu_fem.pub_simp_solver import KE_UNIT_3D, _edof_table_3d
    spec = get_preset(size_entry["preset"])
    dims = (spec.nelx, spec.nely, spec.nelz)
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    free_gpu = cp.asarray(free)
    edof_gpu = cp.asarray(_edof_table_3d(*dims).astype(np.int32))
    ke_gpu = cp.asarray(KE_UNIT_3D)
    op = kff_adapters.make_operator_class()(edof_gpu=edof_gpu, KE_unit_gpu=ke_gpu, free_gpu=free_gpu,
                                            n_free=len(free), ndof=int(bc.ndof),
                                            core=kff_adapters.ArmCore(arm, dims, KE_UNIT_3D))
    s = PROTOCOL["schedules"][schedule]
    fused = None
    if s["fused_fine"]:
        from gpu_fem.cuda_fused_matvec import FusedMatvec
        fused = FusedMatvec(edof_gpu=edof_gpu, KE_unit_gpu=ke_gpu, ndof=int(bc.ndof))
    gmg = GalerkinMatFreeGMG(mf_op=op, free=free, free_gpu=free_gpu, nelx=dims[0], nely=dims[1], nelz=dims[2],
                             KE_UNIT=KE_UNIT_3D, n_levels=s["n_levels"], fine_smoother=s["fine_smoother"],
                             smoother_type=s["smoother_type"], fused_op=fused)
    E = W2A["e_min"] + (1 - W2A["e_min"]) * cp.full(int(np.prod(dims)), W2A["density"]) ** W2A["penal"]
    F = cp.asarray(bc.F[free].astype(np.float64))
    t0 = sync()
    gmg.setup(E)
    setup = sync() - t0
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    return {"op": op, "gmg": gmg, "E": E, "F": F, "dims": dims, "ndof": int(bc.ndof),
            "setup_seconds": setup, "gpu_used_bytes": int(total_b - free_b)}


def pcg(state, tol, maxiter):
    from gpu_fem.solver_v2 import _cupy_pcg
    op, E = state["op"], state["E"]
    t0 = sync()
    x, iters, conv = _cupy_pcg(lambda v: op.matvec(v, E), state["F"], state["gmg"].apply, tol=tol, maxiter=maxiter)
    return x, int(iters), bool(conv), sync() - t0


def main():
    import cupy as cp
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--deadline-seconds", type=float, required=True)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.deadline_seconds
    result = {"started_utc": now(), "status": "running", "records": [], "errors": []}
    rng = np.random.default_rng(PROTOCOL["order_seed"])
    total_mem = cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"]
    if total_mem < W2A["minimum_gpu_bytes"]:
        result.update(status="skipped", reason=f"GPU memory {total_mem} below {W2A['minimum_gpu_bytes']}")
        write(args.out / "result.json", result)
        return
    try:
        for size in W2A["sizes"]:
            for schedule in W2A["schedules"]:
                for block in range(W2A["blocks"]):
                    for arm in [W2A["arms"][int(i)] for i in rng.permutation(len(W2A["arms"]))]:
                        if time.monotonic() + W2A["estimated_arm_seconds"] > deadline:
                            result["errors"].append({"size": size["id"], "schedule": schedule, "arm": arm,
                                                     "skipped": "job deadline"})
                            continue
                        print(f"{now()} W2a {size['id']} {schedule} block {block} {arm}", flush=True)
                        rec = {"size": size["id"], "schedule": schedule, "block": block, "arm": arm}
                        state = None
                        try:
                            state = build(size, schedule, arm)
                            rec.update(dims=state["dims"], ndof=state["ndof"], setup_seconds=state["setup_seconds"],
                                       gpu_used_bytes=state["gpu_used_bytes"])
                            if block == 0:
                                _, it, conv, secs = pcg(state, W2A["tol"], W2A["maxiter"])
                                rec.update(to_tolerance={"iterations": it, "converged": conv, "seconds": secs})
                            rounds = []
                            for _ in range(W2A["rounds"]):
                                _, it, _, secs = pcg(state, 0.0, W2A["fixed_iterations"])
                                rounds.append({"iterations": it, "seconds": secs})
                            rec["fixed_work"] = rounds
                        except Exception as exc:
                            rec.update(error=repr(exc)[:2000], traceback=traceback.format_exc()[-3000:],
                                       out_of_memory="OutOfMemory" in type(exc).__name__)
                        finally:
                            del state
                            cp.get_default_memory_pool().free_all_blocks()
                        result["records"].append(rec)
                        write(args.out / "result.json", result)
    except Exception as exc:
        result["errors"].append({"fatal": repr(exc)[:2000], "traceback": traceback.format_exc()[-3000:]})
    result.update(status="completed", finished_utc=now())
    write(args.out / "result.json", result)


if __name__ == "__main__":
    main()
