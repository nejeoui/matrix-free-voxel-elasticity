"""Nsight Compute target: launch each FP64 product kernel a few times on one grid.

`ncu --kernel-name K --launch-skip 2 --launch-count 1 python ncu_target.py` profiles the third
launch of kernel K. The same seeded inputs are used for every kernel.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
for p in (SRC / "experiments/combined", SRC / "experiments/session_e/dependencies"):
    sys.path.insert(0, str(p))
from common import read  # noqa: E402

LAUNCHES = 3


def main():
    import cupy as cp
    import kff_adapters
    from cuda import CudaModal
    from modal import Modal
    protocol = read(HERE / "protocol.json")
    shape = tuple(protocol["ncu"]["grid"])
    ke = np.load(HERE / "ke_unit_3d.npy")
    n_elem = int(np.prod(shape))
    ndof = 3 * int(np.prod(np.array(shape) + 1))
    rng = np.random.default_rng(7)
    u = cp.asarray(rng.standard_normal(ndof))
    E = cp.asarray(1e-9 + (1 - 1e-9) * rng.uniform(0, 1, n_elem) ** 3)
    modal = CudaModal(shape, ke, Modal(ke).blocks)
    fused = kff_adapters._fused_operators().OperatorSuite(*shape, ke, build_edof=False)
    for _ in range(LAUNCHES):
        for path in ("dense8", "modal8", "modal8_muladd"):
            modal.matvec_full(u, E, path)
        fused.matvec_full(u, E, path="fused_ai_fp64")
    cp.cuda.Stream.null.synchronize()
    print("launched", LAUNCHES, "x", ["dense8", "modal8", "modal8_muladd", "fused_ai_fp64"], "on", shape)


if __name__ == "__main__":
    main()
