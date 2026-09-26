"""W1 integration: swap the FP64 fine operator inside Yang et al.'s Galerkin GMG solver.

Their solver (pinned donor `donor_yang_gmg`, commit fd9ddecd) builds one `MatrixFreeKff` and passes
it both to the outer PCG/FGMRES and to the multigrid as its FP64 fine operator (`mf_op`). This module
replaces the *core* of that operator's FP64 product and leaves everything else unchanged:

  expand  u_full[free] = u_free, fixed DOFs 0      (theirs, identical in every arm)
  core    y_full = K(E) u_full                    (the only difference between arms)
  restrict y = y_full[free]                       (theirs, identical in every arm)

Arms (COMBINED_PAPER_PLAN.md, Amendment 1):
  stock          their gather + DGEMM + bincount core (reproduction arm)
  fused_ai_fp64  the strongest published FP64 kernel (their fused-kernel repository, Sessions A-E)
  modal8_muladd  the parity kernel with the signed multiply-add butterfly (Session E)

Non-FP64 calls (their FP32/BF16 smoothing paths) and the diagonal extraction are always theirs,
so the arms differ only in FP64 fine products.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE / "donor_yang_gmg" / "src", ROOT / "experiments/session_e/dependencies"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
# Yang's GMG and Yang's fused-kernel repository both name their package `gpu_fem`; the fused one is
# loaded from its own directory under a distinct name so the two never shadow each other.
FUSED_PACKAGE = ROOT / "experiments/session_e/donor/src/gpu_fem"


def _fused_operators():
    import importlib
    import importlib.util
    if "fused_gpu_fem" not in sys.modules:
        spec = importlib.util.spec_from_file_location("fused_gpu_fem", FUSED_PACKAGE / "__init__.py",
                                                      submodule_search_locations=[str(FUSED_PACKAGE)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["fused_gpu_fem"] = module
        spec.loader.exec_module(module)
    return importlib.import_module("fused_gpu_fem.cuda_operators")

ARMS = ("stock", "fused_ai_fp64", "modal8_muladd")


def _yang():
    """Import their modules lazily (they import CuPy at use, not at import, where possible)."""
    from gpu_fem.solver_v2 import MatrixFreeKff
    from gpu_fem.solver_v4 import SolverV4
    return MatrixFreeKff, SolverV4


class ArmCore:
    """y_full = K(E) u_full for one arm, on the donor's node numbering (element e = (ex*ny+ey)*nz+ez)."""

    def __init__(self, arm, grid_dims, ke_unit):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
        self.arm, self.grid_dims = arm, tuple(int(n) for n in grid_dims)
        self.ke = np.asarray(ke_unit, dtype=np.float64)
        self._impl = None

    def build(self):
        if self.arm == "stock" or self._impl is not None:
            return self
        if self.arm == "fused_ai_fp64":
            self._impl = _fused_operators().OperatorSuite(*self.grid_dims, self.ke, build_edof=False)
        else:
            from cuda import CudaModal
            from modal import Modal
            self._impl = CudaModal(self.grid_dims, self.ke, Modal(self.ke).blocks)
        return self

    def __call__(self, u_full, E):
        if self.arm == "fused_ai_fp64":
            return self._impl.matvec_full(u_full, E, path="fused_ai_fp64")
        return self._impl.matvec_full(u_full, E, self.arm)


def make_operator_class():
    MatrixFreeKff, _ = _yang()

    class ArmKff(MatrixFreeKff):
        """Their MatrixFreeKff with the FP64 core replaced by `core` (unless the arm is stock)."""

        def __init__(self, *args, core, **kwargs):
            super().__init__(*args, **kwargs)
            self.core = core.build()
            self.fp64_calls = 0

        def matvec(self, u_free, E_e):
            import cupy as cp
            if self.core.arm == "stock" or u_free.dtype != cp.float64:
                return super().matvec(u_free, E_e)
            self.fp64_calls += 1
            u_full = cp.zeros(self._ndof, dtype=cp.float64)          # expand: as theirs
            u_full[self._free_gpu] = u_free
            y_full = self.core(u_full, cp.ascontiguousarray(E_e, dtype=cp.float64))
            return y_full[self._free_gpu]                            # restrict: as theirs

    return ArmKff


def make_solver_class():
    """SolverV4 subclass whose single FP64 fine operator (outer Krylov and GMG) is an ArmKff."""
    _, SolverV4 = _yang()
    ArmKff = make_operator_class()

    class ArmSolverV4(SolverV4):
        def __init__(self, *args, arm="stock", **kwargs):
            self._arm = arm
            super().__init__(*args, **kwargs)

        def _ensure_matfree_components(self):
            if self._matfree_op is None:
                import cupy as cp
                # the solver's own element matrix, so every arm uses exactly what their operator uses
                core = ArmCore(self._arm, self._grid_dims, cp.asnumpy(self._KE_unit_gpu))
                self._matfree_op = ArmKff(edof_gpu=self._edof_gpu, KE_unit_gpu=self._KE_unit_gpu,
                                          free_gpu=self._free_gpu, n_free=self._n_free,
                                          ndof=self.ndof, core=core)
            super()._ensure_matfree_components()

    return ArmSolverV4
