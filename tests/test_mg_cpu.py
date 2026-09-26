"""CPU checks of the multigrid preconditioner (no GPU). Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/session_a/dependencies"))
import voxel_mg as mg  # noqa: E402
from common import to_author  # noqa: E402

KE = np.load(ROOT / "experiments/session_a/inputs/smoke-q4.npz")["ke"]


def smoke(q=4, seed=None):
    """The real smoke-q4 input, or a synthetic cantilever of the same kind at size 2q x q x q."""
    d = np.load(ROOT / "experiments/session_a/inputs/smoke-q4.npz")
    if q == 4 and seed is None:
        shape = tuple(int(s) for s in d["shape"])
        rho, mask, force = d["rho"], d["mask"], d["force"]
    else:
        shape = (2 * q, q, q)
        nx, ny, nz = shape
        rng = np.random.default_rng(seed or 0)
        rho = 0.05 + 0.95 * rng.random(nx * ny * nz) ** 2
        m = np.ones((nz + 1, ny + 1, nx + 1, 3)); m[:, :, 0, :] = 0
        f = np.zeros((nz + 1, ny + 1, nx + 1, 3)); f[0, :, nx, 2] = -1.0 / (ny + 1)
        mask, force = m.ravel(), f.ravel()
    E = to_author(1e-6 + (1 - 1e-6) * rho ** 3, shape)
    return shape, E, to_author(mask, shape, True), to_author(force, shape, True)


def build(shape, E, mask, dtype=np.float64, max_coarse_x=2, pre=4, post=4):
    shapes = mg.hierarchy_shapes(shape, max_coarse_x)
    products = [mg.NumpyProduct(s, KE * 2 ** i, dtype) for i, s in enumerate(shapes)]
    Ec, mc = E, mask
    for i in range(1, len(shapes)):
        Ec, mc = mg.coarsen_modulus(np, Ec, shapes[i - 1]), mg.coarsen_mask(np, mc, shapes[i - 1])
    A = mg.assemble_dense(shapes[-1], KE * 2 ** (len(shapes) - 1), Ec, mc)
    factor = np.linalg.cholesky(A)
    M = mg.Multigrid(np, shapes, products, E, mask, factor, dtype, pre=pre, post=post)
    M.omegas, _ = mg.spectral_omegas(np, M.levels)
    return shapes, M


def P_matrix(coarse):
    n = 3 * int(np.prod(np.array(coarse) + 1))
    return np.stack([mg.prolong(np, np.eye(n)[i], coarse) for i in range(n)], axis=1)


def test_restriction_is_the_exact_transpose_of_prolongation():
    rng = np.random.default_rng(1)
    coarse, fine = (2, 1, 1), (4, 2, 2)
    c = rng.standard_normal(3 * 3 * 2 * 2)
    f = rng.standard_normal(3 * 5 * 3 * 3)
    assert np.isclose(mg.prolong(np, c, coarse) @ f, c @ mg.restrict(np, f, fine), rtol=1e-13)


def test_rediscretisation_equals_galerkin_for_uniform_modulus():
    """P^T K_fine P == K_coarse with ke_coarse = 2 ke: validates transfers, scaling and layout."""
    fine, coarse = (4, 2, 2), (2, 1, 1)
    Kf = mg.assemble_dense(fine, KE, np.ones(16), np.ones(3 * 5 * 3 * 3)) - 0   # unconstrained
    Kc = mg.assemble_dense(coarse, 2 * KE, np.ones(2), np.ones(3 * 3 * 2 * 2))
    P = P_matrix(coarse)
    assert np.abs(P.T @ Kf @ P - Kc).max() / np.abs(Kc).max() < 1e-13


def test_vcycle_is_symmetric():
    shape, E, mask, _ = smoke()
    _, M = build(shape, E, mask)
    rng = np.random.default_rng(2)
    r1, r2 = rng.standard_normal(E.size * 0 + mask.size), rng.standard_normal(mask.size)
    r1, r2 = r1 * mask, r2 * mask
    a, b = M(r1) @ r2, r1 @ M(r2)
    assert abs(a - b) / abs(a) < 1e-10


def _apply_fine(shape, E, mask):
    prod = mg.NumpyProduct(shape, KE)
    return lambda u: prod.raw(u * mask, E) * mask + u * (1 - mask)


def test_mg_fcg_reaches_1e8_true_residual_on_the_real_smoke_input():
    shape, E, mask, force = smoke()
    _, M = build(shape, E, mask)
    x, it, ok, rel, _ = mg.fcg(np, _apply_fine(shape, E, mask), force, M, 1e-8, 200)
    assert ok and rel <= 1e-8 and it <= 30, (it, rel)
    assert np.all(x[mask == 0] == 0)


@pytest.mark.parametrize("q", [4, 8])
def test_iteration_count_is_roughly_mesh_independent(q):
    shape, E, mask, force = smoke(q=q, seed=3)
    _, M = build(shape, E, mask)
    _, it, ok, rel, _ = mg.fcg(np, _apply_fine(shape, E, mask), force, M, 1e-8, 300)
    assert ok and it <= 60, (q, it, rel)


def test_fp32_preconditioner_still_reaches_1e8_in_fp64_outer():
    shape, E, mask, force = smoke(q=8, seed=4)
    _, M32 = build(shape, E, mask, dtype=np.float32)
    _, it, ok, rel, _ = mg.fcg(np, _apply_fine(shape, E, mask), force, M32, 1e-8, 300)
    assert ok and rel <= 1e-8, (it, rel)


def test_fixed_iterations_do_exactly_the_requested_work():
    shape, E, mask, force = smoke()
    _, M = build(shape, E, mask)
    _, it, _, _, mv = mg.fcg(np, _apply_fine(shape, E, mask), force, M, 1e-8, 999, fixed_iterations=7)
    assert it == 7 and mv == 8          # 7 CG products + 1 final true-residual product


def test_single_level_degenerates_to_a_direct_solve():
    shape, E, mask, force = smoke()
    _, M = build(shape, E, mask, max_coarse_x=64)          # one level: exact coarse solve
    x, it, ok, rel, _ = mg.fcg(np, _apply_fine(shape, E, mask), force, M, 1e-8, 10, check_every=1)
    assert ok and it <= 2, (it, rel)


def test_fixed_omega_0_6_is_unstable_on_heterogeneous_density():
    """Negative control documenting why the damping is chosen spectrally: omega * lambda_max > 2."""
    shape, E, mask, _ = smoke(q=8, seed=3)
    _, M = build(shape, E, mask)
    _, lambdas = mg.spectral_omegas(np, M.levels)
    assert 0.6 * max(lambdas) > 2.0
    assert all(w * lam == pytest.approx(4 / 3) for w, lam in zip(M.omegas, lambdas))
