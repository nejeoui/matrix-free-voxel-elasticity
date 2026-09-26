"""Geometric multigrid-preconditioned flexible CG for voxel elasticity (Träff-style), backend-agnostic.

Only the FINEST-level product is injected per experiment arm; every coarse level uses one common
product, so a comparison between arms isolates the finest kernel.

Hierarchy: element grid (nx, ny, nz) halved in every direction per level. Coarse moduli are 2x2x2
averages of the child moduli. The coarse element matrix is the fine one times 2**level (3-D
elasticity: K_e scales with element size h). Coarse boundary masks are injected from the fine
nodes coinciding with coarse nodes. Transfers are trilinear prolongation P and restriction P^T.
For a uniform modulus this rediscretisation equals the Galerkin product P^T K P exactly (checked in
tests/test_mg_cpu.py).

Smoother: damped Jacobi, `pre` and `post` sweeps, with a per-level damping omega_l = 4 / (3 lambda_l),
lambda_l ~ lambda_max(D^-1 A_l) from a seeded power iteration (`spectral_omegas`). A fixed omega = 0.6
was found unstable on heterogeneous densities (omega * lambda_max = 2.8-2.9 > 2) in the CPU tests. Coarsest level: dense Cholesky of the constrained
operator, assembled once on the host. Outer solver: flexible (Polak-Ribière) CG in FP64, so an FP32
or otherwise inexact preconditioner is handled consistently. The stopping test uses the TRUE
residual ||b - A x|| / ||b||, recomputed every `check_every` iterations and at exit.

Arrays are in the donor's author layout: nodal (nx+1, ny+1, nz+1, 3) and element (nx, ny, nz),
flattened, with x the slowest index.
"""
from __future__ import annotations

import numpy as np

XYZ = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])   # native Hex8 node order of `ke`


# ----------------------------------------------------------------------------------- CPU reference
def edof_author(shape):
    """(n_el, 24) global DOF indices in the author layout for the native node order of `ke`."""
    nx, ny, nz = shape
    ex, ey, ez = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    ex, ey, ez = ex.ravel(), ey.ravel(), ez.ravel()
    nodes = [((ex + x) * (ny + 1) + ey + y) * (nz + 1) + ez + z for x, y, z in XYZ]
    return np.stack([3 * n + c for n in nodes for c in range(3)], axis=1)


class NumpyProduct:
    """Element-by-element K(E) u on the CPU. `raw(u, E)` expects constrained DOFs already zeroed."""

    def __init__(self, shape, ke, dtype=np.float64):
        self.shape, self.dtype = tuple(shape), dtype
        self.ke = np.asarray(ke, dtype=dtype)
        self.edof = edof_author(shape)
        self.ndof = 3 * int(np.prod(np.array(shape) + 1))

    def raw(self, u, E):
        y = np.zeros(self.ndof, dtype=self.dtype)
        np.add.at(y, self.edof, (E[:, None] * (u[self.edof] @ self.ke.T)).astype(self.dtype))
        return y

    def diagonal(self, E):
        d = np.zeros(self.ndof, dtype=self.dtype)
        np.add.at(d, self.edof, (E[:, None] * np.diag(self.ke)[None, :]).astype(self.dtype))
        return d


def assemble_dense(shape, ke, E, mask):
    """Dense constrained matrix M K M + (I - M), FP64, on the host (coarsest level only)."""
    edof = edof_author(shape)
    n = 3 * int(np.prod(np.array(shape) + 1))
    K = np.zeros((n, n))
    for e in range(edof.shape[0]):
        idx = edof[e]
        K[np.ix_(idx, idx)] += E[e] * ke
    m = np.asarray(mask, dtype=np.float64)
    return m[:, None] * K * m[None, :] + np.diag(1.0 - m)


# ------------------------------------------------------------------------------------- transfers
def _prolong_axis(xp, v, axis):
    v = xp.moveaxis(v, axis, 0)
    nc = v.shape[0]
    out = xp.empty((2 * nc - 1,) + v.shape[1:], dtype=v.dtype)
    out[0::2] = v
    out[1::2] = 0.5 * (v[:-1] + v[1:])
    return xp.moveaxis(out, 0, axis)


def _restrict_axis(xp, f, axis):
    f = xp.moveaxis(f, axis, 0)
    out = f[0::2].copy()
    half = 0.5 * f[1::2]
    out[:-1] += half
    out[1:] += half
    return xp.moveaxis(out, 0, axis)


def prolong(xp, uc, coarse_shape):
    nx, ny, nz = coarse_shape
    v = uc.reshape(nx + 1, ny + 1, nz + 1, 3)
    for axis in range(3):
        v = _prolong_axis(xp, v, axis)
    return xp.ascontiguousarray(v).ravel()


def restrict(xp, rf, fine_shape):
    nx, ny, nz = fine_shape
    v = rf.reshape(nx + 1, ny + 1, nz + 1, 3)
    for axis in range(3):
        v = _restrict_axis(xp, v, axis)
    return xp.ascontiguousarray(v).ravel()


def coarsen_modulus(xp, E, shape):
    nx, ny, nz = shape
    return xp.ascontiguousarray(
        E.reshape(nx // 2, 2, ny // 2, 2, nz // 2, 2).mean(axis=(1, 3, 5))).ravel()


def coarsen_mask(xp, mask, shape):
    nx, ny, nz = shape
    return xp.ascontiguousarray(mask.reshape(nx + 1, ny + 1, nz + 1, 3)[::2, ::2, ::2, :]).ravel()


# ------------------------------------------------------------------------------------- hierarchy
class Level:
    def __init__(self, xp, shape, product, E, mask, dtype):
        self.xp, self.shape, self.product, self.dtype = xp, tuple(shape), product, dtype
        self.E = E.astype(dtype)
        self.mask = mask.astype(dtype)
        self.fixed = (1 - self.mask).astype(dtype)
        d = product.diagonal(self.E)
        self.inv_diag = (1 / xp.where(self.mask != 0, d, xp.asarray(1, dtype=dtype))).astype(dtype)

    def apply(self, u):
        return self.product.raw(u * self.mask, self.E) * self.mask + u * self.fixed


class Multigrid:
    """V-cycle preconditioner. `products` is a list of per-level product objects (finest first)
    exposing raw(u, E) and diagonal(E); `coarse_solve` solves the coarsest constrained system."""

    def __init__(self, xp, shapes, products, E_fine, mask_fine, coarse_factor, dtype,
                 pre=4, post=4, omegas=None):
        self.xp, self.dtype, self.pre, self.post = xp, dtype, pre, post
        self.levels = []
        E, mask = E_fine, mask_fine
        for i, (shape, prod) in enumerate(zip(shapes, products)):
            if i:
                E = coarsen_modulus(xp, E, shapes[i - 1])
                mask = coarsen_mask(xp, mask, shapes[i - 1])
            self.levels.append(Level(xp, shape, prod, E, mask, dtype))
        self.coarse_factor = coarse_factor.astype(dtype)
        n_smooth = len(self.levels) - 1
        self.omegas = list(omegas) if omegas is not None else [0.6] * n_smooth
        assert len(self.omegas) == n_smooth

    def coarse_solve(self, r):
        xp = self.xp
        if xp is np:
            from scipy.linalg import solve_triangular
        else:
            from cupyx.scipy.linalg import solve_triangular
        y = solve_triangular(self.coarse_factor, r, lower=True)
        return solve_triangular(self.coarse_factor.T, y, lower=False)

    def vcycle(self, r, level=0):
        L = self.levels[level]
        if level == len(self.levels) - 1:
            return self.coarse_solve(r) * L.mask
        w = self.omegas[level]
        x = w * L.inv_diag * r
        for _ in range(self.pre - 1):
            x = x + w * L.inv_diag * (r - L.apply(x))
        res = (r - L.apply(x)) * L.mask
        rc = restrict(self.xp, res, L.shape) * self.levels[level + 1].mask
        ec = self.vcycle(rc, level + 1)
        x = x + prolong(self.xp, ec, self.levels[level + 1].shape) * L.mask
        for _ in range(self.post):
            x = x + w * L.inv_diag * (r - L.apply(x))
        return x

    def __call__(self, r):
        return self.vcycle(r.astype(self.dtype)).astype(r.dtype)


def fcg(xp, apply_A, b, precondition, tol, maxiter, check_every=5, fixed_iterations=None):
    """Flexible PCG (Polak-Ribière beta), FP64 outer. Stops on the TRUE relative residual.

    With `fixed_iterations`, performs exactly that many iterations (no early stop), so every arm does
    identical work. Returns (x, iterations, converged, true_relative_residual, outer_matvecs).
    """
    nb = float(xp.linalg.norm(b))
    x = xp.zeros_like(b)
    r = b.copy()
    z = precondition(r)
    p = z.copy()
    rz = float(r @ z)
    matvecs = 0
    limit = fixed_iterations if fixed_iterations is not None else maxiter
    it = 0
    for it in range(1, limit + 1):
        q = apply_A(p)
        matvecs += 1
        pq = float(p @ q)
        alpha = rz / pq
        x = x + alpha * p
        r_new = r - alpha * q
        if fixed_iterations is None and it % check_every == 0:
            true = b - apply_A(x)
            matvecs += 1
            if float(xp.linalg.norm(true)) / nb <= tol:
                return x, it, True, float(xp.linalg.norm(true)) / nb, matvecs
            r_new = true                              # residual replacement, as the donor PCG does
        z_new = precondition(r_new)
        rz_new = float(r_new @ z_new)
        beta = float(r_new @ (z_new - z)) / rz
        p = z_new + beta * p
        r, z, rz = r_new, z_new, rz_new
    true = b - apply_A(x)
    matvecs += 1
    rel = float(xp.linalg.norm(true)) / nb
    return x, it, bool(rel <= tol), rel, matvecs


def hierarchy_shapes(shape, max_coarse_x=16):
    """Halve every direction until the x extent is <= max_coarse_x (all extents stay integers)."""
    shapes = [tuple(int(s) for s in shape)]
    while shapes[-1][0] > max_coarse_x and all(s % 2 == 0 for s in shapes[-1]):
        shapes.append(tuple(s // 2 for s in shapes[-1]))
    return shapes


def spectral_omegas(xp, levels, iterations=30, seed=20260923):
    """omega_l = 4 / (3 lambda_l) for every non-coarsest level, lambda_l from a seeded power iteration
    on D^-1 A restricted to free DOFs. Deterministic for a given operator, so every arm that shares
    the estimating operator gets identical omegas. Returns (omegas, lambdas)."""
    omegas, lambdas = [], []
    for L in levels[:-1]:
        rng = np.random.default_rng(seed)
        v = xp.asarray(rng.standard_normal(L.mask.size)).astype(L.dtype) * L.mask
        v = v / xp.linalg.norm(v)
        lam = 0.0
        for _ in range(iterations):
            w = L.inv_diag * L.apply(v) * L.mask
            lam = float(xp.linalg.norm(w))
            v = w / lam
        lambdas.append(lam)
        omegas.append(4.0 / (3.0 * lam))
    return omegas, lambdas
