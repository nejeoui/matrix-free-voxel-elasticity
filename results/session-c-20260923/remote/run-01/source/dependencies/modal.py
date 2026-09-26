"""FP64 parity-block action for affine rectangular isotropic Hex8 cells.

The transform is a change of basis, not reduced integration or truncation of
physical modes. The input element matrix is rejected if it violates the
prescribed block structure beyond the independently declared roundoff bound.
"""
import numpy as np

XYZ = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
BITS = XYZ @ np.array([1, 2, 4])
BINARY_TO_NATIVE = np.argsort(BITS)
NATIVE_TO_BINARY = np.argsort(BINARY_TO_NATIVE)
H = np.array([[(-1.) ** ((p & n).bit_count()) for n in range(8)] for p in range(8)])
T = np.kron(H[:, BITS], np.eye(3))
GROUPS = np.array([[3 * (g ^ (1 << a)) + a for a in range(3)] for g in range(8)])


def quadrature(lengths, nu):
    """Separate eight-point integration, using engineering-strain Voigt order."""
    lengths = np.asarray(lengths, dtype=float)
    lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    signs = 2 * XYZ - 1
    ke = np.zeros((24, 24))
    for point in (2 * XYZ - 1) / np.sqrt(3.):
        derivatives = np.empty((8, 3))
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            derivatives[:, axis] = (signs[:, axis] *
                np.prod(1 + signs[:, others] * point[others], axis=1) / (4 * lengths[axis]))
        B = np.zeros((6, 24))
        for i, (dx, dy, dz) in enumerate(derivatives):
            B[:, 3 * i:3 * i + 3] = [[dx, 0, 0], [0, dy, 0], [0, 0, dz],
                                      [dy, dx, 0], [0, dz, dy], [dz, 0, dx]]
        ke += B.T @ D @ B * (np.prod(lengths) / 8)
    return ke


def butterfly(values):
    """Unnormalised eight-point transform; 24 additions per component."""
    result = values.copy()
    for stride in (1, 2, 4):
        for start in range(0, 8, 2 * stride):
            a = result[:, start:start + stride].copy()
            b = result[:, start + stride:start + 2 * stride].copy()
            result[:, start:start + stride] = a + b
            result[:, start + stride:start + 2 * stride] = a - b
    return result


class Modal:
    def __init__(self, ke, tolerance=2e-14):
        self.ke = np.asarray(ke, dtype=np.float64).reshape(24, 24)
        full = T @ self.ke @ T.T / 64
        sparse = np.zeros_like(full)
        self.blocks = np.empty((8, 3, 3))
        for g, indices in enumerate(GROUPS):
            self.blocks[g] = full[np.ix_(indices, indices)]
            sparse[np.ix_(indices, indices)] = self.blocks[g]
        self.discarded_relative = float(np.linalg.norm(full - sparse) / np.linalg.norm(full))
        if not np.isfinite(self.discarded_relative) or self.discarded_relative > tolerance:
            raise ValueError(f'Element is outside the declared parity-block model: {self.discarded_relative}')
        self.reconstructed = T.T @ sparse @ T
        self.full_modal = full
        self.sparse_modal = sparse

    def apply(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 8, 3)
        transformed = butterfly(values[:, BINARY_TO_NATIVE]).reshape(-1, 24)
        result = np.zeros_like(transformed)
        for g, indices in enumerate(GROUPS):
            result[:, indices] = transformed[:, indices] @ self.blocks[g].T
        return butterfly(result.reshape(-1, 8, 3))[:, NATIVE_TO_BINARY].reshape(-1, 24)


def geometry(shape):
    nx, ny, nz = shape
    z, y, x = np.indices((nz, ny, nx))
    cells = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    nodes = cells[:, None] + XYZ[None]
    ids = (nodes[:, :, 2] * (ny + 1) + nodes[:, :, 1]) * (nx + 1) + nodes[:, :, 0]
    return (3 * ids[:, :, None] + np.arange(3)).reshape(-1, 24)


def product(shape, rho, u, mask, floor, operator):
    edof = geometry(shape)
    local = (u * mask)[edof]
    forces = operator.apply(local)
    energy = np.sum(local * forces, axis=1)
    modulus = floor + (1 - floor) * rho ** 3
    out = np.bincount(edof.ravel(), weights=(forces * modulus[:, None]).ravel(), minlength=len(u))
    out[mask == 0] = u[mask == 0]
    gradient = -3 * (1 - floor) * rho ** 2 * energy
    return out, energy, gradient
