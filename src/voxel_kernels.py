"""Precision-templated parity-block (modal8) and dense8 CUDA kernels, plus a CPU lane emulator.

The FP64 source generated here is required to be byte-identical to the measured prototype in
code/hex_modal_cuda/cuda.py (checked by tests/test_kernels_cpu.py), so FP64 results remain those of
the imported evidence. The FP32 variant is the same text with `double` replaced by `float` and
kernel names suffixed `_fp32`; it exists for the FP32 falsification experiment (H4) and has NOT
been compiled or executed on a GPU yet.

`emulate` reproduces the kernels' lane logic (8 lanes per element, xor-butterfly transform, the
exact shuffle partners and block indices, per-lane atomic scatter) in NumPy at a chosen dtype, so
the index and shuffle algebra can be checked on a machine without CUDA.
"""
import importlib.util
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prototype = _load("voxel_cuda_prototype", "code/hex_modal_cuda/cuda.py")
_modal = _load("voxel_modal_reference", "code/hex_modal/modal.py")
PROTOTYPE_SOURCE = _prototype.SOURCE
Modal, quadrature, XYZ = _modal.Modal, _modal.quadrature, _modal.XYZ


def make_source(real: str) -> str:
    """CUDA source for `real` in {'double', 'float'}; 'double' returns the prototype unchanged."""
    if real == "double":
        return PROTOTYPE_SOURCE
    if real != "float":
        raise ValueError(real)
    src = PROTOTYPE_SOURCE.replace("double", "float")
    src = src.replace("__global__ void modal8(", "__global__ void modal8_fp32(")
    src = src.replace("__global__ void dense8(", "__global__ void dense8_fp32(")
    return src


def element_nodes(shape):
    """Global node index for each (element, binary lane), exactly as the kernels compute it."""
    nx, ny, nz = shape
    e = np.arange(nx * ny * nz)
    ez, ey, ex = e % nz, (e // nz) % ny, e // (nz * ny)
    lanes = np.arange(8)
    return (((ex[:, None] + (lanes & 1)) * (ny + 1) + ey[:, None] + ((lanes >> 1) & 1)) * (nz + 1)
            + ez[:, None] + ((lanes >> 2) & 1))


def _transform(v):
    v = v.copy()
    lanes = np.arange(8)
    for bit in (1, 2, 4):
        other = v[:, lanes ^ bit]
        v = np.where((lanes & bit)[None, :] != 0, other - v, v + other)
    return v


def emulate(path, shape, ke, E, u, dtype=np.float64):
    """NumPy replica of modal8/dense8 lane logic at `dtype`. Returns y (length 3*nodes)."""
    nodes = element_nodes(shape)
    n_nodes = int(np.prod(np.array(shape) + 1))
    E = np.asarray(E, dtype=dtype)
    u = np.asarray(u, dtype=dtype)
    vx, vy, vz = (u[3 * nodes + c] for c in range(3))
    lanes = np.arange(8)
    if path == "modal8":
        C = np.asarray(Modal(ke).blocks, dtype=np.float64).reshape(72).astype(dtype)
        vx, vy, vz = _transform(vx), _transform(vy), _transform(vz)
        gx, gy, gz = (lanes ^ 1) * 9, (lanes ^ 2) * 9, (lanes ^ 4) * 9
        fx = C[gx] * vx + C[gx + 1] * vy[:, lanes ^ 3] + C[gx + 2] * vz[:, lanes ^ 5]
        fy = C[gy + 3] * vx[:, lanes ^ 3] + C[gy + 4] * vy + C[gy + 5] * vz[:, lanes ^ 6]
        fz = C[gz + 6] * vx[:, lanes ^ 5] + C[gz + 7] * vy[:, lanes ^ 6] + C[gz + 8] * vz
        fx, fy, fz = _transform(fx), _transform(fy), _transform(fz)
    elif path == "dense8":
        K = np.asarray(ke, dtype=np.float64).reshape(576).astype(dtype)
        native = lanes ^ ((lanes & 2) >> 1)
        fx = np.zeros_like(vx); fy = np.zeros_like(vx); fz = np.zeros_like(vx)
        for j in range(8):
            src = j ^ ((j & 2) >> 1)
            ax, ay, az = vx[:, [src]], vy[:, [src]], vz[:, [src]]
            base = (3 * native) * 24 + 3 * j
            fx = fx + K[base] * ax + K[base + 1] * ay + K[base + 2] * az
            fy = fy + K[base + 24] * ax + K[base + 25] * ay + K[base + 26] * az
            fz = fz + K[base + 48] * ax + K[base + 49] * ay + K[base + 50] * az
    else:
        raise ValueError(path)
    y = np.zeros(3 * n_nodes, dtype=dtype)
    for c, f in enumerate((fx, fy, fz)):
        np.add.at(y, 3 * nodes + c, (E[:, None] * f).astype(dtype))
    return y


def assembled_reference(shape, ke, E, u):
    """Independent global K u in FP64 using the native node order of `ke`."""
    nx, ny, nz = shape
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    y = np.zeros(3 * n_nodes)
    for e in range(nx * ny * nz):
        ez, ey, ex = e % nz, (e // nz) % ny, e // (nz * ny)
        gn = [((ex + x) * (ny + 1) + ey + yy) * (nz + 1) + ez + z for x, yy, z in XYZ]
        dofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in gn]).reshape(24)
        y[dofs] += E[e] * (ke @ u[dofs])
    return y
