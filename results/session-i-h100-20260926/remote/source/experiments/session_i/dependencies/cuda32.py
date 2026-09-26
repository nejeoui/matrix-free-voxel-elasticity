"""FP32 parity-block and dense-mapping kernels for experiment E3.

The source is derived at import time from the measured FP64 prototype (`cuda.SOURCE`) by a pure
`double -> float` substitution and `_fp32` kernel names. It is the same transformation as
`src/voxel_kernels.make_source('float')`, which is checked by the Voxel CPU tests.
"""
import numpy as np

from cuda import SOURCE as FP64_SOURCE

SOURCE32 = (FP64_SOURCE.replace("double", "float")
            .replace("__global__ void modal8(", "__global__ void modal8_fp32(")
            .replace("__global__ void dense8(", "__global__ void dense8_fp32("))


class CudaModal32:
    BLOCK = 128

    def __init__(self, shape, ke, blocks):
        import cupy as cp
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = int(np.prod(shape))
        self.ke = cp.asarray(np.asarray(ke, dtype=np.float64).astype(np.float32))
        self.blocks = cp.asarray(np.asarray(blocks, dtype=np.float64).astype(np.float32))
        self.y = cp.zeros(3 * int(np.prod(np.array(shape) + 1)), dtype=cp.float32)
        self.module = cp.RawModule(code=SOURCE32, options=('-std=c++14',))
        self.kernels = {name: self.module.get_function(name) for name in ('modal8_fp32', 'dense8_fp32')}

    def matvec_full(self, u, E, path):
        self.y.fill(0)
        first = self.blocks if path == 'modal8_fp32' else self.ke
        self.kernels[path](((8 * self.n_elem + self.BLOCK - 1) // self.BLOCK,), (self.BLOCK,),
                           (first, E, u, self.y, *self.shape))
        return self.y
