"""CUDA prototypes: parity blocks and a dense control with the same mapping.

Both use eight threads per element, analytic structured indices, FP64 values,
and atomic assembly. Author-layout arrays have z as the fastest spatial index.
"""
import numpy as np

SOURCE = r'''
__device__ __forceinline__ double transform(double v, unsigned active, int lane) {
    #pragma unroll
    for (int bit=1;bit<8;bit*=2) {
        const double other=__shfl_xor_sync(active,v,bit,8);
        v=(lane&bit)?other-v:v+other;
    }
    return v;
}

extern "C" __global__ void modal8(
    const double* __restrict__ blocks,const double* __restrict__ E,
    const double* __restrict__ u,double* __restrict__ y,
    int nx,int ny,int nz) {
    __shared__ double C[72];
    for(int j=threadIdx.x;j<72;j+=blockDim.x) C[j]=blocks[j];
    __syncthreads();
    int thread=blockIdx.x*blockDim.x+threadIdx.x;
    int e=thread/8,lane=thread&7;
    unsigned active=__ballot_sync(0xffffffffu,e<nx*ny*nz);
    if(e>=nx*ny*nz) return;
    int ez=e%nz, ey=(e/nz)%ny, ex=e/(nz*ny);
    int node=((ex+(lane&1))*(ny+1)+ey+((lane>>1)&1))*(nz+1)+ez+((lane>>2)&1);
    double vx=transform(u[3*node],active,lane);
    double vy=transform(u[3*node+1],active,lane);
    double vz=transform(u[3*node+2],active,lane);
    int gx=(lane^1)*9,gy=(lane^2)*9,gz=(lane^4)*9;
    double fx=C[gx]*vx+C[gx+1]*__shfl_xor_sync(active,vy,3,8)+C[gx+2]*__shfl_xor_sync(active,vz,5,8);
    double fy=C[gy+3]*__shfl_xor_sync(active,vx,3,8)+C[gy+4]*vy+C[gy+5]*__shfl_xor_sync(active,vz,6,8);
    double fz=C[gz+6]*__shfl_xor_sync(active,vx,5,8)+C[gz+7]*__shfl_xor_sync(active,vy,6,8)+C[gz+8]*vz;
    fx=transform(fx,active,lane);
    fy=transform(fy,active,lane);
    fz=transform(fz,active,lane);
    double modulus=E[e];
    atomicAdd(y+3*node,modulus*fx);
    atomicAdd(y+3*node+1,modulus*fy);
    atomicAdd(y+3*node+2,modulus*fz);
}

extern "C" __global__ void dense8(
    const double* __restrict__ ke,const double* __restrict__ E,
    const double* __restrict__ u,double* __restrict__ y,
    int nx,int ny,int nz) {
    __shared__ double K[576];
    for(int j=threadIdx.x;j<576;j+=blockDim.x) K[j]=ke[j];
    __syncthreads();
    int thread=blockIdx.x*blockDim.x+threadIdx.x;
    int e=thread/8,lane=thread&7;
    unsigned active=__ballot_sync(0xffffffffu,e<nx*ny*nz);
    if(e>=nx*ny*nz) return;
    int ez=e%nz, ey=(e/nz)%ny, ex=e/(nz*ny);
    int node=((ex+(lane&1))*(ny+1)+ey+((lane>>1)&1))*(nz+1)+ez+((lane>>2)&1);
    double vx=u[3*node],vy=u[3*node+1],vz=u[3*node+2];
    double fx=0.,fy=0.,fz=0.;
    // Native node order is 0,1,3,2,4,5,7,6 in binary-coordinate numbering.
    int native=lane^((lane&2)>>1);
    #pragma unroll
    for(int j=0;j<8;++j) {
        int source=j^((j&2)>>1);
        double ax=__shfl_sync(active,vx,source,8);
        double ay=__shfl_sync(active,vy,source,8);
        double az=__shfl_sync(active,vz,source,8);
        int base=(3*native)*24+3*j;
        fx+=K[base]*ax;fx+=K[base+1]*ay;fx+=K[base+2]*az;
        fy+=K[base+24]*ax;fy+=K[base+25]*ay;fy+=K[base+26]*az;
        fz+=K[base+48]*ax;fz+=K[base+49]*ay;fz+=K[base+50]*az;
    }
    double modulus=E[e];
    atomicAdd(y+3*node,modulus*fx);
    atomicAdd(y+3*node+1,modulus*fy);
    atomicAdd(y+3*node+2,modulus*fz);
}
'''


# Session E: two butterfly variants of modal8 (JPDC sections 63/77/88). Only the transform body and
# the kernel name differ from modal8; layout, block product, modulus scaling and atomic assembly are
# the identical text. Both are algebraically and IEEE-bitwise identical to modal8's butterfly
# (other-v == (-1)*v+other exactly; flipping the sign bit is exact negation).
#  modal8_muladd: FP64 adaptation of the signed multiply-add butterfly of Tri Dao / Dao AI Lab
#    fast-hadamard-transform (commit e7706faf..., BSD-3-Clause; ATTRIBUTION.md, DAO-LICENSE).
#  modal8_signbit: sign-bit flip then add.
_BUTTERFLY = "v=(lane&bit)?other-v:v+other;"
_VARIANT_BUTTERFLY = {
    'modal8_muladd': "const double sign=(lane&bit)?-1.0:1.0;\n        v=sign*v+other;",
    'modal8_signbit': ("const unsigned mask=(lane&bit)?0x80000000u:0u;\n"
                       "        const int hi=__double2hiint(v)^int(mask);\n"
                       "        v=other+__hiloint2double(hi,__double2loint(v));"),
}


def _variant(name):
    head = SOURCE[:SOURCE.index('extern "C" __global__ void dense8(')]
    assert head.count(_BUTTERFLY) == 1 and head.count('__global__ void modal8(') == 1
    head = head.replace('transform(', f'transform_{name}(')
    head = head.replace(_BUTTERFLY, _VARIANT_BUTTERFLY[name])
    return head.replace(f'__global__ void modal8(', f'__global__ void {name}(')


VARIANT_SOURCE = ''.join(_variant(n) for n in _VARIANT_BUTTERFLY)
MODAL_PATHS = ('modal8',) + tuple(_VARIANT_BUTTERFLY)


class CudaModal:
    BLOCK = 128

    def __init__(self, shape, ke, blocks):
        import cupy as cp
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = int(np.prod(shape))
        self.ke = cp.asarray(ke, dtype=cp.float64)
        self.blocks = cp.asarray(blocks, dtype=cp.float64)
        self.y = cp.zeros(3 * int(np.prod(np.array(shape) + 1)), dtype=cp.float64)
        self.module = cp.RawModule(code=SOURCE + VARIANT_SOURCE, options=('-std=c++14',))
        self.kernels = {name: self.module.get_function(name) for name in MODAL_PATHS + ('dense8',)}

    def matvec_full(self, u, E, path):
        self.y.fill(0)
        first = self.blocks if path in MODAL_PATHS else self.ke
        self.kernels[path](((8 * self.n_elem + self.BLOCK - 1) // self.BLOCK,), (self.BLOCK,),
                           (first, E, u, self.y, *self.shape))
        return self.y
