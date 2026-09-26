"""Symmetric-half control and one-warp Hex8 layouts; no author-code equivalence claim.

All three kernels use analytic structured indices, FP64 and atomic assembly.
The two warp layouts assign 24 active lanes per element, one scalar per lane.
"""
import numpy as np


def pack_symmetric(ke):
    ke = np.asarray(ke, dtype=np.float64).reshape(24, 24)
    if not np.isfinite(ke).all() or np.linalg.norm(ke - ke.T) > 2e-14 * np.linalg.norm(ke):
        raise ValueError('Symmetric-half storage requires a symmetric element matrix')
    return np.r_[np.diag(ke), ke[np.tril_indices(24, -1)]]


def symmetric_index(i, j):
    if i == j:
        return i
    hi, lo = max(i, j), min(i, j)
    return 24 + hi * (hi - 1) // 2 + lo


def emulate_modal32(values, blocks):
    """CPU simulation of the lane/shuffle schedule, independent of Modal.apply."""
    values = np.asarray(values).reshape(8, 3)
    v = np.array([values[(lane & 7) ^ (((lane & 7) & 2) >> 1), lane // 8]
                  for lane in range(24)])
    def transform(a):
        for bit in (1, 2, 4):
            old = a.copy()
            for lane in range(24):
                other = old[lane ^ bit]
                a[lane] = other - old[lane] if lane & bit else old[lane] + other
        return a
    v = transform(v)
    f = np.zeros(24)
    for lane in range(24):
        component, mode = divmod(lane, 8)
        group = mode ^ (1 << component)
        for d in range(3):
            source = 8*d + (mode ^ (1 << component) ^ (1 << d))
            f[lane] += blocks[group, component, d] * v[source]
    f = transform(f)
    out = np.zeros((8, 3))
    for lane in range(24):
        binary = lane & 7
        out[binary ^ ((binary & 2) >> 1), lane // 8] = f[lane]
    return out.ravel()


SOURCE = r'''
__device__ __forceinline__ int symidx(int i,int j) {
    if(i==j) return i;
    int hi=max(i,j),lo=min(i,j);
    return 24+hi*(hi-1)/2+lo;
}
__device__ __forceinline__ double wht8(double v,unsigned active,int mode) {
    #pragma unroll
    for(int bit=1;bit<8;bit*=2) {
        double other=__shfl_xor_sync(active,v,bit,8);
        v=(mode&bit)?other-v:v+other;
    }
    return v;
}
extern "C" __global__ void sym8(
    const double* __restrict__ packed,const double* __restrict__ E,
    const double* __restrict__ u,double* __restrict__ y,int nx,int ny,int nz) {
    __shared__ double S[300];
    for(int j=threadIdx.x;j<300;j+=blockDim.x) S[j]=packed[j];
    __syncthreads();
    int thread=blockIdx.x*blockDim.x+threadIdx.x;
    int e=thread/8,lane=thread&7;
    unsigned active=__ballot_sync(0xffffffffu,e<nx*ny*nz);
    if(e>=nx*ny*nz) return;
    int ez=e%nz,ey=(e/nz)%ny,ex=e/(nz*ny);
    int node=((ex+(lane&1))*(ny+1)+ey+((lane>>1)&1))*(nz+1)+ez+((lane>>2)&1);
    int row=3*(lane^((lane&2)>>1));
    double vx=u[3*node],vy=u[3*node+1],vz=u[3*node+2];
    double fx=0.,fy=0.,fz=0.;
    #pragma unroll
    for(int j=0;j<8;++j) {
        int source=j^((j&2)>>1),col=3*j;
        double ax=__shfl_sync(active,vx,source,8);
        double ay=__shfl_sync(active,vy,source,8);
        double az=__shfl_sync(active,vz,source,8);
        fx+=S[symidx(row,col)]*ax;fx+=S[symidx(row,col+1)]*ay;fx+=S[symidx(row,col+2)]*az;
        fy+=S[symidx(row+1,col)]*ax;fy+=S[symidx(row+1,col+1)]*ay;fy+=S[symidx(row+1,col+2)]*az;
        fz+=S[symidx(row+2,col)]*ax;fz+=S[symidx(row+2,col+1)]*ay;fz+=S[symidx(row+2,col+2)]*az;
    }
    double modulus=E[e];
    atomicAdd(y+3*node,modulus*fx);
    atomicAdd(y+3*node+1,modulus*fy);
    atomicAdd(y+3*node+2,modulus*fz);
}
extern "C" __global__ void modal32(
    const double* __restrict__ blocks,const double* __restrict__ E,
    const double* __restrict__ u,double* __restrict__ y,int nx,int ny,int nz) {
    __shared__ double C[72];
    for(int j=threadIdx.x;j<72;j+=blockDim.x) C[j]=blocks[j];
    __syncthreads();
    int lane=threadIdx.x&31,e=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    unsigned active=__ballot_sync(0xffffffffu,e<nx*ny*nz&&lane<24);
    if(e>=nx*ny*nz||lane>=24) return;
    int component=lane/8,mode=lane&7;
    int ez=e%nz,ey=(e/nz)%ny,ex=e/(nz*ny);
    int node=((ex+(mode&1))*(ny+1)+ey+((mode>>1)&1))*(nz+1)+ez+((mode>>2)&1);
    double v=wht8(u[3*node+component],active,mode);
    int group=mode^(1<<component);
    double f=0.;
    #pragma unroll
    for(int d=0;d<3;++d) {
        int source=8*d+(group^(1<<d));
        f+=C[group*9+component*3+d]*__shfl_sync(active,v,source,32);
    }
    f=wht8(f,active,mode);
    atomicAdd(y+3*node+component,E[e]*f);
}
extern "C" __global__ void dense32(
    const double* __restrict__ ke,const double* __restrict__ E,
    const double* __restrict__ u,double* __restrict__ y,int nx,int ny,int nz) {
    __shared__ double K[576];
    for(int j=threadIdx.x;j<576;j+=blockDim.x) K[j]=ke[j];
    __syncthreads();
    int lane=threadIdx.x&31,e=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    unsigned active=__ballot_sync(0xffffffffu,e<nx*ny*nz&&lane<24);
    if(e>=nx*ny*nz||lane>=24) return;
    int component=lane/8,mode=lane&7;
    int ez=e%nz,ey=(e/nz)%ny,ex=e/(nz*ny);
    int node=((ex+(mode&1))*(ny+1)+ey+((mode>>1)&1))*(nz+1)+ez+((mode>>2)&1);
    int row=3*(mode^((mode&2)>>1))+component;
    double v=u[3*node+component],f=0.;
    #pragma unroll
    for(int j=0;j<8;++j) {
        int source=j^((j&2)>>1),base=24*row+3*j;
        f+=K[base]*__shfl_sync(active,v,source,32);
        f+=K[base+1]*__shfl_sync(active,v,8+source,32);
        f+=K[base+2]*__shfl_sync(active,v,16+source,32);
    }
    atomicAdd(y+3*node+component,E[e]*f);
}
'''


class CudaVariants:
    BLOCK = 128
    NAMES = ('sym8', 'modal32', 'dense32')

    def __init__(self, shape, ke, blocks):
        import cupy as cp
        self.shape = tuple(map(int, shape))
        self.ne = int(np.prod(shape))
        self.coefficients = {'sym8': cp.asarray(pack_symmetric(ke)),
                             'modal32': cp.asarray(blocks), 'dense32': cp.asarray(ke)}
        self.y = cp.zeros(3 * int(np.prod(np.array(shape) + 1)), dtype=cp.float64)
        self.module = cp.RawModule(code=SOURCE, options=('-std=c++14',))
        self.kernels = {name: self.module.get_function(name) for name in self.NAMES}

    def matvec_full(self, u, E, path):
        self.y.fill(0)
        threads = 8 if path == 'sym8' else 32
        self.kernels[path](((threads*self.ne+self.BLOCK-1)//self.BLOCK,), (self.BLOCK,),
                           (self.coefficients[path], E, u, self.y, *self.shape))
        return self.y
