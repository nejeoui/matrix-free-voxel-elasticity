// Independent natural-order assembly, using the pinned PETSc element routine.
// The four DTU/TopOpt math functions are extracted unchanged by build_reference.py.
#include <cstring>
#include <cmath>
using PetscScalar = double;
using PetscInt = int;
PetscScalar Dot(PetscScalar*, PetscScalar*, PetscInt);
void DifferentiatedShapeFunctions(PetscScalar, PetscScalar, PetscScalar,
                                  PetscScalar*, PetscScalar*, PetscScalar*);
PetscScalar Inverse3M(PetscScalar[][3], PetscScalar[][3]);
#include "petsc_element.inc"
extern "C" void element(double h, double nu, double *ke) {
    double X[8]={0,h,h,0,0,h,h,0},Y[8]={0,0,h,h,0,0,h,h},Z[8]={0,0,0,0,h,h,h,h};
    Hex8Isoparametric(X,Y,Z,nu,0,ke);
}
extern "C" void product(int nx,int ny,int nz,const double *rho,const double *u,
                         const double *ke,const double *mask,double emin,double emax,double penal,
                         double *y,double *ce,double *diag) {
    int nd=3*(nx+1)*(ny+1)*(nz+1);
    std::memset(y,0,nd*sizeof(double));
    std::memset(diag,0,nd*sizeof(double));
    const int dx[8]={0,1,1,0,0,1,1,0},dy[8]={0,0,1,1,0,0,1,1},dz[8]={0,0,0,0,1,1,1,1};
    for(int z=0;z<nz;++z) for(int yy=0;yy<ny;++yy) for(int x=0;x<nx;++x) {
        int e=(z*ny+yy)*nx+x,ids[24];double ue[24];
        double E=emin+(emax-emin)*std::pow(rho[e],penal);
        for(int n=0;n<8;++n) for(int c=0;c<3;++c) {
            int k=3*n+c;
            ids[k]=3*(((z+dz[n])*(ny+1)+yy+dy[n])*(nx+1)+x+dx[n])+c;
            ue[k]=mask[ids[k]]?u[ids[k]]:0;
        }
        ce[e]=0;
        for(int r=0;r<24;++r) {
            double v=0;
            for(int c=0;c<24;++c) v+=ke[24*r+c]*ue[c];
            y[ids[r]]+=E*v;ce[e]+=ue[r]*v;diag[ids[r]]+=E*ke[24*r+r];
        }
    }
    for(int i=0;i<nd;++i) if(!mask[i]) { y[i]=u[i];diag[i]=1; }
}
