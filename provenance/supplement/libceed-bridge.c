// Draft device-resident bridge. Unexecuted; requires a new prospective GPU study.
#include "elasticity.h"
#include "bridge.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CALL(expression) do {int error=(expression);if(error){fprintf(stderr,"libCEED error %d at %s:%d\n",error,__FILE__,__LINE__);abort();}} while(0)
#define REQUIRE(condition) do {if(!(condition)){fprintf(stderr,"Requirement failed at %s:%d\n",__FILE__,__LINE__);abort();}} while(0)
typedef struct {
  Ceed ceed;
  CeedBasis basis, material_basis;
  CeedElemRestriction restriction, material_restriction;
  CeedQFunction qfunction;
  CeedQFunctionContext context;
  CeedOperator op;
  CeedVector input, output, material;
  CeedInt nx, ny, nz, ne, nd;
} Problem;

static void create(Problem *p,const char *resource,int nx,int ny,int nz,const double *rho) {
  memset(p,0,sizeof(*p));p->nx=nx;p->ny=ny;p->nz=nz;p->ne=nx*ny*nz;p->nd=3*(nx+1)*(ny+1)*(nz+1);
  CALL(CeedInit(resource,&p->ceed));
  const char *resolved;CeedMemType memory;
  CALL(CeedGetResource(p->ceed,&resolved));CALL(CeedGetPreferredMemType(p->ceed,&memory));
  REQUIRE(!strncmp(resource,"/gpu/cuda/",10) && !strncmp(resolved,resource,strlen(resource)) && memory==CEED_MEM_DEVICE);
  CALL(CeedView(p->ceed,stdout));
  printf("requested_resource=%s resolved_resource=%s preferred_memory=device\n",resource,resolved);
  CeedInt *offsets=malloc(8ull*p->ne*sizeof(*offsets));REQUIRE(offsets);
  for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++) {
    int e=(z*ny+y)*nx+x;
    for(int lane=0;lane<8;lane++) {
      int node=((z+((lane>>2)&1))*(ny+1)+y+1-((lane>>1)&1))*(nx+1)+x+(lane&1);
      offsets[8*e+lane]=3*node;
    }
  }
  CALL(CeedElemRestrictionCreate(p->ceed,p->ne,8,3,1,p->nd,CEED_MEM_HOST,CEED_COPY_VALUES,offsets,&p->restriction));
  free(offsets);
  CALL(CeedBasisCreateTensorH1Lagrange(p->ceed,3,3,2,2,CEED_GAUSS,&p->basis));
  // A discontinuous constant scalar basis carries one physical density per cell.
  // This avoids duplicating it into eight stored quadrature values.
  CeedScalar qref[2],qweight[2],interpolation[2]={1,1},gradient[2]={0,0};
  CeedInt strides[3]={1,1,1};
  CALL(CeedGaussQuadrature(2,qref,qweight));
  CALL(CeedBasisCreateTensorH1(p->ceed,3,1,1,2,interpolation,gradient,qref,qweight,&p->material_basis));
  CALL(CeedElemRestrictionCreateStrided(p->ceed,p->ne,1,1,p->ne,strides,&p->material_restriction));
  CALL(CeedVectorCreate(p->ceed,p->ne,&p->material));
  CALL(CeedVectorSetArray(p->material,CEED_MEM_HOST,CEED_COPY_VALUES,(CeedScalar *)rho));
  CALL(CeedVectorCreate(p->ceed,p->nd,&p->input));CALL(CeedVectorCreate(p->ceed,p->nd,&p->output));
  CALL(CeedQFunctionCreateInterior(p->ceed,1,Elasticity,Elasticity_loc,&p->qfunction));
  CALL(CeedQFunctionAddInput(p->qfunction,"gradient",9,CEED_EVAL_GRAD));
  CALL(CeedQFunctionAddInput(p->qfunction,"rho",1,CEED_EVAL_INTERP));
  CALL(CeedQFunctionAddInput(p->qfunction,"weight",1,CEED_EVAL_WEIGHT));
  CALL(CeedQFunctionAddOutput(p->qfunction,"flux",9,CEED_EVAL_GRAD));
  CALL(CeedQFunctionContextCreate(p->ceed,&p->context));
  CALL(CeedQFunctionSetContext(p->qfunction,p->context));
  CALL(CeedOperatorCreate(p->ceed,p->qfunction,CEED_QFUNCTION_NONE,CEED_QFUNCTION_NONE,&p->op));
  CALL(CeedOperatorSetField(p->op,"gradient",p->restriction,p->basis,CEED_VECTOR_ACTIVE));
  CALL(CeedOperatorSetField(p->op,"rho",p->material_restriction,p->material_basis,p->material));
  CALL(CeedOperatorSetField(p->op,"weight",CEED_ELEMRESTRICTION_NONE,p->basis,CEED_VECTOR_NONE));
  CALL(CeedOperatorSetField(p->op,"flux",p->restriction,p->basis,CEED_VECTOR_ACTIVE));
}
static void destroy(Problem *p) {
  CALL(CeedOperatorDestroy(&p->op));CALL(CeedQFunctionDestroy(&p->qfunction));
  CALL(CeedQFunctionContextDestroy(&p->context));
  CALL(CeedVectorDestroy(&p->input));CALL(CeedVectorDestroy(&p->output));CALL(CeedVectorDestroy(&p->material));
  CALL(CeedBasisDestroy(&p->basis));CALL(CeedBasisDestroy(&p->material_basis));
  CALL(CeedElemRestrictionDestroy(&p->restriction));CALL(CeedElemRestrictionDestroy(&p->material_restriction));
  CALL(CeedDestroy(&p->ceed));
}
void jpdc_libceed_set_fault(void *handle,int fault) {
  Problem *p=handle;
  REQUIRE(fault>=0 && fault<=3);
  ElasticContext context={.lambda=.3/(1.3*.4),.mu=1/(2*1.3),.fault=fault};
  CALL(CeedQFunctionContextSetData(p->context,CEED_MEM_HOST,CEED_COPY_VALUES,sizeof(context),&context));
}
void jpdc_libceed_set_input(void *handle,const double *input) {
  Problem *p=handle;
  CALL(CeedVectorSetArray(p->input,CEED_MEM_HOST,CEED_COPY_VALUES,(CeedScalar *)input));
}
void *jpdc_libceed_create(const char *resource,int nx,int ny,int nz,const double *rho,const double *input) {
  _Static_assert(sizeof(CeedScalar)==8 && DBL_MANT_DIG==53,"FP64 libCEED required");
  Problem *p=malloc(sizeof(*p));REQUIRE(p);
  create(p,resource,nx,ny,nz,rho);
  jpdc_libceed_set_fault(p,0);jpdc_libceed_set_input(p,input);
  CALL(CeedVectorSetValue(p->output,NAN));
  // Force initial input/material transfers during setup, before any timing.
  const CeedScalar *device;
  CALL(CeedVectorGetArrayRead(p->input,CEED_MEM_DEVICE,&device));
  CALL(CeedVectorRestoreArrayRead(p->input,&device));
  CALL(CeedVectorGetArrayRead(p->material,CEED_MEM_DEVICE,&device));
  CALL(CeedVectorRestoreArrayRead(p->material,&device));
  return p;
}
void jpdc_libceed_begin(void *handle,const double **input,double **output) {
  Problem *p=handle;
  // CeedOperatorApply zeros the reused output before assembled addition.
  CALL(CeedOperatorApply(p->op,p->input,p->output,CEED_REQUEST_IMMEDIATE));
  CALL(CeedVectorGetArrayRead(p->input,CEED_MEM_DEVICE,input));
  CALL(CeedVectorGetArray(p->output,CEED_MEM_DEVICE,output));
}
void jpdc_libceed_end(void *handle,const double **input,double **output) {
  Problem *p=handle;
  // Caller launches the identity-row kernel before releasing these accesses.
  CALL(CeedVectorRestoreArray(p->output,output));
  CALL(CeedVectorRestoreArrayRead(p->input,input));
}
static CeedVector field_vector(Problem *p,int field) {
  REQUIRE(field>=0 && field<=2);
  return field==0?p->material:field==1?p->input:p->output;
}
void jpdc_libceed_read(void *handle,int field,const double **values) {
  CALL(CeedVectorGetArrayRead(field_vector(handle,field),CEED_MEM_DEVICE,values));
}
void jpdc_libceed_restore_read(void *handle,int field,const double **values) {
  CALL(CeedVectorRestoreArrayRead(field_vector(handle,field),values));
}
void jpdc_libceed_destroy(void *handle) {
  destroy(handle);free(handle);
}
