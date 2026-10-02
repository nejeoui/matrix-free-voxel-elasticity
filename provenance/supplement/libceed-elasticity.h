// Exact unit-cube Q1 isotropic elasticity for the libCEED operator interface.
// libCEED API conventions are attributed in ATTRIBUTION.md.
#include <ceed.h>

typedef struct {
  CeedScalar lambda, mu;
  CeedInt fault;
} ElasticContext;

CEED_QFUNCTION(Elasticity)(void *raw_context, const CeedInt Q,
                         const CeedScalar *const *in, CeedScalar *const *out) {
  const ElasticContext *context = (const ElasticContext *)raw_context;
  const CeedScalar *gradient = in[0], *rho = in[1], *weight = in[2];
  CeedScalar *flux = out[0];
  for (CeedInt q = 0; q < Q; q++) {
    const CeedScalar trace = gradient[q] + gradient[4 * Q + q] + gradient[8 * Q + q];
    const CeedScalar material = 1e-6 + (1 - 1e-6) * rho[q] * rho[q] * rho[q];
    // J = I/2: two derivative factors of 2 and det(J)=1/8 give 1/2.
    const CeedScalar scale = (context->fault == 2 ? 1.0 : 0.5) * material * weight[q];
    const CeedScalar lambda = context->lambda * (context->fault == 1 ? 1.01 : 1.0);
    for (CeedInt d = 0; d < 3; d++) {
      for (CeedInt c = 0; c < 3; c++) {
        const CeedScalar stress = context->mu * (gradient[(3*d+c)*Q+q] + gradient[(3*c+d)*Q+q])
                                + (d == c ? lambda * trace : 0.0);
        flux[(3*d+c)*Q+q] = scale * stress;
      }
    }
  }
  return 0;
}
