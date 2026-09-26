PetscInt Hex8Isoparametric(PetscScalar* X, PetscScalar* Y, PetscScalar* Z, PetscScalar nu,
                                             PetscInt redInt, PetscScalar* ke) {
    // HEX8_ISOPARAMETRIC - Computes HEX8 isoparametric element matrices
    // The element stiffness matrix is computed as:
    //
    //       ke = int(int(int(B^T*C*B,x),y),z)
    //
    // For an isoparameteric element this integral becomes:
    //
    //       ke = int(int(int(B^T*C*B*det(J),xi=-1..1),eta=-1..1),zeta=-1..1)
    //
    // where B is the more complicated expression:
    // B = [dx*alpha1 + dy*alpha2 + dz*alpha3]*N
    // where
    // dx = [invJ11 invJ12 invJ13]*[dxi deta dzeta]
    // dy = [invJ21 invJ22 invJ23]*[dxi deta dzeta]
    // dy = [invJ31 invJ32 invJ33]*[dxi deta dzeta]
    //
    // Remark: The elasticity modulus is left out in the below
    // computations, because we multiply with it afterwards (the aim is
    // topology optimization).
    // Furthermore, this is not the most efficient code, but it is readable.
    //
    /////////////////////////////////////////////////////////////////////////////////
    //////// INPUT:
    // X, Y, Z  = Vectors containing the coordinates of the eight nodes
    //               (x1,y1,z1,x2,y2,z2,...,x8,y8,z8). Where node 1 is in the
    //               lower left corner, and node 2 is the next node
    //               counterclockwise (looking in the negative z-dir). Finish the
    //               x-y-plane and then move in the positive z-dir.
    // redInt   = Reduced integration option boolean (here an integer).
    //           	redInt == 0 (false): Full integration
    //           	redInt == 1 (true): Reduced integration
    // nu 		= Poisson's ratio.
    //
    //////// OUTPUT:
    // ke  = Element stiffness matrix. Needs to be multiplied with elasticity
    // modulus
    //
    //   Written 2013 at
    //   Department of Mechanical Engineering
    //   Technical University of Denmark (DTU).
    /////////////////////////////////////////////////////////////////////////////////

    //// COMPUTE ELEMENT STIFFNESS MATRIX
    // Lame's parameters (with E=1.0):
    PetscScalar lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    PetscScalar mu     = 1.0 / (2.0 * (1.0 + nu));
    // Constitutive matrix
    PetscScalar C[6][6] = {{lambda + 2.0 * mu, lambda, lambda, 0.0, 0.0, 0.0},
                           {lambda, lambda + 2.0 * mu, lambda, 0.0, 0.0, 0.0},
                           {lambda, lambda, lambda + 2.0 * mu, 0.0, 0.0, 0.0},
                           {0.0, 0.0, 0.0, mu, 0.0, 0.0},
                           {0.0, 0.0, 0.0, 0.0, mu, 0.0},
                           {0.0, 0.0, 0.0, 0.0, 0.0, mu}};
    // Gauss points (GP) and weigths
    // Two Gauss points in all directions (total of eight)
    PetscScalar GP[2] = {-0.577350269189626, 0.577350269189626};
    // Corresponding weights
    PetscScalar W[2] = {1.0, 1.0};
    // If reduced integration only use one GP
    if (redInt) {
        GP[0] = 0.0;
        W[0]  = 2.0;
    }
    // Matrices that help when we gather the strain-displacement matrix:
    PetscScalar alpha1[6][3];
    PetscScalar alpha2[6][3];
    PetscScalar alpha3[6][3];
    memset(alpha1, 0, sizeof(alpha1[0][0]) * 6 * 3); // zero out
    memset(alpha2, 0, sizeof(alpha2[0][0]) * 6 * 3); // zero out
    memset(alpha3, 0, sizeof(alpha3[0][0]) * 6 * 3); // zero out
    alpha1[0][0] = 1.0;
    alpha1[3][1] = 1.0;
    alpha1[5][2] = 1.0;
    alpha2[1][1] = 1.0;
    alpha2[3][0] = 1.0;
    alpha2[4][2] = 1.0;
    alpha3[2][2] = 1.0;
    alpha3[4][1] = 1.0;
    alpha3[5][0] = 1.0;
    PetscScalar  dNdxi[8];
    PetscScalar  dNdeta[8];
    PetscScalar  dNdzeta[8];
    PetscScalar  J[3][3];
    PetscScalar  invJ[3][3];
    PetscScalar  beta[6][3];
    PetscScalar  B[6][24]; // Note: Small enough to be allocated on stack
    PetscScalar* dN;
    // Make sure the stiffness matrix is zeroed out:
    memset(ke, 0, sizeof(ke[0]) * 24 * 24);
    // Perform the numerical integration
    for (PetscInt ii = 0; ii < 2 - redInt; ii++) {
        for (PetscInt jj = 0; jj < 2 - redInt; jj++) {
            for (PetscInt kk = 0; kk < 2 - redInt; kk++) {
                // Integration point
                PetscScalar xi   = GP[ii];
                PetscScalar eta  = GP[jj];
                PetscScalar zeta = GP[kk];
                // Differentiated shape functions
                DifferentiatedShapeFunctions(xi, eta, zeta, dNdxi, dNdeta, dNdzeta);
                // Jacobian
                J[0][0] = Dot(dNdxi, X, 8);
                J[0][1] = Dot(dNdxi, Y, 8);
                J[0][2] = Dot(dNdxi, Z, 8);
                J[1][0] = Dot(dNdeta, X, 8);
                J[1][1] = Dot(dNdeta, Y, 8);
                J[1][2] = Dot(dNdeta, Z, 8);
                J[2][0] = Dot(dNdzeta, X, 8);
                J[2][1] = Dot(dNdzeta, Y, 8);
                J[2][2] = Dot(dNdzeta, Z, 8);
                // Inverse and determinant
                PetscScalar detJ = Inverse3M(J, invJ);
                // Weight factor at this point
                PetscScalar weight = W[ii] * W[jj] * W[kk] * detJ;
                // Strain-displacement matrix
                memset(B, 0, sizeof(B[0][0]) * 6 * 24); // zero out
                for (PetscInt ll = 0; ll < 3; ll++) {
                    // Add contributions from the different derivatives
                    if (ll == 0) {
                        dN = dNdxi;
                    }
                    if (ll == 1) {
                        dN = dNdeta;
                    }
                    if (ll == 2) {
                        dN = dNdzeta;
                    }
                    // Assemble strain operator
                    for (PetscInt i = 0; i < 6; i++) {
                        for (PetscInt j = 0; j < 3; j++) {
                            beta[i][j] =
                                invJ[0][ll] * alpha1[i][j] + invJ[1][ll] * alpha2[i][j] + invJ[2][ll] * alpha3[i][j];
                        }
                    }
                    // Add contributions to strain-displacement matrix
                    for (PetscInt i = 0; i < 6; i++) {
                        for (PetscInt j = 0; j < 24; j++) {
                            B[i][j] = B[i][j] + beta[i][j % 3] * dN[j / 3];
                        }
                    }
                }
                // Finally, add to the element matrix
                for (PetscInt i = 0; i < 24; i++) {
                    for (PetscInt j = 0; j < 24; j++) {
                        for (PetscInt k = 0; k < 6; k++) {
                            for (PetscInt l = 0; l < 6; l++) {

                                ke[j + 24 * i] = ke[j + 24 * i] + weight * (B[k][i] * C[k][l] * B[l][j]);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
PetscScalar Dot(PetscScalar* v1, PetscScalar* v2, PetscInt l) {
    // Function that returns the dot product of v1 and v2,
    // which must have the same length l
    PetscScalar result = 0.0;
    for (PetscInt i = 0; i < l; i++) {
        result = result + v1[i] * v2[i];
    }
    return result;
}

void DifferentiatedShapeFunctions(PetscScalar xi, PetscScalar eta, PetscScalar zeta,
                                                    PetscScalar* dNdxi, PetscScalar* dNdeta, PetscScalar* dNdzeta) {
    // differentiatedShapeFunctions - Computes differentiated shape functions
    // At the point given by (xi, eta, zeta).
    // With respect to xi:
    dNdxi[0] = -0.125 * (1.0 - eta) * (1.0 - zeta);
    dNdxi[1] = 0.125 * (1.0 - eta) * (1.0 - zeta);
    dNdxi[2] = 0.125 * (1.0 + eta) * (1.0 - zeta);
    dNdxi[3] = -0.125 * (1.0 + eta) * (1.0 - zeta);
    dNdxi[4] = -0.125 * (1.0 - eta) * (1.0 + zeta);
    dNdxi[5] = 0.125 * (1.0 - eta) * (1.0 + zeta);
    dNdxi[6] = 0.125 * (1.0 + eta) * (1.0 + zeta);
    dNdxi[7] = -0.125 * (1.0 + eta) * (1.0 + zeta);
    // With respect to eta:
    dNdeta[0] = -0.125 * (1.0 - xi) * (1.0 - zeta);
    dNdeta[1] = -0.125 * (1.0 + xi) * (1.0 - zeta);
    dNdeta[2] = 0.125 * (1.0 + xi) * (1.0 - zeta);
    dNdeta[3] = 0.125 * (1.0 - xi) * (1.0 - zeta);
    dNdeta[4] = -0.125 * (1.0 - xi) * (1.0 + zeta);
    dNdeta[5] = -0.125 * (1.0 + xi) * (1.0 + zeta);
    dNdeta[6] = 0.125 * (1.0 + xi) * (1.0 + zeta);
    dNdeta[7] = 0.125 * (1.0 - xi) * (1.0 + zeta);
    // With respect to zeta:
    dNdzeta[0] = -0.125 * (1.0 - xi) * (1.0 - eta);
    dNdzeta[1] = -0.125 * (1.0 + xi) * (1.0 - eta);
    dNdzeta[2] = -0.125 * (1.0 + xi) * (1.0 + eta);
    dNdzeta[3] = -0.125 * (1.0 - xi) * (1.0 + eta);
    dNdzeta[4] = 0.125 * (1.0 - xi) * (1.0 - eta);
    dNdzeta[5] = 0.125 * (1.0 + xi) * (1.0 - eta);
    dNdzeta[6] = 0.125 * (1.0 + xi) * (1.0 + eta);
    dNdzeta[7] = 0.125 * (1.0 - xi) * (1.0 + eta);
}

PetscScalar Inverse3M(PetscScalar J[][3], PetscScalar invJ[][3]) {
    // inverse3M - Computes the inverse of a 3x3 matrix
    PetscScalar detJ = J[0][0] * (J[1][1] * J[2][2] - J[2][1] * J[1][2]) -
                       J[0][1] * (J[1][0] * J[2][2] - J[2][0] * J[1][2]) +
                       J[0][2] * (J[1][0] * J[2][1] - J[2][0] * J[1][1]);
    invJ[0][0] = (J[1][1] * J[2][2] - J[2][1] * J[1][2]) / detJ;
    invJ[0][1] = -(J[0][1] * J[2][2] - J[0][2] * J[2][1]) / detJ;
    invJ[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) / detJ;
    invJ[1][0] = -(J[1][0] * J[2][2] - J[1][2] * J[2][0]) / detJ;
    invJ[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) / detJ;
    invJ[1][2] = -(J[0][0] * J[1][2] - J[0][2] * J[1][0]) / detJ;
    invJ[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) / detJ;
    invJ[2][1] = -(J[0][0] * J[2][1] - J[0][1] * J[2][0]) / detJ;
    invJ[2][2] = (J[0][0] * J[1][1] - J[1][0] * J[0][1]) / detJ;
    return detJ;
}
