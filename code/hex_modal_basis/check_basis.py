"""Desk check: is modal8's Walsh-Hadamard basis the classical Hex8 monomial basis, and how exact is
its parity block structure off the axis-aligned rectangular brick?

No GPU, no timing, no novelty claim. Writes results.json next to --out.
"""
import argparse, itertools, json, pathlib, platform
import numpy as np

NODES = np.array([(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)], float)
# Monomials {1, x, y, z, xy, yz, zx, xyz} evaluated at the nodes: the Flanagan-Belytschko base and
# hourglass vectors for a brick in natural coordinates.
H = np.array([[1, p[0], p[1], p[2], p[0]*p[1], p[1]*p[2], p[2]*p[0], p[0]*p[1]*p[2]] for p in NODES])
ODD = ((1,4,6,7), (2,4,5,7), (3,5,6,7))    # monomials odd in x, y, z


def stiffness(X, E=1.0, nu=0.3):
    """Fully integrated (2x2x2 Gauss) isotropic trilinear hexahedron with nodal coordinates X."""
    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
    D = np.zeros((6,6)); D[:3,:3] = lam; D[range(3),range(3)] += 2*mu; D[3:,3:] = mu*np.eye(3)
    g = 1/np.sqrt(3); K = np.zeros((24,24))
    for xi, eta, zeta in itertools.product([-g, g], repeat=3):
        dN = np.array([[n[0]*(1+n[1]*eta)*(1+n[2]*zeta), n[1]*(1+n[0]*xi)*(1+n[2]*zeta),
                        n[2]*(1+n[0]*xi)*(1+n[1]*eta)] for n in NODES]) / 8
        J = dN.T @ X; dNx = dN @ np.linalg.inv(J).T
        B = np.zeros((6,24))
        for i in range(8):
            x, y, z = dNx[i]
            B[:, 3*i:3*i+3] = [[x,0,0],[0,y,0],[0,0,z],[y,x,0],[0,z,y],[z,0,x]]
        K += B.T @ D @ B * abs(np.linalg.det(J))
    return K


def parity(m, c):
    return tuple((m in ODD[a]) ^ (c == a) for a in range(3))


LABELS = [parity(m, c) for m in range(8) for c in range(3)]
OFF = np.array([[LABELS[i] != LABELS[j] for j in range(24)] for i in range(24)])
T = np.kron(H, np.eye(3)) / np.sqrt(8)


def off_share(X):
    Km = T.T @ stiffness(X) @ T
    return float(np.linalg.norm(Km[OFF]) / np.linalg.norm(Km))


def main():
    a = argparse.ArgumentParser(); a.add_argument("--out", type=pathlib.Path, required=True)
    out = a.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    sylvester = np.array([[1]])
    for _ in range(3):
        sylvester = np.block([[sylvester, sylvester], [sylvester, -sylvester]])
    brick = NODES * np.array([0.5, 0.35, 0.65])
    rng = np.random.default_rng(20260923)
    jitter = {}
    for eps in (0.0, 1e-3, 1e-2, 0.05, 0.1, 0.2):
        jitter[str(eps)] = float(np.median([off_share(brick + eps*0.7*rng.uniform(-1,1,brick.shape))
                                            for _ in range(20)]))
    shear = np.array([[1,0.3,0],[0,1,0.2],[0,0,1]])
    classes = {}
    for i, l in enumerate(LABELS):
        classes.setdefault(l, []).append(i)
    res = {
        "question": "Is modal8's basis the classical Hex8 monomial/hourglass basis, and where is its "
                    "parity block structure exact?",
        "basis_entries_are_pm1": bool(set(np.unique(H)) == {-1.0, 1.0}),
        "basis_orthogonal_HtH_eq_8I": bool(np.allclose(H.T @ H, 8*np.eye(8))),
        "basis_equals_sylvester_wht_up_to_order_and_sign": bool(
            np.allclose(np.sort(np.abs(H.T @ sylvester), axis=1)[:, -1], 8)),
        "parity_classes": len(classes),
        "class_sizes": sorted(len(v) for v in classes.values()),
        "dense_entries": 576,
        "block_entries": sum(len(v)**2 for v in classes.values()),
        "off_block_share_rectangular_brick": off_share(brick),
        "off_block_share_median_by_node_jitter_fraction": jitter,
        "off_block_share_affine_shear": off_share(brick @ shear.T),
        "interpretation": "Exact parity decoupling holds for axis-aligned rectangular bricks only "
                          "(structured voxel grids). Distortion leaks roughly linearly; an affine "
                          "shear already gives a large off-block share. Desk analysis only.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
