"""CPU revision checks of explicit parity permutation and composite quadrature.

Run: python3 analysis/revision_algebra.py
Writes paper/revision/algebra-checks.json, algebra-invariants.tex and quadrature-checks.tex.
Uses the retained Modal implementation without modifying the imported code snapshot.
This is an algebra/quadrature calculation, not GPU validation or fresh performance data.
"""
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/hex_modal"))
from modal import GROUPS, T, XYZ, Modal, quadrature  # noqa: E402


def midpoint_stiffness(lengths, nu, points_per_axis):
    """Independent tensor composite midpoint integration of the engineering-strain form."""
    lengths = np.asarray(lengths, dtype=float)
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1 / (2 * (1 + nu))
    material = np.diag([2 * mu] * 3 + [mu] * 3)
    material[:3, :3] += lam
    signs = 2 * XYZ - 1
    axis_points = -1 + (2 * np.arange(points_per_axis) + 1) / points_per_axis
    matrix = np.zeros((24, 24))
    for point in itertools.product(axis_points, repeat=3):
        point = np.array(point)
        gradient = np.empty((8, 3))
        for d in range(3):
            others = [a for a in range(3) if a != d]
            gradient[:, d] = (signs[:, d] * np.prod(1 + signs[:, others] * point[others], axis=1)
                              / (4 * lengths[d]))
        strain = np.zeros((6, 24))
        for node, (dx, dy, dz) in enumerate(gradient):
            strain[:, 3 * node:3 * node + 3] = [[dx, 0, 0], [0, dy, 0], [0, 0, dz],
                                             [dy, dx, 0], [0, dz, dy], [dz, 0, dx]]
        matrix += strain.T @ material @ strain * np.prod(lengths) / points_per_axis ** 3
    return matrix


def spectrum(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 1e-10 * np.linalg.norm(matrix, 2)
    return {"eigenvalue_tolerance": float(tolerance),
            "near_zero_eigenvalues": int(np.count_nonzero(np.abs(eigenvalues) <= tolerance)),
            "positive_eigenvalues": int(np.count_nonzero(eigenvalues > tolerance)),
            "negative_eigenvalues": int(np.count_nonzero(eigenvalues < -tolerance)),
            "smallest_positive_eigenvalue": float(np.min(eigenvalues[eigenvalues > tolerance])),
            "eigenvalues": eigenvalues.tolist()}


def check_cell(lengths):
    matrix = quadrature(lengths, 0.3)
    modal = Modal(matrix)
    permutation = np.zeros((24, 24))
    for g in range(8):
        for c in range(3):
            permutation[3 * g + c, 3 * (g ^ (1 << c)) + c] = 1
    assert np.array_equal(GROUPS.ravel(), np.argmax(permutation, axis=1))
    assert np.array_equal(T @ T.T, 8 * np.eye(24))
    transformed = permutation @ T @ matrix @ T.T @ permutation.T
    block_diagonal = np.zeros((24, 24))
    for g in range(8):
        block_diagonal[3 * g:3 * g + 3, 3 * g:3 * g + 3] = transformed[3 * g:3 * g + 3, 3 * g:3 * g + 3]
        np.testing.assert_array_equal(modal.blocks[g], transformed[3 * g:3 * g + 3, 3 * g:3 * g + 3] / 64)
    reconstructed = T.T @ permutation.T @ block_diagonal @ permutation @ T / 64
    np.testing.assert_array_equal(reconstructed, modal.reconstructed)
    # Analytic translations and rotations around the cell centre must span the nullspace.
    positions = (XYZ - .5) * np.asarray(lengths)
    rigid = np.column_stack([np.tile(np.eye(3)[c], 8) for c in range(3)] +
                            [np.cross(np.eye(3)[c], positions).ravel() for c in range(3)])
    rigid, _ = np.linalg.qr(rigid)
    rigid_error = np.linalg.norm(matrix @ rigid, "fro") / np.linalg.norm(matrix, "fro")
    rng = np.random.default_rng(20260927)
    vectors = np.concatenate((np.eye(24), rng.standard_normal((32, 24))))
    action_error = np.linalg.norm(modal.apply(vectors) - vectors @ matrix.T) / np.linalg.norm(vectors @ matrix.T)
    row = {"lengths": list(lengths), "young_modulus": 1, "poisson_ratio": .3,
           "reconstruction_relative_frobenius": float(np.linalg.norm(matrix - reconstructed, "fro") / np.linalg.norm(matrix, "fro")),
           "off_block_relative_frobenius": float(np.linalg.norm(transformed - block_diagonal, "fro") / np.linalg.norm(transformed, "fro")),
           "rigid_basis_residual_relative_frobenius": float(rigid_error),
           "product_relative_l2_aggregate": float(action_error), "action_vectors": len(vectors),
           "stored_blocks_C_over_64": modal.blocks.tolist(),
           "reconstructed_spectrum": spectrum(reconstructed), **spectrum(matrix)}
    assert row["reconstruction_relative_frobenius"] < 2e-14
    assert row["off_block_relative_frobenius"] < 2e-14
    assert rigid_error < 2e-14 and action_error < 2e-14
    assert row["near_zero_eigenvalues"] == 6 and row["positive_eigenvalues"] == 18 and row["negative_eigenvalues"] == 0
    assert row["reconstructed_spectrum"]["near_zero_eigenvalues"] == 6
    assert row["reconstructed_spectrum"]["positive_eigenvalues"] == 18
    assert row["reconstructed_spectrum"]["negative_eigenvalues"] == 0
    return row


def main():
    retained_path = ROOT / "evidence/companion/results/rescope/hex-modal-priorart-20260923/cpu-02/results.json"
    retained = json.loads(retained_path.read_text())
    invariant_rows = [check_cell(lengths) for lengths in [(1, 1, 1), (1, 2, 3)]]
    quadrature_rows = []
    for nu in (.2, .3, .45):
        exact = quadrature([1, 1, 1], nu)
        composite = midpoint_stiffness([1, 1, 1], nu, 10)
        discrepancy = np.linalg.norm(composite - exact, "fro") / np.linalg.norm(exact, "fro")
        source = next(r for r in retained["openmp_cases"] if r["nu"] == nu and r["level"] == 0)
        source_discrepancy = source["source_sum_vs_exact_gauss"]["relative_frobenius"]
        row = {"lengths": [1, 1, 1], "poisson_ratio": nu, "young_modulus": 1,
               "points_per_axis": 10, "points_per_fine_element": 1000,
               "relative_frobenius_difference_from_exact_gauss": float(discrepancy),
               "retained_source_relative_frobenius_difference": source_discrepancy,
               "difference_from_retained_source_discrepancy": float(abs(discrepancy - source_discrepancy)),
               "composite_spectrum": spectrum(composite),
               "literal_one_point_negative_control_spectrum": spectrum(midpoint_stiffness([1, 1, 1], nu, 1))}
        assert abs(discrepancy - source_discrepancy) < 1e-12
        assert row["composite_spectrum"]["near_zero_eigenvalues"] == 6
        assert row["composite_spectrum"]["positive_eigenvalues"] == 18
        assert row["literal_one_point_negative_control_spectrum"]["near_zero_eigenvalues"] == 18
        quadrature_rows.append(row)
    source_files = [Path(__file__).resolve(), ROOT / "code/hex_modal/modal.py", retained_path,
                    retained_path.parents[1] / "cpu-protocol-02.json"]
    result = {"scope": "Fresh CPU algebra/quadrature checks; no fresh GPU execution", "passed": True,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
              "norms": {"reconstruction": "||K - T^T P^T blockdiag(P T K T^T P^T) P T /64||_F / ||K||_F",
                        "off_block": "||C-blockdiag(C)||_F/||C||_F", "eigenvalue_tolerance": "1e-10 * ||K||_2"},
              "invariants": invariant_rows, "quadrature": quadrature_rows,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    out = ROOT / "paper/revision"
    out.mkdir(exist_ok=True)
    (out / "algebra-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    tex = [r"\begin{tabular}{lrrrr}", r"\hline", r"Cell & Reconstruction & Off-block & Nullity & Min. positive $\lambda$\\", r"\hline"]
    for r in invariant_rows:
        cell = "$" + r"\times".join(map(str, r["lengths"])) + "$"
        tex.append(f"{cell} & {r['reconstruction_relative_frobenius']:.3e} & {r['off_block_relative_frobenius']:.3e} & {r['near_zero_eigenvalues']} & {r['smallest_positive_eigenvalue']:.5f}" + r"\\")
    tex += [r"\hline", r"\end{tabular}"]
    (out / "algebra-invariants.tex").write_text("\n".join(tex) + "\n")
    tex = [r"\begin{tabular}{rrrr}", r"\hline", r"$\nu$ & Composite discrepancy (\%) & Composite nullity & Single-point nullity\\", r"\hline"]
    for r in quadrature_rows:
        tex.append(f"{r['poisson_ratio']:.2f} & {100*r['relative_frobenius_difference_from_exact_gauss']:.6f} & {r['composite_spectrum']['near_zero_eigenvalues']} & {r['literal_one_point_negative_control_spectrum']['near_zero_eigenvalues']}" + r"\\")
    tex += [r"\hline", r"\end{tabular}"]
    (out / "quadrature-checks.tex").write_text("\n".join(tex) + "\n")
    print(json.dumps({"passed": True, "invariant_cells": len(invariant_rows), "quadrature_cases": len(quadrature_rows),
                      "reconstruction_relative_frobenius": [r["reconstruction_relative_frobenius"] for r in invariant_rows],
                      "composite_discrepancy_percent": [100*r["relative_frobenius_difference_from_exact_gauss"] for r in quadrature_rows]}, indent=2))


if __name__ == "__main__":
    main()
