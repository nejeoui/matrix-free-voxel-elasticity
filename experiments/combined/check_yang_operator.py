"""W0 CPU check: the modal8/dense8 lane logic reproduces Yang et al.'s GMG FP64 operator core.

Loads `_build_ke_unit_3d` and `_edof_table_3d` from the pinned donor (fd9ddecd) without importing
CuPy, applies their gather + GEMM + bincount core in NumPy, and compares it with
`src/voxel_kernels.emulate` on small and irregular grids.
"""
import ast, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import voxel_kernels as vk  # noqa: E402

DONOR = ROOT / "experiments/combined/donor_yang_gmg/src/gpu_fem/pub_simp_solver.py"


def donor_functions():
    ns = {"np": np}
    for node in ast.parse(DONOR.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_build_ke_unit_3d", "_edof_table_3d"):
            exec(compile(ast.Module([node], []), str(DONOR), "exec"), ns)
    return ns["_build_ke_unit_3d"](), ns["_edof_table_3d"]


def main():
    ke, edof = donor_functions()
    rng = np.random.default_rng(7)
    rows = []
    for shape in [(1, 1, 1), (2, 3, 1), (3, 5, 7), (8, 4, 6)]:
        nx, ny, nz = shape
        ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
        E = rng.uniform(1e-9, 1, nx * ny * nz)
        u = rng.standard_normal(ndof)
        ed = edof(nx, ny, nz)
        y_ref = np.bincount(ed.ravel(), weights=((ke @ u[ed].T) * E).T.ravel(), minlength=ndof)
        for path in ("modal8", "dense8"):
            y = vk.emulate(path, shape, ke, E, u)
            rows.append({"shape": shape, "path": path,
                         "relative_l2": float(np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref))})
    worst = max(r["relative_l2"] for r in rows)
    result = {"donor_commit": "fd9ddecda039fdfb7ffb8cf49c58f602318ed2ef", "rows": rows,
              "worst_relative_l2": worst, "limit": 1e-12, "passed": worst <= 1e-12}
    out = ROOT / "experiments/combined/check_yang_operator.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"worst relative L2 {worst:.2e}; passed={result['passed']}")


if __name__ == "__main__":
    main()
