"""CPU checks for session G before freezing (no CuPy or GPU needed)."""
import ast
import json
import py_compile
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_package  # noqa: E402

E3 = ROOT / "experiments/combined/donor_yang_gmg/experiments/paper4/run_experiments_e1_e10.py"


def _body(func):
    return [ast.dump(n) for n in func.body]


def test_sources_compile():
    for f in build_package.FILES + ["experiments/session_g/build_package.py"]:
        if f.endswith(".py"):
            py_compile.compile(str(ROOT / f), doraise=True)


def test_oc_update_is_theirs_verbatim():
    e3 = next(n for n in ast.parse(E3.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "e3_simp_speedup")
    theirs = next(n for n in ast.walk(e3) if isinstance(n, ast.FunctionDef) and n.name == "_oc_update")
    ours = next(n for n in ast.parse((HERE / "run_w1.py").read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "oc_update")
    assert _body(theirs) == _body(ours)[1:]          # ours adds only a docstring
    assert [a.arg for a in theirs.args.args] == [a.arg for a in ours.args.args]


def test_mixed_schedule_is_their_e3_configuration():
    src = E3.read_text()
    block = src[src.index("# SolverV4 with FP32-GMG"):src.index("rho = np.full(n_elem, spec.volfrac)", src.index("# SolverV4 with FP32-GMG"))]
    for fragment in ("enable_matrix_free=True", "enable_fused_cuda=True", "enable_matfree_gmg=True",
                     "matfree_gmg_levels=4", 'gmg_fine_smoother="fp32"', 'gmg_smoother_type="chebyshev"'):
        assert fragment in block, fragment
    p = json.loads((HERE / "protocol_template.json").read_text())
    m = p["schedules"]["mixed"]
    assert (m["levels"], m["fine_smoother"], m["smoother_type"], m["enable_fused_cuda"]) == (4, "fp32", "chebyshev", True)
    assert "move=0.15" in src and "penal=3.0" in src and "n_simp_iters: int = 30" in src
    assert p["simp"]["steps"] == 30 and p["simp"]["move"] == 0.15 and p["simp"]["penal"] == 3.0


def test_protocol_consistent():
    p = json.loads((HERE / "protocol_template.json").read_text())
    sys.path.insert(0, str(ROOT / "experiments/combined"))
    import kff_adapters
    assert tuple(p["arms"]) == kff_adapters.ARMS
    assert p["complete_optimization"]["case"] in {c["id"] for c in p["cases"]}
    assert p["limits"]["maximum_rental_seconds"] >= p["limits"]["maximum_job_seconds"] + 1200


def test_package_layout_imports(tmp_path):
    files = build_package.packaged_files()
    tar_path = tmp_path / "p.tar"
    with tarfile.open(tar_path, "w") as tar:
        for f in files:
            tar.add(ROOT / f, arcname=f"source/{f}")
        tar.add(HERE / "protocol_template.json", arcname="source/experiments/session_g/protocol.json")
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_path / "x", filter="data")
    src = tmp_path / "x/source"
    (src / "experiments/combined/donor_yang_gmg/scripts").mkdir()
    code = textwrap.dedent("""
        import sys; sys.path.insert(0, 'experiments/session_g')
        import run_w1, numpy as np
        for c in run_w1.PROTOCOL['cases']:
            case = run_w1.Case(c['preset'])
            assert case.n_elem > 0 and case.common['ndof'] == 3 * np.prod(np.array(case.dims) + 1)
        rho = np.full(10, 0.5); dc = -np.linspace(1, 2, 10)
        theirs = run_w1.oc_update(rho, dc, 0.5, 0.15)
        assert abs(theirs.mean() - 0.35) < 1e-12        # their E3 OC: multiplier stuck ~8e9 -> every x moves to x - move
        ours = run_w1.oc_top88(rho, dc, np.ones(10), 0.5, 0.15)
        assert abs(ours.mean() - 0.5) < 1e-3            # top88 OC meets the volume constraint
        f = run_w1.DensityFilter((4, 3, 2), 1.5)
        assert np.allclose(f.density(np.ones(24)), 1.0) and f.H.nnz > 24
        g = np.random.default_rng(1).standard_normal(24); v = np.random.default_rng(2).standard_normal(24)
        assert abs(v @ f.density(g) - g @ f.gradient(v)) < 1e-12   # gradient is the adjoint of density
        print('ok')
    """)
    out = subprocess.run([sys.executable, "-c", code], cwd=src, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-3000:]
