"""CPU checks for session F before freezing (no CuPy or GPU needed)."""
import dataclasses
import json
import py_compile
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_package  # noqa: E402


def test_sources_compile():
    for f in build_package.FILES + ["experiments/session_f/build_package.py"]:
        if f.endswith(".py"):
            py_compile.compile(str(ROOT / f), doraise=True)


def test_protocol_template_consistent():
    p = json.loads((HERE / "protocol_template.json").read_text())
    sys.path.insert(0, str(ROOT / "experiments/combined"))
    import kff_adapters
    assert tuple(p["arms"]) == kff_adapters.ARMS
    assert set(r["schedule"] for r in p["sizing"]["runs"]) <= set(p["schedules"])
    assert set(r["arm"] for r in p["sizing"]["runs"]) <= set(p["arms"])
    sizes = [np.prod(e.get("nel", [0, 0, 0])) for e in p["sizing"]["ladder"] if "nel" in e]
    assert sizes == sorted(sizes)
    assert p["limits"]["maximum_rental_seconds"] > p["limits"]["maximum_job_seconds"] + 900


def test_their_element_matrix_passes_the_parity_check():
    sys.path.insert(0, str(ROOT / "experiments/session_e/dependencies"))
    from modal import Modal
    Modal(build_package.yang_ke())          # raises if the parity block structure is violated


def test_package_layout_imports(tmp_path):
    """Build the tar into a temporary copy and check both gpu_fem packages resolve independently."""
    files = build_package.packaged_files()
    tar_path = tmp_path / "p.tar"
    np.save(tmp_path / "ke.npy", build_package.yang_ke())
    with tarfile.open(tar_path, "w") as tar:
        for f in files:
            if f != "experiments/session_f/ke_unit_3d.npy":
                tar.add(ROOT / f, arcname=f"source/{f}")
        tar.add(HERE / "protocol_template.json", arcname="source/experiments/session_f/protocol.json")
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_path / "x")
    src = tmp_path / "x/source"
    (src / "experiments/combined/donor_yang_gmg/scripts").mkdir()
    code = ("import sys; sys.path.insert(0, 'experiments/combined'); import kff_adapters as k;"
            "m = k._fused_operators(); assert hasattr(m, 'OperatorSuite');"
            "sys.path.insert(0, 'experiments/combined/donor_yang_gmg/src');"
            "import gpu_fem.pub_simp_solver as y, gpu_fem.presets as pr, gpu_fem.bc_generator as bg;"
            "assert 'donor_yang_gmg' in y.__file__;"
            "import dataclasses; s = dataclasses.replace(pr.get_preset('cantilever_gpu_medium'), nelx=37, nely=23, nelz=17);"
            "b = bg.generate_bc(s); assert b.ndof == 3*38*24*18;"
            "sys.path.insert(0, 'experiments/session_f'); import run_probe; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], cwd=src, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]


def test_ladder_specs_are_valid():
    sys.path.insert(0, str(ROOT / "experiments/combined/donor_yang_gmg/src"))
    scripts = ROOT / "experiments/combined/donor_yang_gmg/scripts"
    made = not scripts.exists()
    scripts.mkdir(exist_ok=True)
    try:
        from gpu_fem.presets import get_preset
        p = json.loads((HERE / "protocol_template.json").read_text())
        for e in p["sizing"]["ladder"] + p["integration"]["grids"]:
            spec = get_preset(e["preset"])
            if "nel" in e:
                spec = dataclasses.replace(spec, nelx=e["nel"][0], nely=e["nel"][1], nelz=e["nel"][2])
            assert spec.nelx * spec.nely * spec.nelz > 0
    finally:
        if made:
            scripts.rmdir()
