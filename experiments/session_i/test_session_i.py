"""CPU checks for session I before freezing (no CuPy or GPU): input regeneration, and a full CPU dry run of
run_gpu.py (NumPy products) on a tiny upsampled case with the declared arms and stages."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_inputs  # noqa: E402


def test_regeneration_is_exact_at_factor_one():
    for f in ("optimized-q64.npz", "optimized-q96.npz", "smoke-q4.npz"):
        d = dict(np.load(HERE / "inputs" / f))
        mask, force = make_inputs.regenerate(d["shape"])
        assert np.array_equal(mask, d["mask"]) and np.array_equal(force, d["force"]), f


def test_upsampling_preserves_design():
    d = dict(np.load(HERE / "inputs/optimized-q64.npz"))
    up = make_inputs.upsample(d, 2)
    nx, ny, nz = d["shape"]
    r = up["rho"].reshape(2 * nz, 2 * ny, 2 * nx)
    assert np.array_equal(r[1::2, ::2, 1::2], d["rho"].reshape(nz, ny, nx))
    assert abs(up["force"].sum() + 1) < 1e-12 and up["mask"].size == 3 * (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1)


def test_protocol_cases_are_known():
    p = json.loads((HERE / "protocol_template.json").read_text())
    assert all(c["id"] in make_inputs.CASES for c in p["cases"])
    assert p["products"]["variants"] == []
    assert {a["id"] for a in p["arms"]} >= set(p["profile"]["arms"])


def test_cpu_dry_run(tmp_path):
    work = tmp_path / "session_i"
    shutil.copytree(HERE, work, ignore=shutil.ignore_patterns("__pycache__", "package.tar", "protocol.json"))
    p = json.loads((work / "protocol_template.json").read_text())
    p["cases"] = [{"id": "q4x2", "q": 8, "primary": True}]
    (work / "protocol_template.json").write_text(json.dumps(p))
    gen = tmp_path / "gen"
    subprocess.run([sys.executable, work / "make_inputs.py", "--base", work / "inputs", "--out", gen,
                    "--cases", "q4x2"], check=True, capture_output=True)
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, work / "run_gpu.py", "--inputs", gen, "--out", out, "--cpu-dry-run",
                        "--dry-max-coarse-x", "4"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / "result.json").read_text())
    assert res["status"] == "completed" and res["ladder"] == [{"case": "q4x2", "status": "ok"}]
    assert len(res["to_tolerance"]) == len(p["arms"]) * p["to_tolerance"]["repetitions"]
    assert all(t["physical_passed"] for t in res["to_tolerance"]), [t for t in res["to_tolerance"] if not t["physical_passed"]][:1]
    assert all("solution_sha256" in t for t in res["to_tolerance"])
    assert len(res["fixed_work"]) == len(p["fixed_work"]["arms"]) * p["fixed_work"]["rounds"]


def test_package_layout_imports(tmp_path):
    import tarfile
    sys.path.insert(0, str(HERE))
    import build_package
    tar_path = tmp_path / "p.tar"
    with tarfile.open(tar_path, "w") as tar:
        for f in build_package.packaged():
            tar.add(build_package.ROOT / f, arcname=f"source/{f}")
        tar.add(HERE / "protocol_template.json", arcname="source/experiments/session_i/protocol.json")
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_path / "x", filter="data")
    src = tmp_path / "x/source"
    (src / "experiments/combined/donor_yang_gmg/scripts").mkdir()
    code = ("import sys; sys.path.insert(0, 'experiments/session_i'); import run_yang_scale as w;"
            "import kff_adapters as k; assert 'node_fp64' in k.SUITE_ARMS; k.ArmCore('node_fp64', (2, 2, 2), __import__('numpy').eye(24));"
            "m = k._fused_operators(); assert hasattr(m, 'OperatorSuite') and 'node_fp64' in m.PATH_SPEC;"
            "import gpu_fem.presets as pr; [pr.get_preset(s['preset']) for s in w.W2A['sizes']]; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], cwd=src, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-3000:]
