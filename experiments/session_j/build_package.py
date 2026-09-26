"""Freeze Voxel session J: pin every packaged file in protocol.json, then build package.tar.

The package mirrors the Voxel tree under `source/` so the adapters' relative paths hold remotely.
Refuses to overwrite a frozen protocol unless --redeclare is given (record an amendment if so).

    python3 experiments/session_j/build_package.py
"""
import argparse
import ast
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                  # Voxel repository root
FILES = ["experiments/session_j/ncu_target.py",
         "experiments/session_j/job.py", "experiments/session_j/ke_unit_3d.npy",
         "experiments/combined/kff_adapters.py", "experiments/session_e/collect.py",
         "experiments/session_e/dependencies/common.py", "experiments/session_e/dependencies/cuda.py",
         "experiments/session_e/dependencies/modal.py", "experiments/session_e/dependencies/ATTRIBUTION.md",
         "experiments/session_e/dependencies/DAO-LICENSE"]
TREES = ["experiments/combined/donor_yang_gmg", "experiments/session_e/donor"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def yang_ke():
    """Their element matrix, computed by their own function without importing CuPy."""
    src = ROOT / "experiments/combined/donor_yang_gmg/src/gpu_fem/pub_simp_solver.py"
    ns = {"np": np}
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_build_ke_unit_3d":
            exec(compile(ast.Module([node], []), str(src), "exec"), ns)
    return ns["_build_ke_unit_3d"]()


def packaged_files():
    files = list(FILES)
    for tree in TREES:
        files += sorted(str(f.relative_to(ROOT)) for f in (ROOT / tree).rglob("*")
                        if f.is_file() and "__pycache__" not in f.parts and f.name != ".DS_Store")
    return files


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--redeclare", action="store_true")
    args = a.parse_args()
    if (HERE / "protocol.json").exists() and not args.redeclare:
        raise SystemExit("protocol.json already frozen; use --redeclare and record an amendment")
    np.save(HERE / "ke_unit_3d.npy", yang_ke())
    files = packaged_files()
    protocol = json.loads((HERE / "protocol_template.json").read_text())
    protocol["declared_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["package_source_pins"] = {f: sha(ROOT / f) for f in files + ["experiments/session_j/bootstrap.sh"]}
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n")
    psha = sha(HERE / "protocol.json")
    package = HERE / "package.tar"
    with tarfile.open(package, "w") as tar:
        tar.add(HERE / "bootstrap.sh", arcname="source/bootstrap.sh")
        tar.add(HERE / "protocol.json", arcname="source/experiments/session_j/protocol.json")
        for f in files:
            tar.add(ROOT / f, arcname=f"source/{f}")
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": psha,
               "package_sha256": sha(package), "bytes": package.stat().st_size, "files": len(files) + 2}
    (HERE / "package-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
