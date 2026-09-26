"""Freeze Voxel session I: pin every packaged file in protocol.json (paths relative to this directory),
then build package.tar mirroring the Voxel tree under source/. Refuses to overwrite a frozen protocol
unless --redeclare is given (record an amendment if so).

    python3 experiments/session_i/build_package.py
"""
import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OWN = ["run_gpu.py", "run_yang_scale.py", "make_inputs.py", "job.py", "collect.py",
       "inputs/optimized-q64.npz", "inputs/optimized-q96.npz", "inputs/reference.cc", "inputs/petsc_element.inc"]
OWN_TREES = ["dependencies", "donor"]
SHARED = ["experiments/combined/kff_adapters.py",
          "experiments/session_e/dependencies/common.py", "experiments/session_e/dependencies/cuda.py",
          "experiments/session_e/dependencies/modal.py", "experiments/session_e/dependencies/ATTRIBUTION.md",
          "experiments/session_e/dependencies/DAO-LICENSE"]
SHARED_TREES = ["experiments/combined/donor_yang_gmg", "experiments/session_e/donor"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files_in(base, tree):
    return sorted(str(f.relative_to(base)) for f in (base / tree).rglob("*")
                  if f.is_file() and "__pycache__" not in f.parts and f.name != ".DS_Store")


def packaged():
    """(path relative to the Voxel root) for every packaged file except protocol.json and bootstrap.sh."""
    own = [f"experiments/session_i/{f}" for f in OWN]
    for t in OWN_TREES:
        own += [f"experiments/session_i/{f}" for f in files_in(HERE, t)]
    shared = list(SHARED)
    for t in SHARED_TREES:
        shared += files_in(ROOT, t)
    return own + shared


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--redeclare", action="store_true")
    args = a.parse_args()
    if (HERE / "protocol.json").exists() and not args.redeclare:
        raise SystemExit("protocol.json already frozen; use --redeclare and record an amendment")
    files = packaged()
    protocol = json.loads((HERE / "protocol_template.json").read_text())
    protocol["declared_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["package_source_pins"] = {os.path.relpath(ROOT / f, HERE): sha(ROOT / f) for f in files}
    protocol["bootstrap_sha256"] = sha(HERE / "bootstrap.sh")
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n")
    psha = sha(HERE / "protocol.json")
    package = HERE / "package.tar"
    with tarfile.open(package, "w") as tar:
        tar.add(HERE / "bootstrap.sh", arcname="source/bootstrap.sh")
        tar.add(HERE / "protocol.json", arcname="source/experiments/session_i/protocol.json")
        for f in files:
            tar.add(ROOT / f, arcname=f"source/{f}")
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": psha,
               "package_sha256": sha(package), "bytes": package.stat().st_size, "files": len(files) + 2}
    (HERE / "package-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
