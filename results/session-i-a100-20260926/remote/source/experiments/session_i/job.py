"""Voxel session I job: C reference, generated inputs, W2b (run_gpu.py), W2a (run_yang_scale.py), collect.

Generated inputs (up to ~2.7 GB for q96x3) are written outside the evidence root; only their manifest
(generated.json, with SHA-256 of every file) is copied into the evidence.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dependencies"))
from common import now, read, sha, write  # noqa: E402

NAME = "session_i"
GENERATED = Path("/root/voxel-session_i-generated")


def execute(command, log, timeout):
    start = time.monotonic()
    with log.open("x") as stream:
        child = subprocess.Popen([str(c) for c in command], stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        expired = False
        try:
            code = child.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            expired = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    return {"command": [str(c) for c in command], "returncode": code, "timeout": expired,
            "elapsed_seconds": time.monotonic() - start, "log": log.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    p = read(HERE / "protocol.json")
    job = {"started_utc": now(), "state": "running", "stages": []}
    write(root / "job.json", job)
    deadline = time.monotonic() + p["limits"]["maximum_job_seconds"]

    def stage(command, log, limit):
        receipt = execute(command, root / log, min(limit, deadline - time.monotonic() - 120))
        job["stages"].append(receipt)
        write(root / "job.json", job)
        return receipt

    try:
        (HERE.parent / "combined/donor_yang_gmg/scripts").mkdir(exist_ok=True)    # session F finding
        build = root / "build"
        build.mkdir()
        r = stage(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", HERE / "inputs/reference.cc", "-o",
                   build / "reference.so"], "reference-build.log", 300)
        assert r["returncode"] == 0, "reference build failed"
        write(build / "build.json", {"utc": now(), "binary_sha256": sha(build / "reference.so"),
                                     "reference_sha256": sha(HERE / "inputs/reference.cc")})
        for name, digest in p["inputs"]["base_files"].items():
            assert sha(HERE / "inputs" / name) == digest, name
        r = stage([sys.executable, HERE / "make_inputs.py", "--base", HERE / "inputs", "--out", GENERATED,
                   "--cases", *[c["id"] for c in p["cases"]]], "make-inputs.log", 1800)
        assert r["returncode"] == 0, "input generation failed"
        shutil.copy2(GENERATED / "generated.json", root / "generated.json")
        for f in ("reference.cc", "petsc_element.inc"):
            shutil.copy2(HERE / "inputs" / f, GENERATED / f)
        stage([sys.executable, HERE / "run_gpu.py", "--inputs", GENERATED, "--reference", build / "reference.so",
               "--out", root / "run-01"], "gpu-run.log", p["limits"]["suite_seconds"] + 1800)
        remaining = deadline - time.monotonic() - 600
        if remaining > 600:
            stage([sys.executable, HERE / "run_yang_scale.py", "--out", root / "w2a-01",
                   "--deadline-seconds", min(p["limits"]["w2a_seconds"], remaining - 300)],
                  "w2a.log", min(p["limits"]["w2a_seconds"] + 600, remaining))
        else:
            job["w2a_skipped"] = "job deadline"
    except Exception as exc:
        job["error"] = repr(exc)
    finally:
        job.update(state="finished", finished_utc=now())
        write(root / "job.json", job)
        receipt = execute([sys.executable, HERE / "collect.py", "--root", root,
                           "--out", f"/root/voxel-{NAME}-evidence.tar.gz"],
                          root.parent / f"voxel-{NAME}-collection.log", 1800)
        write(root.parent / f"voxel-{NAME}-collection.json", receipt)


if __name__ == "__main__":
    main()
