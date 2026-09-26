"""Voxel session F job: Nsight Compute permission/counter probe, then run_probe.py; collect everything."""
import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC / "experiments/session_e/dependencies"))
from common import now, read, write  # noqa: E402

NAME = "session_f"
PERMISSION_MARKERS = ("ERR_NVGPUCTRPERM", "permission to access NVIDIA GPU Performance Counters")


def execute(command, log, timeout, cwd=None):
    start = time.monotonic()
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True, cwd=cwd)
        expired = False
        try:
            code = child.wait(timeout=timeout)
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


def find_ncu():
    for candidate in [shutil.which("ncu"), "/usr/local/cuda/bin/ncu",
                      *sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))]:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def ncu_probe(root, p, deadline):
    out = root / "ncu"
    out.mkdir()
    ncu = find_ncu()
    result = {"ncu": ncu, "rows": []}
    if ncu is None:
        result["counters_available"] = False
        result["reason"] = "ncu not found (see installation/nsight-install.log)"
        return result
    result["version"] = subprocess.run([ncu, "--version"], capture_output=True, text=True).stdout.strip()
    c = p["ncu"]
    for kernel in c["kernels"]:
        remaining = deadline - time.monotonic()
        if remaining < 120:
            result["rows"].append({"kernel": kernel, "status": "skipped_job_deadline"})
            continue
        command = [ncu, "--target-processes", "all", "--kernel-name", kernel, "--launch-skip", str(c["launch_skip"]),
                   "--launch-count", str(c["launch_count"]), "--csv", "--page", "details",
                   *sum([["--section", s] for s in c["sections"]], []), "--metrics", ",".join(c["metrics"]),
                   sys.executable, str(HERE / "ncu_target.py")]
        receipt = execute(command, out / f"ncu-{kernel}.csv",
                          min(p["limits"]["ncu_timeout_seconds"], remaining - 60))
        text = (out / f"ncu-{kernel}.csv").read_text(errors="replace")
        permission_error = any(m in text for m in PERMISSION_MARKERS)
        receipt.update(kernel=kernel, permission_error=permission_error,
                       profiled=receipt["returncode"] == 0 and not permission_error and f'"{kernel}' in text)
        result["rows"].append(receipt)
    rows = [r for r in result["rows"] if "returncode" in r]
    result["counters_available"] = bool(rows) and all(r["returncode"] == 0 and not r["permission_error"]
                                                      for r in rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    p = read(HERE / "protocol.json")
    job = {"started_utc": now(), "state": "running", "stages": []}
    write(root / "job.json", job)
    deadline = time.monotonic() + p["limits"]["maximum_job_seconds"]
    try:
        try:
            job["ncu"] = ncu_probe(root, p, deadline)
        except Exception as exc:
            job["ncu"] = {"error": repr(exc), "counters_available": False}
        write(root / "job.json", job)
        remaining = deadline - time.monotonic()
        command = [sys.executable, str(HERE / "run_probe.py"), "--out", str(root / "run-01"),
                   "--deadline-seconds", str(max(60, remaining - 300))]
        job["stages"].append(execute(command, root / "gpu-run.log", remaining - 120))
    except Exception as exc:
        job["error"] = repr(exc)
    finally:
        job.update(state="finished", finished_utc=now())
        write(root / "job.json", job)
        receipt = execute([sys.executable, str(SRC / "experiments/session_e/collect.py"), "--root", str(root),
                           "--out", f"/root/voxel-{NAME}-evidence.tar.gz"],
                          root.parent / f"voxel-{NAME}-collection.log", 900)
        write(root.parent / f"voxel-{NAME}-collection.json", receipt)


if __name__ == "__main__":
    main()
