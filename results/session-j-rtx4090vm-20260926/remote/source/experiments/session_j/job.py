"""Voxel session J job (combined paper W3): Nsight Compute counters for the FP64 product kernels in a VM.

1. Record the driver's profiling restriction (/proc/driver/nvidia/params RmProfilingAdminOnly) and ncu.
2. Query the metrics this ncu supports; keep only declared metrics that exist (all recorded).
3. For each kernel: ncu --kernel-name K --launch-skip 2 --launch-count 1 with the declared sections and
   metrics on ncu_target.py; raw CSV kept. A permission refusal is recorded, not retried.
"""
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

NAME = "session_j"
PERMISSION_MARKERS = ("ERR_NVGPUCTRPERM", "permission to access NVIDIA GPU Performance Counters")


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
            code = child.wait()
    return {"command": [str(c) for c in command], "returncode": code, "timeout": expired,
            "elapsed_seconds": time.monotonic() - start, "log": log.name}


def find_ncu():
    for c in [shutil.which("ncu"), "/usr/local/cuda/bin/ncu", *sorted(glob.glob("/usr/local/cuda-*/bin/ncu")),
              *sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))]:
        if c and Path(c).exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    p = read(HERE / "protocol.json")
    c = p["ncu"]
    out = root / "ncu"
    out.mkdir()
    job = {"started_utc": now(), "state": "running", "rows": []}
    deadline = time.monotonic() + p["limits"]["maximum_job_seconds"]
    try:
        params = Path("/proc/driver/nvidia/params")
        job["driver_params_profiling"] = [l for l in params.read_text().splitlines() if "Profiling" in l] \
            if params.exists() else "not present"
        ncu = find_ncu()
        job["ncu"] = ncu
        write(root / "job.json", job)
        if ncu is None:
            job["counters_available"] = False
            job["reason"] = "ncu not found (installation/nsight-install.log)"
            return
        job["version"] = subprocess.run([ncu, "--version"], capture_output=True, text=True).stdout.strip()
        q = subprocess.run([ncu, "--query-metrics", "--csv"], capture_output=True, text=True, timeout=300)
        (out / "query-metrics.csv").write_text(q.stdout + q.stderr)
        available = {line.split(",")[0].strip('"') for line in q.stdout.splitlines()}
        base = lambda m: m.rsplit(".", 1)[0] if m.rsplit(".", 1)[-1] in ("sum", "avg", "max", "min") or "pct_of" in m else m
        metrics = [m for m in c["metrics"] if base(m).split(".")[0] in available or m in available]
        job["metrics_requested"], job["metrics_used"] = c["metrics"], metrics
        for kernel in c["kernels"]:
            if deadline - time.monotonic() < 120:
                job["rows"].append({"kernel": kernel, "status": "skipped_job_deadline"})
                continue
            command = [ncu, "--target-processes", "all", "--kernel-name", kernel, "--launch-skip",
                       str(c["launch_skip"]), "--launch-count", str(c["launch_count"]), "--csv", "--page", "raw",
                       *sum([["--section", s] for s in c["sections"]], []),
                       *(["--metrics", ",".join(metrics)] if metrics else []),
                       sys.executable, HERE / "ncu_target.py"]
            receipt = execute(command, out / f"ncu-{kernel}.csv", min(900, deadline - time.monotonic() - 60))
            text = (out / f"ncu-{kernel}.csv").read_text(errors="replace")
            perm = any(m in text for m in PERMISSION_MARKERS)
            receipt.update(kernel=kernel, permission_error=perm,
                           profiled=receipt["returncode"] == 0 and not perm and kernel in text)
            job["rows"].append(receipt)
            write(root / "job.json", job)
        rows = [r for r in job["rows"] if "returncode" in r]
        job["counters_available"] = bool(rows) and all(r["profiled"] for r in rows)
    except Exception as exc:
        job["error"] = repr(exc)
    finally:
        job.update(state="finished", finished_utc=now())
        write(root / "job.json", job)
        receipt = execute([sys.executable, SRC / "experiments/session_e/collect.py", "--root", root,
                           "--out", f"/root/voxel-{NAME}-evidence.tar.gz"],
                          root.parent / f"voxel-{NAME}-collection.log", 900)
        write(root.parent / f"voxel-{NAME}-collection.json", receipt)


if __name__ == "__main__":
    main()
