"""Voxel session G job: run_w1b.py under the job deadline, then collect everything."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC / "experiments/session_e/dependencies"))
from common import now, read, write  # noqa: E402

NAME = "session_h"


def execute(command, log, timeout):
    start = time.monotonic()
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    p = read(HERE / "protocol.json")
    job = {"started_utc": now(), "state": "running", "stages": []}
    write(root / "job.json", job)
    try:
        (SRC / "experiments/combined/donor_yang_gmg/scripts").mkdir(exist_ok=True)   # session F finding
        limit = p["limits"]["maximum_job_seconds"]
        command = [sys.executable, str(HERE / "run_w1b.py"), "--out", str(root / "run-01"),
                   "--deadline-seconds", str(limit - 600)]
        job["stages"].append(execute(command, root / "gpu-run.log", limit - 120))
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
