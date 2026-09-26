#!/usr/bin/env bash
# Voxel session I bootstrap: Python stack (CuPy, NumPy; SciPy/PyAMG for Yang et al.'s GMG), then the job.
set -euo pipefail
root=/root/voxel-session_i
here=$root/source/experiments/session_i
venv=/opt/voxel-session_i-venv
export DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUPY_CACHE_DIR=/opt/voxel-session_i-cupy-cache
mkdir "$root/installation"
collect_failure() {
    status=$?
    if [ "$status" -ne 0 ]; then
        printf '%s\n' "$status" > "$root/installation/bootstrap-exit.txt"
        python3 "$here/collect.py" --root "$root" --out /root/voxel-session_i-evidence.tar.gz \
            > /root/voxel-session_i-bootstrap-collection.log 2>&1 || true
        printf '{"returncode": %s, "bootstrap_failed": true}\n' "$status" > /root/voxel-session_i-collection.json
    fi
}
trap collect_failure EXIT
timeout 600 bash -c 'apt-get update && apt-get install -y python3-venv g++' > "$root/installation/apt.log" 2>&1
python3 -m venv "$venv"
timeout 900 "$venv/bin/python" -m pip install --disable-pip-version-check \
    numpy==2.2.6 cupy-cuda12x==13.6.0 scipy==1.15.3 pyamg==5.3.0 > "$root/installation/pip.log" 2>&1
"$venv/bin/python" -m pip freeze > "$root/installation/pip-freeze.txt"
"$venv/bin/python" -c 'import cupy; cupy.show_config()' > "$root/installation/cupy-config.txt" 2>&1
uname -a > "$root/installation/uname.txt"
nvidia-smi -q > "$root/installation/nvidia-smi.txt" 2>&1 || true
free -g > "$root/installation/host-memory.txt" 2>&1 || true
trap - EXIT
exec "$venv/bin/python" "$here/job.py" --root "$root"
