#!/usr/bin/env bash
set -euo pipefail
root=/root/voxel-session_e
venv=/opt/voxel-session_e-venv
export DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUPY_CACHE_DIR=/opt/voxel-session_e-cupy-cache
mkdir "$root/installation"
collect_failure() {
    status=$?
    if [ "$status" -ne 0 ]; then
        printf '%s\n' "$status" > "$root/installation/bootstrap-exit.txt"
        python3 "$root/source/collect.py" --root "$root" --out /root/voxel-session_e-evidence.tar.gz > /root/voxel-session_e-bootstrap-collection.log 2>&1 || true
    fi
}
trap collect_failure EXIT
timeout 600 bash -c 'apt-get update && apt-get install -y python3-venv g++' > "$root/installation/apt.log" 2>&1
python3 -m venv "$venv"
timeout 600 "$venv/bin/python" -m pip install --disable-pip-version-check numpy==2.2.6 cupy-cuda12x==13.6.0 > "$root/installation/pip.log" 2>&1
"$venv/bin/python" -m pip freeze > "$root/installation/pip-freeze.txt"
"$venv/bin/python" -c 'import cupy; cupy.show_config()' > "$root/installation/cupy-config.txt" 2>&1
uname -a > "$root/installation/uname.txt"
trap - EXIT
exec "$venv/bin/python" "$root/source/job.py" --root "$root"
