#!/usr/bin/env bash
# Voxel session F bootstrap: Python stack pinned to Yang et al.'s environment.yml where it matters,
# Nsight Compute from the CUDA image or NVIDIA's apt repository, then the job.
set -euo pipefail
root=/root/voxel-session_f
venv=/opt/voxel-session_f-venv
export DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUPY_CACHE_DIR=/opt/voxel-session_f-cupy-cache
mkdir "$root/installation"
collect_failure() {
    status=$?
    if [ "$status" -ne 0 ]; then
        printf '%s\n' "$status" > "$root/installation/bootstrap-exit.txt"
        python3 "$root/source/experiments/session_e/collect.py" --root "$root" \
            --out /root/voxel-session_f-evidence.tar.gz > /root/voxel-session_f-bootstrap-collection.log 2>&1 || true
        printf '{"returncode": %s, "bootstrap_failed": true}\n' "$status" > /root/voxel-session_f-collection.json
    fi
}
trap collect_failure EXIT
timeout 600 bash -c 'apt-get update && apt-get install -y python3-venv g++' > "$root/installation/apt.log" 2>&1
# Nsight Compute: use the image's copy if present, otherwise try NVIDIA's apt package (not fatal).
if ! command -v ncu >/dev/null && [ ! -x /usr/local/cuda/bin/ncu ] && ! ls /opt/nvidia/nsight-compute/*/ncu >/dev/null 2>&1; then
    timeout 900 apt-get install -y cuda-nsight-compute-12-6 > "$root/installation/nsight-install.log" 2>&1 || true
fi
ls -la /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute 2>&1 > "$root/installation/nsight-location.txt" || true
python3 -m venv "$venv"
# Yang et al. pin Python 3.10.18; the image has Ubuntu 24.04's Python 3.12. Package pins are theirs.
timeout 900 "$venv/bin/python" -m pip install --disable-pip-version-check \
    numpy==2.2.6 cupy-cuda12x==13.6.0 scipy==1.15.3 pyamg==5.3.0 > "$root/installation/pip.log" 2>&1
"$venv/bin/python" -m pip freeze > "$root/installation/pip-freeze.txt"
"$venv/bin/python" -c 'import cupy; cupy.show_config()' > "$root/installation/cupy-config.txt" 2>&1
python3 --version > "$root/installation/python-version.txt" 2>&1
uname -a > "$root/installation/uname.txt"
nvidia-smi -q > "$root/installation/nvidia-smi.txt" 2>&1 || true
trap - EXIT
exec "$venv/bin/python" "$root/source/experiments/session_f/job.py" --root "$root"
