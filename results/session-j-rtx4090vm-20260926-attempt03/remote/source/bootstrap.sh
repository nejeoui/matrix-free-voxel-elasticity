#!/usr/bin/env bash
# Voxel session J bootstrap (runs as root inside a vast.ai KVM virtual machine): Python stack, Nsight
# Compute (image copy or NVIDIA apt package), then the job.
set -euo pipefail
root=/root/voxel-session_j
here=$root/source/experiments/session_j
venv=/opt/voxel-session_j-venv
export DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 HOME=/root
export CUPY_CACHE_DIR=/opt/voxel-session_j-cupy-cache
mkdir "$root/installation"
collect_failure() {
    status=$?
    if [ "$status" -ne 0 ]; then
        printf '%s\n' "$status" > "$root/installation/bootstrap-exit.txt"
        python3 "$root/source/experiments/session_e/collect.py" --root "$root" \
            --out /root/voxel-session_j-evidence.tar.gz > /root/voxel-session_j-bootstrap-collection.log 2>&1 || true
        printf '{"returncode": %s, "bootstrap_failed": true}\n' "$status" > /root/voxel-session_j-collection.json
    fi
}
trap collect_failure EXIT
nvidia-smi -q > "$root/installation/nvidia-smi.txt" 2>&1 || true
cat /proc/driver/nvidia/params > "$root/installation/driver-params.txt" 2>&1 || true
timeout 900 bash -c 'apt-get update && apt-get install -y python3-venv g++ wget' > "$root/installation/apt.log" 2>&1
# Attempt 02 (amendment): Ubuntu's own nsight-compute (2021.3.1) cannot load the Ada driver, so install
# NVIDIA's package matching the driver's CUDA version from NVIDIA's apt repository.
cuda_mm=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1-\2/p' | head -1)
timeout 1200 bash -c "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && apt-get update && apt-get install -y cuda-nsight-compute-${cuda_mm}" \
    > "$root/installation/nsight-install.log" 2>&1 || true
ls -la /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute 2>&1 > "$root/installation/nsight-location.txt" || true
python3 -m venv "$venv"
timeout 900 "$venv/bin/python" -m pip install --disable-pip-version-check numpy==2.2.6 cupy-cuda12x==13.6.0 \
    > "$root/installation/pip.log" 2>&1
"$venv/bin/python" -m pip freeze > "$root/installation/pip-freeze.txt"
uname -a > "$root/installation/uname.txt"
trap - EXIT
exec "$venv/bin/python" "$here/job.py" --root "$root"
