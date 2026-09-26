# Session A runbook (RTX 4090) — execute only after the researcher approves and provides SSH access

Frozen protocol SHA-256 `4fd17ba44153f4a8a6d3b78357b3a803e901fe52d5d2917ec8ef5fc347ce868b`;
package `package.tar` SHA-256 `0484d9ee00929a4a1ed26f9b475509c0283a7d7a9b3f3c5ca9337a894fe53184`
(see `package-receipt.json`). Any change to a pinned file requires a dated, recorded amendment
and a re-declaration with `build_package.py --redeclare`.

## Instance

One **NVIDIA GeForce RTX 4090** (24 GB), ≥128 GB host RAM, ≥8 CPU cores, 80 GB disk, image
`nvidia/cuda:12.6.3-devel-ubuntu24.04` (or any Ubuntu with a CUDA 12 driver and root). Declared
price cap: USD 0.60/h. Expected wall time: about 45–60 min of job plus about 10 min of set-up and
transfer. Declared limits: suite 3,600 s, job 4,500 s, rental 7,200 s.

## Steps (replace HOST/PORT)

```sh
cd /Users/nejeoui/Voxel/experiments/session_a
shasum -a 256 package.tar                                   # must match the receipt
scp -P PORT package.tar root@HOST:/root/
ssh -p PORT root@HOST 'nvidia-smi --query-gpu=name,utilization.gpu,power.draw --format=csv && \
  mkdir /root/voxel-session-a && tar -xf /root/package.tar -C /root/voxel-session-a && \
  nohup bash /root/voxel-session-a/source/bootstrap.sh > /root/voxel-session-a-bootstrap.log 2>&1 &'
```

Refuse to start if another process is using the GPU (utilisation above 5 % or unexpected power
draw); an earlier kernel experiment was voided by a co-tenant.

Poll until `/root/voxel-session-a-collection.json` exists:

```sh
ssh -p PORT root@HOST 'cat /root/voxel-session-a/job.json; tail -3 /root/voxel-session-a/gpu-run.log'
```

Collect and verify locally before destroying the instance:

```sh
OUT=/Users/nejeoui/Voxel/results/session-a-$(date -u +%Y%m%d)
mkdir -p $OUT && scp -P PORT root@HOST:/root/voxel-session-a-evidence.tar.gz \
  root@HOST:/root/voxel-session-a-evidence.tar.receipt.json root@HOST:/root/voxel-session-a-collection.json $OUT/
shasum -a 256 $OUT/voxel-session-a-evidence.tar.gz          # must match the receipt
mkdir $OUT/remote && tar -xzf $OUT/voxel-session-a-evidence.tar.gz -C $OUT/remote
clang++ -O3 -std=c++17 -shared -fPIC inputs/reference.cc -o $OUT/reference.dylib
/Users/nejeoui/JPDC/.venv/bin/python verify.py --inputs inputs --reference $OUT/reference.dylib \
  --run $OUT/remote/run-01 --out $OUT/local-verification.json
```

Then ask the researcher to destroy the instance, and record the rental cost after the shutdown is
confirmed. A failed or partial run is kept and reported, never re-run silently.
