# CUDA Hex8 parity-block kernel, RTX 4090

Part of experiment `exp02-products-rtx4090`. This is the code that ran the first GPU comparison of the
parity-block kernel against the published FP64 kernels, under the frozen [protocol](protocol.json).
The candidate uses eight parity blocks; `dense8` uses the same eight-thread element mapping with a dense local
product. Both keep FP64 arithmetic, full quadrature, material scaling and atomic assembly.

All 126 products, 72 zero-start PCG solves and six intentional one-iteration failures passed their checks.
Against the fastest published baseline (`fused_ai_fp64`), the paired speed ratios were 1.259121 and 1.223604 at
524,288 and 1,769,472 elements; against `dense8`, 1.302955 and 1.261428. Small-mesh solver time was nearly
unchanged. These are eleven randomized timing rounds within one rental; they do not establish between-device
uncertainty or whole-application performance (later experiments address both).

Evidence in this repository: `evidence/kernel-rtx4090/` (protocol, complete result, independent CPU replay,
cost and confirmed instance destruction).

## Files

`prepare.py` freezes independent reference inputs and donor sources. `run_gpu.py` executes all six paths and
keeps raw arrays. `verify.py` replays saved outcomes with independent CPU assembly. `job.py` bounds execution
and collects all outcomes; `monitor.py` verifies the archive locally and destroys only the recorded rental.
The code is kept as it was executed, so it contains the paths of the machine it ran on. The protocol's
`gpu_executed: false` records its state when declared; execution is recorded in the separate result.
