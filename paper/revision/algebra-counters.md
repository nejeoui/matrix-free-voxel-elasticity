# Algebra, quadrature and counter methods

This supplement specifies the operator construction and measurement definitions.
Its numerical work consists of CPU algebra and reanalysis of retained counters;
it contains no fresh GPU measurements.

## Explicit parity permutation and coefficient convention

Let the native Hex8 node order be
`(000,100,110,010,001,101,111,011)`, and encode a binary node as
`n = nx + 2 ny + 4 nz`. Let `p=0,...,7` label coordinate modes and
`c=0,1,2` label the x, y and z displacement components. Define

\[
H_{pn}=(-1)^{\operatorname{popcount}(p\mathbin{\&}n)},\qquad
T_{3p+c,3i+d}=H_{p,n_i}\delta_{cd},\qquad TT^\top=8I.
\]

Thus `T` includes the native-to-binary node-column permutation and its output uses
`(p,c)` order, with flat index `3p+c`. The block order is `(g,c)`, with
`g = p xor 2^c` and flat index `3g+c`. The permutation matrix is

\[
P_{3g+c,\;3(g\oplus2^c)+c}=1
\]

with all other entries zero. Write `A = T K0 Tᵀ`. Each block is completely specified by

\[
(C_g)_{cd}=A_{3(g\oplus2^c)+c,\;3(g\oplus2^d)+d},
\quad c,d\in\{0,1,2\}.
\]

For the stated element class,

\[
C=PTK_0T^\top P^\top=\operatorname{diag}(C_0,\ldots,C_7),\qquad
K_0u=\frac1{64}T^\top P^\top CPTu.
\]

The factor `1/64` arises because `T⁻¹=Tᵀ/8` appears on both sides of
the transformed matrix. The implementation stores `C_g/64`, not `C_g`:
`code/hex_modal/modal.py` first forms `full=T@K0@T.T/64`, then selects
`GROUPS[g,c]=3*(g xor (1<<c))+c`. Consequently its two unnormalised
butterflies need no final division. In the eight-lane CUDA implementation, a lane
owns mode `p` for all three components; the x/y/z coefficient blocks have group
indices `p xor 1`, `p xor 2` and `p xor 4`. The required partner for component `d`
in output component `c` is mode `p xor 2^c xor 2^d`. This implements `P` and `Pᵀ`
through indexing/shuffles without materialising a permutation array.

The exactness class is axis-aligned rectangular trilinear Hex8 elements, with an
isotropic elasticity tensor constant within each element. Density multiplies the
element tensor by a scalar modulus; it does not alter this symmetry. Under reflection
of coordinate `a`, the scalar coordinate mode has character `(-1)^p_a`, while
component `c` receives the additional vector sign `(-1)^[a=c]`. The vector-mode
character is therefore `(-1)^(p_a+[a=c])`. The elastic energy bilinear form is
invariant under each reflection. For two modes with different characters, applying
that reflection changes the sign of their coupling while preserving its value,
so their coupling must vanish. Equality of all three characters is precisely equality
of `g=p xor 2^c`. Each of the eight characters has three component-mode pairs.
All 24 modes are retained. This argument does not extend the exactness claim to
distorted or sheared cells, arbitrary rotations with untransformed material/components,
anisotropy or material variation within the cell.

The fresh check uses `E=1`, `nu=0.3`, exact `2×2×2` Gauss integration and FP64.
Reconstruction means `||K0-Kreconstructed||F/||K0||F`; off-block means
`||C-blockdiag(C)||F/||C||F`. Eigenvalues with absolute value at most
`1e-10 ||K0||2` are counted as zero. The six analytic translations/rotations
are separately checked as a null basis, and 24 coordinate vectors plus 32 seeded
normal vectors check the implemented product.

| Cell side lengths | Reconstruction relative Frobenius error | Off-block relative Frobenius norm | Near-zero / positive / negative eigenvalues | Smallest positive eigenvalue |
|---|---:|---:|---:|---:|
| 1×1×1 | 2.3077393e-16 | 1.6064287e-16 | 6 / 18 / 0 | 0.06410256 |
| 1×2×3 | 1.7818459e-16 | 1.3207403e-16 | 6 / 18 / 0 | 0.02548896 |

The reconstruction errors are approximately `2.31e-16` and `1.78e-16`, respectively.
All blocks, eigenvalues, actual thresholds and additional residuals are generated in
[algebra-checks.json](algebra-checks.json); the compact manuscript table is
[algebra-invariants.tex](algebra-invariants.tex).

The local plain, sign-bit and signed-multiply-add butterflies agree bitwise on the
tested CPU FP64 corpus under the tested round-to-nearest arithmetic assumptions,
including signed-zero and subnormal inputs. Multiplication by `±1` is exact in this
scope, which explains the local identity. The CPU emulation is not a universal
compiler/rounding/NaN guarantee. Complete assembled GPU products include floating-point
atomic scatter, whose order may differ, and are described as numerically equivalent
within the declared tolerances. They are not claimed bitwise identical.

The source comparison must identify its scope. In
`experiments/session_e/dependencies/cuda.py`, `modal8_muladd` and `modal8_signbit`
are generated from `modal8` by changing the transform body and names; the layout,
block product, modulus scaling and atomic scatter remain the same source text.
The existing source-equality test checks precisely this. These variants deliberately
change floating-point instruction selection. The `dense8` control in the same file
uses the same eight-lane mapping but a dense local matrix product, and therefore
does not differ from parity only in index arithmetic. The frozen
`code/hex_modal_cuda/cuda.py` is the earlier plain/dense prototype.

## Composite midpoint quadrature in the pinned release

The retained source observation identifies `getKEsubspace` in the pinned OpenMP GPU
release (`b46ed453`) as using ten midpoint samples per fine-subcell axis. At the
fine level this is **composite 10×10×10 midpoint quadrature per fine element**,
namely 1000 samples. The coarse-level construction integrates its fine subcells
with the same per-subcell resolution. It is not single-point reduced integration
of an entire Hex8 element.

For the unit cell at `E=1`, the independent CPU tensor-rule calculation gives:

| Poisson ratio | `||Kcomposite-KGauss||F/||KGauss||F` (%) | Composite near-zero / positive modes |
|---|---:|---:|
| 0.20 | 0.548361476 | 6 / 18 |
| 0.30 | 0.516454157 | 6 / 18 |
| 0.45 | 0.416469489 | 6 / 18 |

Here `KGauss` is the exact `2×2×2` Gauss matrix and the eigenvalue tolerance is
again `1e-10 ||K||2`. These differences agree with the retained source replay
within `1e-12` in relative Frobenius discrepancy. A literal one-point negative
control has eighteen near-zero eigenvalues, but that pathology is not attributed
to the pinned composite-rule release. The finding is a mismatch with this study's
exact-Gauss operator contract. It does not by itself invalidate the audited
mechanics or the conclusions of the original paper. The pinned Futhark constants
(`c3cd2b00`) match exact Gauss to the recorded roundoff tolerance.

Sources: [protocol and pre-execution source observation](../../evidence/companion/results/rescope/hex-modal-priorart-20260923/cpu-protocol-02.json)
(`midpoint_points_per_subcell_axis=10`), [retained CPU source replay](../../evidence/companion/results/rescope/hex-modal-priorart-20260923/cpu-02/results.json)
and [new independent calculation](../../analysis/revision_algebra.py).
The legacy protocol and raw replay remain unchanged. The generated compact table
is [quadrature-checks.tex](quadrature-checks.tex).

## Exact counter quantities and profiling boundary

Session J attempt 07 profiled the third launch after skipping two matching launches
for each of five FP64 kernels on one RTX 4090 VM and a `128×64×64` grid
(`524288` elements). This is one selected launch per kernel; each selected launch
was replayed in twelve profiler passes, not twelve independent observations.
The input vector is seeded normal (`numpy.default_rng(7)`), and element moduli
are `1e-9+(1-1e-9)*rho³` for seeded uniform `rho` in `[0,1)`.

The three counted metrics are:

```
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
```

Their sum counts **executed, predicate-enabled FP64 thread instructions**.
Each DFMA contributes one instruction here; this is not the roofline's FLOP count,
where an FMA contributes two operations. Static SASS instruction sites describe
compiled code positions and are a separate quantity. Launched threads, active
lanes and element count are also distinct.

| Kernel | Executed FP64 thread-instruction total | Per element | FP64 pipe active-cycle peak (%) | DRAM elapsed-cycle peak (%) | Profiled kernel duration (ns) |
|---|---:|---:|---:|---:|---:|
| dense8 | 314572800 | 600 | 93.48 | 5.19 | 594688 |
| plain modal8 | 201326592 | 384 | 83.00 | 7.21 | 429152 |
| mul-add modal8 | 125829120 | 240 | 88.01 | 12.14 | 254304 |
| fused-ai | 314572800 | 600 | 97.47 | 5.42 | 569888 |
| node | 402653184 | 768 | 94.74 | 2.26 | 769568 |

All totals are divided by `524288` elements. Eight threads per element applies
to dense8 and the two modal variants, but fused-ai launches one thread per element
and node owns output nodes (including boundary work). Dividing every total by
`8*524288` would impose a fictitious shared thread layout. The corrected dense8/mul-add
and fused-ai/mul-add count ratios both remain `2.5`.

The utilization and time metrics are exactly:

```
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active
dram__throughput.avg.pct_of_peak_sustained_elapsed
gpu__time_duration.sum
```

The FP64 percentage normalizes to the unit's active cycles; the DRAM percentage
normalizes to elapsed cycles. Their percentages therefore have different
denominators, and the FP64 value is not an elapsed-time occupancy fraction.
The duration is that of the selected profiled kernel, with the raw CSV unit
`nsecond`. It excludes wrapper output zeroing, separate mask/projection kernels,
host overhead, allocations, transfers and compilation. It must not be substituted
for the ordinary complete constrained-product times that include their separate
zeroing and projection operations.

The recorded profiler is NVIDIA Nsight Compute CLI **2023.2.2.0, build 33188574**,
from `/usr/local/cuda-12.2/bin/ncu`. The commands selected `SpeedOfLight`,
`ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis`, `Occupancy` and
`InstructionStats`, together with the requested exact metrics. No clock-control,
cache-control or replay-mode override was supplied. The
[version-specific CLI documentation](https://archive.docs.nvidia.com/nsight-compute/2023.2/NsightComputeCli/index.html)
specifies defaults `clock-control=base`, `cache-control=all` and
`replay-mode=kernel`: base-clock control is attempted, caches are flushed for
replay, and kernel memory is saved/restored as required. This is a statement of
the recorded command and documented defaults. The retained run does not
independently establish successful clock locking, thermal/power stability or a
profiler configuration/environment override audit. We do not invent that metadata.

Source: [job/version/command receipt](../../results/session-j-rtx4090vm-20260926-attempt07/remote/job.json),
[target](../../experiments/session_j/ncu_target.py), the five raw CSVs in
`results/session-j-rtx4090vm-20260926-attempt07/remote/ncu/`, and
[regenerator](../../analysis/session_j_counters.py).
`paper/session_j_numbers.json` schema 2 identifies the exact metrics and units,
keeps per-metric totals, and records actual launch-thread and replay counts.

## Post-hoc accounting and bounded interpretation

For kernel `k`, define `N_k` as the count above and `U_k` as the active-cycle
FP64 pipe fraction. The descriptive ratio is
`(N_k/U_k)/(N_muladd/U_muladd)`. Its maximum relative discrepancy from the
same-launch profiled-duration ratio is `1.768%` (approximately `1.8%`).
Because the utilization denominator already contains active execution-cycle
information from that launch, dividing counts by utilization partly recovers
cycle accounting. This is a post-hoc consistency result, not an independent
prediction of runtime, a held-out validation or an explanation of every timing
component.

H15's utilization subcondition passes. Its static-count-ratio subcondition fails:
the dense8/mul-add executed ratio `2.5` differs from `78/34` by `8.974%`, beyond
the declared 5% threshold. This failed declaration remains visible. The smaller
plain/mul-add discrepancy (`4.615%`) does not rescue the joint hypothesis.

High FP64 pipe activity, low DRAM throughput and fewer executed FP64 instructions
support FP64 throughput as a dominant constraint for these profiled workloads.
They do not independently exclude cache behavior, latency or atomic costs, nor
establish a bottleneck on the unprofiled A100, H100 or RTX 3090. Recommendations
outside this RTX 4090 workload must instead be bounded by their own measured
product and solver endpoints.

## Hardware basis and idealized traffic of the roofline

The original declared roofline used the A100 80 GB **SXM** specification:
9.7 non-tensor FP64 TFLOP/s and 2039 GB/s. Its original FP64 ridge
`4.7572 FLOP/B` is retained and labeled. For the measured A100 80 GB **PCIe**,
the SKU-specific calculation uses 9.7 FP64 TFLOP/s, 19.5 FP32 TFLOP/s and 1935 GB/s
from [NVIDIA's A100 specification table](https://www.nvidia.com/en-us/data-center/a100/).
The corresponding ridges are `5.0129 FLOP/B` (FP64) and `10.0775 FLOP/B` (FP32).
The ideal dense/parity ratio remains `1.0` in both precisions. The added row is
a revised SKU calculation; it is not relabeled as a prospective prediction.
Both hardware bases and their status are emitted by [roofline.py](../../analysis/roofline.py)
to [roofline.json](../roofline.json) and [roofline.md](../roofline.md).

The 48 values per element count one gather of 24 displacement values and one
scatter of 24 results. This is idealized traffic accounting, not measured DRAM
bytes or a universal traffic lower bound. Shared-node reuse and cache hits can
reduce DRAM loads; atomic read/modify/write transactions, modulus reads and
constraint masks add traffic. The model excludes these effects and cannot
establish their measured share of runtime.

## Regeneration and verification record

Executed on the local CPU on 2026-09-27:

```
.venv/bin/python analysis/revision_algebra.py
.venv/bin/python analysis/session_j_counters.py
.venv/bin/python analysis/roofline.py
.venv/bin/python -m pytest tests/test_kernels_cpu.py experiments/session_e/test_session_e.py -q
```

The algebra generator passed both geometry checks and all three quadrature cases.
The existing tests passed **16/16**. The butterfly hard corpus emits six expected
overflow/invalid-operation warnings while testing extreme inputs; all comparisons
pass. The tests include FP64/FP32 lane emulation against assembled products,
shuffle negative control, source equality for the butterfly variants, and the local
bitwise corpus. This does not replace GPU execution or GPU accuracy replay.
Counter totals were read directly from all five retained CSVs, then normalized by
the protocol element count; H15's failed count condition remains failed.
The parser additionally checks one data row per selected kernel and the raw units
`inst`, `%` and `nsecond`. Direct checks of the generated JSON against the five
raw totals passed; both dense8/mul-add and fused-ai/mul-add ratios equal `2.5`.
Both original and reconstructed stiffness matrices have six near-zero and eighteen
positive eigenvalues under the stated tolerance. The revised PCIe and original SXM
roofline records retain their distinct bandwidths and both give ratio `1.0`.
