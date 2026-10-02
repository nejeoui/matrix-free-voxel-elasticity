---
title: 'Supporting Information: matrix-free voxel elasticity'
author: 'Abderrazzak Nejeoui and Aissam Bekkari'
date: '27 September 2026'
fontsize: 10pt
geometry: [a4paper, landscape, margin=18mm]
mainfont: "texgyreheros-regular.otf"
mainfontoptions:
  - BoldFont=texgyreheros-bold.otf
  - ItalicFont=texgyreheros-italic.otf
  - BoldItalicFont=texgyreheros-bolditalic.otf
monofont: "lmmono10-regular.otf"
colorlinks: true
toc: true
---

This supporting document accompanies the matrix-free voxel elasticity study. It contains
operator construction, methods, accuracy, outcome, capacity, comparator, audit
and hypothesis evidence. Numerical regeneration and selected CPU checks use
retained records; no new GPU performance experiment is represented. Source
paths are relative to the accompanying artifact. CSV/JSON files retain exact
per-observation values and references where PDF tables use rounded summaries.



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
and [new independent calculation](../../analysis/supplement_algebra.py).
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
an added SKU calculation; it is not relabeled as a prospective prediction.
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
.venv/bin/python analysis/supplement_algebra.py
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
positive eigenvalues under the stated tolerance. The added PCIe and original SXM
roofline records retain their distinct bandwidths and both give ratio `1.0`.


# Physical problems and solver accuracy

This supplement specifies the implemented problems, including differences between
experiment families. Paths refer to the accompanying artifact.

## Constitutive and discretization contract

Small-strain isotropic elasticity uses
`sigma = 2 mu epsilon + lambda tr(epsilon) I`, with
`mu = E/[2(1+nu)]` and `lambda = E nu/[(1+nu)(1-2nu)]`.
The reference element uses `E=1`, `nu=0.3`, trilinear Hex8 interpolation and
exact tensor 2×2×2 Gauss integration. Each element has a spatially constant
scalar modulus `E_e = Emin + (1-Emin) rho_e^3`; this scalar multiplies its
reference stiffness. The parity exactness claim does not cover distorted cells,
anisotropic constitutive laws, or a tensor varying within an element.

| Family | Element geometry/material | Densities and boundary/loading contract |
|---|---|---|
| Sessions A–E, base cantilevers | Cubes of side `1/q`, grid `2q × q × q`; `Emin=1e-6`, except explicitly labeled Session C contrast cases `1e-9` | Clamp all displacement components on `x=0`. On the index line `x=nx,z=0` (physical x=2,z=0), apply negative z load with interior nodal weights `1/ny` and endpoint weights `1/(2ny)`, total force −1. Retained optimized fields at q=64/96; uniform control rho=0.12; smoke control q=4 and rho=0.12. |
| Session I rediscretized scale | `make_inputs.upsample` repeats each density and preserves the base `ke` byte for byte; the effective fine-element stiffness scale remains that of the q=64 or q=96 base, with coarse matrices scaled by `2^level`; `Emin=1e-6` | Same clamp and total unit line-load rule on the enlarged grid. These are repeated coefficient fields, not new optimized designs. In particular, no extra `1/factor` stiffness scaling is applied during upsampling. |
| Sessions F, G, H and I Galerkin branch | Donor `KE_UNIT_3D`, the unit-cube matrix (not rescaled using the preset's domain coordinates); `Emin=1e-9`, penalty 3, nu=0.3 | Preset coordinates select supports/load nodes; force vectors and dimensions follow the pinned donor presets below. F integration: rho=0.5 or uniform random rho in [0.3,1], NumPy seed 20260926. F sizing and I Galerkin: rho=0.5. G/H start from the preset volume fraction and use a density filter of radius 1.5 element spacings, move 0.15, top88 update without continuation/projection. |
| Session J profiling | Unit-cube `ke_unit_3d.npy`; `Emin=1e-9`, penalty 3 | NumPy `default_rng(7)`: standard-normal displacement vector followed by uniform [0,1] densities. This is a product workload; no physical solve or load case is inferred from it. |

The base force/mask are reconstructed by `experiments/session_i/make_inputs.py:regenerate`;
factor-one equality is checked in `test_session_i.py`. The uniform density is
read directly from `experiments/session_e/inputs/uniform-q64.npz`.
Material and stopping parameters are in each frozen protocol.

| G/H case (preset) | Grid | Preset domain | Initial/target volume | Supports and force |
|---|---|---|---:|---|
| cantilever-216k (`cantilever_gpu_large`) | 120×60×30 | 2×1×0.5 | 0.30 | Clamp left face; `(0,-1,0)` point force nearest `(2,0.5,0.25)` |
| cantilever-512k (`cantilever_gpu_xlarge`) | 160×80×40 | 2×1×0.5 | 0.30 | Same normalized location and unit force |
| mbb-514k (`mbb_gpu_xlarge`) | 210×70×35 | 3×1×0.5 | 0.50 | Left-face `pin_x` fixes only ux; point `pin_y` fixes only uy nearest `(3,0,0.25)`; `(0,-1,0)` force nearest `(0,1,0.25)`. No uz constraint is added by this preset. |
| torsion-499k (`torsion_gpu_500k`) | 165×55×55 | 3×1×1 | 0.25 | Clamp left face; forces `(0,-0.5,+0.5)` at `(3,1,0.5)` and `(0,+0.5,-0.5)` at `(3,0,0.5)` |

These are the exact pinned `presets.py` and `bc_generator.py` conventions,
including nearest-node snapping via `argmin` and removal of loads on constrained
DOFs. Domain coordinates specify that snapping; the unit-element stiffness in
the donor runner is a separate convention and is not silently replaced by
physical-domain-scaled stiffness.

F integration uses the same cantilever domain/support/load convention, with the
80×40×20 `cantilever_gpu_medium` grid and an overridden 37×23×17 odd grid.
F sizing uses 252×126×63 (2M) and 342×171×86 (5M) first, with later declared
sizes 432×216×108, 544×272×136 and 640×320×160 subject to the stop-on-failure
ladder. I's Galerkin branch uses 200×100×50 and 252×126×63. All of these retain
the same unit-cube stiffness convention and point force; they do not use the
rediscretized family's line load. These are actual implementation conventions,
not a claim that the dimensioned preset boxes set each stiffness length scale.

## Residual and energy definitions

Let `M` be the diagonal free-DOF mask and `A=M K M+(I-M)`. The full-vector
residual is `||A u-f||_2/||f||_2`. Saved states must be finite and exactly zero
at constrained DOFs; `f` is also zero there, so this equals the free-space norm.
For the rediscretized CPU/C-reference gate, energy–work discrepancy is
`|sum_e E_e u_e^T K0 u_e/(f^T u)-1|`; the numerator
is twice the strain energy, with no extra 1/2. These checks compare the same
physical units. `analysis/supplement_accuracy.py` regenerates maxima, counts and
source paths in `accuracy.json` and `accuracy.md`. Their E/I maxima cover all
retained in-run independent checks, not just saved-state replay: 440 states for
Session E across two hosts, 36 for RTX scale and 96 for A100/H100 scale.
Separate retained local Session E replay covers the first repetition for each
case/arm (44 states per host, 88 total); its largest residual is
4.35296445e-9. Session I retains in-run CPU/C-reference checks and solution hashes,
without a separate saved-state replay being claimed here.

| Experiment/endpoint | Requested stop | Independent acceptance | Coverage/exception |
|---|---|---|---|
| A–C solves to tolerance; D/E rediscretized multigrid | true relative residual 1e-8 | CPU reference residual <=1e-7; energy–work <=1e-6; finite state and exact fixed zeros | Accuracy gates are separate from timing; fixed-work runs intentionally do not stop by tolerance. |
| I rediscretized multigrid | true relative residual 1e-8 | C-reference residual <=1e-7; energy–work <=1e-6; exact fixed zeros | Checks executed during retained GPU runs; reported maxima are from those records. |
| F Galerkin integration | 1e-10, max 300 iterations | stock-operator true residual <=1e-10 and solution relative L2 agreement <=1e-8 | Product agreement <=1e-12. Mixed random-density cases on A100/H100 hit cap; integration gate failed. |
| F Galerkin sizing | 1e-8, max 40 iterations | native convergence flag/iteration count only in sizing records; no separate stock residual replay | A successful allocation/sizing run is not proof that the capped solve met tolerance. |
| G/H Galerkin optimization | donor default CG tolerance 1e-5, max 1000, warm starts; automatic method resolves to PCG for these FP64/FP32 smoother schedules | Recorded linear-solve gate uses the iteration-cap proxy (`last_cg_iters < cg_maxiter`); final compliance relative difference <=1e-3 against stock on repetition 1, final physical volume within 1e-3 of target on all fixed trajectories | Donor PCG tests recursive residuals (plus its initial true residual), rather than recomputing true residual every iteration. G/H outputs do not carry the E/I independent C-reference residual/energy check. Capped designs are not converged optimizations. |
| I Galerkin branch | 1e-8, max 1000 for preliminary solves | retained donor convergence records; fixed-work endpoint uses 50 iterations | Do not infer the rediscretized branch's independent accuracy gate for this branch. |
| J product counters | no iterative stop | no solver residual endpoint | One profiled kernel launch, not a solve. |

A–C test the Jacobi/iterative-refinement outer solution's true residual at their
declared checks (every 50 PCG iterations); they are not instances of the D/E/I
flexible-CG recurrence below. The donor G/H solver also retries failed warm-start
solves from zero. Its cap proxy and compliance/volume comparisons are reported
as those checks, without converting them into a broader independently verified
residual claim.

## Flexible CG and product accounting

The rediscretized hierarchy halves all three extents while `nx>16` and every
extent is even. It stops early if another halving would be nonintegral: the
largest q96x3 hierarchy ends at 18×9×9, for example. Noncoarsest operators are
matrix-free, with averaged child moduli and `K0_coarse=2^level K0_fine`;
the coarsest constrained matrix is assembled and factored by dense Cholesky.

`src/voxel_mg.py:fcg` implements FP64 Polak–Ribière flexible CG, including when
the preconditioner computes its V-cycle in FP32. Start with `x=0, r=b`,
`z=B(r), p=z, delta=r^T z`. Each iteration computes:

```
q = A p
alpha = delta / (p^T q)
x = x + alpha p
r_new = r - alpha q
if a requested true-residual check is due:
    r_new = b - A x
    stop if ||r_new|| / ||b|| <= tolerance
z_new = B(r_new)
beta = r_new^T (z_new-z) / delta
p = z_new + beta p
delta = r_new^T z_new
r, z = r_new, z_new
```

The checked residual replaces the recursive residual. The routine defaults to
`check_every=5`, but D/E/I frozen protocols pass `check_every=1`: a successful
to-tolerance iteration therefore costs two outer products (search direction and
true residual). Fixed-work mode skips intermediate replacement and uses K
direction products plus one final true-residual product, K+1 total. A capped
to-tolerance exit executes the final check as well: with `check_every=1`, a cap
of K iterations costs `2K+1` outer products if no early return succeeds. This recurrence describes
the rediscretized solver; G/H retain the separate donor solver and its tolerance.

## Damping estimator and limits

`spectral_omegas` estimates the spectral radius of constrained/free-space
`D^-1 A` on each noncoarsest level. It initializes a NumPy standard-normal vector,
masks fixed DOFs and normalizes it. For 30 iterations it computes
`w=D^-1 A v`, masks fixed DOFs, sets `lambda=||w||_2` and `v=w/lambda`.
Then `omega=4/(3 lambda)`. The random generator is restarted on each level with
the same seed, not advanced continuously across levels. Default seed is
20260923; runners pass their protocol `order_seed` (D/E: 2026092320;
I: 2026092621). The reference FP64 hierarchy uses node-owned products; every arm
shares its estimated weights and the same host-computed coarsest factor, with
that factor cast to the arm's preconditioner precision on construction.

Thirty power steps are an estimate, not a certified upper eigenvalue bound.
Empirical evidence consists of CPU smoother/transfer tests and the retained
successful residual/energy checks for the tested coefficient fields. A fixed
0.6 damping failed heterogeneous CPU cases. Neither the estimator nor those
tests establish universal smoother stability or mixed-precision convergence.
The exact rational factorization certificates in the separate pinned-release
audit concern different ideal operators and must not be attributed to this
power estimate.

Verification on 2026-09-27: `.venv/bin/python -m pytest tests/test_mg_cpu.py -q`
passed all 10 existing CPU checks, including transpose transfers, uniform-modulus
Galerkin equality, tested V-cycle symmetry, FP32-preconditioned convergence and
fixed-work product accounting. The five existing Session I checks also passed
(`.venv/bin/python -m pytest experiments/session_i/test_session_i.py -q`),
including input regeneration and the CPU dry-run scale path. Input inspection independently confirmed base
`ke` against exact Gauss cells of side `1/q` to about 3.1e-16 relative Frobenius
error, and exactly reproduced every base mask and force using `regenerate`.
These are CPU checks and source/record inspection, not new GPU validation.


# Retained accuracy results

Generated by `python analysis/supplement_accuracy.py`. This inspects retained independent-check records; it is not a new GPU experiment or fresh state replay.

| Experiment | In-run checked states | Requested | Independent acceptance | Observed residual maximum | Energy–work maximum (limit 1e-6) |
|---|---:|---:|---:|---:|---:|
| Session E, two RTX 4090 hosts | 440 | 1e-08 | 1e-07 | 4.35301436e-09 | 7.94719845e-12 |
| Session I, RTX 4090 | 36 | 1e-08 | 1e-07 | 4.43192315e-09 | 3.06175085e-11 |
| Session I, A100 and H100 | 96 | 1e-08 | 1e-07 | 7.44563488e-09 | 6.84184931e-11 |

Source records:

- `results/session-e-host1-20260925/remote/run-01/result.json`
- `results/session-e-host2-20260925/remote/run-01/result.json`
- `results/session-i-rtx4090-20260926/remote/run-01/result.json`
- `results/session-i-a100-20260926/remote/run-01/result.json`
- `results/session-i-h100-20260926/remote/run-01/result.json`

Separate retained local CPU replay of saved Session E states (first repetition only):

- `results/session-e-host1-20260925/local-verification.json`: 44 states, maximum residual 4.35290392e-09; all passed.
- `results/session-e-host2-20260925/local-verification.json`: 44 states, maximum residual 4.35296445e-09; all passed.

Session I records CPU/C-reference checks during each GPU run and solution hashes; this summary does not claim a separate saved-state replay for that family.


# Timing, optimization outcomes and measured scale

This supplement documents optimization outcomes, solver performance and measured capacity. Its [generated tables](outcomes-scale-tables.md), [exact JSON](outcomes-scale.json) and CSV files are regenerated from retained evidence by `python3 analysis/supplement_outcomes.py`. This is numerical regeneration on the CPU, not fresh GPU execution. Every optimization, scale-solve and Galerkin record includes its raw file and JSON pointer; input hashes are in the JSON. Historical fields named `complete` or `complete_optimization` are retained as evidence identifiers and do not establish design convergence.

## Experiment manifest and timing boundaries

The [manifest CSV](experiment-manifest.csv) and generated hardware table list GPU SKU/capacity, CPU, driver, CUDA runtime, CuPy, compiler, host/allocation count, cases, arms, repetitions, warmup and endpoints for 20 measured allocations across A–J, synthetic replication and libCEED. An unavailable entry is NR/null. NVRTC is the CUDA just-in-time compiler; a Python build's GCC version does not identify the C++ reference compiler. The native libCEED harness retained nvcc/GCC version commands; reference-compiler versions were not separately retained in the CuPy session records. CUDA driver capability, locally installed runtime and CuPy-linked runtime are different fields. Where the installation record distinguishes them, its full value is preserved. Clock telemetry is a snapshot; no common fixed-clock policy is claimed. Manifest rows identify the metadata files examined. The only missing CPU model is Session A: its manual rental/preflight records 32 CPUs and 188 GB RAM, but no CPU model; the retained installation and result records do not supply one.

The synthetic FP64 product study directly measured one RTX 4090 and an independent RTX 3090, with plain parity, dense8 and the three donor products. It used five warmup calls, eleven paired rounds and twenty products per timed batch. Session A used one RTX 4090 for physical-state products/Jacobi-PCG/refinement, B one **A100 80GB PCIe**, and C another RTX 4090 for refinement and modulus-floor sensitivity. A/B had three tolerance repetitions; C had two. Their product measurements used eleven rounds of twenty products, and fixed-work solves ten paired rounds. Session D used one RTX 4090 for rediscretized multigrid, five tolerance repetitions and ten fixed-work rounds. Session E used two RTX 4090 hosts, eleven product rounds of twenty products, five tolerance repetitions and ten fixed-work rounds. Full cases and kernel variants are preserved in the manifest's structured columns and the named protocols.

Session F's feasibility cards were RTX 4090, A100 80GB PCIe and **H100 NVL (95830 MiB reported physical memory)**. The Session I scale H100 was **H100 80GB HBM3 (81559 MiB)**. They are different hardware. No measured A100 PCIe result is relabeled as an A100 SXM measurement; the historical SXM specification used in the roofline calculation is a separate model input.

Session G ran three paired 30-step trajectories per case/schedule on each of two RTX 4090 hosts. H ran three paired **100-step** trajectories on each of two other RTX 4090 hosts. Each warmed one step per case/schedule/arm. Each also retained one additional trajectory per arm/schedule for the 216k cantilever, stopping on design change or the 150-step cap. There are three arms (stock gather/DGEMM/scatter, fused-ai FP64, parity mul-add) and two four-level Galerkin schedules (FP64 and mixed). Session I used one allocation per GPU type: RTX 4090, A100 80GB PCIe and H100 80GB HBM3. It used five rounds of ten products, two tolerance solves per arm and three paired fixed-work rounds, plus one instrumented profile per selected arm/case. Its W2a Galerkin check used two blocks with three fixed 50-iteration solves per block/arm/schedule and one separate tolerance attempt in block zero.

Session J used one RTX 4090 VM for the successful counter collection, five FP64 kernels at 128×64×64 elements, and one selected launch per kernel after skipping two matching launches. Nsight Compute 2023.2.2.0 replayed the selected launch in twelve metric passes; these are not twelve independent timing samples. Profiler kernel duration excludes wrapper zeroing/projection and host overhead. The installation logs identify cudart/NVRTC 12.2.140; no separate loaded-runtime version query is retained. [Counter methods](algebra-counters.md) retain clock/cache/replay policy and distinguish unavailable achieved-clock metadata. The libCEED native CUDA/C harness used one RTX 4090 allocation, FP64, six timed shape/pattern combinations from 64×32×32 to 192×96×96, five seconds of warmup and twelve randomized blocks of 256 actions. Each action zeroes output, applies the operator and identity rows, then synchronizes; host steady-clock time is primary. Setup, allocations, conversion, transfers and compilation are excluded. Byte-verified metadata extracted from the deposited files archive records the Ryzen 5 5600X CPU, driver 595.58.03, 24564 MiB GPU memory, nvcc 12.6.85, GCC 11.4.0 and CUDA 12.6 SDK probe. Extraction member names and hashes are in `provenance/supplement/manifest-metadata/extractions.json`; the two original synthetic product hosts' rental/CuPy metadata was recovered by the same method. See [comparator provenance](provenance-audit.md).

| Endpoint | Included in its timer | Excluded or separately recorded |
|---|---|---|
| Synthetic and A–C complete constrained product | Output zeroing where needed, mask/projection, launch, assembly/atomic scatter and constrained identity restoration | Input generation, host/device transfer, setup/JIT, validation; resident buffers reused |
| E/I isolated raw product | Output zeroing and operator action; CUDA-event elapsed time | Mask/projection is outside the raw product call; constrained entries of the probe were zeroed; transfers/setup/JIT/replay excluded |
| D/E/I fixed-work solve | Synchronized solver wall time, vector allocations performed by the solver, prescribed Krylov iterations, all V-cycle/outer actions, vector operations and reductions | Hierarchy construction, coarse factorization, spectral estimate, compilation warmup, host transfer and independent physical check |
| D/E/I solve to tolerance | Same solver boundary, including each true-residual recomputation/replacement and stopping test | Same setup/replay exclusions; not the same product count as a fixed-work run |
| G/H total optimization wall | Solver construction, initial allocations/transfers, every solve, density-dependent hierarchy setup, filtering, sensitivities, OC update and synchronization | One-step compilation warmup precedes timed trajectories; state saving and final cleanup occur after timer stops |
| F capacity probe | Cold constructor/setup and a separately timed capped solve; initial compilation may occur | It is not a warm solve to tolerance |
| I W2a | `setup_seconds` times `gmg.setup(E)`; fixed-work time brackets PCG and its synchronized allocations/actions | BC generation, operator/GMG construction precede setup timer; setup is not the full construction cost; GPU memory sample is after setup |

The G/H `gmg_setup_ms` sum is the repeated density-dependent setup time. Initial solver/hierarchy construction is in the total wall time but not isolated by a retained timer; it cannot be recovered by subtracting CG/setup/sensitivity totals because the remainder also includes filtering, OC, transfers and other work. Thus the table reports that construction component as NR instead of inventing a split. For H, density filtering and OC are GPU resident; G performs these on the CPU and includes associated transfers. Warmup removes many JIT costs but does not prove that every later call is compilation-free.

Session I shared setup includes transfers, construction of reference hierarchy, host dense coarsest assembly/Cholesky and 30-step spectral estimation. Per-arm setup is the recorded incremental construction time with possible reuse of compilation/cache state, not six independent cold costs. All six arms are resident in the measured harness before solves. Its reported fitting size and OOM are therefore implementation/harness capacity observations, not isolated per-arm optimal capacity limits.

## Uncertainty and decisions

Session E computes a paired ratio for each **round**: numerator time divided by candidate time. The statistic is the median of these ratios, not the ratio of medians. Its one-sided 95% lower bound is the 5th percentile of 10,000 bootstrap sample medians, resampling the round ratios with replacement using NumPy `default_rng(order_seed)` (`2026092320`); sample size equals the number of paired rounds. Twenty products within one round are repeated work in a batch, not twenty independent observations. See `experiments/session_e/analysis.py:bootstrap_lower` and the frozen protocol's H8/H9 rules. H8 requires median ≥1.05 and lower bound ≥1.02 against the fastest declared donor on both primary cases, with all rounds and checks present. H9 requires its separately declared all-FP64 fixed-work thresholds. The synthetic study uses its own bootstrap seed (`20260924`) and declared thresholds; A–D use their frozen analysis/protocol rules.

A narrow interval such as 1.259 with lower bound 1.258 describes within-session round variability conditional on the measured host/configuration. It does not quantify uncertainty over rental hosts, GPU populations, arbitrary designs or the choice of implementation. The two-host E replication is a separate outcome, not a bootstrap population sample.

G/H use **median-threshold decisions**, not one-sided bootstrap tests. For each primary all-FP64 case on a host, H11/H11b requires median paired wall ratio ≥1.10, at least two completed repetitions, total Krylov counts matched within 3% in every pair, and the compliance/volume/solve gate. Replication requires both hosts to pass. The generated paired table prints all three observations, including variability and unfavorable results. Mixed-schedule H12/H12b comparisons are descriptive medians with a declared expectation below 1.10; they are not confidence-bound decisions. Single capped trajectories have no repeated-sampling uncertainty estimate. Session I H13/H14 also use declared median thresholds on three fixed-work rounds; two tolerance times support a descriptive ordering among the tested routes, not a statistical universal winner.

## Optimization outcomes and validation coverage

The [outcome CSV](optimization-outcomes.csv) contains all 288 fixed-length trajectories (four hosts × four cases × two schedules × three arms × three repetitions), 24 capped trajectories and two Session G auxiliary diagnostics. No diagnostic, cap or unfavorable ratio is discarded. It records step count, exit reason, total Krylov count, wall time, CG time, hierarchy re-setup, sensitivity/diagonal times, final logged compliance, volume and design change. The readable table gives separately computed medians per host/case/schedule/arm, with n=3 for fixed runs and n=1 for capped runs. Exact per-repetition values and sources remain in CSV/JSON.

The donor log's final compliance is evaluated at the density **before the last OC update**. Its final volume and design change refer to the physical density/new design after that update. These are the retained endpoint conventions, not a new equilibrium solve on the saved final density. Design change is the infinity norm of the unfiltered design-variable update `max(abs(x_new-x))`; the cap criterion implemented in the frozen runner is `change <= 0.01`. All 150-step trajectories exited at the iteration cap. For example, H host 2 FP64 parity ends at change **0.1484386706** (G host 1: **0.1484407262**), far above 0.01. Successful linear solves and matching compliance do not imply that the design has converged. These are capped end-to-end trajectories; no time-to-converged-design claim is made.

The fixed-run agreement gate compares each nonstock arm's repetition-1 final logged compliance to stock, using `abs(C_arm-C_stock)/abs(C_stock) <= 1e-3`. Every fixed run is checked for final physical volume within absolute `1e-3` of the preset volume fraction and no solve whose iteration count reaches the 1000-iteration cap. This is compliance/volume agreement under that coverage; it is not density-field equivalence. The current analysis does not compute a density-field norm, maximum per-step compliance difference, or a same-state independent residual for every optimization solve. Stored design arrays cover repetition 1 and the capped trajectories; repetitions 2/3 have scalar histories, not stored fields. The frozen protocol's intention to report field norms must not be confused with completed verification. The exposed `equivalence` compatibility field has precisely this narrower meaning.

For H's **three primary 100-step all-FP64 cases**, median fused-ai/parity wall ratios span **1.080900–1.181485** over both hosts. Adding the 216k cantilever gives **1.023073–1.181485**. The fourth-case ratios are **1.025590 / 1.023073** on hosts 1/2. Its single capped FP64 trajectory gives **1.020899 / 1.021647**. Mixed runs and G's CPU-optimizer pipeline are separate populations: see the complete paired table, which includes H host 1's capped mixed ratio **0.850511**. None of these selected ranges describes every case, schedule and trajectory length.

The earlier “about 25 s instead of 66 s” values are the medians of the six capped runs across three arms and two schedules, **H host 2: 24.939278861 s; G host 1: 66.342864434 s**. They illustrate different pipelines on different rental hosts. They are not paired observations and do not isolate the causal effect of moving the optimizer to the GPU. Within-host, within-schedule arm comparisons remain the controlled performance evidence.

## Practical kernel map and scope

“Fastest” below means fastest among the retained tested implementations, sizes, precision schedules and endpoint definitions. All statements are limited to axis-aligned isotropic Hex8 with the tested scalar SIMP modulus fields; enlarged coefficients are nearest-neighbor copies of the q64/q96 design family. This is neither arbitrary-contrast validation nor a general mixed-precision convergence result.

| Hardware/evidence | Isolated FP64 product | Finest/outer choice in all-FP64 rediscretized solve | Fastest observed mixed tolerance route |
|---|---|---|---|
| RTX 3090 synthetic replication; 0.52/1.77 M elements | Plain parity directly measured | Mul-add/multigrid unmeasured; any suggestion is an extrapolation from 4090 | Unmeasured |
| RTX 4090 E, two hosts; 0.52/1.77 M elements, and I, one host; 1.77–14.16 M | Mul-add directly measured, including comparison with plain parity | Mul-add, fixed work; I 43.0 M total DOFs gives fused-ai/parity **1.352031**, all-FP64 | FP32 node V-cycle + FP64 parity mul-add outer, directly measured |
| A100 80GB PCIe B/I; up to 47.78 M elements in I | Node fastest among tested I FP64 products; fused-ai is not the fastest A100 product comparator | Node; I parity/node fixed-work ratio **1.189658–1.200122** | At q96x1, fused-ai outer **0.316095714 s**, node outer **0.319631294 s**; node outer wins enlarged cases |
| H100 80GB HBM3 I; 1.77–47.78 M elements | Node in tested I product timings | Node; parity/node **1.179589** at q96x1 and **1.091656–1.104500** in enlarged primary cases | FP32 node V-cycle + FP64 node outer in tested cases |

The A100 q96x1 difference is about 1.1% and uses only two tolerance repetitions per route on one host. It is a small observed advantage for fused-ai outer, not evidence that node always wins, nor proof of statistical equality. Product, all-FP64 fixed-work and mixed time-to-tolerance rankings must not be substituted for one another. The H100 1.09–1.10 range applies only to the enlarged primary cases; the continuity case is about 1.18.

## Measured capacity and allocation semantics

The [scale table](outcomes-scale-tables.md) reports every successful hardware/case/arm pair: exact elements, total/free DOFs, hierarchy depth, iteration counts, two warm solve times in the CSV, median in the readable table, shared and incremental setup. Rediscretized multigrid means matrix-free rediscretization on **noncoarsest** levels and a small assembled/factored coarsest operator solved by dense Cholesky. It does not mean no coarse operator is stored.

The largest successful RTX 4090 case was q96x2: **14,155,776 elements; 43,022,595 total DOFs; 42,910,848 free DOFs**. Its fastest observed warm tolerance route was FP32 node V-cycle / FP64 parity outer, median **2.116005612 s**. This endpoint differs from the approximately 1.35× **fixed-work all-FP64** speed ratio above. The q96x3 construction failed with OOM in the six-arm resident configuration. A100 and H100 completed q96x3: **47,775,744 elements; 144,574,851 total DOFs; 144,324,288 free DOFs**, with fastest mixed medians **4.536915969 s** and **3.977970041 s**, respectively. Counts are calculated from retained grid dimensions, not rounded protocol descriptions.

Nearest-neighbor upsampling of the retained design fields demonstrates capacity and solver behavior for those fields. There is no independently optimized design at each resolution, general mesh-independent convergence result, or parallel strong/weak-scaling experiment. Different q64/q96 base fields also preclude treating the entire ladder as one identical-coefficient refinement sequence.

Session F's pinned Galerkin implementation allocated about **46.3 GiB device-used / 46.3 GiB pool-reserved** for its successful 2.000376 M-element setup on A100/H100 NVL, or about **23 GiB per million elements**. This is a property of that implementation, construction arrays and allocator behavior, not an intrinsic requirement of all Galerkin methods. Device-used is total-minus-free CUDA memory; pool-reserved includes cached blocks and does not equal live arrays. Separate live-array totals and transient construction peaks were not measured. The allocation-failure message identifies a requested temporary allocation, not a successfully measured peak. The 2 M setup and 40-iteration solve succeeded as operations but did **not** reach tolerance. The 5.029452 M setup failed on both large cards; RTX 4090 failed already at 2 M. Approximately 3 M on an 80 GB card is only an estimate between observed trials, not a measured maximum.

W2a records every 1 M/2 M arm/schedule/block, including setup, device-used memory, 50-step fixed-work timing and the first tolerance attempt. The 1000-iteration cap failures remain labeled failures to converge; capacity or fixed-work timing is not converted into time to tolerance. The tables explicitly distinguish the feasibility H100 NVL from scale H100 HBM3 and the different setup timing boundaries.

## Comparator denominators and session-specific ratios

The product comparison reports **fused-ai time / plain-parity time**. The practical kernel-map figure uses speed relative to the fastest tested donor product, which is node on A100; these denominators differ and are labeled. The RTX 4090 Nsight Compute examination in [counter methods](algebra-counters.md) provides the retained hardware-counter evidence for the throughput interpretation.

For Session E host 1 q96, the approximately `0.752 / 0.326 = 2.31` gain combines an FP32 node V-cycle **and** replacement of fused-ai outer by parity mul-add. Keeping fused-ai outer, `0.752 / 0.369 = 2.04` is the corresponding V-cycle precision/configuration change. The generator now supplies `FpThirtyTwoVcycleGainQnineSix` as well as the combined macro.

The FP64 parity-PCG **20.3401218606 s** value belongs to Session C q96/Emin=1e-6 (median of two repetitions); **19.9638886160 s** belongs to Session A q96 (median of three repetitions). Different RTX 4090 allocations and session statistics explain the two values; they are not two representations of one measurement.

## Regeneration

```sh
.venv/bin/python analysis/session_g_tables.py results/session-g-host1-20260926 results/session-g-host2-20260926
.venv/bin/python analysis/session_h_tables.py results/session-h-host1-20260926 results/session-h-host2-20260926
.venv/bin/python analysis/session_i_tables.py \
  results/session-i-rtx4090-20260926 results/session-i-a100-20260926 results/session-i-h100-20260926
.venv/bin/python analysis/supplement_outcomes.py
.venv/bin/python paper/manuscript/make_figures.py
```

These commands preserve all raw results, saved states, frozen code and protocols. Generated summaries retain every case and expose capped exits, setup components and statistical coverage rather than silently strengthening the original observations.


# Comparator provenance, release audit and claim evidence

Evidence check date: 2026-09-27. This supplement documents comparator provenance, pinned-release audits and claim evidence. Paths are relative to the artifact repository unless explicitly described as members of a deposited archive. Historical protocols and results remain unchanged. Numerical values are summarized from retained records, not from new GPU runs.

## Contribution and closest comparisons

The engineering contribution is an exact low-order Hex8 parity implementation, comparison of its floating-point butterfly forms, and controlled measurement of where its product advantage survives in solver and optimization contexts. The tensor sign basis, reflection decomposition and Hadamard butterfly are established mathematics. The signed multiply-add implementation is an FP64 adaptation of Tri Dao's `fast-hadamard-transform`, commit `e7706faf8d1c3b9f241e36860640ad1dac644ede`, `csrc/fast_hadamard_transform_common.h`; attribution and BSD notice are retained in `experiments/session_e/dependencies/ATTRIBUTION.md` and `DAO-LICENSE`. No claim that all earlier Q1 methods lack a basis change is needed.

| Implementation / closest work | Element and operator | Integration / coefficient | Layout and accumulation | Precision and reduction |
|---|---|---|---|---|
| This parity product | Axis-aligned rectangular Hex8, isotropic elasticity, constant tensor per cell times scalar modulus | Exact 2×2×2 Gauss law; per-cell scalar SIMP modulus | Eight lanes per element, analytic indices, shared-node atomic scatter | FP64 local parity blocks; three butterflies tested; FP32 counterpart tested separately |
| Kiran, Gautam and Sharma (2020) | **Four-node quadrilaterals**; elasticity and scalar heat examples | Supplied element stiffness; the selected method screen did not establish the integration rule | Unstructured mesh data; element submatrices, warp communication, coloring to prevent conflicting updates | Double precision; triangular symmetry/storage reduction, rather than this Hex8 reflection-block construction |
| Yang fused release, `fused_fp64` / `fused_ai_fp64` | Cartesian Hex8 elasticity, dense 24×24 element action | Same caller-supplied exact-Gauss matrix and per-element modulus in this comparison | One thread per element; explicit edof / analytic indexing; atomic scatter | Supplied FP64 code paths; no parity basis reduction |
| Same release, `node_fp64` | Same physical operator | Same matrix and modulus | One thread per output node, incident-element gather, no scatter atomics | Supplied FP64 instantiation; duplicated incident-element reads and fixed accumulation order |
| Yang Galerkin stock FP64 | Same low-order elasticity family | Caller-supplied fine matrix; assembled Galerkin coarse levels | CuPy gather, DGEMM, `bincount` scatter in `MatrixFreeKff` | FP64 stock fine action; donor FP32/BF16 fused paths are different schedules |
| Tested libCEED configuration | Unit-cube Q1 vector elasticity | Tensor 2×2×2 Gauss; scalar density per cell | Natural interleaved node layout and element restrictions; CUDA restriction transpose performs assembly | FP64 `/gpu/cuda/gen`, with `/ref` and `/shared` controls; tensor basis action, not an implementation-wide library ranking |

The Kiran element class, double precision and colored element assembly are documented in the [author-hosted manuscript](https://fac.iitg.ac.in/dsharma/papers/journal/2020_EBE_SYM_FEA.pdf), selected methods and test-problem sections. The table explicitly leaves unverified details unidentified instead of inferring them from the element name.

## Exact Yang source identities and adaptations

The cited versions are [arXiv:2604.18020v1](https://arxiv.org/abs/2604.18020v1) and [arXiv:2604.26441v1](https://arxiv.org/abs/2604.26441v1), both preprints. The first paper emphasizes an FP32 fused framework. That paper's FP32 performance claims do not establish the FP64 performance measured here.

The retained fused donor matches [release commit `3a6e37af137c3f8ba65ef1d298602a6390da6466`](https://github.com/nbbllxx0/Fused-Gather-GEMM-Scatter-Kernels/tree/3a6e37af137c3f8ba65ef1d298602a6390da6466). A read-only source comparison found the following exact byte identities (retained in `provenance/supplement/comparator-identity.json`):

| Retained file under `experiments/session_e/donor/src/gpu_fem/` | SHA-256 | Symbols / status |
|---|---|---|
| `cuda_operators.py` | `d36cc55cf762e6f701056aeecc18977f8f4e0914be278c6971162fca0db7dc6f` | `fused_matvec_fp64`, `fused_matvec_ai_fp64`, `node_matvec_fp64` are supplied in this release. `_instantiate` generates the latter two from release-owned `__T__` templates with `double`; this substitution is upstream code, not a new investigator adaptation. |
| `cuda_fused_matvec.py` | `7e9f7002451b1b118243410ad62a3a6f19e9fc5c099ad2e68ca6af0575f2aa0f` | Supplied FP32/BF16 fused kernels; the FP64 baseline should be attributed to `cuda_operators.py`. |

The Galerkin donor is pinned to `fd9ddecda039fdfb7ffb8cf49c58f602318ed2ef` in `experiments/combined/donor_yang_gmg/PINNED.txt`. Its `src/gpu_fem/solver_v2.py:MatrixFreeKff.matvec` implements the stock FP64 gather/DGEMM/`bincount` path. `experiments/combined/kff_adapters.py:ArmKff` replaces the FP64 fine core, including the fine smoother's FP64 calls; it preserves the donor free/full expansion and restriction. Non-FP64 products and diagonal extraction remain donor calls. Thus “unchanged” describes the imported source files, not the complete adapted experiment. The common path-discovery workaround, optimizer/filter changes and schedule choices are separately declared in Sessions F–H.

## Resident libCEED experiment

The tested library is v0.12.0, commit `4018a20a98d451fac24765d3ddb936861647ce8d`, archive SHA-256 `4c642ae1563e0363aa48f5d794832ad6e9716789405e33511213b837d1528a23`. `evidence/companion/results/rescope/hex-modal-libceed-resident-20260924/protocol.json` pins the source and configuration; no current-library or unreleased-main claim is made.

The extracted bridge and QFunction are retained as `provenance/supplement/libceed-bridge.c` and `libceed-elasticity.h`, with archive/member hashes in `provenance/supplement/extractions.json`. The bridge creates a 3D, three-component Q1 basis with two Gauss points per axis. A discontinuous constant scalar basis stores one density per cell; the QFunction evaluates `E=1e-6+(1-1e-6)rho^3`, with unit modulus, Poisson ratio 0.3 and exact unit-cube geometry scaling. It does not store eight replicated density coefficients per cell. Library nodes are `(z,y,x,component)` while the custom padded layout is `(x,z,y,component)`; a reflected-y restriction maps the same physical nodes. Constraints are exact identity rows imposed by a GPU kernel after the library action.

The requested backends are `/gpu/cuda/ref`, `/gpu/cuda/shared` and `/gpu/cuda/gen`, with device-memory preference checked. One action includes output clearing/assembly, the density-scaled elastic law, identity rows and `cudaDeviceSynchronize`. Host steady-clock batched time is primary; CUDA-event time is secondary. Setup, conversion, allocation, transfers, compilation, JIT warmup and numerical output are excluded. Input/material vectors remain resident. There are 12 randomized blocks of 256 calls, five seconds of warmup, and a minimum five-millisecond batch criterion. Six primary cases are 64×32×32, 128×64×64 and 192×96×96, each with two coefficient patterns; 18 total shapes/patterns and separate sanitizer runs receive correctness checks.

The 2.5768–3.0083 ratio is generated-backend time divided by signed-multiply-add time for those six cases (`figures/ratios.csv`; regenerate via `analysis/session_e_tables.py`). It comes from one RTX 4090 allocation and a different harness from the main CuPy study. It does not establish superiority over libCEED as a whole, other backend choices, or tensor implementations in general. The declared sign-test/Holm rule covers all 30 case/control comparisons and must not be replaced with Session E's paired-bootstrap rule.

## Pinned release audit

The detailed finding table is [the reproducibility record](../combined/reproducibility_record.md). `companion/NAME` there abbreviates **`evidence/companion/results/rescope/NAME`**. The OpenMP pin is `b46ed453ad5a05818b563e540ce176c9d85c6fe2`; the Futhark pin is `c3cd2b00632ad57cd3d8917246552b2fb4e99e01`. Their repository URLs, commits and per-file hashes are retained in `provenance/supplement/traff-source-provenance.json`. These are pinned GitHub trees; the original source-provenance record does not establish byte identity with the authors' separate Zenodo version archive. The source pins and any adapted-source hashes are in each row's frozen protocol.

**The 12.17% value is an initial candidate integration error.** The archived `diagnosis.json` identifies published `stencil_methods.c:426` as copying input on fixed rows, while the first candidate's `interpose.c` set those rows to zero. Free-DOF probes could not distinguish them. The initial CPU experiment has 18 rejected state gates, including small nonzero fixed displacements from the source and much larger candidate errors. Its result must not be presented as a 12.17% error in the unmodified published solver, or as the result of a candidate that already honored identity rows. The next CPU experiment restored candidate identity rows and added shared exact projection of returned fixed values; its 18/18 passing states concern that declared adaptation.

The diagnosis has been copied byte-for-byte to `provenance/supplement/host-diagnosis.json` from the deposited `companion--hex-modal-mg-host-20260923--files.tar.gz`, whose SHA-256 is recorded in the extraction manifest. No original evidence file has been overwritten.

| Quantity | Definition, denominator and units | Input and interpretation |
|---|---|---|
| 12.170770009% state error | `100*norm(u-u_ref,2)/norm(u_ref,2)`, full DOFs; dimensionless percentage; direct constrained source-matched reference | Initial modal candidate, 24×12×8, step 1. Rescaling a linear right-hand side scales both numerator and denominator; conditioning and rounding can still change observed relative error. |
| 1.139422914e-5 bilinear defect | `abs(aᵀB(b)-bᵀB(a))/(norm(a)*norm(B(b))+norm(b)*norm(B(a)))`; dimensionless | CPU source-transfer V-cycle, 16×8×8, uniform density, pair 2. A sampled reciprocity error, sensitive to finite-precision summation and cancellation; not a displacement error or a universal SPD proof. |
| 66.593264989 fixed-DOF leakage | `max(abs(B(a)[fixed]), abs(B(b)[fixed]))`; no denominator; absolute correction/displacement units of this benchmark | 24×12×8, uniform density, pair 0, zero fixed entries in both input residuals. Magnitude scales with input amplitude and material/force units. Nonzero output contradicts the exact free-space-preservation contract; it cannot be compared numerically with the two relative quantities. |

`python3 analysis/supplement_registry.py` verifies the extracted member hashes and recomputes all three reported maxima from their saved vectors, producing `audit-units.json`. The selected residual probes are bounded by about 0.5 in magnitude; no normalization to unit residual norm was applied. The source-transfer defect and leakage are distinct diagnostics. Projection after restriction/prolongation gives 24/24 passing sampled tests and a maximum defect about 1.19e-17; it does not certify arbitrary coefficient fields or alone establish the cause of an earlier GPU failure.

The auxiliary E3 OC diagnosis is confined to `experiments/paper4/run_experiments_e1_e10.py:_oc_update` at the Galerkin pin: its arithmetic midpoint bisection starts at `[1e-40,1e40]` and takes 100 steps. Even repeated upper-bound halving leaves `1e40/2^100 ≈ 7.888609e9`, preventing the needed small multiplier. The original preprint already identifies E3 divergence and excludes it from its main acceleration evidence. The new contribution is this source-level localization and full-size reproduction, not discovery that the authors' entire optimizer fails. The separate full `run_simp` OC update is unaffected. Session G uses the explicitly declared filtered top88 update.

## Hypothesis convention and chronology

[The generated registry](hypothesis-registry.md) and [full JSON](hypothesis-registry.json) enumerate endpoints, primary cases, thresholds, declaration timestamps, byte-linked execution records and outcomes for Sessions A–J. H3 was not confirmed; H5 was not supported; H11, H11b and H15 failed their recorded decisions. H11b changes the optimizer and step count and remains a separate hypothesis. Packaging and integration gates are separate; H12/H12b are descriptive expectations with work-match qualifications. Therefore “three failed hypotheses” is not a complete whole-study count.

Protocol hashes establish identity. The registry checks the recorded declaration time against the start of the job that used the same protocol bytes (or a verifier's matching protocol hash). These are retained author/system clock records, not independent timestamp certification or proof against retrospective authorship. Session J's final declaration must only be paired with attempt 07; earlier attempts used their separately preserved amendments. Historical release tags identify artifact versions, not proof of prospective declaration. This chronology assessment uses the retained records rather than a reconstruction of Git history.

## Principal-claim evidence map

The initial software release is [Zenodo 23001863](https://doi.org/10.5281/zenodo.23001863); raw evidence is [Zenodo 22998404](https://doi.org/10.5281/zenodo.22998404). Tags below are historical release locators listed in the artifact README. The inspected source baseline is `a60700fe9df55d4b2e597e4a4014cee189ac5e43`. This supplement records subsequent evidence checks and regenerated summaries; the historical tags identify the original experiment snapshots.

| Claim / retained failure | Historical tag | Protocol and raw/summarized evidence | Exact analysis command (repository root) |
|---|---|---|---|
| Exact parity operator and local lane equivalence | `exp01-parity-algebra`, `base` | `code/hex_modal/protocol.json`; `code/hex_modal/modal.py`; CPU tests; algebra supplement | `python3 -m pytest -q tests` |
| Plain FP64 product on RTX 4090 and independent RTX 3090 | `exp02-products-rtx4090`, `exp03-products-rtx3090` | `code/hex_modal_cuda/protocol.json`; `evidence/kernel-rtx4090/`, `evidence/kernel-rtx3090-replication/`; `paper/numbers.json` | `python3 analysis/paper_numbers.py` |
| Fixed-work PCG and no useful FP32 parity gain; failed A100 dense-control prediction | `exp07-session-a`, `exp08-session-b` | `experiments/session_{a,b}/protocol.json`; `results/session-{a,b}-20260923/local-verification.json`; `paper/session_{a,b}_numbers.json` | `python3 analysis/session_a_tables.py results/session-a-20260923`; `python3 analysis/session_a_tables.py results/session-b-20260923 session_b` |
| Tested high-contrast FP64-only niche unsupported (H5) | `exp09-session-c` | `experiments/session_c/protocol.json`; `results/session-c-20260923/local-verification.json` | `python3 analysis/session_a_tables.py results/session-c-20260923 session_c` |
| Rediscretized multigrid route comparison | `exp10-session-d` | `experiments/session_d/protocol.json`; `results/session-d-20260923/local-verification.json` | `python3 analysis/session_d_tables.py results/session-d-20260923` |
| Two-host FP64 product ratio 2.17–2.25 and solver effects; narrow libCEED result | `exp11-session-e` | `experiments/session_e/protocol.json`; `results/session-e-host{1,2}-20260925/local-verification.json`; companion libCEED protocol/ratios above | `python3 analysis/session_e_tables.py` |
| Packaging failure, integration gate failure, Galerkin capacity | `exp12-session-f` | `experiments/session_f/protocol.json`; `results/session-f-*/remote/{job.json,run-01/result.json}`; `paper/session_f_summary.json` | `python3 analysis/session_f_summary.py` |
| 30-step and capped 150-step trajectories; failed H11 | `exp13-session-g` | `experiments/session_g/protocol.json`; `results/session-g-host{1,2}-20260926/remote/run-01/result.json` | `python3 analysis/session_g_tables.py results/session-g-host1-20260926 results/session-g-host2-20260926` |
| All four 100-step cases and capped 150-step outcomes; failed H11b | `exp14-session-h` | `experiments/session_h/protocol.json`; `results/session-h-host{1,2}-20260926/remote/run-01/result.json` | `python3 analysis/session_h_tables.py results/session-h-host1-20260926 results/session-h-host2-20260926` |
| Measured capacity and kernel-map exceptions; RTX fixed-work FP64 gain ~1.35 at 43 M DOFs | `exp15-session-i` | `experiments/session_i/protocol.json`; `results/session-i-*/remote/run-01/result.json`; `paper/session_i_numbers.json` | `python3 analysis/session_i_tables.py results/session-i-rtx4090-20260926 results/session-i-a100-20260926 results/session-i-h100-20260926` |
| Executed counter ratio 2.5; H15 count failure; post-hoc timing agreement | `exp16-session-j` | `experiments/session_j/protocol.json`; attempt-07 `remote/ncu/ncu-*.csv`; all failed attempts retained | `python3 analysis/session_j_counters.py` |
| Pinned quadrature, boundary, transfer and damping audit | `exp17-traff-audit` | `evidence/companion/results/rescope/hex-modal-*/protocol*.json` and result/replay files in reproducibility record; selected deposited inputs in `provenance/supplement/` | `python3 analysis/repro_numbers.py`; `python3 analysis/supplement_registry.py` |

Brace and wildcard paths above denote families, not literal filenames. The registry generator reads the freshly regenerated summaries; run the combined numerical build before it when other analysis sources have changed.

## Reproducibility levels and bibliography metadata

**Numerical regeneration** reads retained results and recomputes summaries, tables, macros and figures on a CPU. **Saved-state replay** reloads recorded coefficient/displacement arrays and checks a reference operator or acceptance decision; it is not a fresh performance measurement. **Fresh execution** recreates the declared GPU software/hardware environment, runs the frozen workload and collects new timings. The public repository supports numerical regeneration; larger saved-state and raw-source members are accessible in the existing deposits through their manifests. The manuscript text and its build are not part of the public repository.

The [PLOS publisher record](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0240813) confirms volume **15(10)** and DOI `10.1371/journal.pone.0240813`. The [Springer publisher record](https://link.springer.com/article/10.1007/s00158-026-04280-3) gives **Long Hu, Tianyuan Qi, Junpeng Zhao and Chunjie Wang**, volume 69, article 90 and DOI `10.1007/s00158-026-04280-3`. These fields are recorded in `paper/manuscript/refs.bib`; the records were checked on 2026-09-27. The linked September 2026 Hu correction concerns interchanged ellipse semi-axes in Fig. 25 and does not change these bibliography fields.


# Hypothesis and gate registry

Five named hypotheses lack support: H3, H5, H11, H11b, H15. H5 retains its original not-supported label; H11b is a separate changed-pipeline hypothesis. Descriptive H12/H12b expectations and packaging/integration gates are separate. This count is for Sessions A-J, not every exploratory companion study.

Hashes identify content, not temporal priority. These dates are author/system recorded timestamps, checked against byte-linked execution records where available, not independent preregistration timestamps. No local git history is present in this workspace.

Frozen Session B h3.rule writes modal8/dense8, but the question and recorded analysis concern advantage dense8 time / modal8 time. The retained failed decision uses that latter ratio (about 2.24), not the literal reversed label. The historical protocol is preserved.

The JSON companion preserves the exact primary cases, decision rules, hashes and every inspected execution record. The table abbreviates these fields.

| ID | Type | Cases / endpoint and decision | Declaration UTC | Executed UTC (byte-linked final protocol) | Outcome |
|---|---|---|---|---|---|
| H2 | hypothesis | optimized q64/q96; fixed Jacobi-PCG work, median ≥1.08 and one-sided 95% lower ≥1.05 | 2026-09-23 17:15:38 | 2026-09-23 17:26:40 | supported |
| H3 | hypothesis | optimized q64/q96; FP64 dense8/parity product gain <1.10 expected | 2026-09-23 18:05:36 | 2026-09-23 18:41:35 | not confirmed |
| H4 | hypothesis | optimized q64/q96; FP32 gain over fastest baseline <1.10 expected | 2026-09-23 17:15:38 | 2026-09-23 17:26:40 | supported |
| H5 | hypothesis | optimized q64/q96 at Emin=1e-9; at least one refinement failure while all FP64 PCG repetitions pass | 2026-09-23 18:38:14 | 2026-09-23 18:39:47 | not supported |
| H6 | hypothesis | optimized q64/q96; fixed MG work, median ≥1.05, lower ≥1.02 | 2026-09-23 19:35:12 | 2026-09-23 19:38:27 | supported |
| H7 | descriptive question | fastest physically accepted solve to tolerance; descriptive | 2026-09-23 19:35:12 | 2026-09-23 19:38:27 | descriptive |
| H8 | hypothesis | optimized q64/q96 on both hosts; product median ≥1.05, lower ≥1.02 | 2026-09-25 10:54:43 | 2026-09-25 10:58:45, 2026-09-25 11:09:30 | supported on two hosts |
| H9 | hypothesis | same cases; fixed MG work median ≥1.05, lower ≥1.02 | 2026-09-25 10:54:43 | 2026-09-25 10:58:45, 2026-09-25 11:09:30 | supported on two hosts |
| H10 | descriptive question | same cases; fastest solve and mixed outer-product effect, descriptive | 2026-09-25 10:54:43 | 2026-09-25 10:58:45, 2026-09-25 11:09:30 | descriptive |
| F-stack | packaging gate | three GPUs; unchanged donor smoke-test exit code 0 | 2026-09-26 00:42:40 | 2026-09-26 00:59:35, 2026-09-26 01:18:10, 2026-09-26 00:44:22 | failed as released; directory workaround passes |
| F-integration | integration gate | all grids, densities, schedules and arms; product ≤1e-12, convergence, residual ≤1e-10, cross-state ≤1e-8 | 2026-09-26 00:42:40 | 2026-09-26 00:59:35, 2026-09-26 01:18:10, 2026-09-26 00:44:22 | not passed |
| H11 | hypothesis | three primary cases, FP64 30-step trajectories; all medians ≥1.10, ≥2 repetitions, work/agreement gates | 2026-09-26 01:41:32 | 2026-09-26 01:44:43, 2026-09-26 02:44:46 | failed on two hosts |
| H12 | descriptive expectation | same primary cases, mixed trajectories; descriptive expectation <1.10 | 2026-09-26 01:41:32 | 2026-09-26 01:44:43, 2026-09-26 02:44:46 | not uniformly below 1.10; work-match qualification required |
| H11b | new hypothesis on changed pipeline | same three primary cases, GPU optimizer, 100 steps; same ≥1.10 and qualification rules | 2026-09-26 04:15:46 | 2026-09-26 04:17:38, 2026-09-26 04:18:54 | failed on two hosts |
| H12b | descriptive expectation | GPU optimizer mixed trajectories; descriptive expectation <1.10 | 2026-09-26 04:15:46 | 2026-09-26 04:17:38, 2026-09-26 04:18:54 | not uniformly below 1.10; work-match qualification required |
| H13 | hypothesis | fitting primary enlarged cases on each 80 GB card; parity/node fixed-work time ≥1/1.05 | 2026-09-26 04:22:52 | 2026-09-26 04:40:45, 2026-09-26 05:27:10, 2026-09-26 04:30:46 | supported on fitting primary cases |
| H14 | hypothesis | largest fitting primary RTX case; fused/parity fixed-work median ≥1.18 | 2026-09-26 04:22:52 | 2026-09-26 04:40:45, 2026-09-26 05:27:10, 2026-09-26 04:30:46 | supported at largest fitting primary case |
| H15 | hypothesis | 128×64×64 RTX4090 VM; FP64 active utilization ≥70%; executed count ratios within 5% of static 78:52:34 | 2026-09-26 05:25:58 | 2026-09-26 06:04:06 | failed count criterion; utilization part passed |


# Reproducibility record for the combined paper

Consolidates the audit findings on the published codes used as baselines and hosts. Each row gives the
finding, how it was established, what repaired it (if anything), and the sealed evidence. Every evidence
entry is a path in this repository: the Träff et al. audit files under `evidence/companion/` (abbreviated
`companion/` below means `evidence/companion/results/rescope/`; their protocols and results keep their original layout) and the
session folders under `results/`.
Declared protocols and dated execution records are linked in [the hypothesis registry](hypothesis-registry.md). Hashes establish content identity; the recorded timestamps support chronology without constituting independent preregistration. Failures are kept.

These findings are about specific pinned releases and paths. They are not claims about other versions,
and they do not say whether the published papers' conclusions hold.

## A. Träff et al. topology-optimization codes (OpenMP GPU and Futhark releases)

| # | Finding | Established by | Repair / consequence | Evidence |
|---|---|---|---|---|
| A1 | The OpenMP release uses **composite 10×10×10 midpoint quadrature per fine element**: `getKEsubspace` samples ten points per fine-subcell axis. Relative Frobenius differences from exact 2×2×2 Gauss Q1 are **0.548361% / 0.516454% / 0.416469%** at ν = **0.20 / 0.30 / 0.45**. The tested composite matrices retain six rigid modes; this is not the eighteen-mode pathology of literal single-point Hex8. The pinned Futhark matrix matches exact Q1. | Independent exact-Q1/source-matched composite integration replay on pinned sources (OpenMP `b46ed453`, Futhark `c3cd2b00`); fresh CPU quadrature and eigenvalue checks in `paper/supplement/algebra-checks.json` use null tolerance `1e-10 ‖K‖₂`. | This is a mismatch with this study's exact-Gauss operator contract, not evidence by itself that the audited mechanics or original conclusions are invalid. Baselines in this study use exact Gauss Q1; composite-rule comparisons are labeled source-matched. | `companion/hex-modal-priorart-20260923/` (`cpu-protocol-02.json`, `cpu-02/results.json`); [quadrature definitions and check](algebra-counters.md#composite-midpoint-quadrature-in-the-pinned-release) |
| A2 | Both releases run mixed precision (FP32 vectors, correction or CG in parts), not all-FP64. The pinned trees contain no licence file. | Source trace | All-FP64 claims use separately declared adaptations. No redistribution permission is inferred. | pinned sources (commits above) |
| A3 | **Initial candidate boundary mismatch:** the candidate zeroed fixed rows instead of copying the input as the published operator does; its state error reached **12.17 %**. Separately, source runs leave constrained displacement up to **5.27e-7**, violating this audit's exact-zero acceptance gate. Free-input probes cannot distinguish zero from identity fixed rows. | Serial CPU execution of source C with OpenMP pragmas ignored; 18 captured solves on 16×8×8, 24×12×8 and 16×8×12 over three steps; all 36 source/candidate free-product probes pass, all 18 state gates fail | Candidate identity rows are restored, and a common returned-state projection is declared for both arms. The adapted CPU experiment passes 18/18 states, 9/9 paired and 3/3 endpoint comparisons. **12.17 % is not an unmodified-source solver error.** | `companion/hex-modal-mg-host-20260923/{protocol.json,run-01/result.json}`; archived diagnosis now at `provenance/supplement/host-diagnosis.json`; `companion/hex-modal-mg-host-v2-20260923/` |
| A4 | **Source transfer boundary contract:** unprojected restriction/prolongation in the declared adapted host fail 24/24 sampled free-space/symmetry probes (dimensionless bilinear defect up to 1.14e-5; absolute fixed correction leakage 66.6). | CPU execution of C `VcyclePreconditioner`, not an original GPU rerun; three meshes, uniform/random density, four residual pairs each | Zero fixed outputs after each restriction and prolongation: 24/24 pass, dimensionless defect ≤1.19e-17. This tests necessary properties, not universal SPD or causation of an earlier GPU failure. | `companion/hex-modal-mg-bc-audit-20260923/{protocol.json,run-01/verification.json}`; exact normalization and saved-array replay in `paper/supplement/provenance-audit.md` |
| A5 | **Damping:** fixed weighted-Jacobi ω=0.6 admits unstable modes at level two on retained designs under this study's exact-Q1 operator contract. | CPU independent spectral diagnostic: 9/9 converged eigen-estimates; exact local bounds on 73 subcell matrices; subsequently a separately declared GPU replay | Weights [0.3,0.25,0.2] have exact rational $LDL^T$ local certification for ideal Q1. The matched GPU replay accepts 9/9 saved states versus 3/9 with ω=0.6. This is not a universal mixed-precision convergence proof. | `companion/hex-modal-mg-smoother-audit-20260923/`; `companion/hex-modal-mg-damping-control-20260923/run-03/exact-summary.json`; `companion/hex-modal-mg-damping-gpu-20260923/rental-01/local-verification.json` |

## B. Yang et al. Galerkin GMG release (`nbbllxx0/Mixed-Precision-GMG-SIMP`, `fd9ddecd`)

| # | Finding | Established by | Repair / consequence | Evidence |
|---|---|---|---|---|
| B1 | **Package/path-discovery failure:** `paths.py` requires a `scripts/` directory beside `src/`, absent from the pinned public tree. | Original `ci/smoke_test.py` attempted on RTX 4090, A100 PCIe 80 GB and H100 NVL; nonzero exit before numerical smoke solve | Creating an empty directory allows smoke tests to run and pass. This is a packaging workaround, not repair of a numerical algorithm. RTX smoke: 25 iterations, residual 5.95e-11. | `experiments/combined/donor_yang_gmg/src/gpu_fem/paths.py`; `results/session-f-*/remote/job.json`; `paper/session_f_summary.json` |
| B2 | **The auxiliary E3 loop's OC update collapses the design** (confirmed at full size on two RTX 4090 hosts, session G: volume 0.30 → 0.15 → 0.001, compliance → 6.9e8). E3's own `_oc_update` bisects the multiplier arithmetically over [1e-40, 1e40] for 100 iterations, so it cannot fall below ≈ 7.9e9. Volume went 0.30 → 0.15 → 0.001 and compliance 140 → 1.9e9 on a CPU replica of their cantilever. The paper already treats E3 as auxiliary: "not used as main acceleration evidence because the design trajectories diverge" (§4.4, App. A). It attributes the divergence to capped Jacobi solves, and its E3 compliance plots are at $10^8$–$10^9$. Their full SIMP driver (`run_simp`) uses a correct OC (bisection on [0, 1e9] to 1e-9, volume on filtered/projected densities) and is not affected. | CPU replica; paper full text read; full-size diagnostic in session G | Only the auxiliary E3 schedule is affected; the paper's headline claims do not rest on it. Session G uses a top88 OC with a density filter. | session G (`results/session-g-*`); arXiv 2604.26441 §4.4, App. A |
| B3 | **FP64 comparator provenance:** the Galerkin stock core is CuPy gather + DGEMM + `bincount`; its fused kernels are FP32/BF16. | Pinned `solver_v2.py:MatrixFreeKff.matvec` and `cuda_fused_matvec.py` source | The supplied FP64 fused and node paths come from the separate fused-kernel release at `3a6e37af137c3f8ba65ef1d298602a6390da6466`. The FP64 core substitution is an explicit adapter; it does not make the complete solver unmodified. | `experiments/combined/kff_adapters.py`; `experiments/session_f/protocol.json`; comparator symbol/hash table in `paper/supplement/provenance-audit.md` |
| B4 | **Measured allocation footprint:** about 23 GiB per million elements for this pinned Galerkin implementation. 2 M fits, whereas 5 M fails on A100 PCIe 80 GB and H100 NVL 94 GB. | Session F sizing ladder, per schedule/arm, including out-of-memory outcomes | About 3 M is an extrapolated ceiling, not a tested maximum. These records do not fully separate live arrays, transient construction allocations and cached pool memory. Larger studies use the separately specified rediscretized hierarchy. | `experiments/session_f/protocol.json`; `results/session-f-*/remote/run-01/result.json`; `paper/session_f_summary.json` |
| B5 | **Mixed-schedule integration gate:** the stock FP32 fine-smoother schedule reaches 300 iterations on random-density 64 k cantilever states on A100/H100; stock takes 246 on RTX 4090. Other adapted arms also fail parts of the compound gate. | Original and adapted GPU solver execution in Session F; product agreement, convergence, residual and cross-state requirements recorded separately | The gate remains failed on every host despite FP64 core product agreement ~2.2e-16. The donor paper already discloses difficult heterogeneous convergence; this audit localizes failure in its own declared workload, not a new universal defect. Session G changes to SIMP-generated trajectories and records non-converged solves. | `experiments/session_f/protocol.json`; `results/session-f-*/remote/run-01/result.json`; `paper/session_f_summary.json` |

## C. Process and units

[The supplement](provenance-audit.md) defines the state-error denominator,
normalized bilinear defect and absolute correction leakage, and reproduces their selected maxima
from hash-verified saved arrays. The three quantities have different units and cannot be compared
as percentages. In particular, the magnitude of absolute fixed-DOF leakage scales with the residual
input; preservation of the free space is the violated invariant.

The [registry](hypothesis-registry.md) distinguishes named hypotheses from descriptive
expectations, packaging failures, integration gates and operational attempts. H3 was not confirmed,
H5 was not supported, and H11/H11b/H15 failed their recorded decisions. H11b concerns a changed
pipeline; it does not erase H11. No global count of three failures is claimed.

GPU operational failures (upload timeouts, rate limits, SSH, startup, counter permission and tooling)
remain separate attempts, including Session J's seven attempts. Each final protocol is linked to
matching execution records; earlier failed attempts use their own amendments. Hashes alone do not
prove temporal priority. Historical thresholds and outcomes remain preserved. New adaptations are
explicitly labeled, including the boundary projection, corrected OC and GPU optimizer.

## D. Source, input and contract locator

This table supplements the observed-error/adaptation/outcome columns above. Source paths are
relative to each pinned upstream tree unless identified as an audit script. The Träff host/transfer
entries are CPU source execution; the damping follow-up and Yang sessions explicitly executed GPUs.

| Finding | Pinned source / location | Exercised input and mathematical contract | Evidence type / result locator |
|---|---|---|---|
| A1 quadrature | OpenMP `b46ed453…`, `local_matrix.c:getKEsubspace`; Futhark `c3cd2b00…`, `src/keConstants.fut` | Three Poisson ratios, levels/subcells in `cpu-protocol-02.json`; exact-Gauss Q1 comparison, not whole-element reduced integration | CPU matrix comparison; `hex-modal-priorart-20260923/cpu-02/results.json` |
| A2 precision | Same pins; OpenMP `definitions.h`, solver vectors; Futhark `src/solvers.fut` | Source scalar/vector types versus this study's FP64-throughout contract | Source trace, not a speed claim; source hashes in A1 protocol |
| A3 boundary | OpenMP `b46ed453…`, `stencil_methods.c:426`; initial audit `interpose.c` | Three meshes × three steps × source/candidate; source identity rows versus candidate zero rows; full-state norm and exact fixed values | CPU source execution; diagnosis and raw/saved state checks above |
| A4 transfer | Same base, declared adapted-02 source hashes; `stencil_solvers.c:VcyclePreconditioner`, restriction/prolongation wrappers | 16×8×8, 24×12×8 (3 levels), 32×16×16 (4); uniform 0.2/random void patterns; free-space preservation and reciprocity | CPU C source transfers; `hex-modal-mg-bc-audit-20260923` protocol and verification |
| A5 damping | Same base; fixed-Jacobi source semantics, independent `spectrum.py` audit | Retained 128×64×64 states 8/9 and 64×32×32 state 21; Jacobi stability and subsequently true-residual/state gates | CPU spectral and rational certification, then separately adapted GPU replay; three audit folders above |
| B1 path discovery | Galerkin `fd9ddecd…`, `src/gpu_fem/paths.py`, `ci/smoke_test.py` | Three GPU hosts; package root discovery and zero-exit smoke gate | Original GPU-host launch failure, then directory-only workaround |
| B2 auxiliary OC | Same pin, `experiments/paper4/run_experiments_e1_e10.py:_oc_update` | Auxiliary E3 216k cantilever, mixed stock schedule; multiplier bisection must span volume-preserving update | CPU arithmetic diagnosis; Session G full-size GPU diagnostic on two hosts; full `run_simp` unaffected |
| B3 FP64 core | Galerkin `fd9ddecd…:solver_v2.py`; fused `3a6e37af…:cuda_operators.py` | Same caller-supplied Ke, modulus and constraints, stock versus substituted fine actions | Supplied-code identity plus explicit adapter and GPU product checks |
| B4 memory | Galerkin `fd9ddecd…`, `multigrid_v4.py` | 2 M/5 M element ladder; allocation feasibility by schedule/arm | GPU sizing, not exact capacity limit or universal Galerkin-memory bound |
| B5 integration | Same pin, `solver_v4.py` / mixed fine smoother | Small/medium uniform and seeded random fields; product ≤1e-12, residual ≤1e-10, convergence and cross-state ≤1e-8 | Original/adapted GPU runs; failed compound gate retained |
