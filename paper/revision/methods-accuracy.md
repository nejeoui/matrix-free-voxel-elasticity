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
physical units. `analysis/revision_accuracy.py` regenerates maxima, counts and
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
