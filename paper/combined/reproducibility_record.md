# Reproducibility record for the combined paper

Consolidates the audit findings on the published codes used as baselines and hosts. Each row gives the
finding, how it was established, what repaired it (if anything), and the sealed evidence. Every evidence
entry is a path in this repository: the Träff et al. audit files under `evidence/companion/` (abbreviated
`companion/` below means `evidence/companion/results/rescope/`; their protocols and results keep their original layout) and the
session folders under `results/`.
Declared protocols and dated execution records are linked in [the hypothesis registry](../supplement/hypothesis-registry.md). Hashes establish content identity; the recorded timestamps support chronology without constituting independent preregistration. Failures are kept.

These findings are about specific pinned releases and paths. They are not claims about other versions,
and they do not say whether the published papers' conclusions hold.

## A. Träff et al. topology-optimization codes (OpenMP GPU and Futhark releases)

| # | Finding | Established by | Repair / consequence | Evidence |
|---|---|---|---|---|
| A1 | The OpenMP release uses **composite 10×10×10 midpoint quadrature per fine element**: `getKEsubspace` samples ten points per fine-subcell axis. Relative Frobenius differences from exact 2×2×2 Gauss Q1 are **0.548361% / 0.516454% / 0.416469%** at ν = **0.20 / 0.30 / 0.45**. The tested composite matrices retain six rigid modes; this is not the eighteen-mode pathology of literal single-point Hex8. The pinned Futhark matrix matches exact Q1. | Independent exact-Q1/source-matched composite integration replay on pinned sources (OpenMP `b46ed453`, Futhark `c3cd2b00`); fresh CPU quadrature and eigenvalue checks in `paper/supplement/algebra-checks.json` use null tolerance `1e-10 ‖K‖₂`. | This is a mismatch with this study's exact-Gauss operator contract, not evidence by itself that the audited mechanics or original conclusions are invalid. Baselines in this study use exact Gauss Q1; composite-rule comparisons are labeled source-matched. | `companion/hex-modal-priorart-20260923/` (`cpu-protocol-02.json`, `cpu-02/results.json`); [quadrature definitions and check](../supplement/algebra-counters.md#composite-midpoint-quadrature-in-the-pinned-release) |
| A2 | Both releases run mixed precision (FP32 vectors, correction or CG in parts), not all-FP64. The pinned trees contain no licence file. | Source trace | All-FP64 claims use separately declared adaptations. No redistribution permission is inferred. | pinned sources (commits above) |
| A3 | **Initial candidate boundary mismatch:** the candidate zeroed fixed rows instead of copying the input as the published operator does; its state error reached **12.17 %**. Separately, source runs leave constrained displacement up to **5.27e-7**, violating this audit's exact-zero acceptance gate. Free-input probes cannot distinguish zero from identity fixed rows. | Serial CPU execution of source C with OpenMP pragmas ignored; 18 captured solves on 16×8×8, 24×12×8 and 16×8×12 over three steps; all 36 source/candidate free-product probes pass, all 18 state gates fail | Candidate identity rows are restored, and a common returned-state projection is declared for both arms. The adapted CPU experiment passes 18/18 states, 9/9 paired and 3/3 endpoint comparisons. **12.17 % is not an unmodified-source solver error.** | `companion/hex-modal-mg-host-20260923/{protocol.json,run-01/result.json}`; archived diagnosis now at `provenance/supplement/host-diagnosis.json`; `companion/hex-modal-mg-host-v2-20260923/` |
| A4 | **Source transfer boundary contract:** unprojected restriction/prolongation in the declared adapted host fail 24/24 sampled free-space/symmetry probes (dimensionless bilinear defect up to 1.14e-5; absolute fixed correction leakage 66.6). | CPU execution of C `VcyclePreconditioner`, not an original GPU rerun; three meshes, uniform/random density, four residual pairs each | Zero fixed outputs after each restriction and prolongation: 24/24 pass, dimensionless defect ≤1.19e-17. This tests necessary properties, not universal SPD or causation of an earlier GPU failure. | `companion/hex-modal-mg-bc-audit-20260923/{protocol.json,run-01/verification.json}`; exact normalization and saved-array replay in `paper/supplement/provenance-audit.md` |
| A5 | **Damping:** fixed weighted-Jacobi ω=0.6 admits unstable modes at level two on retained designs under this study's exact-Q1 operator contract. | CPU independent spectral diagnostic: 9/9 converged eigen-estimates; exact local bounds on 73 subcell matrices; subsequently a separately declared GPU replay | Weights [0.3,0.25,0.2] have exact rational LDLᵀ local certification for ideal Q1. The matched GPU replay accepts 9/9 saved states versus 3/9 with ω=0.6. This is not a universal mixed-precision convergence proof. | `companion/hex-modal-mg-smoother-audit-20260923/`; `companion/hex-modal-mg-damping-control-20260923/run-03/exact-summary.json`; `companion/hex-modal-mg-damping-gpu-20260923/rental-01/local-verification.json` |

## B. Yang et al. Galerkin GMG release (`nbbllxx0/Mixed-Precision-GMG-SIMP`, `fd9ddecd`)

| # | Finding | Established by | Repair / consequence | Evidence |
|---|---|---|---|---|
| B1 | **Package/path-discovery failure:** `paths.py` requires a `scripts/` directory beside `src/`, absent from the pinned public tree. | Original `ci/smoke_test.py` attempted on RTX 4090, A100 PCIe 80 GB and H100 NVL; nonzero exit before numerical smoke solve | Creating an empty directory allows smoke tests to run and pass. This is a packaging workaround, not repair of a numerical algorithm. RTX smoke: 25 iterations, residual 5.95e-11. | `experiments/combined/donor_yang_gmg/src/gpu_fem/paths.py`; `results/session-f-*/remote/job.json`; `paper/session_f_summary.json` |
| B2 | **The auxiliary E3 loop's OC update collapses the design** (confirmed at full size on two RTX 4090 hosts, session G: volume 0.30 → 0.15 → 0.001, compliance → 6.9e8). E3's own `_oc_update` bisects the multiplier arithmetically over [1e-40, 1e40] for 100 iterations, so it cannot fall below ≈ 7.9e9. Volume went 0.30 → 0.15 → 0.001 and compliance 140 → 1.9e9 on a CPU replica of their cantilever. The paper already treats E3 as auxiliary: "not used as main acceleration evidence because the design trajectories diverge" (§4.4, App. A). It attributes the divergence to capped Jacobi solves, and its E3 compliance plots are at 10⁸–10⁹. Their full SIMP driver (`run_simp`) uses a correct OC (bisection on [0, 1e9] to 1e-9, volume on filtered/projected densities) and is not affected. | CPU replica; paper full text read; full-size diagnostic in session G | Only the auxiliary E3 schedule is affected; the paper's headline claims do not rest on it. Session G uses a top88 OC with a density filter. | session G (`results/session-g-*`); arXiv 2604.26441 §4.4, App. A |
| B3 | **FP64 comparator provenance:** the Galerkin stock core is CuPy gather + DGEMM + `bincount`; its fused kernels are FP32/BF16. | Pinned `solver_v2.py:MatrixFreeKff.matvec` and `cuda_fused_matvec.py` source | The supplied FP64 fused and node paths come from the separate fused-kernel release at `3a6e37af137c3f8ba65ef1d298602a6390da6466`. The FP64 core substitution is an explicit adapter; it does not make the complete solver unmodified. | `experiments/combined/kff_adapters.py`; `experiments/session_f/protocol.json`; comparator symbol/hash table in `paper/supplement/provenance-audit.md` |
| B4 | **Measured allocation footprint:** about 23 GiB per million elements for this pinned Galerkin implementation. 2 M fits, whereas 5 M fails on A100 PCIe 80 GB and H100 NVL 94 GB. | Session F sizing ladder, per schedule/arm, including out-of-memory outcomes | About 3 M is an extrapolated ceiling, not a tested maximum. These records do not fully separate live arrays, transient construction allocations and cached pool memory. Larger studies use the separately specified rediscretized hierarchy. | `experiments/session_f/protocol.json`; `results/session-f-*/remote/run-01/result.json`; `paper/session_f_summary.json` |
| B5 | **Mixed-schedule integration gate:** the stock FP32 fine-smoother schedule reaches 300 iterations on random-density 64 k cantilever states on A100/H100; stock takes 246 on RTX 4090. Other adapted arms also fail parts of the compound gate. | Original and adapted GPU solver execution in Session F; product agreement, convergence, residual and cross-state requirements recorded separately | The gate remains failed on every host despite FP64 core product agreement ~2.2e-16. The donor paper already discloses difficult heterogeneous convergence; this audit localizes failure in its own declared workload, not a new universal defect. Session G changes to SIMP-generated trajectories and records non-converged solves. | `experiments/session_f/protocol.json`; `results/session-f-*/remote/run-01/result.json`; `paper/session_f_summary.json` |

## C. Process and units

[The supplement](../supplement/provenance-audit.md) defines the state-error denominator,
normalized bilinear defect and absolute correction leakage, and reproduces their selected maxima
from hash-verified saved arrays. The three quantities have different units and cannot be compared
as percentages. In particular, the magnitude of absolute fixed-DOF leakage scales with the residual
input; preservation of the free space is the violated invariant.

The [registry](../supplement/hypothesis-registry.md) distinguishes named hypotheses from descriptive
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
