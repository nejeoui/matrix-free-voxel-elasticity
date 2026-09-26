# Reproducibility record for the combined paper

Consolidates the audit findings on the published codes used as baselines and hosts. Each row gives the
finding, how it was established, what repaired it (if anything), and the sealed evidence. Every evidence
entry is a path in this repository: the Träff et al. audit files under `evidence/companion/` (abbreviated
`companion/` below; their protocols and results keep the layout of the earlier project they come from) and the
session folders under `results/`.
Every experiment was declared before execution, and failures are kept.

These findings are about specific pinned releases and paths. They are not claims about other versions,
and they do not say whether the published papers' conclusions hold.

## A. Träff et al. topology-optimization codes (OpenMP GPU and Futhark releases)

| # | Finding | Established by | Repair / consequence | Evidence |
|---|---|---|---|---|
| A1 | The OpenMP release uses midpoint (one-point) integration for its fine element matrix. It differs from the exact trilinear (Q1) matrix by **0.416–0.548 %**. The Futhark matrix matches exact Q1. | Independent exact-Q1 reference on pinned sources (OpenMP `b46ed453`, Futhark `c3cd2b00`) | Baselines in this study use exact 2×2×2 Gauss Q1. Midpoint-based comparisons are labelled source-matched. | `companion/hex-modal-priorart-20260923/` (`cpu-protocol-02.json`, `cpu-02/results.json`) |
| A2 | Both releases run mixed precision (FP32 vectors, correction or CG in parts), not all-FP64. The pinned trees contain no licence file. | Source trace | All-FP64 claims use separately declared adaptations. No redistribution permission is inferred. | pinned sources (commits above) |
| A3 | **Boundary contract:** a finest-level replacement that honours the published operator's identity rows exactly exposed state errors up to **12.17 %**. The unmodified source leaves constrained values up to **5.27e-7** rather than exactly zero. Free-DOF product probes missed this. | 18 captured solves on three meshes; all 36 product probes pass; all 18 state gates fail | A shared projection of returned constrained values restores **18/18** states, 9/9 paired and 3/3 endpoint comparisons. It is labelled an adaptation. | `companion/hex-modal-mg-host-20260923/` (failure), `companion/hex-modal-mg-host-v2-20260923/` (adapted) |
| A4 | **Multigrid transfers are not boundary-consistent:** the published V-cycle fails **24/24** free-space and symmetry probes (bilinear defect up to 1.14e-5, fixed-DOF leakage up to 66.6). | Unmodified C `VcyclePreconditioner` executed on the CPU | Zeroing fixed DOFs after each restriction and prolongation passes 24/24 (defect ≤ 1.19e-17). This is the usual projection, not claimed as new. | `companion/hex-modal-mg-bc-audit-20260923/` |
| A5 | **Damping:** fixed weighted-Jacobi ω = 0.6 admits unstable smoothing modes (level two) on retained designs. | 9/9 converged eigen-estimates; exact local bounds on 73 subcell matrices | Conservative weights [0.3, 0.25, 0.2] are certified by exact rational LDLᵀ for ideal Q1. The matched GPU replay of a failed large solve accepts **9/9** states, against 3/9 with ω = 0.6. | `companion/hex-modal-mg-smoother-audit-20260923/` (spectrum), `companion/hex-modal-mg-damping-control-20260923/` (exact bounds, `run-03/exact-summary.json`), `companion/hex-modal-mg-damping-gpu-20260923/` (GPU replay) |

## B. Yang et al. Galerkin GMG release (`nbbllxx0/Mixed-Precision-GMG-SIMP`, `fd9ddecd`)

| # | Finding | Established by | Repair / consequence | Evidence |
|---|---|---|---|---|
| B1 | **The release fails its own smoke test as published.** `paths.py` requires a `scripts/` directory beside `src/`, and the public repository has none. | Unmodified `ci/smoke_test.py` on RTX 4090, A100 80 GB and H100 NVL | An empty `scripts/` directory makes it pass (25 iterations, residual 5.95e-11 on the RTX 4090). | session F (`results/session-f-*`) |
| B2 | **The auxiliary E3 loop's OC update collapses the design** (confirmed at full size on two RTX 4090 hosts, session G: volume 0.30 → 0.15 → 0.001, compliance → 6.9e8). E3's own `_oc_update` bisects the multiplier arithmetically over [1e-40, 1e40] for 100 iterations, so it cannot fall below ≈ 7.9e9. Volume went 0.30 → 0.15 → 0.001 and compliance 140 → 1.9e9 on a CPU replica of their cantilever. The paper already treats E3 as auxiliary: "not used as main acceleration evidence because the design trajectories diverge" (§4.4, App. A). It attributes the divergence to capped Jacobi solves, and its E3 compliance plots are at 10⁸–10⁹. Their full SIMP driver (`run_simp`) uses a correct OC (bisection on [0, 1e9] to 1e-9, volume on filtered/projected densities) and is not affected. | CPU replica; paper full text read; full-size diagnostic in session G | Only the auxiliary E3 schedule is affected; the paper's headline claims do not rest on it. Session G uses a top88 OC with a density filter. | session G (`results/session-g-*`); arXiv 2604.26441 §4.4, App. A |
| B3 | Their FP64 fine operator is CuPy gather + DGEMM + `bincount`, not fused. Their fused kernels are FP32/BF16 only. | Code reading | FP64 comparisons use the strongest published FP64 kernel (`fused_ai_fp64`) as the baseline (declared in the session F protocol). | `experiments/session_f/protocol.json` |
| B4 | **Memory:** the GMG needs about **23 GiB per million elements**. 2 M elements fits in 80 GB; 5 M does not, on either the A100 80 GB or the H100 NVL 94 GB. | Session F sizing ladder | Scale beyond about 3 M elements (≈ 10 M DOFs) needs a leaner multigrid. | session F (`results/session-f-*`) |
| B5 | Their default mixed schedule (FP32 fused fine smoother) did not converge within 300 iterations on a random-density 64 k cantilever with their own operator, on the A100 and H100. It took 246 iterations on the RTX 4090. | Session F integration stage | Session G uses SIMP-generated designs and records every non-converged solve as a failure. | session F (`results/session-f-*`) |

## C. Process record

- Every GPU session was declared first (protocol SHA-256, pinned package), then run on rentals that
  were destroyed after hash-verified collection.
- Operational failures (upload timeouts, provider rate limits, SSH failures, hosts that could not be
  rented) are kept as their own attempts beside the successful ones, for example the seven attempts of
  session J (`results/session-j-*`).
- Numerical thresholds were never relaxed after a failure. Adaptations are declared as new protocols
  and labelled as such.
