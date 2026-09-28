# Hypothesis and gate registry

Five named hypotheses lack support: H3, H5, H11, H11b, H15. H5 retains its original not-supported label; H11b is a separate changed-pipeline hypothesis. Descriptive H12/H12b expectations and packaging/integration gates are separate. This count is for Sessions A-J, not every exploratory companion study.

Hashes identify content, not temporal priority. These dates are author/system recorded timestamps, checked against byte-linked execution records where available, not independent preregistration timestamps. No local git history is present in this revision workspace.

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
