# Changelog

## v1.2 — Supporting-information reorganization

- Rename the supplement generators to `analysis/supplement_*.py` and move their
  sources and outputs to `paper/supplement/` and `provenance/supplement/`; the
  supplement PDF is now `paper/supplement/supporting-information.pdf`.
- Regenerate the supplement, registry and roofline summaries with the renamed
  paths and clarified wording. Numerical results are unchanged.
- Clarify in the README that `JPDC` in historical paths and protocol notes names
  the original local research workspace.

No new GPU experiments were performed. Raw measurements, frozen protocols and
experimental source snapshots are unchanged.

## v1.1 — Corrections and expanded reproducibility documentation

This version corrects numerical reporting and expands the evidence supporting
*Matrix-free voxel elasticity on GPUs: a reproducible kernel and multigrid study
across precisions and hardware*.

- Normalize executed, predicate-enabled FP64 thread instructions per element:
  dense8/plain-parity/mul-add/fused-ai/node are 600/384/240/600/768. The dense8 and
  fused-ai ratios to mul-add remain 2.5.
- Clarify the parity factorization, permutation and normalization, with cubic
  and rectangular element checks.
- Identify the audited rule as composite 10×10×10 midpoint quadrature.
- Include every optimization comparison case, capped exit, timing boundary and
  hardware-specific exception in generated summaries.
- Separate fixed-length/capped trajectories from converged designs and
  same-launch timing accounting from independent predictive validation.
- Add the scientific evidence supplement, accuracy and hardware/software
  manifests, comparator provenance, hypothesis registry and claim/evidence map.
- Read G/H preset metadata without importing the GPU solver or changing donor
  directories, so numerical regeneration can run alongside package tests.
- Update numerical generators, figures, tables, bibliography and reproduction
  instructions. See `paper/supplement/release-validation.md` for checks.

No new GPU experiments were performed. Raw measurements, frozen protocols and
experimental source snapshots are unchanged. The existing raw-evidence deposit,
[10.5281/zenodo.22998404](https://doi.org/10.5281/zenodo.22998404), remains applicable.

## v1.0

Original software artifact:
[10.5281/zenodo.23001863](https://doi.org/10.5281/zenodo.23001863).
