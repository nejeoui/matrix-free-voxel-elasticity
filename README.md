# Matrix-free voxel elasticity on GPUs

Artifact repository of the paper **"Matrix-free voxel elasticity on GPUs: a reproducible kernel and
multigrid study across precisions and hardware"** by Abderrazzak Nejeoui (corresponding author,
a.nejeoui@uca.ac.ma, ORCID [0009-0000-6004-1777](https://orcid.org/0009-0000-6004-1777)) and Aissam Bekkari (ORCID
[0009-0000-1400-6962](https://orcid.org/0009-0000-1400-6962)), National School of Applied Sciences, Cadi Ayyad University, Marrakesh, Morocco. It contains the code, the frozen protocols,
the retained evidence and scripts for numerical regeneration of the paper's results, tables and
figures. This CPU regeneration path uses the files in this repository; larger saved-state inputs
and source snapshots are mapped to the existing raw-evidence deposit below.

## Historical release tags

The artifact organizes experiments with the tags below, in paper order. These tags
identify the retained experimental snapshots. Release corrections and supporting
documentation are described in [CHANGELOG.md](CHANGELOG.md); the original v1.0
software archive remains available at
[doi:10.5281/zenodo.23001863](https://doi.org/10.5281/zenodo.23001863).
Use the release-specific citation for the version you use.

| tag | paper section | content |
|---|---|---|
| `base` | – | FP64 kernels (`src/`), CPU lane tests, licences, import provenance |
| `exp01-parity-algebra` | §3 | exactness of the parity-block Hex8 decomposition |
| `exp02-products-rtx4090` | §6.1 | complete FP64 products against the published kernels, RTX 4090 |
| `exp03-products-rtx3090` | §6.1 | independent RTX 3090 replication |
| `exp04-solve-profile` | §6.3 | warm physical solves, product share, readiness audit |
| `exp05-product-layout` | §6.1 | symmetric-half and warp-per-element layout controls |
| `exp06-roofline` | §4 | roofline predictions declared before measurement |
| `exp07-session-a` | §6.1, §6.3 | session A: products, Jacobi-PCG, refinement (RTX 4090) |
| `exp08-session-b` | §6.5 | session B: A100 80 GB, the failed full-rate FP64 prediction |
| `exp09-session-c` | §6.3 | session C: mixed-precision refinement ranking (all attempts) |
| `exp10-session-d` | §6.4 | session D: kernel ranking inside multigrid-preconditioned CG |
| `exp11-session-e` | §3, §6.2 | session E: butterfly forms on two hosts; static SASS counts; resident libCEED comparison |
| `tools-lifecycle` | – | rental lifecycle tools (create, verify, collect, destroy) |
| `exp12-session-f` | §7 | session F: feasibility probe on RTX 4090, A100, H100; Galerkin GMG adapter |
| `exp13-session-g` | §7 | session G: kernel inside a published Galerkin multigrid, full SIMP runs |
| `exp14-session-h` | §7 | session H: GPU-resident filter and OC, 100-step runs |
| `exp15-session-i` | §8 | session I: scale to 144.6 M DOFs on RTX 4090, A100, H100 |
| `exp16-session-j` | §9 | session J: Nsight Compute counters in a VM (all seven attempts) |
| `exp17-traff-audit` | §10 | reproducibility audit of the Träff et al. multigrid codes |
| `paper` | all | generated numbers, tables and figures |

Each experiment retains its declared protocol (`experiments/session_*/protocol.json` or `evidence/*/protocol.json`),
executed code, collected results (`results/`) and lifecycle records. Independent saved-state CPU replays,
where performed, are recorded in `local-verification.json`; verification coverage is session-specific
and documented in [the accuracy supplement](paper/supplement/methods-accuracy.md).
Failed and unexecuted attempts are kept beside successful ones.

## Reproduce the paper

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                 # NumPy, SciPy, Matplotlib, pytest
python -m pytest -q tests                       # CPU checks of the kernel source and lane algebra
PY=python paper/combined/build.sh               # regenerate numerical results, tables and figures
python analysis/supplement_registry.py            # audit units, hypothesis outcomes and dated provenance
```

The build writes result values as LaTeX macros (`paper/combined/combined_numbers.tex`,
`paper/combined/repro_numbers.tex`, `paper/manuscript/numbers.tex`), plus the tables (`paper/manuscript/tab_*.tex`)
and figures (`paper/manuscript/fig_*.pdf`). The [evidence supplement](paper/supplement/supporting-information.pdf) and
[provenance and claim-evidence map](paper/supplement/provenance-audit.md) document these outputs. They include source identities,
audit contracts and units, measured claims and retained failed outcomes, with exact analysis commands.
The numerical build requires only the Python packages in `requirements.txt`.
To rebuild the optional evidence-supplement PDF, additionally install Pandoc and
Tectonic and run `python paper/supplement/build_supplement.py`. This uses the bundled
TeX Gyre Heros and Latin Modern Mono fonts by default; `--main-font` and
`--mono-font` select alternatives.
Validation commands and scope are recorded in
[release-validation.md](paper/supplement/release-validation.md).

The manuscript text is not part of this repository. The public numerical build regenerates every
reported number, table and figure; it does not create the manuscript PDF.

These are three different reproducibility levels:

* **Numerical regeneration** reads saved results and recomputes statistics, tables, macros and figures.
  The commands above do this on a CPU; they do not rerun GPU timing experiments.
* **Saved-state replay** reloads saved densities/displacements and applies independent reference
  checks. Session verifier scripts and `local-verification.json` identify the checked states and
  tolerances. Larger inputs can be restored from the deposited archives and checked against the
  manifests. The supplement registry also recomputes selected audit norms from small extracted arrays.
* **Fresh GPU execution** runs the frozen workload again on identified hardware with its declared
  environment, warmup, timing boundary and accuracy gates. It needs a suitable GPU/CUDA environment
  and any source/large-input restoration specified by that session. Historical launcher paths are
  provenance, not a claim that the CPU build provisions fresh GPU experiments. In historical paths,
  labels and protocol notes, `JPDC` names the original local research workspace (`~/JPDC`) from which
  these experiments were prepared and launched.

The [hypothesis registry](paper/supplement/hypothesis-registry.md) links protocol hashes to dated
execution records when their identities match. Hashes prove content identity, not temporal priority;
self-recorded timestamps are distinguished from independently timestamped preregistration.

## Large files

Files of 5 MB or more are not in git; they are archived on Zenodo,
[doi:10.5281/zenodo.22998404](https://doi.org/10.5281/zenodo.22998404). Nothing in the build above
needs them; they allow the raw evidence to be audited and the independent replays to be rerun.

* `LARGE_FILES.tsv` lists every large file of the session folders (`results/`) with its path, size and SHA-256.
  They are published as the sealed evidence archive of each GPU host (about 6 GB).
* The earlier experiments whose summaries are in `evidence/` and `evidence/companion/` are published as complete
  folders (about 13.5 GB): each sealed evidence archive plus one archive of all other files.
* `provenance/zenodo/` holds the record's manifests: `MANIFEST.tsv` and `COMPANION_MANIFEST.tsv` map every
  Zenodo file to its path, size and SHA-256; `EXCLUDED.tsv` lists every file left out, with the reason
  (byte-identical copies of files inside a sealed archive, Traeff et al. source copies, API-key redactions);
  `REPACKED.json` records the one sealed archive published without the Traeff et al. sources it contained.

## Provenance notes

* `evidence/` and `code/` hold the earlier kernel experiments; `provenance/import-manifest.json` records their
  original location and SHA-256. `evidence/companion/` holds the few files of that earlier project the paper reads
  (static SASS counts, the resident libCEED comparison and the Träff audit), in their original layout;
  `evidence/companion/EXTRACTED.json` records the one member extracted from a sealed archive.
* The sealed protocols, code snapshots and run records keep the internal project names and working labels
  they were written with (for example the earlier project's working name), because their SHA-256 values are
  recorded; the rest of the repository calls that earlier project the *companion project*.
* Historical scripts (package builders, rental launchers, input manifests) keep the absolute paths of the
  machine they ran on, because they document what was executed. Only the reproduction path above was made
  repository-relative.
* Per-instance cloud API keys returned by the provider have been replaced by `REDACTED` in the files listed,
  with their original SHA-256, in `REDACTIONS.md`; the instances were destroyed after collection. Every other
  file is published as recorded.
* Third-party code: Yang et al.'s solver sources under `experiments/*/donor*/` keep their BSD 3-Clause
  licence. The Träff et al. sources are not redistributed (their public trees carry no licence); they are
  referenced by commit.

## Licence

Copyright (c) 2026 Abderrazzak Nejeoui and Aissam Bekkari. The repository is dual-licensed by content type:

* **Code — MIT** (`LICENSE`): all source code and scripts, including `src/`, `tests/`, `analysis/`, `code/`,
  `experiments/` (runners, package builders, lifecycle tools, CUDA kernels), the `make_*.py` generators and the
  `build.sh` scripts.
* **Data and text — CC BY 4.0** (`LICENSE-DATA`): protocols, evidence, results, verification receipts,
  generated numbers and tables, figures, and the documentation
  (`README.md`).

Third-party material keeps its own terms and is **not** covered by either licence:

* Yang et al.'s solver sources in every `donor/` and `donor_yang_gmg/` directory (including the copies inside
  `results/*/remote/source/`): BSD 3-Clause, see the `LICENSE` file beside them.

## Citation

See `CITATION.cff`.
