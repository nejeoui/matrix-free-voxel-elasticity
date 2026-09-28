# v1.1 validation

Validation date: 2026-09-28. The changed-file release package was applied to a
fresh public-repository export at commit
`a60700fe9df55d4b2e597e4a4014cee189ac5e43` and checked without private manuscript
sources or additional raw-evidence downloads. A subsequent check used a fresh
virtual environment in the actual Git checkout, including all current Session
A–I CPU suites. The commands below run from the repository root with
`requirements.txt` installed.

| Command | Result |
|---|---|
| `python -m pytest tests -q` | 22 passed, including five preset-metadata regression checks |
| `python -m pytest experiments/session_a/test_session_a.py -q` | 12 passed |
| `python -m pytest experiments/session_b/test_extended.py -q` | 4 passed |
| `python -m pytest experiments/session_c/test_extended.py -q` | 6 passed |
| `python -m pytest experiments/session_d/test_session_d.py -q` | 6 passed |
| `python -m pytest experiments/session_e/test_session_e.py -q` | 9 passed; six expected overflow/invalid-value warnings from the extreme-value corpus |
| `python -m pytest experiments/session_f/test_session_f.py -q` | 5 passed |
| `python -m pytest experiments/session_g/test_session_g.py -q` | 5 passed |
| `python -m pytest experiments/session_h/test_session_h.py -q` | 5 passed |
| `python -m pytest experiments/session_i/test_session_i.py -q` | 5 passed |
| `PY=python sh paper/combined/build.sh` | Numerical regeneration passed |
| `python paper/revision/build_supplement.py` | Supporting PDF regenerated with Pandoc and Tectonic; text bounds and missing glyphs checked |

All 79 current CPU tests pass. Session test modules reuse names such as `analysis` and `build_package`; run
each suite in a separate pytest invocation, as above. Combining those suites
in one Python process causes import-cache collisions.

An initial concurrent test/build run exposed a temporary-directory race in the
G/H analysis: loading preset metadata imported the donor GPU package and
created/removed its `scripts` directory, also used by package tests. The public
analysis now reads literal preset metadata without loading that package or
mutating its source tree. The final concurrent check validates this correction.

Archived tests under `code/` preserve their historical layout assumptions and
are separate from the current suites above. Direct runs fail where the original
`rescope` paths or C reference library are absent. In a temporary historical
layout, using byte-identical tests and hash-verified retained reference sources
and receipts, 20 checks passed (layout 5, profile 9, readiness 6). One readiness
source-pin test could not run because its historical `hex_modal_gpu_application`
source/protocol is not retained in this repository. No frozen test or evidence
file was edited to bypass that limitation.

The public numerical build regenerates algebra invariants, independent-check
summaries, optimization/scale tables, hardware counters, hypothesis outcomes,
LaTeX macros and figures. All 26 supplied generated JSON/CSV/TeX files checked
against the clean regeneration were byte-identical. The outcomes audit checks
122 source hashes, 314 optimization records, 24 capped exits, 132 scale solves
and 20 allocation identities. The extraction manifests verify all newly
included provenance inputs. Historical measurements, frozen protocols and
experimental source snapshots were preserved.

These checks are CPU computations, retained-record analyses and document
builds. No fresh GPU timing experiment or new saved-state GPU replay is claimed.
The Python environment used NumPy, SciPy, Matplotlib and pytest; the optional
supplement PDF additionally requires Pandoc, Tectonic and its selected fonts.
