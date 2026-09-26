# Solve-profile readiness audit

Part of experiment `exp04-solve-profile`. A CPU-only audit of the saved kernel evidence before the
solve-profile measurement; it executes no CUDA and establishes no solver speedup.

The archived product batches do not identify the product fraction of a complete
physical solve. Under unchanged work, unchanged other costs, and a kernel ratio
that transfers from the synthetic benchmark, the conditional episode ratio is

`S = 1 / (1 - f + f/r + d)`.

Here `f` is product time divided by the baseline's complete charged episode and
`d` is the candidate's additional cost divided by that same baseline time.
With `d=0`, the RTX 4090 observations require `f >= 0.441746` and `0.497471`
for a 1.10 ratio at the two primary sizes. Neither share had been measured, so the product timings alone
could not predict the effect on a complete solve; this does not refute the kernel or its product-time result.
The solve-profile experiment therefore measured the product share on physical states directly.

The audit replays all 40 archived timing comparisons and checks all 144 normal solver records. These are
record checks, not a new finite-element replay of the GPU solves. Its outputs are in `evidence/readiness/`.

## Files

`analyze.py` performs the audit, with unit tests in `test_analysis.py`; `verify_inputs.py` checks the
prepared inputs; `reconcile_costs.py` reconciled rental receipts. The code is kept as it was executed, so it
refers to the paths of the machine it ran on.
