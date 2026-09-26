# Rectangular Hex8 parity-block action (CPU screen)

Part of experiment `exp01-parity-algebra`. This is the code of the CPU screen that preceded the GPU kernel.
Declared in [protocol.json](protocol.json) before execution, it passed all 15 saved states, 15
cell-geometry/material controls, six assembled-product controls and three intentional-error controls. It is a
CPU correctness check only: it measures no GPU time.

## Algebra and scope

For each node let its three binary coordinates be `n`, and define
`H[p,n] = (-1)^popcount(p & n)`. For three displacement components, use
`T = H ⊗ I3`, with columns permuted into the original element-node order.
Then `T Tᵀ = 8 I`. Store `C = T K Tᵀ / 64`, so that
`K u = Tᵀ C T u`. Each component transform uses a three-stage butterfly;
normalization is already included in `C`.

On an axis-aligned rectangular cell with isotropic elasticity, reflection in
coordinate `a` multiplies displacement mode `(p,c)` by
`(-1)^(p_a + [a=c])`. The integrated elastic-energy bilinear form is invariant
under each coordinate reflection. Two modes with different reflection signs
therefore have zero coupling. Grouping by `g = p XOR (1 << c)` gives eight
independent groups of three modes, or eight 3×3 blocks. All 24 modes are retained;
the six rigid modes remain null modes. This is full integration and a change of
basis, not removal of physical modes or a change in material floor.

Floating-point quadrature leaves tiny off-block entries. The implementation
rejects an input if their relative Frobenius norm exceeds `2e-14`, and records
the discarded norm. On the pinned element matrix it is about `1.96e-16`.
This bound does not authorize use on arbitrary distorted, rotated or anisotropic
elements. Original constraints, loads, element moduli and FP64 scatter are retained.

The static work model counts an FMA as two operations. Dense multiplication
uses 576 FMAs; two butterflies use 144 additions and eight dense 3×3 blocks use
72 FMAs. Thus the counted work is 288 versus 1,152, excluding common material
scaling, indexing, movement and scatter. That ratio is not a runtime prediction.
GPU shuffles, occupancy, FP64 throughput and atomics can dominate.

## Files

`modal.py` builds the parity-block matrices and the butterfly action; `run.py` executes the screen. The code
is kept as it was executed; its saved input states and pinned reference build are not part of this
repository. The algebra is re-checked on the CPU by `tests/` in this repository, and the related algebra
evidence is in `evidence/algebra/`.
