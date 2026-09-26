# Arithmetic-control attribution

The signed multiply-add form follows the warp butterfly in Tri Dao / Dao AI Lab, `fast-hadamard-transform`, commit `e7706faf8d1c3b9f241e36860640ad1dac644ede`, `csrc/fast_hadamard_transform_common.h`, lines 108–123. The original copyright is (c) 2023 Tri Dao; the repository BSD-3-Clause notice is retained in DAO-LICENSE.

[Commit-pinned upstream source](https://github.com/Dao-AILab/fast-hadamard-transform/blob/e7706faf8d1c3b9f241e36860640ad1dac644ede/csrc/fast_hadamard_transform_common.h#L108-L123).

This independent FP64 adaptation uses eight-lane shuffle width and the existing finite-element kernel body. It is not the upstream package, which uses float arithmetic in the inspected helper. There is no novelty claim for signed multiplication or the Hadamard butterfly. CUDA compilation and execution require a separate declared GPU test.
