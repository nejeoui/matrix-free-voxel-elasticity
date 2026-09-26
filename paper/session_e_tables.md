# Session E: butterfly variants on two RTX 4090 hosts

## host1: machine 51613 (AMD EPYC 7742 64-Core Processor, The Netherlands, NL), instance 52578700

evidence valid True; H8 True; H9 True; max variant-vs-modal8 relative L2 9.3e-17

| case | product: X / modal8_muladd | median | lower 95 | X ms | muladd ms |
|---|---|---:|---:|---:|---:|
| optimized-q64 | fused_ai_fp64 | 2.169 | 2.168 | 0.470 | 0.217 |
| optimized-q64 | fused_fp64 | 2.221 | 2.218 | 0.482 | 0.217 |
| optimized-q64 | node_fp64 | 2.912 | 2.910 | 0.631 | 0.217 |
| optimized-q64 | dense8 | 2.262 | 2.258 | 0.490 | 0.217 |
| optimized-q64 | modal8 | 1.640 | 1.638 | 0.356 | 0.217 |
| optimized-q64 | modal8_signbit | 1.045 | 1.041 | 0.226 | 0.217 |
| optimized-q96 | fused_ai_fp64 | 2.198 | 2.196 | 1.577 | 0.718 |
| optimized-q96 | fused_fp64 | 2.239 | 2.238 | 1.607 | 0.718 |
| optimized-q96 | node_fp64 | 2.908 | 2.907 | 2.088 | 0.718 |
| optimized-q96 | dense8 | 2.287 | 2.286 | 1.642 | 0.718 |
| optimized-q96 | modal8 | 1.656 | 1.655 | 1.189 | 0.718 |
| optimized-q96 | modal8_signbit | 1.047 | 1.045 | 0.750 | 0.718 |
| uniform-q64 | fused_ai_fp64 | 2.165 | 2.164 | 0.470 | 0.217 |
| uniform-q64 | fused_fp64 | 2.215 | 2.214 | 0.481 | 0.217 |
| uniform-q64 | node_fp64 | 2.904 | 2.903 | 0.631 | 0.217 |
| uniform-q64 | dense8 | 2.258 | 2.257 | 0.490 | 0.217 |
| uniform-q64 | modal8 | 1.637 | 1.636 | 0.356 | 0.217 |
| uniform-q64 | modal8_signbit | 1.043 | 1.042 | 0.226 | 0.217 |

| case | multigrid fixed work: arm / mg64-modal8_muladd | median | lower 95 | identical work |
|---|---|---:|---:|---|
| optimized-q64 | mg64-fused_ai | 1.178 | 1.171 | True |
| optimized-q64 | mg64-fused | 1.229 | 1.215 | True |
| optimized-q64 | mg64-node | 1.346 | 1.342 | True |
| optimized-q64 | mg64-dense8 | 1.185 | 1.181 | True |
| optimized-q64 | mg64-modal8 | 1.092 | 1.086 | True |
| optimized-q64 | mg64-modal8_signbit | 1.011 | 1.003 | True |
| optimized-q64 | mg32-node64-node32 | 0.934 | 0.930 | True |
| optimized-q64 | mg32-fused_ai64-node32 | 0.917 | 0.913 | True |
| optimized-q64 | mg32-modal8-node32 | 0.897 | 0.886 | True |
| optimized-q64 | mg32-modal8_muladd-node32 | 0.878 | 0.875 | True |
| optimized-q96 | mg64-fused_ai | 1.368 | 1.368 | True |
| optimized-q96 | mg64-fused | 1.383 | 1.382 | True |
| optimized-q96 | mg64-node | 1.579 | 1.578 | True |
| optimized-q96 | mg64-dense8 | 1.396 | 1.395 | True |
| optimized-q96 | mg64-modal8 | 1.202 | 1.201 | True |
| optimized-q96 | mg64-modal8_signbit | 1.015 | 1.014 | True |
| optimized-q96 | mg32-node64-node32 | 0.621 | 0.617 | True |
| optimized-q96 | mg32-fused_ai64-node32 | 0.596 | 0.593 | True |
| optimized-q96 | mg32-modal8-node32 | 0.581 | 0.577 | True |
| optimized-q96 | mg32-modal8_muladd-node32 | 0.557 | 0.552 | True |
| uniform-q64 | mg64-fused_ai | 1.185 | 1.178 | True |
| uniform-q64 | mg64-fused | 1.224 | 1.215 | True |
| uniform-q64 | mg64-node | 1.357 | 1.347 | True |
| uniform-q64 | mg64-dense8 | 1.191 | 1.185 | True |
| uniform-q64 | mg64-modal8 | 1.096 | 1.089 | True |
| uniform-q64 | mg64-modal8_signbit | 1.010 | 1.002 | True |
| uniform-q64 | mg32-node64-node32 | 0.938 | 0.935 | True |
| uniform-q64 | mg32-fused_ai64-node32 | 0.926 | 0.917 | True |
| uniform-q64 | mg32-modal8-node32 | 0.904 | 0.901 | True |
| uniform-q64 | mg32-modal8_muladd-node32 | 0.892 | 0.879 | True |

| case | arm | time to 1e-8 (s) | iterations | all passed |
|---|---|---:|---|---|
| optimized-q64 | mg32-modal8_muladd-node32 | 0.0925 | [13] | True |
| optimized-q64 | mg32-modal8-node32 | 0.0943 | [13] | True |
| optimized-q64 | mg32-fused_ai64-node32 | 0.0976 | [13] | True |
| optimized-q64 | mg32-node64-node32 | 0.1014 | [13] | True |
| optimized-q64 | mg64-modal8_muladd | 0.1019 | [13] | True |
| optimized-q64 | mg64-modal8_signbit | 0.1021 | [13] | True |
| optimized-q64 | mg64-modal8 | 0.1121 | [13] | True |
| optimized-q64 | mg64-fused_ai | 0.1221 | [13] | True |
| optimized-q64 | mg64-dense8 | 0.1236 | [13] | True |
| optimized-q64 | mg64-fused | 0.1269 | [13] | True |
| optimized-q64 | mg64-node | 0.1404 | [13] | True |
| optimized-q96 | mg32-modal8_muladd-node32 | 0.3260 | [24] | True |
| optimized-q96 | mg32-modal8-node32 | 0.3479 | [24] | True |
| optimized-q96 | mg32-fused_ai64-node32 | 0.3689 | [24] | True |
| optimized-q96 | mg32-node64-node32 | 0.3966 | [24] | True |
| optimized-q96 | mg64-modal8_muladd | 0.5443 | [24] | True |
| optimized-q96 | mg64-modal8_signbit | 0.5528 | [24] | True |
| optimized-q96 | mg64-modal8 | 0.6582 | [24] | True |
| optimized-q96 | mg64-fused_ai | 0.7516 | [24] | True |
| optimized-q96 | mg64-fused | 0.7597 | [24] | True |
| optimized-q96 | mg64-dense8 | 0.7667 | [24] | True |
| optimized-q96 | mg64-node | 0.8701 | [24] | True |
| uniform-q64 | mg32-modal8_muladd-node32 | 0.0629 | [9] | True |
| uniform-q64 | mg32-modal8-node32 | 0.0648 | [9] | True |
| uniform-q64 | mg32-fused_ai64-node32 | 0.0676 | [9] | True |
| uniform-q64 | mg32-node64-node32 | 0.0711 | [9] | True |
| uniform-q64 | mg64-modal8_muladd | 0.0712 | [9] | True |
| uniform-q64 | mg64-modal8_signbit | 0.0714 | [9] | True |
| uniform-q64 | mg64-modal8 | 0.0792 | [9] | True |
| uniform-q64 | mg64-fused_ai | 0.0857 | [9] | True |
| uniform-q64 | mg64-dense8 | 0.0877 | [9] | True |
| uniform-q64 | mg64-fused | 0.0906 | [9] | True |
| uniform-q64 | mg64-node | 0.0982 | [9] | True |

outer-kernel effect (mg32-fused_ai64-node32 / mg32-modal8_muladd-node32): optimized-q64 1.044 (lb 1.041), optimized-q96 1.074 (lb 1.064), uniform-q64 1.025 (lb 1.016)

## host2: machine 120270 (AMD Ryzen Threadripper 2950X 16-Core Processor, France, FR), instance 52580099

evidence valid True; H8 True; H9 True; max variant-vs-modal8 relative L2 9.1e-17

| case | product: X / modal8_muladd | median | lower 95 | X ms | muladd ms |
|---|---|---:|---:|---:|---:|
| optimized-q64 | fused_ai_fp64 | 2.221 | 2.220 | 0.610 | 0.275 |
| optimized-q64 | fused_fp64 | 2.260 | 2.260 | 0.621 | 0.275 |
| optimized-q64 | node_fp64 | 2.987 | 2.985 | 0.821 | 0.275 |
| optimized-q64 | dense8 | 2.313 | 2.313 | 0.636 | 0.275 |
| optimized-q64 | modal8 | 1.678 | 1.678 | 0.461 | 0.275 |
| optimized-q64 | modal8_signbit | 1.064 | 1.064 | 0.293 | 0.275 |
| optimized-q96 | fused_ai_fp64 | 2.248 | 2.247 | 2.046 | 0.910 |
| optimized-q96 | fused_fp64 | 2.280 | 2.279 | 2.075 | 0.910 |
| optimized-q96 | node_fp64 | 2.984 | 2.983 | 2.716 | 0.910 |
| optimized-q96 | dense8 | 2.343 | 2.342 | 2.132 | 0.910 |
| optimized-q96 | modal8 | 1.697 | 1.696 | 1.544 | 0.910 |
| optimized-q96 | modal8_signbit | 1.067 | 1.066 | 0.971 | 0.910 |
| uniform-q64 | fused_ai_fp64 | 2.220 | 2.220 | 0.610 | 0.275 |
| uniform-q64 | fused_fp64 | 2.260 | 2.259 | 0.621 | 0.275 |
| uniform-q64 | node_fp64 | 2.986 | 2.986 | 0.821 | 0.275 |
| uniform-q64 | dense8 | 2.314 | 2.313 | 0.636 | 0.275 |
| uniform-q64 | modal8 | 1.679 | 1.678 | 0.461 | 0.275 |
| uniform-q64 | modal8_signbit | 1.064 | 1.064 | 0.293 | 0.275 |

| case | multigrid fixed work: arm / mg64-modal8_muladd | median | lower 95 | identical work |
|---|---|---:|---:|---|
| optimized-q64 | mg64-fused_ai | 1.279 | 1.272 | True |
| optimized-q64 | mg64-fused | 1.331 | 1.324 | True |
| optimized-q64 | mg64-node | 1.507 | 1.504 | True |
| optimized-q64 | mg64-dense8 | 1.309 | 1.300 | True |
| optimized-q64 | mg64-modal8 | 1.121 | 1.117 | True |
| optimized-q64 | mg64-modal8_signbit | 1.015 | 1.010 | True |
| optimized-q64 | mg32-node64-node32 | 0.963 | 0.936 | True |
| optimized-q64 | mg32-fused_ai64-node32 | 0.918 | 0.913 | True |
| optimized-q64 | mg32-modal8-node32 | 0.900 | 0.890 | True |
| optimized-q64 | mg32-modal8_muladd-node32 | 0.874 | 0.867 | True |
| optimized-q96 | mg64-fused_ai | 1.425 | 1.425 | True |
| optimized-q96 | mg64-fused | 1.437 | 1.437 | True |
| optimized-q96 | mg64-node | 1.669 | 1.669 | True |
| optimized-q96 | mg64-dense8 | 1.459 | 1.458 | True |
| optimized-q96 | mg64-modal8 | 1.238 | 1.237 | True |
| optimized-q96 | mg64-modal8_signbit | 1.023 | 1.022 | True |
| optimized-q96 | mg32-node64-node32 | 0.607 | 0.606 | True |
| optimized-q96 | mg32-fused_ai64-node32 | 0.581 | 0.579 | True |
| optimized-q96 | mg32-modal8-node32 | 0.558 | 0.557 | True |
| optimized-q96 | mg32-modal8_muladd-node32 | 0.533 | 0.532 | True |
| uniform-q64 | mg64-fused_ai | 1.258 | 1.257 | True |
| uniform-q64 | mg64-fused | 1.311 | 1.309 | True |
| uniform-q64 | mg64-node | 1.486 | 1.482 | True |
| uniform-q64 | mg64-dense8 | 1.286 | 1.284 | True |
| uniform-q64 | mg64-modal8 | 1.113 | 1.107 | True |
| uniform-q64 | mg64-modal8_signbit | 1.008 | 1.006 | True |
| uniform-q64 | mg32-node64-node32 | 0.941 | 0.938 | True |
| uniform-q64 | mg32-fused_ai64-node32 | 0.913 | 0.910 | True |
| uniform-q64 | mg32-modal8-node32 | 0.896 | 0.892 | True |
| uniform-q64 | mg32-modal8_muladd-node32 | 0.874 | 0.872 | True |

| case | arm | time to 1e-8 (s) | iterations | all passed |
|---|---|---:|---|---|
| optimized-q64 | mg32-modal8_muladd-node32 | 0.1013 | [13] | True |
| optimized-q64 | mg32-modal8-node32 | 0.1051 | [13] | True |
| optimized-q64 | mg32-fused_ai64-node32 | 0.1087 | [13] | True |
| optimized-q64 | mg64-modal8_muladd | 0.1141 | [13] | True |
| optimized-q64 | mg32-node64-node32 | 0.1150 | [13] | True |
| optimized-q64 | mg64-modal8_signbit | 0.1155 | [13] | True |
| optimized-q64 | mg64-modal8 | 0.1292 | [13] | True |
| optimized-q64 | mg64-fused_ai | 0.1485 | [13] | True |
| optimized-q64 | mg64-dense8 | 0.1516 | [13] | True |
| optimized-q64 | mg64-fused | 0.1544 | [13] | True |
| optimized-q64 | mg64-node | 0.1753 | [13] | True |
| optimized-q96 | mg32-modal8_muladd-node32 | 0.3580 | [24] | True |
| optimized-q96 | mg32-modal8-node32 | 0.3839 | [24] | True |
| optimized-q96 | mg32-fused_ai64-node32 | 0.4101 | [24] | True |
| optimized-q96 | mg32-node64-node32 | 0.4429 | [24] | True |
| optimized-q96 | mg64-modal8_muladd | 0.6210 | [24] | True |
| optimized-q96 | mg64-modal8_signbit | 0.6356 | [24] | True |
| optimized-q96 | mg64-modal8 | 0.7731 | [24] | True |
| optimized-q96 | mg64-fused_ai | 0.8930 | [24] | True |
| optimized-q96 | mg64-fused | 0.9010 | [24] | True |
| optimized-q96 | mg64-dense8 | 0.9143 | [24] | True |
| optimized-q96 | mg64-node | 1.0498 | [24] | True |
| uniform-q64 | mg32-modal8_muladd-node32 | 0.0705 | [9] | True |
| uniform-q64 | mg32-modal8-node32 | 0.0740 | [9] | True |
| uniform-q64 | mg32-fused_ai64-node32 | 0.0771 | [9] | True |
| uniform-q64 | mg64-modal8_muladd | 0.0802 | [9] | True |
| uniform-q64 | mg32-node64-node32 | 0.0807 | [9] | True |
| uniform-q64 | mg64-modal8_signbit | 0.0809 | [9] | True |
| uniform-q64 | mg64-modal8 | 0.0900 | [9] | True |
| uniform-q64 | mg64-fused_ai | 0.1027 | [9] | True |
| uniform-q64 | mg64-dense8 | 0.1051 | [9] | True |
| uniform-q64 | mg64-fused | 0.1068 | [9] | True |
| uniform-q64 | mg64-node | 0.1216 | [9] | True |

outer-kernel effect (mg32-fused_ai64-node32 / mg32-modal8_muladd-node32): optimized-q64 1.050 (lb 1.041), optimized-q96 1.087 (lb 1.079), uniform-q64 1.050 (lb 1.041)

## Companion evidence (different harness and memory layout)

| kernel | DADD | DMUL | DFMA | FP64 sites |
|---|---:|---:|---:|---:|
| modal | 36 | 9 | 7 | 52 |
| dense | 0 | 5 | 73 | 78 |
| modal_signbit | 18 | 9 | 7 | 34 |
| symmetric_half | 0 | 9 | 70 | 79 |
| modal_signed_muladd | 0 | 9 | 25 | 34 |

| control / modal_signed_muladd (evidence/companion/.../hex-modal-libceed-resident-20260924) | min | max |
|---|---:|---:|
| dense8 | 1.85 | 2.15 |
| symmetric_half | 1.83 | 2.12 |
| ceed_ref | 13.12 | 19.22 |
| ceed_shared | 14.92 | 21.40 |
| ceed_gen | 2.58 | 3.01 |

replication: {'hosts': 2, 'h8_replicated': True, 'h9_replicated': True}
