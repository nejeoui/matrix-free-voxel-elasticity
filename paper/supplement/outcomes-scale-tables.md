# Generated outcome and capacity tables

Regenerate with `python3 analysis/supplement_outcomes.py`. See [methods and interpretation](outcomes-scale.md). NR = not recorded, not zero. CSV/JSON files retain every repetition and exact source record; medians below are descriptive.

## Optimization outcomes

Every row identifies its host, schedule and arm; three fixed-length repetitions are aggregated by the median separately for each numerical column. Single capped trajectories have n=1. Setup is the sum of per-step `gmg_setup_ms`, including density-dependent re-setup; initial construction is included in wall time but not separately recorded. CG/setup/compliance/volume/change columns follow the retained per-step logging convention.

| host | kind | case | schedule | arm | n | exit | steps | Krylov total | wall s | CG s | re-setup s | compliance | volume | change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| session-g-host1-20260926 | fixed_length | cantilever-216k | fp64 | stock | 3 | fixed_length | 30 | 790 | 20.2632 | 6.63421 | 0.852218 | 2.40794 | 0.299996 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-216k | fp64 | fused_ai_fp64 | 3 | fixed_length | 30 | 790 | 16.7415 | 3.14986 | 0.773771 | 2.40794 | 0.299996 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-216k | fp64 | modal8_muladd | 3 | fixed_length | 30 | 790 | 16.5612 | 3.02125 | 0.775102 | 2.40794 | 0.299996 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-216k | mixed | stock | 3 | fixed_length | 30 | 842 | 18.2589 | 4.63008 | 0.852745 | 2.40794 | 0.299996 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-216k | mixed | modal8_muladd | 3 | fixed_length | 30 | 847 | 17.063 | 3.37227 | 0.776675 | 2.40794 | 0.299996 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-216k | mixed | fused_ai_fp64 | 3 | fixed_length | 30 | 846 | 17.0286 | 3.44388 | 0.781636 | 2.40794 | 0.299996 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-512k | fp64 | modal8_muladd | 3 | fixed_length | 30 | 404 | 34.9273 | 2.40579 | 1.986 | 1.96632 | 0.300003 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-512k | fp64 | fused_ai_fp64 | 3 | fixed_length | 30 | 401 | 35.7939 | 3.05646 | 2.01709 | 1.96632 | 0.300003 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-512k | fp64 | stock | 3 | fixed_length | 30 | 401 | 40.0014 | 7.09749 | 2.20572 | 1.96632 | 0.300003 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-512k | mixed | modal8_muladd | 3 | fixed_length | 30 | 411 | 34.8816 | 2.30735 | 1.98625 | 1.96632 | 0.300003 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-512k | mixed | fused_ai_fp64 | 3 | fixed_length | 30 | 413 | 35.1185 | 2.54679 | 2.01711 | 1.96632 | 0.300003 | 0.15 |
| session-g-host1-20260926 | fixed_length | cantilever-512k | mixed | stock | 3 | fixed_length | 30 | 419 | 36.6931 | 3.99922 | 2.20666 | 1.96632 | 0.300003 | 0.15 |
| session-g-host1-20260926 | fixed_length | mbb-514k | fp64 | stock | 3 | fixed_length | 30 | 1829 | 63.5879 | 30.3417 | 2.84191 | 5.79628 | 0.5 | 0.15 |
| session-g-host1-20260926 | fixed_length | mbb-514k | fp64 | fused_ai_fp64 | 3 | fixed_length | 30 | 1848 | 45.8318 | 12.8311 | 2.68122 | 5.79628 | 0.5 | 0.15 |
| session-g-host1-20260926 | fixed_length | mbb-514k | fp64 | modal8_muladd | 3 | fixed_length | 30 | 1836 | 42.9255 | 9.84355 | 2.65498 | 5.79628 | 0.5 | 0.15 |
| session-g-host1-20260926 | fixed_length | mbb-514k | mixed | fused_ai_fp64 | 3 | fixed_length | 30 | 3510 | 63.5615 | 30.4074 | 2.68366 | 5.79628 | 0.5 | 0.15 |
| session-g-host1-20260926 | fixed_length | mbb-514k | mixed | modal8_muladd | 3 | fixed_length | 30 | 3378 | 54.234 | 21.1327 | 2.6564 | 5.79628 | 0.5 | 0.15 |
| session-g-host1-20260926 | fixed_length | mbb-514k | mixed | stock | 3 | fixed_length | 30 | 3589 | 64.7217 | 31.4009 | 2.84372 | 5.79628 | 0.499999 | 0.15 |
| session-g-host1-20260926 | fixed_length | torsion-499k | fp64 | fused_ai_fp64 | 3 | fixed_length | 30 | 1047 | 37.838 | 6.75784 | 1.50804 | 2.0847 | 0.250001 | 0.15 |
| session-g-host1-20260926 | fixed_length | torsion-499k | fp64 | stock | 3 | fixed_length | 30 | 1038 | 47.7006 | 16.4795 | 1.69225 | 2.0847 | 0.250001 | 0.15 |
| session-g-host1-20260926 | fixed_length | torsion-499k | fp64 | modal8_muladd | 3 | fixed_length | 30 | 1020 | 36.1977 | 5.00756 | 1.47768 | 2.0847 | 0.250001 | 0.15 |
| session-g-host1-20260926 | fixed_length | torsion-499k | mixed | modal8_muladd | 3 | fixed_length | 30 | 1170 | 36.5548 | 5.4548 | 1.47747 | 2.0847 | 0.250001 | 0.15 |
| session-g-host1-20260926 | fixed_length | torsion-499k | mixed | stock | 3 | fixed_length | 30 | 1157 | 40.7984 | 9.60556 | 1.6921 | 2.0847 | 0.250001 | 0.15 |
| session-g-host1-20260926 | fixed_length | torsion-499k | mixed | fused_ai_fp64 | 3 | fixed_length | 30 | 1166 | 37.0933 | 5.95928 | 1.50781 | 2.0847 | 0.250001 | 0.15 |
| session-g-host1-20260926 | capped | cantilever-216k | fp64 | stock | 1 | iteration_cap | 150 | 3416 | 80.8094 | 28.8393 | 3.21781 | 2.3702 | 0.3 | 0.148441 |
| session-g-host1-20260926 | capped | cantilever-216k | fp64 | fused_ai_fp64 | 1 | iteration_cap | 150 | 3376 | 65.638 | 13.6702 | 3.19501 | 2.37021 | 0.3 | 0.148433 |
| session-g-host1-20260926 | capped | cantilever-216k | fp64 | modal8_muladd | 1 | iteration_cap | 150 | 3397 | 65.1687 | 13.0684 | 3.25511 | 2.3702 | 0.3 | 0.148441 |
| session-g-host1-20260926 | capped | cantilever-216k | mixed | stock | 1 | iteration_cap | 150 | 3467 | 71.6767 | 19.3597 | 3.23301 | 2.37021 | 0.3 | 0.148455 |
| session-g-host1-20260926 | capped | cantilever-216k | mixed | fused_ai_fp64 | 1 | iteration_cap | 150 | 3539 | 66.7621 | 14.6006 | 3.2437 | 2.3702 | 0.3 | 0.148402 |
| session-g-host1-20260926 | capped | cantilever-216k | mixed | modal8_muladd | 1 | iteration_cap | 150 | 3478 | 65.9237 | 13.9376 | 3.25597 | 2.37021 | 0.3 | 0.148475 |
| session-g-host1-20260926 | auxiliary_diagnostic | cantilever-216k | mixed | stock | 1 | fixed_length | 30 | 339 | 11.8114 | 2.00529 | 1.20152 | 4.05316e+08 | 0.00100003 | 6.24915e-07 |
| session-g-host2-20260926 | fixed_length | cantilever-216k | fp64 | stock | 3 | fixed_length | 30 | 790 | 21.9392 | 6.8944 | 0.894677 | 2.40794 | 0.299996 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-216k | fp64 | fused_ai_fp64 | 3 | fixed_length | 30 | 790 | 17.9811 | 3.22363 | 0.813288 | 2.40794 | 0.299996 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-216k | fp64 | modal8_muladd | 3 | fixed_length | 30 | 790 | 18.0372 | 3.13964 | 0.809907 | 2.40794 | 0.299996 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-216k | mixed | stock | 3 | fixed_length | 30 | 836 | 19.5336 | 4.74325 | 0.888844 | 2.40794 | 0.299996 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-216k | mixed | modal8_muladd | 3 | fixed_length | 30 | 844 | 18.3895 | 3.47133 | 0.809884 | 2.40794 | 0.299996 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-216k | mixed | fused_ai_fp64 | 3 | fixed_length | 30 | 840 | 18.0278 | 3.41807 | 0.807893 | 2.40794 | 0.299996 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-512k | fp64 | modal8_muladd | 3 | fixed_length | 30 | 404 | 37.9168 | 2.49948 | 2.0787 | 1.96632 | 0.300003 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-512k | fp64 | fused_ai_fp64 | 3 | fixed_length | 30 | 403 | 39.1097 | 3.19337 | 2.11371 | 1.96632 | 0.300003 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-512k | fp64 | stock | 3 | fixed_length | 30 | 404 | 43.0509 | 7.37879 | 2.30501 | 1.96632 | 0.300003 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-512k | mixed | modal8_muladd | 3 | fixed_length | 30 | 413 | 37.7677 | 2.39174 | 2.07223 | 1.96632 | 0.300003 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-512k | mixed | fused_ai_fp64 | 3 | fixed_length | 30 | 415 | 38.151 | 2.63664 | 2.10674 | 1.96632 | 0.300003 | 0.15 |
| session-g-host2-20260926 | fixed_length | cantilever-512k | mixed | stock | 3 | fixed_length | 30 | 410 | 40.0374 | 4.07201 | 2.30658 | 1.96632 | 0.300003 | 0.15 |
| session-g-host2-20260926 | fixed_length | mbb-514k | fp64 | stock | 3 | fixed_length | 30 | 1843 | 67.523 | 31.4926 | 2.95461 | 5.79628 | 0.5 | 0.15 |
| session-g-host2-20260926 | fixed_length | mbb-514k | fp64 | fused_ai_fp64 | 3 | fixed_length | 30 | 1827 | 48.7136 | 13.0879 | 2.78025 | 5.79628 | 0.5 | 0.15 |
| session-g-host2-20260926 | fixed_length | mbb-514k | fp64 | modal8_muladd | 3 | fixed_length | 30 | 1820 | 45.7531 | 10.066 | 2.75976 | 5.79628 | 0.5 | 0.15 |
| session-g-host2-20260926 | fixed_length | mbb-514k | mixed | fused_ai_fp64 | 3 | fixed_length | 30 | 3801 | 74.3859 | 38.4941 | 2.78616 | 5.79628 | 0.5 | 0.15 |
| session-g-host2-20260926 | fixed_length | mbb-514k | mixed | modal8_muladd | 3 | fixed_length | 30 | 2905 | 56.5347 | 20.1026 | 2.75041 | 5.79628 | 0.5 | 0.15 |
| session-g-host2-20260926 | fixed_length | mbb-514k | mixed | stock | 3 | fixed_length | 30 | 2949 | 71.7942 | 35.3967 | 2.94465 | 5.79628 | 0.5 | 0.15 |
| session-g-host2-20260926 | fixed_length | torsion-499k | fp64 | fused_ai_fp64 | 3 | fixed_length | 30 | 1034 | 41.3668 | 6.88977 | 1.57593 | 2.0847 | 0.250002 | 0.15 |
| session-g-host2-20260926 | fixed_length | torsion-499k | fp64 | stock | 3 | fixed_length | 30 | 1029 | 51.2874 | 16.846 | 1.76503 | 2.0847 | 0.250002 | 0.15 |
| session-g-host2-20260926 | fixed_length | torsion-499k | fp64 | modal8_muladd | 3 | fixed_length | 30 | 1025 | 39.5788 | 5.22113 | 1.54892 | 2.0847 | 0.250001 | 0.15 |
| session-g-host2-20260926 | fixed_length | torsion-499k | mixed | modal8_muladd | 3 | fixed_length | 30 | 1153 | 40.0446 | 5.57686 | 1.54713 | 2.0847 | 0.250001 | 0.15 |
| session-g-host2-20260926 | fixed_length | torsion-499k | mixed | stock | 3 | fixed_length | 30 | 1162 | 43.8914 | 9.87406 | 1.76712 | 2.0847 | 0.250002 | 0.15 |
| session-g-host2-20260926 | fixed_length | torsion-499k | mixed | fused_ai_fp64 | 3 | fixed_length | 30 | 1171 | 40.2254 | 6.12756 | 1.57321 | 2.0847 | 0.250002 | 0.15 |
| session-g-host2-20260926 | capped | cantilever-216k | fp64 | stock | 1 | iteration_cap | 150 | 3395 | 84.0196 | 29.2236 | 3.32413 | 2.3702 | 0.3 | 0.148439 |
| session-g-host2-20260926 | capped | cantilever-216k | fp64 | fused_ai_fp64 | 1 | iteration_cap | 150 | 3396 | 69.404 | 13.8062 | 3.25901 | 2.3702 | 0.3 | 0.148437 |
| session-g-host2-20260926 | capped | cantilever-216k | fp64 | modal8_muladd | 1 | iteration_cap | 150 | 3402 | 68.6407 | 13.2982 | 3.29533 | 2.3702 | 0.3 | 0.148443 |
| session-g-host2-20260926 | capped | cantilever-216k | mixed | stock | 1 | iteration_cap | 150 | 3585 | 75.7008 | 20.1217 | 3.34383 | 2.3702 | 0.3 | 0.148415 |
| session-g-host2-20260926 | capped | cantilever-216k | mixed | fused_ai_fp64 | 1 | iteration_cap | 150 | 3495 | 71.5652 | 14.987 | 3.34275 | 2.37021 | 0.3 | 0.148368 |
| session-g-host2-20260926 | capped | cantilever-216k | mixed | modal8_muladd | 1 | iteration_cap | 150 | 3463 | 74.9585 | 14.1636 | 3.31915 | 2.37021 | 0.3 | 0.148403 |
| session-g-host2-20260926 | auxiliary_diagnostic | cantilever-216k | mixed | stock | 1 | fixed_length | 30 | 341 | 12.4373 | 2.03677 | 1.25893 | 4.05316e+08 | 0.00100003 | 6.27225e-07 |
| session-h-host1-20260926 | fixed_length | cantilever-216k | fp64 | stock | 3 | fixed_length | 100 | 2257 | 27.6613 | 18.8924 | 2.25191 | 2.37514 | 0.3 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-216k | fp64 | fused_ai_fp64 | 3 | fixed_length | 100 | 2252 | 17.54 | 8.83787 | 2.17239 | 2.37514 | 0.3 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-216k | fp64 | modal8_muladd | 3 | fixed_length | 100 | 2244 | 17.0481 | 8.38323 | 2.16932 | 2.37514 | 0.3 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-216k | mixed | stock | 3 | fixed_length | 100 | 2316 | 21.3463 | 12.575 | 2.25667 | 2.37514 | 0.3 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-216k | mixed | modal8_muladd | 3 | fixed_length | 100 | 2323 | 17.6743 | 9.01827 | 2.16004 | 2.37514 | 0.3 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-216k | mixed | fused_ai_fp64 | 3 | fixed_length | 100 | 2354 | 17.9315 | 9.32175 | 2.18564 | 2.37514 | 0.3 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-512k | fp64 | modal8_muladd | 3 | fixed_length | 100 | 1244 | 26.2724 | 7.51577 | 6.24514 | 1.94243 | 0.299999 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-512k | fp64 | fused_ai_fp64 | 3 | fixed_length | 100 | 1253 | 28.5934 | 9.67772 | 6.28287 | 1.94243 | 0.299999 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-512k | fp64 | stock | 3 | fixed_length | 100 | 1251 | 41.3536 | 22.3919 | 6.47647 | 1.94243 | 0.299999 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-512k | mixed | modal8_muladd | 3 | fixed_length | 100 | 1263 | 26.0624 | 7.15932 | 6.25774 | 1.94243 | 0.299999 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-512k | mixed | fused_ai_fp64 | 3 | fixed_length | 100 | 1260 | 26.665 | 7.8803 | 6.27157 | 1.94243 | 0.299999 | 0.15 |
| session-h-host1-20260926 | fixed_length | cantilever-512k | mixed | stock | 3 | fixed_length | 100 | 1262 | 31.2779 | 12.2644 | 6.47 | 1.94243 | 0.299999 | 0.15 |
| session-h-host1-20260926 | fixed_length | mbb-514k | fp64 | stock | 3 | fixed_length | 100 | 5298 | 109.817 | 88.539 | 8.76318 | 5.77466 | 0.500001 | 0.15 |
| session-h-host1-20260926 | fixed_length | mbb-514k | fp64 | fused_ai_fp64 | 3 | fixed_length | 100 | 5305 | 58.2334 | 37.1563 | 8.59223 | 5.77466 | 0.500001 | 0.15 |
| session-h-host1-20260926 | fixed_length | mbb-514k | fp64 | modal8_muladd | 3 | fixed_length | 100 | 5246 | 49.2982 | 28.3908 | 8.5774 | 5.77466 | 0.500001 | 0.15 |
| session-h-host1-20260926 | fixed_length | mbb-514k | mixed | fused_ai_fp64 | 3 | fixed_length | 100 | 8469 | 79.0772 | 58.1403 | 8.60781 | 5.77466 | 0.499999 | 0.15 |
| session-h-host1-20260926 | fixed_length | mbb-514k | mixed | modal8_muladd | 3 | fixed_length | 100 | 9044 | 74.4005 | 53.4871 | 8.57792 | 5.77466 | 0.499999 | 0.15 |
| session-h-host1-20260926 | fixed_length | mbb-514k | mixed | stock | 3 | fixed_length | 100 | 8194 | 119.997 | 98.2276 | 8.76591 | 5.77466 | 0.499999 | 0.15 |
| session-h-host1-20260926 | fixed_length | torsion-499k | fp64 | fused_ai_fp64 | 3 | fixed_length | 100 | 1890 | 29.1548 | 12.6065 | 4.57715 | 2.08316 | 0.25 | 0.15 |
| session-h-host1-20260926 | fixed_length | torsion-499k | fp64 | stock | 3 | fixed_length | 100 | 1890 | 47.3218 | 30.8184 | 4.75809 | 2.08316 | 0.25 | 0.15 |
| session-h-host1-20260926 | fixed_length | torsion-499k | fp64 | modal8_muladd | 3 | fixed_length | 100 | 1889 | 25.7665 | 9.55962 | 4.54112 | 2.08315 | 0.25 | 0.15 |
| session-h-host1-20260926 | fixed_length | torsion-499k | mixed | modal8_muladd | 3 | fixed_length | 100 | 2064 | 26.1718 | 9.83861 | 4.53673 | 2.08316 | 0.25 | 0.15 |
| session-h-host1-20260926 | fixed_length | torsion-499k | mixed | stock | 3 | fixed_length | 100 | 2078 | 34.5643 | 18.0211 | 4.75272 | 2.08316 | 0.25 | 0.15 |
| session-h-host1-20260926 | fixed_length | torsion-499k | mixed | fused_ai_fp64 | 3 | fixed_length | 100 | 2070 | 27.3526 | 10.9216 | 4.57525 | 2.08316 | 0.25 | 0.15 |
| session-h-host1-20260926 | capped | cantilever-216k | fp64 | stock | 1 | iteration_cap | 150 | 3398 | 39.7441 | 28.5241 | 3.26506 | 2.3702 | 0.3 | 0.148439 |
| session-h-host1-20260926 | capped | cantilever-216k | fp64 | fused_ai_fp64 | 1 | iteration_cap | 150 | 3387 | 24.1837 | 13.2608 | 3.17972 | 2.3702 | 0.3 | 0.14844 |
| session-h-host1-20260926 | capped | cantilever-216k | fp64 | modal8_muladd | 1 | iteration_cap | 150 | 3388 | 23.6886 | 12.6283 | 3.1581 | 2.3702 | 0.3 | 0.148435 |
| session-h-host1-20260926 | capped | cantilever-216k | mixed | stock | 1 | iteration_cap | 150 | 3502 | 29.9849 | 18.9858 | 3.24984 | 2.3702 | 0.3 | 0.148246 |
| session-h-host1-20260926 | capped | cantilever-216k | mixed | fused_ai_fp64 | 1 | iteration_cap | 150 | 3443 | 24.5922 | 13.6766 | 3.16693 | 2.37021 | 0.3 | 0.148449 |
| session-h-host1-20260926 | capped | cantilever-216k | mixed | modal8_muladd | 1 | iteration_cap | 150 | 3689 | 28.9146 | 17.99 | 3.16495 | 2.37021 | 0.3 | 0.148433 |
| session-h-host2-20260926 | fixed_length | cantilever-216k | fp64 | stock | 3 | fixed_length | 100 | 2247 | 27.8284 | 18.9722 | 2.25011 | 2.37514 | 0.3 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-216k | fp64 | fused_ai_fp64 | 3 | fixed_length | 100 | 2251 | 17.7621 | 8.99803 | 2.16705 | 2.37514 | 0.3 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-216k | fp64 | modal8_muladd | 3 | fixed_length | 100 | 2258 | 17.3446 | 8.63535 | 2.14648 | 2.37514 | 0.3 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-216k | mixed | stock | 3 | fixed_length | 100 | 2355 | 21.8156 | 12.9883 | 2.24675 | 2.37514 | 0.3 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-216k | mixed | modal8_muladd | 3 | fixed_length | 100 | 2319 | 18.1535 | 9.26991 | 2.15973 | 2.37514 | 0.3 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-216k | mixed | fused_ai_fp64 | 3 | fixed_length | 100 | 2303 | 18.438 | 9.43522 | 2.17357 | 2.37514 | 0.3 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-512k | fp64 | modal8_muladd | 3 | fixed_length | 100 | 1249 | 26.2169 | 7.4473 | 6.17401 | 1.94243 | 0.299999 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-512k | fp64 | fused_ai_fp64 | 3 | fixed_length | 100 | 1252 | 28.3765 | 9.53025 | 6.20828 | 1.94243 | 0.299999 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-512k | fp64 | stock | 3 | fixed_length | 100 | 1251 | 41.1781 | 22.1821 | 6.39158 | 1.94243 | 0.299999 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-512k | mixed | modal8_muladd | 3 | fixed_length | 100 | 1252 | 25.912 | 7.06626 | 6.17647 | 1.94243 | 0.299999 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-512k | mixed | fused_ai_fp64 | 3 | fixed_length | 100 | 1258 | 26.6517 | 7.79478 | 6.19632 | 1.94243 | 0.299999 | 0.15 |
| session-h-host2-20260926 | fixed_length | cantilever-512k | mixed | stock | 3 | fixed_length | 100 | 1257 | 31.1491 | 12.0436 | 6.39424 | 1.94243 | 0.299999 | 0.15 |
| session-h-host2-20260926 | fixed_length | mbb-514k | fp64 | stock | 3 | fixed_length | 100 | 5335 | 109.297 | 88.1944 | 8.62797 | 5.77466 | 0.500001 | 0.15 |
| session-h-host2-20260926 | fixed_length | mbb-514k | fp64 | fused_ai_fp64 | 3 | fixed_length | 100 | 5275 | 57.4952 | 36.4266 | 8.46075 | 5.77466 | 0.500001 | 0.15 |
| session-h-host2-20260926 | fixed_length | mbb-514k | fp64 | modal8_muladd | 3 | fixed_length | 100 | 5288 | 49.275 | 28.2971 | 8.44748 | 5.77466 | 0.500001 | 0.15 |
| session-h-host2-20260926 | fixed_length | mbb-514k | mixed | fused_ai_fp64 | 3 | fixed_length | 100 | 8746 | 88.1973 | 67.3354 | 8.45937 | 5.77466 | 0.499999 | 0.15 |
| session-h-host2-20260926 | fixed_length | mbb-514k | mixed | modal8_muladd | 3 | fixed_length | 100 | 8493 | 73.9424 | 53.1784 | 8.43608 | 5.77466 | 0.499999 | 0.15 |
| session-h-host2-20260926 | fixed_length | mbb-514k | mixed | stock | 3 | fixed_length | 100 | 8208 | 110.312 | 88.9264 | 8.63442 | 5.77466 | 0.5 | 0.15 |
| session-h-host2-20260926 | fixed_length | torsion-499k | fp64 | fused_ai_fp64 | 3 | fixed_length | 100 | 1891 | 28.7761 | 12.4032 | 4.51765 | 2.08315 | 0.25 | 0.15 |
| session-h-host2-20260926 | fixed_length | torsion-499k | fp64 | stock | 3 | fixed_length | 100 | 1873 | 47.0413 | 30.3963 | 4.71092 | 2.08315 | 0.25 | 0.15 |
| session-h-host2-20260926 | fixed_length | torsion-499k | fp64 | modal8_muladd | 3 | fixed_length | 100 | 1889 | 25.8052 | 9.45622 | 4.49744 | 2.08316 | 0.25 | 0.15 |
| session-h-host2-20260926 | fixed_length | torsion-499k | mixed | modal8_muladd | 3 | fixed_length | 100 | 2051 | 25.9895 | 9.75679 | 4.49259 | 2.08316 | 0.25 | 0.15 |
| session-h-host2-20260926 | fixed_length | torsion-499k | mixed | stock | 3 | fixed_length | 100 | 2106 | 34.2483 | 17.7674 | 4.71046 | 2.08316 | 0.25 | 0.15 |
| session-h-host2-20260926 | fixed_length | torsion-499k | mixed | fused_ai_fp64 | 3 | fixed_length | 100 | 2079 | 27.2077 | 10.8325 | 4.52375 | 2.08315 | 0.25 | 0.15 |
| session-h-host2-20260926 | capped | cantilever-216k | fp64 | stock | 1 | iteration_cap | 150 | 3373 | 39.4033 | 28.3552 | 3.22889 | 2.3702 | 0.3 | 0.14844 |
| session-h-host2-20260926 | capped | cantilever-216k | fp64 | fused_ai_fp64 | 1 | iteration_cap | 150 | 3385 | 24.4485 | 13.4471 | 3.14742 | 2.3702 | 0.3 | 0.148439 |
| session-h-host2-20260926 | capped | cantilever-216k | fp64 | modal8_muladd | 1 | iteration_cap | 150 | 3373 | 23.9304 | 12.9294 | 3.13273 | 2.3702 | 0.3 | 0.148439 |
| session-h-host2-20260926 | capped | cantilever-216k | mixed | stock | 1 | iteration_cap | 150 | 3747 | 31.7178 | 20.6043 | 3.2274 | 2.37021 | 0.3 | 0.148403 |
| session-h-host2-20260926 | capped | cantilever-216k | mixed | fused_ai_fp64 | 1 | iteration_cap | 150 | 3469 | 25.064 | 14.1229 | 3.13603 | 2.3702 | 0.3 | 0.148416 |
| session-h-host2-20260926 | capped | cantilever-216k | mixed | modal8_muladd | 1 | iteration_cap | 150 | 3477 | 24.8146 | 13.867 | 3.12381 | 2.37021 | 0.3 | 0.148421 |

## Paired wall-time observations

| host | kind | case | schedule | fused/parity ratios in repetition order | median | work matched within 3% |
|---|---|---|---|---|---|---|
| session-g-host1-20260926 | fixed_length | cantilever-216k | fp64 | 1.005127, 1.010886, 1.009634 | 1.00963 | True |
| session-g-host1-20260926 | fixed_length | cantilever-216k | mixed | 0.998917, 1.011327, 0.813432 | 0.998917 | False |
| session-g-host1-20260926 | fixed_length | cantilever-512k | fp64 | 1.016909, 1.026540, 1.013651 | 1.01691 | True |
| session-g-host1-20260926 | fixed_length | cantilever-512k | mixed | 0.992756, 1.001764, 1.012876 | 1.00176 | False |
| session-g-host1-20260926 | fixed_length | mbb-514k | fp64 | 1.066810, 1.076000, 1.067664 | 1.06766 | True |
| session-g-host1-20260926 | fixed_length | mbb-514k | mixed | 1.274340, 1.254498, 0.996203 | 1.2545 | False |
| session-g-host1-20260926 | fixed_length | torsion-499k | fp64 | 1.049296, 1.049070, 1.043454 | 1.04907 | False |
| session-g-host1-20260926 | fixed_length | torsion-499k | mixed | 1.019968, 1.015660, 1.012139 | 1.01566 | True |
| session-g-host1-20260926 | capped | cantilever-216k | fp64 | 1.007201 | 1.0072 | True |
| session-g-host1-20260926 | capped | cantilever-216k | mixed | 1.012718 | 1.01272 | True |
| session-g-host2-20260926 | fixed_length | cantilever-216k | fp64 | 1.014525, 0.992938, 0.998888 | 0.998888 | True |
| session-g-host2-20260926 | fixed_length | cantilever-216k | mixed | 0.987663, 0.980327, 1.215701 | 0.987663 | False |
| session-g-host2-20260926 | fixed_length | cantilever-512k | fp64 | 1.011601, 1.048994, 1.003964 | 1.0116 | True |
| session-g-host2-20260926 | fixed_length | cantilever-512k | mixed | 0.976115, 1.025259, 1.033805 | 1.02526 | True |
| session-g-host2-20260926 | fixed_length | mbb-514k | fp64 | 1.038774, 1.075576, 1.071123 | 1.07112 | True |
| session-g-host2-20260926 | fixed_length | mbb-514k | mixed | 1.210373, 1.490167, 3.969804 | 1.49017 | False |
| session-g-host2-20260926 | fixed_length | torsion-499k | fp64 | 1.021728, 1.033604, 2.180829 | 1.0336 | True |
| session-g-host2-20260926 | fixed_length | torsion-499k | mixed | 1.014107, 1.005052, 0.294048 | 1.00505 | False |
| session-g-host2-20260926 | capped | cantilever-216k | fp64 | 1.011120 | 1.01112 | True |
| session-g-host2-20260926 | capped | cantilever-216k | mixed | 0.954732 | 0.954732 | True |
| session-h-host1-20260926 | fixed_length | cantilever-216k | fp64 | 1.022847, 1.028853, 1.025590 | 1.02559 | True |
| session-h-host1-20260926 | fixed_length | cantilever-216k | mixed | 1.015728, 1.011717, 1.183193 | 1.01573 | False |
| session-h-host1-20260926 | fixed_length | cantilever-512k | fp64 | 1.088342, 1.086970, 1.090039 | 1.08834 | True |
| session-h-host1-20260926 | fixed_length | cantilever-512k | mixed | 1.020690, 1.023120, 1.032623 | 1.02312 | True |
| session-h-host1-20260926 | fixed_length | mbb-514k | fp64 | 1.181485, 1.183322, 1.169402 | 1.18148 | True |
| session-h-host1-20260926 | fixed_length | mbb-514k | mixed | 0.915822, 1.104957, 1.307282 | 1.10496 | False |
| session-h-host1-20260926 | fixed_length | torsion-499k | fp64 | 1.133018, 1.114135, 1.118677 | 1.11868 | True |
| session-h-host1-20260926 | fixed_length | torsion-499k | mixed | 1.033498, 1.046382, 1.040187 | 1.04019 | True |
| session-h-host1-20260926 | capped | cantilever-216k | fp64 | 1.020899 | 1.0209 | True |
| session-h-host1-20260926 | capped | cantilever-216k | mixed | 0.850511 | 0.850511 | False |
| session-h-host2-20260926 | fixed_length | cantilever-216k | fp64 | 1.021130, 1.023073, 1.029154 | 1.02307 | True |
| session-h-host2-20260926 | fixed_length | cantilever-216k | mixed | 1.015064, 1.010920, 1.048859 | 1.01506 | False |
| session-h-host2-20260926 | fixed_length | cantilever-512k | fp64 | 1.079972, 1.085736, 1.080900 | 1.0809 | True |
| session-h-host2-20260926 | fixed_length | cantilever-512k | mixed | 1.025210, 1.023747, 1.028546 | 1.02521 | True |
| session-h-host2-20260926 | fixed_length | mbb-514k | fp64 | 1.166824, 1.164462, 1.166560 | 1.16656 | True |
| session-h-host2-20260926 | fixed_length | mbb-514k | mixed | 1.095935, 1.201913, 1.158299 | 1.1583 | False |
| session-h-host2-20260926 | fixed_length | torsion-499k | fp64 | 1.099865, 1.134816, 1.115129 | 1.11513 | True |
| session-h-host2-20260926 | fixed_length | torsion-499k | mixed | 1.046878, 1.056981, 1.027671 | 1.04688 | False |
| session-h-host2-20260926 | capped | cantilever-216k | fp64 | 1.021647 | 1.02165 | True |
| session-h-host2-20260926 | capped | cantilever-216k | mixed | 1.010050 | 1.01005 | True |

## Rediscretized scale, all configurations

n=2 warm solves per row; time is the median. Shared setup is paid once per case and arm setup is the recorded incremental setup in execution order, with cached compilation possible. No per-route live/peak memory measurement was recorded; neither periodic device telemetry nor an OOM bound supplies one. Total/free DOFs are calculated from recorded dimensions and the fully clamped x=0 face.

| host | case | arm | elements | DOFs | free DOFs | levels | iterations | warm solve s | shared setup s | arm setup s | live/peak memory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| session-i-a100-20260926 | q96x1 | mg32-modal8_muladd-node32 | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.328333 | 4.63659 | 0.00300704 | NR / NR |
| session-i-a100-20260926 | q96x1 | mg64-node | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.362571 | 4.63659 | 0.00377242 | NR / NR |
| session-i-a100-20260926 | q96x1 | mg64-fused_ai | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.39474 | 4.63659 | 0.00234682 | NR / NR |
| session-i-a100-20260926 | q96x1 | mg32-node64-node32 | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.319631 | 4.63659 | 0.548132 | NR / NR |
| session-i-a100-20260926 | q96x1 | mg64-modal8_muladd | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.434534 | 4.63659 | 0.00316477 | NR / NR |
| session-i-a100-20260926 | q96x1 | mg32-fused_ai64-node32 | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.316096 | 4.63659 | 0.00324398 | NR / NR |
| session-i-a100-20260926 | q64x2 | mg32-fused_ai64-node32 | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.315307 | 2.13725 | 0.00356933 | NR / NR |
| session-i-a100-20260926 | q64x2 | mg64-fused_ai | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.52528 | 2.13725 | 0.00309237 | NR / NR |
| session-i-a100-20260926 | q64x2 | mg64-node | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.49101 | 2.13725 | 0.00341698 | NR / NR |
| session-i-a100-20260926 | q64x2 | mg32-modal8_muladd-node32 | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.3323 | 2.13725 | 0.00337746 | NR / NR |
| session-i-a100-20260926 | q64x2 | mg32-node64-node32 | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.310142 | 2.13725 | 0.00342207 | NR / NR |
| session-i-a100-20260926 | q64x2 | mg64-modal8_muladd | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.581647 | 2.13725 | 0.00340241 | NR / NR |
| session-i-a100-20260926 | q96x2 | mg64-node | 14155776 | 43022595 | 42910848 | 6 | 24 | 2.65559 | 3.4164 | 0.00527519 | NR / NR |
| session-i-a100-20260926 | q96x2 | mg64-modal8_muladd | 14155776 | 43022595 | 42910848 | 6 | 24 | 3.16844 | 3.4164 | 0.00565851 | NR / NR |
| session-i-a100-20260926 | q96x2 | mg64-fused_ai | 14155776 | 43022595 | 42910848 | 6 | 24 | 2.80562 | 3.4164 | 0.00526788 | NR / NR |
| session-i-a100-20260926 | q96x2 | mg32-modal8_muladd-node32 | 14155776 | 43022595 | 42910848 | 6 | 24 | 1.73745 | 3.4164 | 0.00408522 | NR / NR |
| session-i-a100-20260926 | q96x2 | mg32-fused_ai64-node32 | 14155776 | 43022595 | 42910848 | 6 | 24 | 1.66751 | 3.4164 | 0.00408068 | NR / NR |
| session-i-a100-20260926 | q96x2 | mg32-node64-node32 | 14155776 | 43022595 | 42910848 | 6 | 24 | 1.62406 | 3.4164 | 0.0040793 | NR / NR |
| session-i-a100-20260926 | q96x3 | mg64-fused_ai | 47775744 | 144574851 | 144324288 | 6 | 20 | 7.85584 | 13.6777 | 0.0172575 | NR / NR |
| session-i-a100-20260926 | q96x3 | mg64-node | 47775744 | 144574851 | 144324288 | 6 | 20 | 7.41395 | 13.6777 | 0.0172617 | NR / NR |
| session-i-a100-20260926 | q96x3 | mg32-node64-node32 | 47775744 | 144574851 | 144324288 | 6 | 20 | 4.53692 | 13.6777 | 0.013211 | NR / NR |
| session-i-a100-20260926 | q96x3 | mg32-modal8_muladd-node32 | 47775744 | 144574851 | 144324288 | 6 | 20 | 4.84267 | 13.6777 | 0.0131864 | NR / NR |
| session-i-a100-20260926 | q96x3 | mg32-fused_ai64-node32 | 47775744 | 144574851 | 144324288 | 6 | 20 | 4.61184 | 13.6777 | 0.0131941 | NR / NR |
| session-i-a100-20260926 | q96x3 | mg64-modal8_muladd | 47775744 | 144574851 | 144324288 | 6 | 20 | 8.90154 | 13.6777 | 0.0172548 | NR / NR |
| session-i-h100-20260926 | q96x1 | mg32-modal8_muladd-node32 | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.195178 | 3.31427 | 0.00159388 | NR / NR |
| session-i-h100-20260926 | q96x1 | mg64-node | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.257894 | 3.31427 | 0.00204055 | NR / NR |
| session-i-h100-20260926 | q96x1 | mg64-fused_ai | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.275642 | 3.31427 | 0.00130446 | NR / NR |
| session-i-h100-20260926 | q96x1 | mg32-node64-node32 | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.187181 | 3.31427 | 0.388492 | NR / NR |
| session-i-h100-20260926 | q96x1 | mg64-modal8_muladd | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.300121 | 3.31427 | 0.00167462 | NR / NR |
| session-i-h100-20260926 | q96x1 | mg32-fused_ai64-node32 | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.190261 | 3.31427 | 0.00173248 | NR / NR |
| session-i-h100-20260926 | q64x2 | mg32-fused_ai64-node32 | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.245669 | 1.52094 | 0.00182095 | NR / NR |
| session-i-h100-20260926 | q64x2 | mg64-fused_ai | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.361362 | 1.52094 | 0.00163261 | NR / NR |
| session-i-h100-20260926 | q64x2 | mg64-node | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.340082 | 1.52094 | 0.00188472 | NR / NR |
| session-i-h100-20260926 | q64x2 | mg32-modal8_muladd-node32 | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.245192 | 1.52094 | 0.00190321 | NR / NR |
| session-i-h100-20260926 | q64x2 | mg32-node64-node32 | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.236023 | 1.52094 | 0.00174581 | NR / NR |
| session-i-h100-20260926 | q64x2 | mg64-modal8_muladd | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.393198 | 1.52094 | 0.00183488 | NR / NR |
| session-i-h100-20260926 | q96x2 | mg64-node | 14155776 | 43022595 | 42910848 | 6 | 24 | 2.19961 | 2.23602 | 0.00351693 | NR / NR |
| session-i-h100-20260926 | q96x2 | mg64-modal8_muladd | 14155776 | 43022595 | 42910848 | 6 | 24 | 2.41008 | 2.23602 | 0.00350184 | NR / NR |
| session-i-h100-20260926 | q96x2 | mg64-fused_ai | 14155776 | 43022595 | 42910848 | 6 | 24 | 2.2341 | 2.23602 | 0.00351512 | NR / NR |
| session-i-h100-20260926 | q96x2 | mg32-modal8_muladd-node32 | 14155776 | 43022595 | 42910848 | 6 | 24 | 1.43746 | 2.23602 | 0.00305944 | NR / NR |
| session-i-h100-20260926 | q96x2 | mg32-fused_ai64-node32 | 14155776 | 43022595 | 42910848 | 6 | 24 | 1.40421 | 2.23602 | 0.00305484 | NR / NR |
| session-i-h100-20260926 | q96x2 | mg32-node64-node32 | 14155776 | 43022595 | 42910848 | 6 | 24 | 1.39193 | 2.23602 | 0.00305925 | NR / NR |
| session-i-h100-20260926 | q96x3 | mg64-fused_ai | 47775744 | 144574851 | 144324288 | 6 | 20 | 6.39514 | 9.57086 | 0.0112041 | NR / NR |
| session-i-h100-20260926 | q96x3 | mg64-node | 47775744 | 144574851 | 144324288 | 6 | 20 | 6.19087 | 9.57086 | 0.0111802 | NR / NR |
| session-i-h100-20260926 | q96x3 | mg32-node64-node32 | 47775744 | 144574851 | 144324288 | 6 | 20 | 3.97797 | 9.57086 | 0.0108518 | NR / NR |
| session-i-h100-20260926 | q96x3 | mg32-modal8_muladd-node32 | 47775744 | 144574851 | 144324288 | 6 | 20 | 4.09445 | 9.57086 | 0.0108394 | NR / NR |
| session-i-h100-20260926 | q96x3 | mg32-fused_ai64-node32 | 47775744 | 144574851 | 144324288 | 6 | 20 | 4.00109 | 9.57086 | 0.0108468 | NR / NR |
| session-i-h100-20260926 | q96x3 | mg64-modal8_muladd | 47775744 | 144574851 | 144324288 | 6 | 20 | 6.87379 | 9.57086 | 0.0113654 | NR / NR |
| session-i-rtx4090-20260926 | q96x1 | mg32-modal8_muladd-node32 | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.309639 | 5.15247 | 0.0024345 | NR / NR |
| session-i-rtx4090-20260926 | q96x1 | mg64-node | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.875274 | 5.15247 | 0.00351335 | NR / NR |
| session-i-rtx4090-20260926 | q96x1 | mg64-fused_ai | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.754505 | 5.15247 | 0.0022844 | NR / NR |
| session-i-rtx4090-20260926 | q96x1 | mg32-node64-node32 | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.376364 | 5.15247 | 0.572742 | NR / NR |
| session-i-rtx4090-20260926 | q96x1 | mg64-modal8_muladd | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.544653 | 5.15247 | 0.00255587 | NR / NR |
| session-i-rtx4090-20260926 | q96x1 | mg32-fused_ai64-node32 | 1769472 | 5447811 | 5419584 | 5 | 24 | 0.352387 | 5.15247 | 0.00272217 | NR / NR |
| session-i-rtx4090-20260926 | q64x2 | mg32-fused_ai64-node32 | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.428251 | 3.3784 | 0.00286677 | NR / NR |
| session-i-rtx4090-20260926 | q64x2 | mg64-fused_ai | 4194304 | 12830211 | 12780288 | 5 | 14 | 1.08235 | 3.3784 | 0.00261185 | NR / NR |
| session-i-rtx4090-20260926 | q64x2 | mg64-node | 4194304 | 12830211 | 12780288 | 5 | 14 | 1.246 | 3.3784 | 0.00292494 | NR / NR |
| session-i-rtx4090-20260926 | q64x2 | mg32-modal8_muladd-node32 | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.371259 | 3.3784 | 0.00286274 | NR / NR |
| session-i-rtx4090-20260926 | q64x2 | mg32-node64-node32 | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.462285 | 3.3784 | 0.00271831 | NR / NR |
| session-i-rtx4090-20260926 | q64x2 | mg64-modal8_muladd | 4194304 | 12830211 | 12780288 | 5 | 14 | 0.792731 | 3.3784 | 0.00288135 | NR / NR |
| session-i-rtx4090-20260926 | q96x2 | mg64-node | 14155776 | 43022595 | 42910848 | 6 | 24 | 7.17342 | 7.47631 | 0.00797608 | NR / NR |
| session-i-rtx4090-20260926 | q96x2 | mg64-modal8_muladd | 14155776 | 43022595 | 42910848 | 6 | 24 | 4.59363 | 7.47631 | 0.00807348 | NR / NR |
| session-i-rtx4090-20260926 | q96x2 | mg64-fused_ai | 14155776 | 43022595 | 42910848 | 6 | 24 | 6.26144 | 7.47631 | 0.00797238 | NR / NR |
| session-i-rtx4090-20260926 | q96x2 | mg32-modal8_muladd-node32 | 14155776 | 43022595 | 42910848 | 6 | 24 | 2.11601 | 7.47631 | 0.00514667 | NR / NR |
| session-i-rtx4090-20260926 | q96x2 | mg32-fused_ai64-node32 | 14155776 | 43022595 | 42910848 | 6 | 24 | 2.451 | 7.47631 | 0.0051578 | NR / NR |
| session-i-rtx4090-20260926 | q96x2 | mg32-node64-node32 | 14155776 | 43022595 | 42910848 | 6 | 24 | 2.63486 | 7.47631 | 0.00517961 | NR / NR |

Ladder outcomes (including unsuccessful construction):

| host | case | status |
|---|---|---|
| session-i-a100-20260926 | q96x1 | ok |
| session-i-a100-20260926 | q64x2 | ok |
| session-i-a100-20260926 | q96x2 | ok |
| session-i-a100-20260926 | q96x3 | ok |
| session-i-h100-20260926 | q96x1 | ok |
| session-i-h100-20260926 | q64x2 | ok |
| session-i-h100-20260926 | q96x2 | ok |
| session-i-h100-20260926 | q96x3 | ok |
| session-i-rtx4090-20260926 | q96x1 | ok |
| session-i-rtx4090-20260926 | q64x2 | ok |
| session-i-rtx4090-20260926 | q96x2 | ok |
| session-i-rtx4090-20260926 | q96x3 | out_of_memory |

## Galerkin feasibility and retained memory

| host | size | arm | schedule | elements | DOFs | free DOFs | status | iterations | converged | setup s | capped solve s | device used GiB | pool reserved GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| session-f-a100-20260926 | 2M | stock | fp64 | 2000376 | 6169152 | 6144768 | ok | 40 | False | 24.2402 | 2.39051 | 46.3262 | 46.2666 |
| session-f-a100-20260926 | 5M | stock | fp64 | 5029452 | 15397956 | 15353064 | out_of_memory | NR | NR | NR | NR | NR | NR |
| session-f-a100-20260926 | 2M | modal8_muladd | fp64 | 2000376 | 6169152 | 6144768 | ok | 40 | False | 23.0826 | 1.35942 | 45.5547 | 45.5065 |
| session-f-a100-20260926 | 5M | modal8_muladd | fp64 | 5029452 | 15397956 | 15353064 | out_of_memory | NR | NR | NR | NR | NR | NR |
| session-f-a100-20260926 | 2M | stock | mixed | 2000376 | 6169152 | 6144768 | ok | 40 | False | 23.8522 | 1.6589 | 46.373 | 46.3126 |
| session-f-a100-20260926 | 5M | stock | mixed | 5029452 | 15397956 | 15353064 | out_of_memory | NR | NR | NR | NR | NR | NR |
| session-f-h100-20260926-attempt02 | 2M | stock | fp64 | 2000376 | 6169152 | 6144768 | ok | 40 | False | 37.4949 | 1.53195 | 46.3828 | 46.2664 |
| session-f-h100-20260926-attempt02 | 5M | stock | fp64 | 5029452 | 15397956 | 15353064 | out_of_memory | NR | NR | NR | NR | NR | NR |
| session-f-h100-20260926-attempt02 | 2M | modal8_muladd | fp64 | 2000376 | 6169152 | 6144768 | ok | 40 | False | 36.6743 | 0.868987 | 45.5566 | 45.5063 |
| session-f-h100-20260926-attempt02 | 5M | modal8_muladd | fp64 | 5029452 | 15397956 | 15353064 | out_of_memory | NR | NR | NR | NR | NR | NR |
| session-f-h100-20260926-attempt02 | 2M | stock | mixed | 2000376 | 6169152 | 6144768 | ok | 40 | False | 36.5192 | 1.05772 | 46.4297 | 46.3124 |
| session-f-h100-20260926-attempt02 | 5M | stock | mixed | 5029452 | 15397956 | 15353064 | out_of_memory | NR | NR | NR | NR | NR | NR |
| session-f-rtx4090-20260926 | 2M | stock | fp64 | 2000376 | 6169152 | 6144768 | out_of_memory | NR | NR | NR | NR | NR | NR |
| session-f-rtx4090-20260926 | 2M | modal8_muladd | fp64 | 2000376 | 6169152 | 6144768 | out_of_memory | NR | NR | NR | NR | NR | NR |
| session-f-rtx4090-20260926 | 2M | stock | mixed | 2000376 | 6169152 | 6144768 | out_of_memory | NR | NR | NR | NR | NR | NR |

## Galerkin W2a: every block and arm

Each block has three 50-iteration fixed-work solves; their median is shown. Only block 0 records the separate 1000-iteration tolerance attempt; its outcome is retained, including failures. Device used is CUDA total-minus-free after setup, not peak or live arrays.

| host | size | schedule | arm | block | elements | DOFs | free DOFs | levels | setup s | 50 iterations s | tol. iterations | converged | tol. attempt s | device used GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| session-i-a100-20260926 | 1M | fp64 | fused_ai_fp64 | 0 | 1000000 | 3106053 | 3090600 | 4 | 6.48292 | 2.16087 | 1000 | False | 44.2979 | 23.4989 |
| session-i-a100-20260926 | 1M | fp64 | modal8_muladd | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0904667 | 2.26325 | 1000 | False | 44.1082 | 23.5243 |
| session-i-a100-20260926 | 1M | fp64 | node_fp64 | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0874656 | 2.13737 | 1000 | False | 43.3283 | 23.5009 |
| session-i-a100-20260926 | 1M | fp64 | stock | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.264023 | 2.76706 | 1000 | False | 54.0623 | 23.9149 |
| session-i-a100-20260926 | 1M | fp64 | modal8_muladd | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0987561 | 2.18913 | NR | NR | NR | 23.5321 |
| session-i-a100-20260926 | 1M | fp64 | fused_ai_fp64 | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.104101 | 2.1731 | NR | NR | NR | 23.5087 |
| session-i-a100-20260926 | 1M | fp64 | stock | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.131303 | 2.68601 | NR | NR | NR | 23.9149 |
| session-i-a100-20260926 | 1M | fp64 | node_fp64 | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0939336 | 2.15242 | NR | NR | NR | 23.5087 |
| session-i-a100-20260926 | 1M | mixed | node_fp64 | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0934038 | 2.14479 | 1000 | False | 44.2432 | 23.5321 |
| session-i-a100-20260926 | 1M | mixed | modal8_muladd | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0955859 | 2.15211 | 1000 | False | 43.97 | 23.5555 |
| session-i-a100-20260926 | 1M | mixed | fused_ai_fp64 | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0910952 | 2.19593 | 1000 | False | 43.4559 | 23.5321 |
| session-i-a100-20260926 | 1M | mixed | stock | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.126257 | 2.33555 | 1000 | False | 46.9995 | 23.9384 |
| session-i-a100-20260926 | 1M | mixed | modal8_muladd | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0931305 | 2.15551 | NR | NR | NR | 23.5555 |
| session-i-a100-20260926 | 1M | mixed | node_fp64 | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0893899 | 2.20035 | NR | NR | NR | 23.5321 |
| session-i-a100-20260926 | 1M | mixed | stock | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.127265 | 2.47669 | NR | NR | NR | 23.9384 |
| session-i-a100-20260926 | 1M | mixed | fused_ai_fp64 | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0905512 | 2.14392 | NR | NR | NR | 23.5321 |
| session-i-a100-20260926 | 2M | fp64 | modal8_muladd | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.151047 | 2.28655 | 1000 | False | 46.0958 | 45.9735 |
| session-i-a100-20260926 | 2M | fp64 | fused_ai_fp64 | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.152484 | 2.28372 | 1000 | False | 46.7319 | 45.9266 |
| session-i-a100-20260926 | 2M | fp64 | stock | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.211628 | 3.36301 | 1000 | False | 67.5332 | 46.7372 |
| session-i-a100-20260926 | 2M | fp64 | node_fp64 | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.144474 | 2.23368 | 1000 | False | 45.3022 | 45.9266 |
| session-i-a100-20260926 | 2M | fp64 | modal8_muladd | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.151396 | 2.34562 | NR | NR | NR | 45.9735 |
| session-i-a100-20260926 | 2M | fp64 | stock | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.209133 | 3.32349 | NR | NR | NR | 46.7372 |
| session-i-a100-20260926 | 2M | fp64 | fused_ai_fp64 | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.151888 | 2.31097 | NR | NR | NR | 45.9266 |
| session-i-a100-20260926 | 2M | fp64 | node_fp64 | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.149006 | 2.27862 | NR | NR | NR | 45.9266 |
| session-i-a100-20260926 | 2M | mixed | modal8_muladd | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.15784 | 2.32925 | 1000 | False | 45.7009 | 46.0204 |
| session-i-a100-20260926 | 2M | mixed | fused_ai_fp64 | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.14817 | 2.23121 | 1000 | False | 45.1286 | 45.9735 |
| session-i-a100-20260926 | 2M | mixed | node_fp64 | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.149563 | 2.26484 | 1000 | False | 45.5143 | 45.9735 |
| session-i-a100-20260926 | 2M | mixed | stock | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.212398 | 2.66903 | 1000 | False | 54.2224 | 46.7841 |
| session-i-a100-20260926 | 2M | mixed | fused_ai_fp64 | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.149074 | 2.28877 | NR | NR | NR | 45.9735 |
| session-i-a100-20260926 | 2M | mixed | node_fp64 | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.144727 | 2.23294 | NR | NR | NR | 45.9735 |
| session-i-a100-20260926 | 2M | mixed | stock | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.212044 | 2.72701 | NR | NR | NR | 46.7841 |
| session-i-a100-20260926 | 2M | mixed | modal8_muladd | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.151147 | 2.30762 | NR | NR | NR | 46.0204 |
| session-i-h100-20260926 | 1M | fp64 | fused_ai_fp64 | 0 | 1000000 | 3106053 | 3090600 | 4 | 4.44958 | 1.26039 | 1000 | False | 25.3816 | 23.6004 |
| session-i-h100-20260926 | 1M | fp64 | modal8_muladd | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0724088 | 1.25394 | 1000 | False | 25.1805 | 23.6238 |
| session-i-h100-20260926 | 1M | fp64 | node_fp64 | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0706671 | 1.24297 | 1000 | False | 24.8934 | 23.6004 |
| session-i-h100-20260926 | 1M | fp64 | stock | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.180603 | 1.61826 | 1000 | False | 32.3556 | 24.0692 |
| session-i-h100-20260926 | 1M | fp64 | modal8_muladd | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0724489 | 1.25962 | NR | NR | NR | 23.6863 |
| session-i-h100-20260926 | 1M | fp64 | fused_ai_fp64 | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0717104 | 1.25379 | NR | NR | NR | 23.6629 |
| session-i-h100-20260926 | 1M | fp64 | stock | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0951733 | 1.6155 | NR | NR | NR | 24.0692 |
| session-i-h100-20260926 | 1M | fp64 | node_fp64 | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0712453 | 1.2485 | NR | NR | NR | 23.6629 |
| session-i-h100-20260926 | 1M | mixed | node_fp64 | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0687012 | 1.26482 | 1000 | False | 25.4654 | 23.6863 |
| session-i-h100-20260926 | 1M | mixed | modal8_muladd | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0714196 | 1.25331 | 1000 | False | 25.5586 | 23.7098 |
| session-i-h100-20260926 | 1M | mixed | fused_ai_fp64 | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0704501 | 1.25503 | 1000 | False | 25.1647 | 23.6863 |
| session-i-h100-20260926 | 1M | mixed | stock | 0 | 1000000 | 3106053 | 3090600 | 4 | 0.0956311 | 1.38719 | 1000 | False | 27.7715 | 24.0926 |
| session-i-h100-20260926 | 1M | mixed | modal8_muladd | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0715105 | 1.25465 | NR | NR | NR | 23.7098 |
| session-i-h100-20260926 | 1M | mixed | node_fp64 | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0706428 | 1.24754 | NR | NR | NR | 23.6863 |
| session-i-h100-20260926 | 1M | mixed | stock | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0967827 | 1.37797 | NR | NR | NR | 24.0926 |
| session-i-h100-20260926 | 1M | mixed | fused_ai_fp64 | 1 | 1000000 | 3106053 | 3090600 | 4 | 0.0705703 | 1.25033 | NR | NR | NR | 23.6863 |
| session-i-h100-20260926 | 2M | fp64 | modal8_muladd | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.128882 | 1.36084 | 1000 | False | 27.5935 | 46.1258 |
| session-i-h100-20260926 | 2M | fp64 | fused_ai_fp64 | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.125716 | 1.36226 | 1000 | False | 27.1392 | 46.0789 |
| session-i-h100-20260926 | 2M | fp64 | stock | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.174973 | 2.15616 | 1000 | False | 42.2462 | 46.8895 |
| session-i-h100-20260926 | 2M | fp64 | node_fp64 | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.124157 | 1.34588 | 1000 | False | 27.031 | 46.0789 |
| session-i-h100-20260926 | 2M | fp64 | modal8_muladd | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.127353 | 1.36109 | NR | NR | NR | 46.1258 |
| session-i-h100-20260926 | 2M | fp64 | stock | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.174807 | 2.10462 | NR | NR | NR | 46.8895 |
| session-i-h100-20260926 | 2M | fp64 | fused_ai_fp64 | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.13018 | 1.35934 | NR | NR | NR | 46.0789 |
| session-i-h100-20260926 | 2M | fp64 | node_fp64 | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.127383 | 1.34739 | NR | NR | NR | 46.0789 |
| session-i-h100-20260926 | 2M | mixed | modal8_muladd | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.129552 | 1.34538 | 1000 | False | 26.7892 | 46.1727 |
| session-i-h100-20260926 | 2M | mixed | fused_ai_fp64 | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.125696 | 1.37341 | 1000 | False | 27.0313 | 46.1258 |
| session-i-h100-20260926 | 2M | mixed | node_fp64 | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.124721 | 1.33139 | 1000 | False | 26.6589 | 46.1258 |
| session-i-h100-20260926 | 2M | mixed | stock | 0 | 2000376 | 6169152 | 6144768 | 4 | 0.174892 | 1.62047 | 1000 | False | 32.3635 | 46.9363 |
| session-i-h100-20260926 | 2M | mixed | fused_ai_fp64 | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.127676 | 1.33937 | NR | NR | NR | 46.1258 |
| session-i-h100-20260926 | 2M | mixed | node_fp64 | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.123004 | 1.33385 | NR | NR | NR | 46.1258 |
| session-i-h100-20260926 | 2M | mixed | stock | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.176224 | 1.62321 | NR | NR | NR | 46.9363 |
| session-i-h100-20260926 | 2M | mixed | modal8_muladd | 1 | 2000376 | 6169152 | 6144768 | 4 | 0.128938 | 1.33887 | NR | NR | NR | 46.1727 |

## Hardware and software manifest

One row per measured allocation; full protocol cases, arms, precision, repetitions, warmup and boundaries are in experiment-manifest.csv and outcomes-scale.json. Driver and runtime versions are separate. NR compiler metadata is not guessed from the Python compiler string.

| session/host | GPU | capacity | CPU | driver | CUDA runtime | CuPy | CUDA/host compiler |
|---|---|---|---|---|---|---|---|
| session-a-20260923 | NVIDIA GeForce RTX 4090 | 24564 MiB | NR | 580.173.02 | 12090 (linked to CuPy) / 12080 (locally installed) | 13.6.0 | NVRTC (12, 8) |
| session-b-20260923 | NVIDIA A100 80GB PCIe | 81920 MiB | Xeon® Gold 5218R  | 595.84 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-c-20260923 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7B13 64-Core Processor | 590.48.01 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-d-20260923 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 9684X 96-Core Processor | 580.119.02 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-e-host1-20260925 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7742 64-Core Processor | 580.95.05 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-e-host2-20260925 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD Ryzen Threadripper 2950X 16-Core Processor | 595.71.05 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-f-a100-20260926 | NVIDIA A100 80GB PCIe | 81920 MiB | AMD Ryzen Threadripper PRO 5955WX 16-Cores | 580.126.09 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-f-h100-20260926-attempt02 | NVIDIA H100 NVL | 95830 MiB | XEON® PLATINUM 8558 | 570.211.01 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-f-rtx4090-20260926 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7R13 48-Core Processor | 595.84 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-g-host1-20260926 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7282 16-Core Processor | 570.133.07 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-g-host2-20260926 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7702 64-Core Processor | 580.95.05 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-h-host1-20260926 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7502 32-Core Processor | 560.35.03 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-h-host2-20260926 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7502 32-Core Processor | 570.211.01 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-i-a100-20260926 | NVIDIA A100 80GB PCIe | 81920 MiB | AMD EPYC 7713 64-Core Processor | 595.71.05 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-i-h100-20260926 | NVIDIA H100 80GB HBM3 | 81559 MiB | AMD EPYC 9454 48-Core Processor | 595.71.05 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-i-rtx4090-20260926 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7702 64-Core Processor | 560.35.03 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| kernel-rtx4090 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7B13 64-Core Processor | 595.84 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| kernel-rtx3090-replication | NVIDIA GeForce RTX 3090 | 24576 MiB | AMD EPYC 7532 32-Core Processor | 595.84 | 12090 (linked to CuPy) / 12060 (locally installed) | 13.6.0 | NVRTC (12, 6) |
| session-j-rtx4090vm-20260926-attempt07 | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD EPYC 7K62 48-Core Processor | 535.183.01 | 12.2.140 installed cudart; linked version NR | 13.6.0 | NVRTC 12.2.140 |
| libceed-resident | NVIDIA GeForce RTX 4090 | 24564 MiB | AMD Ryzen 5 5600X 6-Core Processor | 595.58.03 | CUDA 12.6 SDK/header probe | native harness | nvcc 12.6.85; GCC 11.4.0 |
