Idealized 48-value traffic accounting; these are specification-based predictions, not measurements.
The original A100 SXM declaration is preserved. The PCIe row is a revision calculation for the measured SKU,
using [NVIDIA's specifications](https://www.nvidia.com/en-us/data-center/a100/) (9.7 FP64 TFLOP/s, 19.5 FP32 TFLOP/s, 1935 GB/s).
Shared-node reuse, caches, atomics, moduli and masks prevent identifying this traffic with measured DRAM bytes.

| device | precision | model status | ridge (FLOP/B) | dense intensity | dense bound | parity bound | ideal dense/parity |
|---|---|---|---:|---:|---|---|---:|
| RTX 3090 | FP64 | original declared model | 0.59 | 3.0 | compute | compute | 4.00 |
| RTX 3090 | FP32 | original declared model | 38.01 | 6.0 | bandwidth | bandwidth | 1.00 |
| RTX 4090 | FP64 | original declared model | 1.28 | 3.0 | compute | bandwidth | 2.34 |
| RTX 4090 | FP32 | original declared model | 81.92 | 6.0 | bandwidth | bandwidth | 1.00 |
| A100 80GB SXM | FP64 | original declared model | 4.76 | 3.0 | bandwidth | bandwidth | 1.00 |
| A100 80GB SXM | FP32 | original declared model | 9.56 | 6.0 | bandwidth | bandwidth | 1.00 |
| H100 SXM | FP64 | original declared model | 10.15 | 3.0 | bandwidth | bandwidth | 1.00 |
| H100 SXM | FP32 | original declared model | 20.00 | 6.0 | bandwidth | bandwidth | 1.00 |
| A100 80GB PCIe | FP64 | revision for measured SKU (2026-09-27) | 5.01 | 3.0 | bandwidth | bandwidth | 1.00 |
| A100 80GB PCIe | FP32 | revision for measured SKU (2026-09-27) | 10.08 | 6.0 | bandwidth | bandwidth | 1.00 |
