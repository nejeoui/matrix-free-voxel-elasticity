"""Ideal-traffic roofline prediction for the voxel Hex8 element product, dense vs parity (modal8).

A PREDICTION, not a measurement. Intensity uses idealized per-element traffic of a fused
gather-product-scatter kernel: read 24 displacement values and write 24 results
(2 x 24 x bytes_per_value), the model used in Yang et al. (arXiv 2604.18020) for its FP32 kernel.
Shared-node reuse and caches can reduce DRAM loads; atomic read/modify/write operations,
moduli and masks add traffic. Thus 48 scalar values per element are an accounting model,
not measured DRAM bytes or a universal lower bound.
Dense work 2 x 576 flops; parity work 144 butterfly additions + 2 x 72 = 288 flops (implementation
README). Device peaks are vendor specification values. Real kernels carry extra traffic (density,
indices, atomics) and non-flop instructions, so measured ratios are expected to be smaller than
these ideal ratios; the point is WHERE a ratio > 1 is possible at all.
The original A100 80GB SXM declaration is retained and identified separately from the
2026-09-27 revision calculation for the measured A100 80GB PCIe SKU.
"""
import json
from pathlib import Path

DEV = {  # FP64 non-tensor TFLOP/s, FP32 TFLOP/s, DRAM GB/s
    "RTX 3090": (0.556, 35.58, 936), "RTX 4090": (1.290, 82.58, 1008),
    "A100 80GB SXM": (9.7, 19.5, 2039), "H100 SXM": (34.0, 67.0, 3350),
    "A100 80GB PCIe": (9.7, 19.5, 1935),
}
WORK = {"dense": 1152, "parity": 288}


def per_element_ns(flops, bytes_, tflops, gbs):
    return max(flops / (tflops * 1e3), bytes_ / gbs)   # ns per element


rows = []
for dev, (f64, f32, bw) in DEV.items():
    for prec, peak, b in (("FP64", f64, 8), ("FP32", f32, 4)):
        traffic = 2 * 24 * b
        ridge = peak * 1e3 / bw
        td = per_element_ns(WORK["dense"], traffic, peak, bw)
        tp = per_element_ns(WORK["parity"], traffic, peak, bw)
        rows.append({"device": dev, "precision": prec,
                     "calculation_status": "revision for measured SKU (2026-09-27)" if dev == "A100 80GB PCIe" else "original declared model",
                     "peak_tflops_non_tensor": peak, "dram_bandwidth_gb_s": bw,
                     "specification_source": "https://www.nvidia.com/en-us/data-center/a100/" if dev.startswith("A100") else "original declared vendor specification",
                     "traffic_model_bytes_per_element": traffic,
                     "ridge_flop_per_byte": ridge,
                     "dense_intensity": WORK["dense"] / traffic,
                     "dense_bound": "compute" if WORK["dense"] / traffic > ridge else "bandwidth",
                     "parity_bound": "compute" if WORK["parity"] / traffic > ridge else "bandwidth",
                     "ideal_ratio": td / tp})
out = Path(__file__).resolve().parents[1] / "paper"
(out / "roofline.json").write_text(json.dumps(rows, indent=1))
L = ["Idealized 48-value traffic accounting; these are specification-based predictions, not measurements.",
     "The original A100 SXM declaration is preserved. The PCIe row is a revision calculation for the measured SKU,",
     "using [NVIDIA's specifications](https://www.nvidia.com/en-us/data-center/a100/) (9.7 FP64 TFLOP/s, 19.5 FP32 TFLOP/s, 1935 GB/s).",
     "Shared-node reuse, caches, atomics, moduli and masks prevent identifying this traffic with measured DRAM bytes.", "",
     "| device | precision | model status | ridge (FLOP/B) | dense intensity | dense bound | parity bound | ideal dense/parity |",
     "|---|---|---|---:|---:|---|---|---:|"]
for r in rows:
    L.append(f"| {r['device']} | {r['precision']} | {r['calculation_status']} | {r['ridge_flop_per_byte']:.2f} | {r['dense_intensity']:.1f} | "
             f"{r['dense_bound']} | {r['parity_bound']} | {r['ideal_ratio']:.2f} |")
(out / "roofline.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
