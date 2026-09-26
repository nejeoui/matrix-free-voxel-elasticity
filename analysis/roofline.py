"""Ideal-traffic roofline prediction for the voxel Hex8 element product, dense vs parity (modal8).

A PREDICTION, not a measurement. Intensity uses the minimal per-element traffic of a fused
gather-product-scatter kernel: read 24 displacement values and write 24 results
(2 x 24 x bytes_per_value), the model used in Yang et al. (arXiv 2604.18020) for its FP32 kernel.
Dense work 2 x 576 flops; parity work 144 butterfly additions + 2 x 72 = 288 flops (implementation
README). Device peaks are vendor specification values. Real kernels carry extra traffic (density,
indices, atomics) and non-flop instructions, so measured ratios are expected to be smaller than
these ideal ratios; the point is WHERE a ratio > 1 is possible at all.
"""
import json
from pathlib import Path

DEV = {  # FP64 non-tensor TFLOP/s, FP32 TFLOP/s, DRAM GB/s
    "RTX 3090": (0.556, 35.58, 936), "RTX 4090": (1.290, 82.58, 1008),
    "A100 80GB SXM": (9.7, 19.5, 2039), "H100 SXM": (34.0, 67.0, 3350),
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
        rows.append({"device": dev, "precision": prec, "ridge_flop_per_byte": ridge,
                     "dense_intensity": WORK["dense"] / traffic,
                     "dense_bound": "compute" if WORK["dense"] / traffic > ridge else "bandwidth",
                     "parity_bound": "compute" if WORK["parity"] / traffic > ridge else "bandwidth",
                     "ideal_ratio": td / tp})
out = Path(__file__).resolve().parents[1] / "paper"
(out / "roofline.json").write_text(json.dumps(rows, indent=1))
L = ["| device | precision | ridge (FLOP/B) | dense intensity | dense bound | parity bound | ideal dense/parity |",
     "|---|---|---:|---:|---|---|---:|"]
for r in rows:
    L.append(f"| {r['device']} | {r['precision']} | {r['ridge_flop_per_byte']:.2f} | {r['dense_intensity']:.1f} | "
             f"{r['dense_bound']} | {r['parity_bound']} | {r['ideal_ratio']:.2f} |")
(out / "roofline.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
