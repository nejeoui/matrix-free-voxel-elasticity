"""Session I inputs: optimized designs upsampled by an integer factor, with the boundary data regenerated.

The base states (session E, from JPDC profile-v2) are cantilevers on an (nx, ny, nz) element grid in the
natural layout: element arrays (z, y, x), nodal arrays (z, y, x, 3). Their boundary data follow one rule,
which this script applies at any resolution:
  - clamp: every DOF on the x = 0 face;
  - load: a consistent line load in -z along y at x = nx, z = 0, total 1: -1/ny at interior nodes and
    -1/(2 ny) at the two end nodes.
Upsampling repeats each element f times per axis (nearest neighbour), so the design is geometrically the
same at f times the resolution. `regenerate(shape)` at f = 1 reproduces the base mask and force exactly
(tested).

    python3 make_inputs.py --base DIR --out DIR --cases q64x2 q96x2 ...
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CASES = {  # id: (base file, factor)
    "q4x2": ("smoke-q4.npz", 2),          # CPU dry-run checks only (16x8x8)
    "q96x1": ("optimized-q96.npz", 1),
    "q64x2": ("optimized-q64.npz", 2),
    "q96x2": ("optimized-q96.npz", 2),
    "q96x3": ("optimized-q96.npz", 3),
}


def regenerate(shape):
    nx, ny, nz = (int(n) for n in shape)
    mask = np.ones((nz + 1, ny + 1, nx + 1, 3))
    mask[:, :, 0, :] = 0.0
    force = np.zeros((nz + 1, ny + 1, nx + 1, 3))
    line = np.full(ny + 1, -1.0 / ny)
    line[[0, -1]] = -0.5 / ny
    force[0, :, nx, 2] = line
    return mask.ravel(), force.ravel()


def upsample(data, factor):
    nx, ny, nz = (int(n) for n in data["shape"])
    rho = data["rho"].reshape(nz, ny, nx)
    for axis in range(3):
        rho = np.repeat(rho, factor, axis=axis)
    shape = np.array([nx * factor, ny * factor, nz * factor])
    mask, force = regenerate(shape)
    return {"shape": shape, "rho": np.ascontiguousarray(rho.ravel()), "mask": mask, "force": force,
            "ke": data["ke"]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--base", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--cases", nargs="+", required=True)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in args.cases:
        base, factor = CASES[cid]
        with np.load(args.base / base) as f:
            data = {k: f[k] for k in f.files}
        up = upsample(data, factor)
        path = args.out / f"{cid}.npz"
        np.savez(path, **up)                      # uncompressed: fast to write and read on the remote
        rows.append({"id": cid, "base": base, "base_sha256": sha(args.base / base), "factor": factor,
                     "shape": [int(s) for s in up["shape"]], "elements": int(np.prod(up["shape"])),
                     "ndof": int(up["mask"].size), "file": path.name, "sha256": sha(path)})
        print(json.dumps(rows[-1]), flush=True)
    (args.out / "generated.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
