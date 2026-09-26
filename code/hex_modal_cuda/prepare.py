"""Freeze portable inputs and source for the separately declared CUDA study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from rescope.floor_work.common import Reference
from rescope.hex_modal.modal import Modal
from rescope.hex_modal.run import load_case


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads((HERE / 'protocol.json').read_text())
    cpu_path = ROOT / 'results/rescope/hex-modal-20260923/cpu-01/result.json'
    assert sha(cpu_path) == p['prerequisite_cpu_result_sha256']
    assert json.loads(cpu_path.read_text())['development_gate_passed']
    prior = json.loads((ROOT / 'rescope/hex_modal/protocol.json').read_text())
    ref = Reference(ROOT / prior['reference_build'])
    for name in ('reference.cc', 'petsc_element.inc'):
        shutil.copyfile(ROOT / prior['reference_build'] / name, out / name)
    for name in ('prepare.py', 'cuda.py', 'protocol.json'):
        shutil.copyfile(HERE / name, out / name)
    donor = ROOT / 'rescope/matched_topopt/_deps/Fused-Gather-GEMM-Scatter-Kernels'
    for name, digest in p['donor_files'].items():
        assert sha(donor / name) == digest
        dest = out / 'donor' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(donor / name, dest)
    shutil.copyfile(donor / 'LICENSE', out / 'donor/LICENSE')
    rows = []

    def save(identifier, kind, shape, mask, force, rho, u, ke, floor):
        op = Modal(ke)
        target, energy, diagonal = ref.product(shape, rho, u, ke.ravel(), mask, floor)
        path = out / (identifier + '.npz')
        np.savez_compressed(path, shape=shape, mask=mask, force=force, rho=rho, u=u,
                            ke=ke.reshape(24, 24), floor=floor, blocks=op.blocks,
                            target=target, energy=energy, diagonal=diagonal)
        row = {'id': identifier, 'kind': kind, 'shape': list(shape), 'floor': floor,
               'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
               'discarded_modal_relative': op.discarded_relative}
        rows.append(row)
        print(json.dumps(row), flush=True)

    for case in prior['cases']:
        shape, mask, force, rho, u, ke = load_case(case)
        save(case['id'], 'saved_state', shape, mask, force, rho, u, ke, case['floor'])
    rng = np.random.default_rng(p['benchmarks']['seed'])
    for kind, shapes in [('small_probe', p['correctness']['synthetic_small_shapes']),
                         ('benchmark', p['benchmarks']['shapes'])]:
        for shape in shapes:
            nx, ny, nz = shape
            mask = np.ones((nz + 1, ny + 1, nx + 1, 3))
            mask[:, :, 0] = 0
            mask = np.ascontiguousarray(mask.ravel())
            z, y, x = np.indices((nz, ny, nx))
            rho = np.ascontiguousarray((.02 + .98 * (.5 + .5 * np.sin(2 * np.pi * (x + .5) / nx)
                          * np.cos(2 * np.pi * (y + .5) / ny) * np.sin(2 * np.pi * (z + .5) / nz))).ravel())
            u = np.ascontiguousarray(rng.standard_normal(mask.size))
            ke = ref.ke(16)
            force = np.zeros_like(u)
            save(kind + '-' + 'x'.join(map(str, shape)), kind, shape, mask, force, rho, u, ke, p['benchmarks']['floor'])
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'complete': True,
               'protocol_sha256': sha(HERE / 'protocol.json'), 'rows': rows,
               'files': {str(f.relative_to(out)): {'bytes': f.stat().st_size, 'sha256': sha(f)}
                         for f in sorted(out.rglob('*')) if f.is_file()},
               'gpu_executed': False, 'performance_claim': False}
    (out / 'inputs.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'complete': True, 'inputs': len(rows), 'total_bytes': sum(x['bytes'] for x in receipt['files'].values())}))


if __name__ == '__main__':
    main()
