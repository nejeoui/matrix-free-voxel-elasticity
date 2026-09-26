"""Exercise the actual donor wrapper with NumPy buffers and a recording launch.

This checks the kernel argument contract on CPU. It does not emulate CUDA,
execute a numerical product, or establish GPU correctness.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np


class RecordingCuPy(types.ModuleType):
    def __init__(self):
        super().__init__('cupy')
        self.launches = []

    def __getattr__(self, name):
        return getattr(np, name)

    def RawKernel(self, code, name, **kwargs):
        def launch(grid, block, args):
            self.launches.append((name, args))
        return launch

    def RawModule(self, **kwargs):
        return types.SimpleNamespace(get_function=lambda name: self.RawKernel('', name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expect-table', choices=['present', 'missing'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    source = args.source.resolve()
    fake = RecordingCuPy()
    sys.modules['cupy'] = fake
    sys.path.insert(0, str(source / 'donor/src'))
    from gpu_fem.cuda_operators import OperatorSuite
    spec = importlib.util.spec_from_file_location('diagnosed_runner', source / 'run_gpu.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with np.load(args.input) as saved:
        data = {k: saved[k].copy() for k in saved.files}
    ctx = runner.Context(data, 'fused_fp64', fake, OperatorSuite)
    ne = int(np.prod(data['shape']))
    ctx.modulus = np.ones(ne)
    ctx.action(np.zeros_like(ctx.force))
    assert len(fake.launches) == 1 and fake.launches[0][0] == 'fused_matvec_fp64'
    table = fake.launches[0][1][0]
    if args.expect_table == 'missing':
        assert table is None
        checked = 0
    else:
        assert table is not None and table.dtype == np.int32
        nx, ny, nz = map(int, data['shape'])
        expected = []
        offsets = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),
                   (0,0,1),(1,0,1),(1,1,1),(0,1,1)]
        for ex in range(nx):
            for ey in range(ny):
                for ez in range(nz):
                    for dx,dy,dz in offsets:
                        node = ((ex+dx)*(ny+1)+(ey+dy))*(nz+1)+(ez+dz)
                        expected.extend([3*node+c for c in range(3)])
        np.testing.assert_array_equal(table, np.asarray(expected, dtype=np.int32))
        assert table.size == 24*ne and table.min() == 0 and table.max() < ctx.force.size
        checked = int(table.size)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {'passed': True, 'expected_table': args.expect_table,
               'actual_table_present': table is not None, 'indices_checked': checked,
               'actual_donor_wrapper_exercised': True, 'cuda_executed': False,
               'numerical_product_executed': False,
               'source_pins': {str(p): digest(p) for p in [source/'run_gpu.py',
                   source/'donor/src/gpu_fem/cuda_operators.py', Path(__file__).resolve()]}}
    args.out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
