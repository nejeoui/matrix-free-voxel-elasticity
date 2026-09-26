"""Independently check the paused campaign's frozen inputs and source identities."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'pending_gpu_common', ROOT / 'rescope/hex_modal_gpu_application/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--donor', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    p = common.protocol()
    for filename, digest in p['application_source_pins'].items():
        assert common.sha(args.inputs / 'application-source' / filename) == digest, filename
    for filename, digest in p['donor']['files'].items():
        assert common.sha(args.donor / filename) == digest, filename
    receipt = json.loads((args.inputs / 'large-inputs.json').read_text())
    assert receipt['protocol_sha256'] == common.sha(common.HERE / 'protocol.json')
    assert receipt['endpoint_sha256'] == common.sha(args.inputs / 'q8-final.bin')
    assert receipt['endpoint_sha256'] == p['large_solves']['endpoint_sha256']
    coarse = common.reference.vectors(args.inputs / 'q8-final.bin')[1].reshape(8, 8, 16)
    reference = common.reference.Reference(args.reference.resolve())
    assert len(receipt['rows']) == len(p['large_solves']['cases'])
    rows = []
    for case, saved in zip(p['large_solves']['cases'], receipt['rows']):
        assert all(saved[k] == v for k, v in case.items())
        filename = args.inputs / saved['file']
        assert common.sha(filename) == saved['sha256']
        with np.load(filename) as data:
            q = case['q']
            assert tuple(data['shape']) == (2 * q, q, q)
            rho = data['rho']
            assert rho.dtype == np.float64 and rho.size == 2 * q ** 3
            assert np.isfinite(rho).all() and rho.min() >= 0 and rho.max() <= 1
            # Direct coarse-cell lookup is independent of the generator's
            # repeated-array construction.
            if case['field'] == 'uniform_0.12':
                assert np.all(rho == .12)
            else:
                iz, iy, ix = np.indices((q, q, 2 * q))
                expected = coarse[iz // (q // 8), iy // (q // 8), ix // (q // 8)]
                assert np.array_equal(rho.reshape(q, q, 2 * q), expected)
            mask = data['mask'].reshape(q + 1, q + 1, 2 * q + 1, 3)
            assert np.all(mask[:, :, 0, :] == 0) and np.all(mask[:, :, 1:, :] == 1)
            force = data['force'].reshape(q + 1, q + 1, 2 * q + 1, 3)
            expected_force = np.zeros_like(force)
            expected_force[0, :, -1, 2] = -1 / q
            expected_force[0, [0, -1], -1, 2] /= 2
            assert np.array_equal(force, expected_force)
            assert np.array_equal(data['ke'], reference.ke(1 / q, p['material']['nu']))
            assert np.isclose(rho.mean(), saved['physical_volume_fraction'], rtol=0, atol=1e-14)
            rows.append({'case': case['id'], 'elements': int(rho.size),
                         'dofs': int(force.size), 'physical_volume_fraction': float(rho.mean()),
                         'arrays_verified': True, 'sha256': common.sha(filename)})
    common.write(args.out, {'verified_utc': datetime.now(timezone.utc).isoformat(),
                           'passed': True, 'protocol_sha256': common.sha(common.HERE / 'protocol.json'),
                           'application_sources_checked': len(p['application_source_pins']),
                           'donor_sources_checked': len(p['donor']['files']),
                           'dependencies_checked': len(p['dependencies']) + len(p['additional_source_pins']),
                           'input_cases': rows, 'gpu_executed': False,
                           'performance_claim': False, 'novelty_claim': False})
    print(json.dumps({'passed': True, 'input_cases': len(rows)}))


if __name__ == '__main__':
    main()
