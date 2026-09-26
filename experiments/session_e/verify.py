"""Independent CPU replay of Voxel session E states, then the declared decisions (H8, H9, H10)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dependencies'))
from common import Reference, now, read, relative, sha, write
from analysis import analyze


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--inputs', type=Path, required=True)
    a.add_argument('--reference', type=Path, required=True)
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    assert not args.out.exists()
    p, result = read(HERE / 'protocol.json'), read(args.run / 'result.json')
    assert result['protocol_sha256'] == sha(HERE / 'protocol.json')
    for filename, digest in p['package_source_pins'].items():
        assert sha(HERE / filename) == digest, filename
    manifest = read(args.inputs / 'inputs.json')
    assert manifest['protocol_sha256'] == result['protocol_sha256']
    ref = Reference(args.reference)
    solver = p['solver']
    replays, mismatches = [], []
    for case in p['cases']:
        src = case.get('input', case['id'])
        row = next(r for r in manifest['rows'] if r['id'] == src)
        assert sha(args.inputs / row['file']) == row['sha256']
        with np.load(args.inputs / row['file']) as f:
            data = {k: np.ascontiguousarray(f[k]) if f[k].ndim else f[k].copy() for k in f.files}
        data['floor'] = np.asarray(case.get('Emin', p['material']['Emin']))
        modulus = float(data['floor']) + (1 - float(data['floor'])) * data['rho'] ** 3
        for r in [r for r in result['to_tolerance'] if r['case'] == case['id'] and 'file' in r]:
            assert sha(args.run / r['file']) == r['sha256']
            with np.load(args.run / r['file']) as f:
                s = f['solution']
            action, energy, _ = ref.product(data, s)
            res = relative(action, data['force'])
            comp, work = float(modulus @ energy), float(data['force'] @ s)
            passed = bool(r['native_converged'] and r['native_true_residual'] <= solver['requested_rtol']
                          and res <= solver['true_residual_limit']
                          and abs(comp / work - 1) <= solver['energy_force_relative_limit']
                          and np.isfinite(s).all() and np.all(s[data['mask'] == 0] == 0))
            replays.append({'case': case['id'], 'arm': r['arm'], 'independent_true_residual': res,
                            'compliance': comp, 'physical_passed': passed})
            if passed != r['physical_passed']:
                mismatches.append(replays[-1])
    decision = analyze(result, p)
    write(args.out, {'verified_utc': now(), 'evidence_valid': not mismatches, 'result_status': result['status'],
                     'result_sha256': sha(args.run / 'result.json'), 'protocol_sha256': sha(HERE / 'protocol.json'),
                     'replays': replays, 'replay_mismatches': mismatches, 'decision': decision})
    print(json.dumps({'evidence_valid': not mismatches, 'h8': decision['h8_decision'], 'h9': decision['h9_decision'],
                      'product_checks_passed': decision['product_checks_passed'], 'fastest': decision['fastest_passing']}))


if __name__ == '__main__':
    main()
