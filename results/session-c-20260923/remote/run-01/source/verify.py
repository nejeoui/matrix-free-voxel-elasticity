"""Independent CPU replay of every retained Voxel session A state, then the declared decisions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dependencies'))
from common import Reference, now, read, relative, sha, write
from analysis import extended as analyze


def physical(data, solution, ref, solver):
    action, energy, _ = ref.product(data, solution)
    modulus = float(data['floor']) + (1 - float(data['floor'])) * data['rho'] ** 3
    compliance, work = float(modulus @ energy), float(data['force'] @ solution)
    return {'independent_true_residual': relative(action, data['force']), 'compliance': compliance,
            'energy_force_relative_error': abs(compliance / work - 1) if work else float('inf')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    assert not a.out.exists()
    p, result = read(HERE / 'protocol.json'), read(a.run / 'result.json')
    assert result['protocol_sha256'] == sha(HERE / 'protocol.json')
    for filename, digest in p['package_source_pins'].items():
        assert sha(HERE / filename) == digest, filename
    manifest = read(a.inputs / 'inputs.json')
    assert manifest['protocol_sha256'] == result['protocol_sha256']
    ref = Reference(a.reference)
    solver = p['solver']
    checks = {'fixed_work_states': [], 'to_tolerance_states': [], 'work_identity': []}
    for case in p['cases']:
        row = next(r for r in manifest['rows'] if r['id'] == case.get('input', case['id']))
        assert sha(a.inputs / row['file']) == row['sha256']
        with np.load(a.inputs / row['file']) as f:
            data = {k: np.ascontiguousarray(f[k]) if f[k].ndim else f[k].copy() for k in f.files}
        assert tuple(map(int, data['shape'])) == (2 * case['q'], case['q'], case['q'])
        data['floor'] = np.asarray(case.get('Emin', p['material']['Emin']))
        fixed = [r for r in result['fixed_work'] if r['case'] == case['id']]
        work = sorted({(r['iterations'], r['matvec_calls']) for r in fixed})
        checks['work_identity'].append({'case': case['id'], 'distinct_work': work,
                                        'identical': len(work) == 1})
        retained = {r['path']: r for r in fixed if 'file' in r}
        base = retained.get(p['decision']['h2']['baseline'])
        base_state = None
        if base is not None:
            with np.load(a.run / base['file']) as f:
                base_state = f['solution']
            base_phys = physical(data, base_state, ref, solver)
        for path, r in retained.items():
            assert sha(a.run / r['file']) == r['sha256']
            with np.load(a.run / r['file']) as f:
                s = f['solution']
            ph = physical(data, s, ref, solver)
            assert abs(ph['independent_true_residual'] - r['independent_true_residual']) <= \
                1e-12 + 1e-5 * ph['independent_true_residual']
            entry = {'case': case['id'], 'path': path, **ph}
            if base_state is not None:
                entry['state_relative_to_baseline'] = relative(s, base_state)
                entry['compliance_relative_to_baseline'] = abs(ph['compliance'] / base_phys['compliance'] - 1)
                entry['consistent'] = bool(entry['state_relative_to_baseline'] <= solver['cross_state_limit']
                                           and entry['compliance_relative_to_baseline'] <= solver['cross_compliance_limit'])
            checks['fixed_work_states'].append(entry)
        for r in [r for r in result['to_tolerance'] if r['case'] == case['id']]:
            if r.get('timeout'):
                assert r['physical_passed'] is False and 'file' not in r
                checks['to_tolerance_states'].append({'case': case['id'], 'method': r['method'], 'rep': r['rep'],
                                                      'physical_passed': False, 'timeout': True})
                continue
            assert sha(a.run / r['file']) == r['sha256']
            with np.load(a.run / r['file']) as f:
                s = f['solution']
            ph = physical(data, s, ref, solver)
            passed = bool(r['native_converged'] and r['native_true_residual'] <= solver['requested_rtol']
                          and ph['independent_true_residual'] <= solver['true_residual_limit']
                          and ph['energy_force_relative_error'] <= solver['energy_force_relative_limit']
                          and np.isfinite(s).all() and np.all(s[data['mask'] == 0] == 0))
            assert passed == r['physical_passed'], (r['file'], passed)
            checks['to_tolerance_states'].append({'case': case['id'], 'method': r['method'], 'rep': r['rep'],
                                                  'physical_passed': passed, **ph})
    decision = analyze(result, p)
    write(a.out, {'verified_utc': now(), 'evidence_valid': True, 'result_status': result['status'],
                  'result_sha256': sha(a.run / 'result.json'), 'protocol_sha256': sha(HERE / 'protocol.json'),
                  'product_checks': result['product_checks'], 'checks': checks, 'decision': decision})
    print(json.dumps({'evidence_valid': True, **{k: v for k, v in decision.items() if k.endswith('_decision')},
                      'fastest': decision.get('fastest_passing_method')}))


if __name__ == '__main__':
    main()
