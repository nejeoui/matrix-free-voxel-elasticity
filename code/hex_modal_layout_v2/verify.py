"""Replay every retained product and physical state with an independent CPU assembly."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'dependencies'))
from common import Reference, now, product_check, read, relative, sha, write
from run_gpu import load_data
from physical import physical_check
from analysis import analyze


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'reference', 'run', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(); assert not a.out.exists()
    p, result = read(HERE/'protocol.json'), read(a.run/'result.json')
    assert result['protocol_sha256'] == sha(HERE/'protocol.json')
    for name, digest in p['package_source_pins'].items():
        assert sha(HERE/name) == digest, name
    inputs = read(a.inputs/'inputs.json')
    assert inputs['protocol_sha256'] == result['protocol_sha256']
    for name, digest in inputs['reference_source_pins'].items():
        assert sha(a.inputs/name) == digest, name
    ref = Reference(a.reference)
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    ref.lib.element.argtypes = [ctypes.c_double, ctypes.c_double, array]
    ref.lib.element.restype = None
    product_rows, state_rows = [], []
    for case in p['cases']:
        row = next(r for r in inputs['rows'] if r['id'] == case['id'])
        assert sha(a.inputs/row['file']) == row['sha256']
        data = load_data(a.inputs/row['file'])
        ke = np.empty((24,24)); ref.lib.element(float(data['h']), float(data['nu']), ke)
        assert relative(data['ke'], ke) <= 1e-14
        assert tuple(data['shape']) == tuple(case['shape'])
        assert np.isfinite(data['u']).all() and np.isfinite(data['rho']).all()
        assert np.all((data['rho'] >= 0) & (data['rho'] <= 1))
        target, energy, _ = ref.product(data, data['u'])
        modulus = float(data['floor'])+(1-float(data['floor']))*data['rho']**3
        base_compliance = float(modulus @ energy)
        if case['solve']:
            assert relative(target, data['force']) <= p['solver']['true_residual_limit']
        for record in [r for r in result['products'] if r['case'] == case['id']]:
            assert sha(a.run/record['file']) == record['sha256']
            with np.load(a.run/record['file']) as file:
                actual, zero = file['action'], file['zero']
            assert actual.shape == data['u'].shape and actual.dtype == np.float64
            assert zero.shape == actual.shape and zero.dtype == np.float64
            check = product_check(actual, target, p)
            fixed = bool(np.array_equal(actual[data['mask'] == 0], data['u'][data['mask'] == 0]))
            zero_exact = bool(np.all(zero == 0))
            assert check['passed'] == record['action_check']['passed']
            assert fixed == record['fixed_identity_exact'] and zero_exact == record['zero_exact']
            if result['status'] == 'completed':
                assert check['passed'] and fixed and zero_exact
            product_rows.append({'file': record['file'], **check, 'fixed_identity_exact': fixed, 'zero_exact': zero_exact})
        for record in [r for r in result['warmups']+result['solves'] if r['case'] == case['id']]:
            assert sha(a.run/record['file']) == record['sha256']
            with np.load(a.run/record['file']) as file:
                solution = file['solution']
            assert solution.shape == data['u'].shape and solution.dtype == np.float64
            check = physical_check(data, solution, ref, record['native_converged'], record['native_true_residual'], p['solver'])
            state_error = relative(solution, data['u'])
            compliance_error = abs(check['compliance']/base_compliance-1)
            cross = bool(state_error <= p['solver']['cross_state_limit'] and compliance_error <= p['solver']['cross_compliance_limit'])
            assert check['physical_passed'] == record['physical_passed'] and cross == record['cross_passed']
            assert abs(check['independent_true_residual']-record['independent_true_residual']) <= 1e-12+1e-5*check['independent_true_residual']
            assert abs(check['compliance']/record['compliance']-1) <= 1e-11
            if record['cap_one']:
                assert record['iterations'] == 1 and not check['physical_passed']
            elif result['status'] == 'completed':
                assert check['physical_passed'] and cross
            state_rows.append({'file': record['file'], 'capped': record['cap_one'], **check,
                'cross_state_relative': state_error, 'cross_compliance_relative': compliance_error, 'cross_passed': cross})
    decision = analyze(result, p)
    assert decision == read(a.run/'decision.json')
    # Check balanced randomized blocks independently of the ratio calculation.
    for case in [c for c in p['cases'] if c['timed']]:
        for rep in range(p['benchmarks']['rounds']):
            rows = [r for r in result['timings'] if r['case'] == case['id'] and r['round'] == rep]
            if result['status'] == 'completed':
                assert sorted(r['position'] for r in rows) == list(range(len(p['paths'])))
            for row in rows:
                assert row['calls'] == p['benchmarks']['calls_per_round']
                assert row['host_ms_per_call'] > 0 and np.isfinite(row['host_ms_per_call'])
    write(a.out, {'verified_utc': now(), 'evidence_valid': True, 'products_replayed': len(product_rows),
        'states_replayed': len(state_rows), 'product_checks': product_rows, 'state_checks': state_rows,
        'decision': decision, 'result_sha256': sha(a.run/'result.json'), 'protocol_sha256': sha(HERE/'protocol.json')})
    print(json.dumps({'evidence_valid': True, 'products_replayed': len(product_rows),
                     'states_replayed': len(state_rows), 'development_gates': decision['development_gates']}))


if __name__ == '__main__':
    main()
