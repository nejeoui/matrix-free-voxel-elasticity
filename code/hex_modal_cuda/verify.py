"""Verify collected CUDA fields and decisions with independent CPU assembly."""
import argparse
from pathlib import Path

import numpy as np

from common import Reference, now, product_check, read, relative, sha, solver_check, timing_decision, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Use a fresh verification receipt'
    r = read(args.run / 'result.json')
    p = read(args.run / 'protocol.json')
    inputs = read(args.inputs / 'inputs.json')
    assert sha(args.run / 'protocol.json') == r['protocol_sha256'] == inputs['protocol_sha256']
    assert sha(args.inputs / 'inputs.json') == r['inputs_manifest_sha256']
    for name, digest in r['source_hashes'].items():
        assert sha(args.run / name) == digest
    for name, row in inputs['files'].items():
        assert sha(args.inputs / name) == row['sha256']
    ref = Reference(args.reference)
    product_count, normal, negative = 0, 0, 0
    worst_residual = 0.
    failures = []
    for source in inputs['rows']:
        with np.load(args.inputs / source['file']) as file:
            data = {k: np.ascontiguousarray(file[k]) if file[k].ndim else file[k].copy() for k in file.files}
        for row in [v for v in r['products'] if v['id'] == source['id']]:
            assert sha(args.run / row['file']) == row['sha256']
            with np.load(args.run / row['file']) as a:
                action, zero = a['action'], a['zero']
                assert action.dtype == np.float64 and action.shape == data['target'].shape
                check = product_check(action, data['target'], p)
                z = bool(np.count_nonzero(zero) == 0)
                fixed = bool(np.array_equal(action[data['mask'] == 0], data['u'][data['mask'] == 0]))
                passed = check['passed'] and z and fixed
                if source['kind'] == 'saved_state':
                    passed = passed and relative(action, data['force']) <= p['correctness']['state_true_residual_limit']
                assert bool(passed) == row['passed'], (source['id'], row['path'])
                assert z == row['zero_exact'] and fixed == row['fixed_identity_exact']
                if not passed:
                    failures.append({'kind': 'product', 'id': source['id'], 'path': row['path']})
            product_count += 1
        for row in [v for v in r['solves'] if v['id'] == source['id']]:
            assert sha(args.run / row['file']) == row['sha256']
            with np.load(args.run / row['file']) as a:
                check, action, energy = solver_check(a['solution'], data, ref, p, row['native_converged'], row['native_true_residual'])
                assert product_check(a['action'], action, p)['passed']
                assert relative(a['energy'], energy) <= 1e-10
                assert check['physical_passed'] == row['physical_passed']
                passed = not check['physical_passed'] if row['cap_one'] else check['physical_passed']
                assert passed == row['passed']
                if row['cap_one']:
                    negative += 1
                else:
                    normal += 1
                    worst_residual = max(worst_residual, check['independent_true_residual'])
                if not passed:
                    failures.append({'kind': 'solver', 'id': source['id'], 'path': row['path'], 'replicate': row['replicate']})
    comparisons = []
    if r['status'] == 'completed':
        assert product_count == 126 and normal == 72 and negative == 6
        assert len(r['timings']) == 264
        for identifier in {v['id'] for v in r['timings']}:
            for iteration in range(p['benchmarks']['rounds']):
                group = sorted((v for v in r['timings'] if v['id'] == identifier and v['round'] == iteration), key=lambda v: v['position'])
                assert [v['position'] for v in group] == list(range(6))
                assert sorted(v['path'] for v in group) == sorted(p['paths'])
                assert all(v['calls'] == p['benchmarks']['calls_per_batch'] for v in group)
        decision = timing_decision(r['timings'], p)
        assert decision['timing_gate_passed'] == r['timing_gate_passed']
        for original, recomputed in zip(r['comparisons'], decision['comparisons']):
            assert original['id'] == recomputed['id'] and original['baseline'] == recomputed['baseline']
            assert original['passed'] == recomputed['passed']
            for key in ('paired_median_ratio', 'paired_bootstrap_lower_95'):
                assert np.isclose(original[key], recomputed[key], rtol=1e-13, atol=1e-13)
        assert r['correctness_gate_passed'] == (not failures)
        assert r['development_gate_passed'] == (not failures and decision['timing_gate_passed'])
        comparisons = decision['comparisons']
    else:
        assert r['status'] == 'failed' and not r['development_gate_passed'] and r.get('error')
    output = {'checked_utc': now(), 'verified': True, 'execution_status': r['status'],
              'result_sha256': sha(args.run / 'result.json'), 'independent_reference_sha256': sha(args.reference),
              'products_checked': product_count, 'normal_solves_replayed': normal, 'negative_solves_replayed': negative,
              'maximum_normal_true_residual': worst_residual, 'retained_failures': failures,
              'timing_comparisons_recomputed': len(comparisons), 'comparisons': comparisons,
              'development_gate_passed': r['development_gate_passed'], 'publication_goal_achieved': False}
    write(args.out, output)
    print(__import__('json').dumps({k: v for k, v in output.items() if k != 'comparisons'}, indent=2))


if __name__ == '__main__':
    main()
