"""Prospective multi-candidate timing decision; partial runs cannot pass."""
import numpy as np


def analyze(result, protocol):
    p = protocol
    cases, paths = p['cases'], p['paths']
    expected_products = {(c['id'], path, phase) for c in cases for path in paths
                         for phase in ('before', 'after')}
    timed = [c for c in cases if c['timed']]
    expected_timings = {(c['id'], path, rep) for c in timed for path in paths
                       for rep in range(p['benchmarks']['rounds'])}
    expected_states = {(c['id'], path) for c in cases if c['solve'] for path in p['new_paths']}
    products = result.get('products', [])
    timings = result.get('timings', [])
    solves, warmups = result.get('solves', []), result.get('warmups', [])
    def exactly(rows, fields, expected):
        keys = [tuple(r[x] for x in fields) for r in rows]
        return len(keys) == len(expected) and set(keys) == expected
    complete = (result['status'] == 'completed' and
        exactly(products, ('case', 'path', 'phase'), expected_products) and
        exactly(timings, ('case', 'path', 'round'), expected_timings) and
        exactly(solves, ('case', 'path'), expected_states) and
        exactly(warmups, ('case', 'path'), expected_states))
    physics = bool(complete and all(r['action_check']['passed'] and r['zero_exact'] and
        r['fixed_identity_exact'] for r in products) and
        all(r['physical_passed'] and r['cross_passed'] for r in solves) and
        all(not r['physical_passed'] and r['iterations'] == 1 for r in warmups))
    comparisons = []
    rng = np.random.default_rng(p['benchmarks']['bootstrap_seed'])
    if complete:
        for c in timed:
            arrays = {}
            for path in paths:
                rows = sorted((r for r in timings if r['case'] == c['id'] and r['path'] == path),
                              key=lambda r: r['round'])
                arrays[path] = np.array([r['gpu_ms_per_call'] for r in rows])
                assert np.isfinite(arrays[path]).all() and (arrays[path] > 0).all()
            for candidate in p['candidates']:
                baselines = p['baselines'] + (['modal8'] if candidate == 'modal32' else [])
                for baseline in baselines:
                    ratios = arrays[baseline] / arrays[candidate]
                    indices = rng.integers(0, len(ratios),
                        size=(p['benchmarks']['bootstrap_resamples'], len(ratios)))
                    medians = np.median(ratios[indices], axis=1)
                    median, lower = float(np.median(ratios)), float(np.quantile(medians, .05))
                    target = 1.10 if baseline == 'modal8' else 1.15
                    comparisons.append({'case': c['id'], 'primary': c['primary'],
                        'candidate': candidate, 'baseline': baseline,
                        'candidate_median_ms': float(np.median(arrays[candidate])),
                        'baseline_median_ms': float(np.median(arrays[baseline])),
                        'paired_median_ratio': median, 'one_sided_95_lower': lower,
                        'target': target, 'passed': bool(median >= target and lower > 1.05)})
    gates = {}
    for candidate in p['candidates']:
        rows = [r for r in comparisons if r['candidate'] == candidate and r['primary']]
        gates[candidate] = bool(physics and rows and all(r['passed'] for r in rows))
    return {'complete': complete, 'physics_passed': physics, 'comparisons': comparisons,
            'development_gates': gates, 'publication_goal_achieved': False,
            'charged_campaign_gate_changed': False,
            'scope': 'Fixed-input constrained product development; no end-to-end or novelty claim'}
