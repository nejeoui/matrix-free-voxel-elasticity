"""CPU tests for Voxel session A (no GPU). Run from the Voxel root:
    python -m pytest experiments/session_a -q
"""
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'dependencies'))
sys.path.insert(0, str(ROOT / 'src'))
import analysis  # noqa: E402

P = json.loads((HERE / 'protocol_template.json').read_text())


def synthetic(ratio_q64=1.15, ratio_q96=1.14, fp32_ratio=1.01, vary_work=False, drop_round=False):
    """A complete fake result with modal8 `ratio` times faster than every FP64 baseline."""
    fixed, products = [], []
    for case in P['cases']:
        ratio = {'optimized-q64': ratio_q64, 'optimized-q96': ratio_q96}.get(case['id'], 1.1)
        for rnd in range(P['fixed_work']['rounds']):
            if drop_round and rnd == 0 and case['id'] == 'optimized-q64':
                continue
            for path in P['paths_fp64']:
                base = 10.0 + 0.01 * rnd
                seconds = base / ratio if path == 'modal8' else base
                iterations = case['fixed_iterations']
                if vary_work and path == 'modal8' and rnd == 3:
                    iterations += 50
                fixed.append({'case': case['id'], 'path': path, 'round': rnd, 'iterations': iterations,
                              'matvec_calls': iterations + iterations // 50 + 1, 'solver_seconds': seconds})
        for precision, paths, r in (('fp64', P['paths_fp64'], 1.25), ('fp32', P['paths_fp32'], fp32_ratio)):
            rounds = []
            cand = P['candidate_fp64'] if precision == 'fp64' else P['candidate_fp32']
            for rnd in range(P['products']['rounds']):
                ms = {p: (1.0 / r if p == cand else 1.0 + 0.1 * i) for i, p in enumerate(paths)}
                rounds.append({'round': rnd, 'order': paths, 'ms_per_product': ms})
            products.append({'case': case['id'], 'precision': precision, 'rounds': rounds})
    return {'status': 'completed', 'fixed_work': fixed, 'products': products, 'to_tolerance': []}


def test_h2_passes_when_both_primaries_clear_the_rule():
    d = analysis.analyze(synthetic(), P)
    assert d['h2_passed'] == {'optimized-q64': True, 'optimized-q96': True}
    assert d['h2_decision'] == 'continue_modal8'


def test_h2_stops_when_one_primary_is_below_the_median_threshold():
    d = analysis.analyze(synthetic(ratio_q96=1.06), P)
    assert d['h2_passed']['optimized-q96'] is False
    assert d['h2_decision'] == 'stop_modal8'


def test_h2_rejects_unequal_work_even_with_a_large_ratio():
    """The exact flaw that failed the JPDC profile gate must fail here too."""
    d = analysis.analyze(synthetic(ratio_q64=1.5, vary_work=True), P)
    assert d['h2_passed']['optimized-q64'] is False
    assert d['h2_decision'] == 'stop_modal8'


def test_h2_rejects_a_missing_round():
    d = analysis.analyze(synthetic(drop_round=True), P)
    assert d['h2_passed']['optimized-q64'] is False


def test_h2_is_false_on_an_incomplete_run():
    r = synthetic()
    r['status'] = 'failed'
    assert analysis.analyze(r, P)['h2_decision'] == 'stop_modal8'


def test_h4_confirmed_only_when_fp32_gain_is_small():
    assert analysis.analyze(synthetic(fp32_ratio=1.02), P)['h4_decision'] == 'predicted_no_fp32_gain_confirmed'
    assert analysis.analyze(synthetic(fp32_ratio=1.30), P)['h4_decision'] != 'predicted_no_fp32_gain_confirmed'


def test_h4_compares_against_the_fastest_fp32_baseline():
    d = analysis.analyze(synthetic(fp32_ratio=1.02), P)
    fastest = [e for e in d['products'] if e['precision'] == 'fp32' and e['fastest_baseline']]
    assert fastest and all(e['baseline'] == P['paths_fp32'][0] for e in fastest)


def test_bootstrap_lower_bound_is_below_the_median_for_noisy_ratios():
    ratios = [1.10, 1.12, 1.09, 1.15, 1.11, 1.13, 1.08, 1.12, 1.10, 1.14]
    assert analysis.bootstrap_lower(ratios, seed=1) <= sorted(ratios)[len(ratios) // 2]


def test_fp32_kernel_source_matches_the_tested_generator():
    import cuda32
    import voxel_kernels
    assert cuda32.SOURCE32 == voxel_kernels.make_source('float')


def test_fp64_kernel_dependency_is_the_measured_prototype():
    import cuda
    import voxel_kernels
    assert cuda.SOURCE == voxel_kernels.PROTOTYPE_SOURCE


def test_runner_imports_without_a_gpu():
    run_gpu = importlib.import_module('run_gpu')
    assert hasattr(run_gpu, 'main') and hasattr(run_gpu, 'analyze') is False


def test_protocol_template_is_internally_consistent():
    assert P['candidate_fp64'] in P['paths_fp64'] and P['candidate_fp32'] in P['paths_fp32']
    assert P['decision']['h2']['baseline'] in P['paths_fp64']
    ops = set(P['paths_fp64'] + P['paths_fp32'])
    for m in P['to_tolerance']['methods']:
        for key in ('operator', 'outer', 'inner'):
            if key in m:
                assert m[key] in ops, m
    assert sum(c['primary'] for c in P['cases']) == 2
