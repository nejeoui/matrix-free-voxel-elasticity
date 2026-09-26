"""CPU tests of the session D decisions (no GPU)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis  # noqa: E402

P = json.loads((HERE / 'protocol_template.json').read_text())


def result(modal_speed=1.10, vary=False, missing_round=False, status='completed'):
    fixed, tol = [], []
    for c in P['cases']:
        for rnd in range(P['fixed_work']['rounds']):
            if missing_round and c['id'] == 'optimized-q64' and rnd == 0:
                continue
            for arm in P['fixed_work']['arms']:
                s = 1.0 + 0.01 * rnd
                if arm == 'mg64-modal8':
                    s /= modal_speed
                it = 15 + (1 if vary and arm == 'mg64-modal8' and rnd == 2 else 0)
                fixed.append({'case': c['id'], 'arm': arm, 'round': rnd, 'iterations': it,
                              'outer_matvecs': it + 1, 'solver_seconds': s})
        for arm in P['arms']:
            for rep in range(P['to_tolerance']['repetitions']):
                tol.append({'case': c['id'], 'arm': arm['id'], 'rep': rep, 'iterations': 15,
                            'solver_seconds': 0.5 if arm['precision'] == 'fp32' else 1.0, 'physical_passed': True})
    return {'status': status, 'fixed_work': fixed, 'to_tolerance': tol}


def test_h6_survives_with_a_clear_margin():
    assert analysis.analyze(result(1.10), P)['h6_decision'] == 'fp64_ranking_survives_in_multigrid'


def test_h6_fails_below_the_median_threshold():
    assert analysis.analyze(result(1.03), P)['h6_decision'] != 'fp64_ranking_survives_in_multigrid'


def test_h6_fails_on_unequal_work():
    assert analysis.analyze(result(1.5, vary=True), P)['h6_decision'] != 'fp64_ranking_survives_in_multigrid'


def test_h6_fails_on_a_missing_round_or_incomplete_run():
    assert analysis.analyze(result(1.5, missing_round=True), P)['h6_decision'] != 'fp64_ranking_survives_in_multigrid'
    assert analysis.analyze(result(1.5, status='failed'), P)['h6_decision'] != 'fp64_ranking_survives_in_multigrid'


def test_fastest_passing_ignores_failed_arms():
    r = result()
    for x in r['to_tolerance']:
        if x['arm'].startswith('mg32'):
            x['physical_passed'] = x['rep'] != 0        # every mixed arm fails once
    d = analysis.analyze(r, P)
    assert all(v['arm'].startswith('mg64') for v in d['fastest_passing'].values())


def test_protocol_arms_reference_known_kernels():
    known = {'fused_ai_fp64', 'fused_fp64', 'node_fp64', 'dense8', 'modal8',
             'node_fp32', 'fused_ai_fp32', 'modal8_fp32'}
    for a in P['arms']:
        assert a['outer'] in known and a['vcycle'] in known
        assert (a['precision'] == 'fp32') == a['vcycle'].endswith('_fp32')
        assert not a['outer'].endswith('_fp32')
