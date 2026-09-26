"""CPU tests of the session E kernel variants and decisions (no GPU)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'dependencies'))
import analysis  # noqa: E402
import cuda  # noqa: E402

P = json.loads((HERE / 'protocol_template.json').read_text())
PRIMARY = [c['id'] for c in P['cases'] if c['primary']]


# ---------------------------------------------------------------- kernel variants
def _kernel(src, name):
    start = src.index(f'__global__ void {name}(')
    end = src.find('extern "C"', start + 1)
    return src[start:end if end >= 0 else len(src)].strip()


def test_variants_differ_from_modal8_only_in_the_butterfly():
    modal8_head = cuda.SOURCE[:cuda.SOURCE.index('extern "C" __global__ void dense8(')]
    for name, body in cuda._VARIANT_BUTTERFLY.items():
        variant = cuda._variant(name)
        back = variant.replace(body, cuda._BUTTERFLY).replace(f'transform_{name}(', 'transform(') \
                      .replace(f'__global__ void {name}(', '__global__ void modal8(')
        assert back == modal8_head


def test_every_modal_path_is_compiled_and_dispatched_with_blocks():
    assert set(cuda.MODAL_PATHS) == {'modal8', 'modal8_muladd', 'modal8_signbit'}
    full = cuda.SOURCE + cuda.VARIANT_SOURCE
    for name in cuda.MODAL_PATHS:
        assert full.count(f'__global__ void {name}(') == 1


def _butterfly(values, variant):
    v = np.array(values, dtype=np.float64)
    lanes = np.arange(8)
    for bit in (1, 2, 4):
        other = v[lanes ^ bit]
        neg = (lanes & bit) != 0
        if variant == 'modal8':
            v = np.where(neg, other - v, v + other)
        elif variant == 'modal8_muladd':      # sign*v+other; with |sign|=1 an FMA rounds identically
            v = np.where(neg, -1.0, 1.0) * v + other
        else:                                 # sign-bit flip, then add
            flipped = (v.view(np.uint64) ^ np.where(neg, np.uint64(1 << 63), np.uint64(0))).view(np.float64)
            v = other + flipped
    return v


def test_butterflies_are_bitwise_identical_on_a_hard_corpus():
    rng = np.random.default_rng(11)
    groups = [rng.standard_normal(8) * 10.0 ** rng.integers(-300, 300, 8) for _ in range(20000)]
    groups += [np.array([0.0, -0.0] * 4), np.array([5e-324, -5e-324] * 4), np.array([1e308, -1e308] * 4)]
    for g in groups:
        ref = _butterfly(g, 'modal8').view(np.uint64)
        for variant in ('modal8_muladd', 'modal8_signbit'):
            assert np.array_equal(_butterfly(g, variant).view(np.uint64), ref), (variant, g)


def test_protocol_arms_reference_known_kernels():
    known = {'fused_ai_fp64', 'fused_fp64', 'node_fp64', 'dense8', 'modal8', 'modal8_muladd',
             'modal8_signbit', 'node_fp32'}
    for a in P['arms']:
        assert a['outer'] in known and a['vcycle'] in known
        assert (a['precision'] == 'fp32') == a['vcycle'].endswith('_fp32')
        assert not a['outer'].endswith('_fp32')
    assert set(P['products']['paths']) <= known
    fp64_kernels = {a['outer'] for a in P['arms']}
    assert set(P['products']['paths']) <= fp64_kernels, 'every timed product must be built by some arm'


# ---------------------------------------------------------------- decisions
def result(product_speed=1.5, mg_speed=1.2, vary=False, missing_round=False, status='completed', bad_check=False):
    fixed, tol, products, checks = [], [], [], []
    for c in P['cases']:
        rounds = []
        for rnd in range(P['products']['rounds']):
            ms = {q: (1.0 + 0.01 * rnd) / (product_speed if q == 'modal8_muladd' else 1.0)
                  for q in P['products']['paths']}
            rounds.append({'round': rnd, 'order': list(ms), 'ms_per_product': ms})
        products.append({'case': c['id'], 'rounds': rounds})
        checks += [{'case': c['id'], 'path': q, 'passed': not (bad_check and q == 'modal8_muladd')}
                   for q in P['products']['paths']]
        for rnd in range(P['fixed_work']['rounds']):
            if missing_round and c['id'] == PRIMARY[0] and rnd == 0:
                continue
            for arm in P['fixed_work']['arms']:
                s = (1.0 + 0.01 * rnd) / (mg_speed if arm == 'mg64-modal8_muladd' else 1.0)
                it = 15 + (1 if vary and arm == 'mg64-modal8_muladd' and rnd == 2 else 0)
                fixed.append({'case': c['id'], 'arm': arm, 'round': rnd, 'iterations': it,
                              'outer_matvecs': it + 1, 'solver_seconds': s})
        for arm in P['arms']:
            for rep in range(P['to_tolerance']['repetitions']):
                tol.append({'case': c['id'], 'arm': arm['id'], 'rep': rep, 'iterations': 15,
                            'solver_seconds': 0.5 if arm['precision'] == 'fp32' else 1.0, 'physical_passed': True})
    return {'status': status, 'fixed_work': fixed, 'to_tolerance': tol, 'products': products,
            'product_checks': checks}


def test_h8_h9_pass_with_clear_margins():
    d = analysis.analyze(result(), P)
    assert d['h8_decision'] and d['h9_decision']


def test_h8_fails_below_threshold_or_on_a_failed_accuracy_check():
    assert not analysis.analyze(result(product_speed=1.03), P)['h8_decision']
    assert not analysis.analyze(result(bad_check=True), P)['h8_decision']


def test_h9_fails_below_threshold_unequal_work_missing_round_or_incomplete():
    assert not analysis.analyze(result(mg_speed=1.03), P)['h9_decision']
    assert not analysis.analyze(result(vary=True), P)['h9_decision']
    assert not analysis.analyze(result(missing_round=True), P)['h9_decision']
    d = analysis.analyze(result(status='failed'), P)
    assert not d['h8_decision'] and not d['h9_decision']


def test_replication_needs_both_hosts():
    good, bad = analysis.analyze(result(), P), analysis.analyze(result(status='failed'), P)
    assert analysis.replicate([good, good])['h8_replicated']
    assert not analysis.replicate([good, bad])['h9_replicated']
    assert not analysis.replicate([good])['h8_replicated']


def test_fastest_passing_ignores_failed_arms():
    r = result()
    for x in r['to_tolerance']:
        if x['arm'].startswith('mg32'):
            x['physical_passed'] = x['rep'] != 0
    d = analysis.analyze(r, P)
    assert all(v['arm'].startswith('mg64') for v in d['fastest_passing'].values())
