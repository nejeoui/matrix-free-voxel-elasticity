"""Pre-declared decisions for Voxel session E (butterfly variants, products and multigrid).

Pure Python/NumPy; testable without a GPU. One call analyses one host's run; replication across
hosts is decided by `replicate` over the per-host outputs.
"""
import statistics

import numpy as np


def bootstrap_lower(ratios, seed, draws=10000, level=0.05):
    ratios = np.asarray(ratios, dtype=float)
    rng = np.random.default_rng(seed)
    return float(np.quantile(np.median(rng.choice(ratios, size=(draws, ratios.size)), axis=1), level))


def _paired(t, num, den, seed):
    common = sorted(set(t[num]) & set(t[den]))
    ratios = [t[num][k] / t[den][k] for k in common]
    return {'numerator': num, 'denominator': den, 'rounds': len(ratios),
            'median_ratio': statistics.median(ratios), 'lower95': bootstrap_lower(ratios, seed)}


def _passes(e, rule, rounds):
    return bool(e['rounds'] == rounds and e['median_ratio'] >= rule['min_median'] and e['lower95'] >= rule['min_lower'])


def analyze(result, protocol):
    seed = protocol['order_seed']
    cases = {c['id']: c for c in protocol['cases']}
    h8, h9 = protocol['decision']['h8'], protocol['decision']['h9']
    pp = protocol['products']
    arm_ids = [a['id'] for a in protocol['arms']]
    out = {'products': [], 'h8_passed': {}, 'fixed_work': [], 'to_tolerance': [], 'fastest_passing': {},
           'h9_passed': {}, 'outer_effect': {}, 'profiles': result.get('profiles', []),
           'product_checks_passed': all(c['passed'] for c in result.get('product_checks', [])),
           'variant_differences': result.get('variant_differences', [])}
    for cid, case in cases.items():
        # ---- H8: products
        prow = next((r for r in result.get('products', []) if r['case'] == cid), None)
        if prow is not None:
            t = {q: {r['round']: r['ms_per_product'][q] for r in prow['rounds'] if q in r['ms_per_product']}
                 for q in pp['paths']}
            med = {q: statistics.median(v.values()) for q, v in t.items() if v}
            for q in pp['paths']:
                if q != h8['candidate'] and q in med:
                    e = _paired(t, q, h8['candidate'], seed)
                    e.update(case=cid, numerator_median_ms=med[q], candidate_median_ms=med[h8['candidate']])
                    out['products'].append(e)
            fastest = min(h8['donors'], key=lambda q: med[q])
            e = _paired(t, fastest, h8['candidate'], seed)
            checks = [c for c in result.get('product_checks', []) if c['case'] == cid]
            out['h8_passed'][cid] = bool(case['primary'] and checks and all(c['passed'] for c in checks)
                                         and _passes(e, h8, pp['rounds']))
            out.setdefault('h8_fastest_donor', {})[cid] = dict(e, fastest_donor=fastest)
        # ---- H9: multigrid fixed work
        rows = [r for r in result['fixed_work'] if r['case'] == cid]
        if rows:
            work = {(r['iterations'], r['outer_matvecs']) for r in rows}
            expected = protocol['fixed_work']['rounds'] * len(protocol['fixed_work']['arms'])
            identical = len(work) == 1 and len(rows) == expected
            t = {}
            for r in rows:
                t.setdefault(r['arm'], {})[r['round']] = r['solver_seconds']
            med = {a: statistics.median(v.values()) for a, v in t.items()}
            for a in arm_ids:
                if a != h9['candidate'] and a in t:
                    e = _paired(t, a, h9['candidate'], seed)
                    e.update(case=cid, identical_work=identical, numerator_median_s=med[a],
                             candidate_median_s=med[h9['candidate']])
                    out['fixed_work'].append(e)
            fastest = min(h9['baselines'], key=lambda a: med[a])
            e = _paired(t, fastest, h9['candidate'], seed)
            out['h9_passed'][cid] = bool(case['primary'] and identical
                                         and _passes(e, h9, protocol['fixed_work']['rounds']))
            out.setdefault('h9_fastest_donor', {})[cid] = dict(e, fastest_donor=fastest)
            if 'mg32-fused_ai64-node32' in t and 'mg32-modal8_muladd-node32' in t:
                out['outer_effect'][cid] = _paired(t, 'mg32-fused_ai64-node32', 'mg32-modal8_muladd-node32', seed)
        # ---- H10: time to tolerance (descriptive)
        for a in arm_ids:
            reps = [r for r in result['to_tolerance'] if r['case'] == cid and r['arm'] == a]
            if reps:
                out['to_tolerance'].append({'case': cid, 'arm': a, 'repetitions': len(reps),
                                            'median_seconds': statistics.median(r['solver_seconds'] for r in reps),
                                            'iterations': sorted({r['iterations'] for r in reps}),
                                            'all_physical_passed': all(r['physical_passed'] for r in reps)})
        passing = [e for e in out['to_tolerance'] if e['case'] == cid and e['all_physical_passed']
                   and e['repetitions'] == protocol['to_tolerance']['repetitions']]
        if passing:
            f = min(passing, key=lambda e: e['median_seconds'])
            out['fastest_passing'][cid] = {'arm': f['arm'], 'median_seconds': f['median_seconds']}
    complete = result.get('status') == 'completed'
    prim = [c for c, v in cases.items() if v['primary']]
    out.update(complete=complete,
               h8_decision=bool(complete and all(out['h8_passed'].get(c, False) for c in prim)),
               h9_decision=bool(complete and all(out['h9_passed'].get(c, False) for c in prim)),
               performance_claim=False, novelty_claim=False,
               scope='One RTX 4090 host; warm contexts after charged, shared setup; fixed work compares identical '
                     'outer iterations. Not a complete-optimization claim.')
    return out


def replicate(per_host):
    """per_host: list of analyze() outputs, one per host (a failed host contributes complete=False)."""
    return {'hosts': len(per_host),
            'h8_replicated': len(per_host) >= 2 and all(h['h8_decision'] for h in per_host),
            'h9_replicated': len(per_host) >= 2 and all(h['h9_decision'] for h in per_host)}
