"""Pre-declared decisions for Voxel session A. Pure Python/NumPy so it can be tested without a GPU.

H2 (continue/stop): fixed-work FP64 PCG, modal8 vs the declared baseline, paired by round. Passes on
a primary case iff every fixed-work solve in that case did identical work, the median paired ratio
is >= h2.min_median and the one-sided 95 % bootstrap lower bound is >= h2.min_lower.
H4 (scope test): FP32 product, modal8_fp32 vs the fastest FP32 baseline, paired by round.
"Confirmed" (no material gain) iff the median ratio < h4.gain_threshold on every primary case.
Time to tolerance is descriptive: medians per method and whether every repetition physically passed.
"""
import statistics

import numpy as np


def bootstrap_lower(ratios, seed, draws=10000, level=0.05):
    ratios = np.asarray(ratios, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(ratios, size=(draws, ratios.size), replace=True)
    return float(np.quantile(np.median(samples, axis=1), level))


def paired(times_a, times_b):
    """Ratios a/b over rounds present in both mappings {round: seconds}."""
    common = sorted(set(times_a) & set(times_b))
    return [times_a[r] / times_b[r] for r in common]


def analyze(result, protocol):
    seed = protocol['order_seed']
    cases = {c['id']: c for c in protocol['cases']}
    h2, h4 = protocol['decision'].get('h2'), protocol['decision'].get('h4')
    out = {'fixed_work': [], 'products': [], 'to_tolerance': [], 'h2_passed': {}, 'h4_confirmed': {}}

    for cid, case in cases.items():
        rows = [r for r in result['fixed_work'] if r['case'] == cid]
        expected = protocol['fixed_work']['rounds'] * len(protocol['paths_fp64'])
        work = {(r['iterations'], r['matvec_calls']) for r in rows}
        identical = (len(rows) == expected and len(work) == 1
                     and next(iter(work))[0] == case['fixed_iterations'])
        by_path = {}
        for r in rows:
            by_path.setdefault(r['path'], {})[r['round']] = r['solver_seconds']
        cand = by_path.get(protocol['candidate_fp64'], {})
        for base in protocol['paths_fp64']:
            if base == protocol['candidate_fp64'] or base not in by_path or not cand:
                continue
            ratios = paired(by_path[base], cand)
            if not ratios:
                continue
            entry = {'case': cid, 'primary': case['primary'], 'baseline': base, 'rounds': len(ratios),
                     'median_ratio': statistics.median(ratios),
                     'lower95': bootstrap_lower(ratios, seed), 'identical_work': identical,
                     'baseline_median_seconds': statistics.median(by_path[base].values()),
                     'candidate_median_seconds': statistics.median(cand.values())}
            out['fixed_work'].append(entry)
            if h2 and base == h2['baseline'] and case['primary']:
                out['h2_passed'][cid] = bool(identical and entry['median_ratio'] >= h2['min_median']
                                             and entry['lower95'] >= h2['min_lower']
                                             and entry['rounds'] == protocol['fixed_work']['rounds'])

    for block in result['products']:
        cid, precision = block['case'], block['precision']
        per = {}
        for row in block['rounds']:
            for path, ms in row['ms_per_product'].items():
                per.setdefault(path, {})[row['round']] = ms
        candidate = protocol['candidate_fp64'] if precision == 'fp64' else protocol['candidate_fp32']
        if candidate not in per:
            continue
        medians = {p: statistics.median(v.values()) for p, v in per.items()}
        baselines = [p for p in per if p != candidate]
        fastest = min(baselines, key=lambda p: medians[p]) if baselines else None
        for base in baselines:
            ratios = paired(per[base], per[candidate])
            out['products'].append({'case': cid, 'precision': precision, 'baseline': base,
                                    'median_ratio': statistics.median(ratios),
                                    'lower95': bootstrap_lower(ratios, seed),
                                    'baseline_median_ms': medians[base], 'candidate_median_ms': medians[candidate],
                                    'fastest_baseline': base == fastest})
        if h4 and precision == 'fp32' and cases[cid]['primary'] and fastest is not None:
            ratio = statistics.median(paired(per[fastest], per[candidate]))
            out['h4_confirmed'][cid] = bool(ratio < h4['gain_threshold'])

    for cid in cases:
        for m in protocol['to_tolerance']['methods']:
            reps = [r for r in result['to_tolerance'] if r['case'] == cid and r['method'] == m['id']]
            if reps:
                out['to_tolerance'].append({
                    'case': cid, 'method': m['id'], 'repetitions': len(reps),
                    'median_seconds': statistics.median(r['solver_seconds'] for r in reps),
                    'all_physical_passed': all(r['physical_passed'] for r in reps)})

    primaries = [cid for cid, c in cases.items() if c['primary'] and 'fixed_work' in c.get('stages', ['fixed_work'])]
    complete = result.get('status') == 'completed'
    h2_all = bool(h2) and complete and bool(primaries) and all(out['h2_passed'].get(cid, False) for cid in primaries)
    h4_prim = [cid for cid, c in cases.items() if c['primary'] and 'products' in c.get('stages', ['products'])]
    h4_all = bool(h4) and complete and bool(h4_prim) and all(out['h4_confirmed'].get(cid, False) for cid in h4_prim)
    out.update(complete=complete,
               h2_decision=('not_declared' if not h2 else 'continue_modal8' if h2_all else 'stop_modal8'),
               h4_decision=('not_declared' if not h4 else 'predicted_no_fp32_gain_confirmed' if h4_all else
                            'fp32_prediction_not_confirmed_or_incomplete'),
               performance_claim=False, novelty_claim=False,
               scope='RTX 4090 development session; warm contexts after charged setup. Fixed-work '
                     'ratios compare identical PCG work. Not a complete-optimization or '
                     'publication claim.')
    return out


def extended(result, protocol):
    """Session B/C decisions layered on `analyze`: H3 (data-centre FP64 product), H5 (refinement
    robustness niche) and the fastest physically passing method per case. Declared in protocol."""
    base = analyze(result, protocol)
    dec = protocol['decision']
    cases = {c['id']: c for c in protocol['cases']}
    complete = result.get('status') == 'completed'
    out = dict(base)
    if 'h3' in dec:
        h3 = {}
        for e in base['products']:
            if (e['precision'] == 'fp64' and e['baseline'] == dec['h3']['baseline']
                    and cases[e['case']]['primary']):
                h3[e['case']] = bool(e['median_ratio'] < dec['h3']['gain_threshold'])
        prim = [c for c, v in cases.items() if v['primary'] and 'products' in v.get('stages', ['products'])]
        out['h3_confirmed'] = h3
        out['h3_decision'] = ('predicted_no_datacentre_fp64_gain_confirmed'
                              if complete and prim and all(h3.get(c, False) for c in prim)
                              else 'datacentre_prediction_not_confirmed_or_incomplete')
    kinds = {m['id']: m['kind'] for m in protocol['to_tolerance']['methods']}
    fastest = {}
    for e in base['to_tolerance']:
        if e['all_physical_passed'] and (e['case'] not in fastest or e['median_seconds'] < fastest[e['case']]['median_seconds']):
            fastest[e['case']] = e
    out['fastest_passing_method'] = {c: {'method': e['method'], 'median_seconds': e['median_seconds']}
                                     for c, e in fastest.items()}
    if 'h5' in dec:
        rows = {}
        for e in base['to_tolerance']:
            if cases[e['case']].get('robustness'):
                rows.setdefault(e['case'], []).append(e)
        per = {}
        for c, es in rows.items():
            ir_fail = any(kinds[e['method']] == 'ir' and not e['all_physical_passed'] for e in es)
            pcg_pass = all(e['all_physical_passed'] for e in es if kinds[e['method']] == 'pcg64')
            per[c] = {'refinement_failed': ir_fail, 'fp64_pcg_passed': pcg_pass,
                      'niche': bool(ir_fail and pcg_pass)}
        out['h5_per_case'] = per
        out['h5_decision'] = ('fp64_throughout_niche_supported'
                              if complete and any(v['niche'] for v in per.values())
                              else 'fp64_throughout_niche_not_supported')
    return out
