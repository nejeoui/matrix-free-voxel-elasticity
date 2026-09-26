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
    h2, h4 = protocol['decision']['h2'], protocol['decision']['h4']
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
            if base == h2['baseline'] and case['primary']:
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
        if precision == 'fp32' and cases[cid]['primary'] and fastest is not None:
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

    primaries = [cid for cid, c in cases.items() if c['primary']]
    complete = result.get('status') == 'completed'
    h2_all = complete and all(out['h2_passed'].get(cid, False) for cid in primaries)
    h4_all = complete and all(out['h4_confirmed'].get(cid, False) for cid in primaries)
    out.update(complete=complete,
               h2_decision='continue_modal8' if h2_all else 'stop_modal8',
               h4_decision=('predicted_no_fp32_gain_confirmed' if h4_all else
                            'fp32_prediction_not_confirmed_or_incomplete'),
               performance_claim=False, novelty_claim=False,
               scope='RTX 4090 development session; warm contexts after charged setup. Fixed-work '
                     'ratios compare identical PCG work. Not a complete-optimization or '
                     'publication claim.')
    return out
