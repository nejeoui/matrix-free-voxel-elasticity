"""Pre-declared decisions for Voxel session D (multigrid). Pure Python/NumPy; testable without a GPU."""
import statistics

import numpy as np


def bootstrap_lower(ratios, seed, draws=10000, level=0.05):
    ratios = np.asarray(ratios, dtype=float)
    rng = np.random.default_rng(seed)
    return float(np.quantile(np.median(rng.choice(ratios, size=(draws, ratios.size)), axis=1), level))


def analyze(result, protocol):
    seed = protocol['order_seed']
    cases = {c['id']: c for c in protocol['cases']}
    h6 = protocol['decision']['h6']
    arm_ids = [a['id'] for a in protocol['arms']]
    out = {'fixed_work': [], 'to_tolerance': [], 'fastest_passing': {}, 'h6_passed': {}, 'outer_effect': {},
           'profiles': result.get('profiles', [])}
    for cid, case in cases.items():
        rows = [r for r in result['fixed_work'] if r['case'] == cid]
        if not rows:
            continue
        work = {(r['iterations'], r['outer_matvecs']) for r in rows}
        expected = protocol['fixed_work']['rounds'] * len(protocol['fixed_work']['arms'])
        identical = len(work) == 1 and len(rows) == expected
        t = {}
        for r in rows:
            t.setdefault(r['arm'], {})[r['round']] = r['solver_seconds']
        med = {a: statistics.median(v.values()) for a, v in t.items()}

        def paired(num, den):
            common = sorted(set(t[num]) & set(t[den]))
            ratios = [t[num][k] / t[den][k] for k in common]
            return {'numerator': num, 'denominator': den, 'rounds': len(ratios),
                    'median_ratio': statistics.median(ratios), 'lower95': bootstrap_lower(ratios, seed)}

        for a in arm_ids:
            if a != h6['candidate'] and a in t:
                e = paired(a, h6['candidate']); e.update(case=cid, identical_work=identical,
                                                        numerator_median_s=med[a], candidate_median_s=med[h6['candidate']])
                out['fixed_work'].append(e)
        fastest_donor = min(h6['baselines'], key=lambda a: med[a])
        e = paired(fastest_donor, h6['candidate'])
        out['h6_passed'][cid] = bool(case['primary'] and identical and e['rounds'] == protocol['fixed_work']['rounds']
                                     and e['median_ratio'] >= h6['min_median'] and e['lower95'] >= h6['min_lower'])
        mixed_no_modal = [a['id'] for a in protocol['arms'] if a['precision'] == 'fp32'
                          and 'modal8' not in (a['outer'] + a['vcycle'])]
        if 'mg32-modal8-node32' in t and mixed_no_modal:
            best = min(mixed_no_modal, key=lambda a: med[a])
            out['outer_effect'][cid] = dict(paired(best, 'mg32-modal8-node32'), best_non_modal8_mixed=best)
        for a in arm_ids:
            reps = [r for r in result['to_tolerance'] if r['case'] == cid and r['arm'] == a]
            if reps:
                out['to_tolerance'].append({'case': cid, 'arm': a, 'repetitions': len(reps),
                                            'median_seconds': statistics.median(r['solver_seconds'] for r in reps),
                                            'iterations': sorted({r['iterations'] for r in reps}),
                                            'all_physical_passed': all(r['physical_passed'] for r in reps)})
        passing = [e for e in out['to_tolerance'] if e['case'] == cid and e['all_physical_passed']]
        if passing:
            f = min(passing, key=lambda e: e['median_seconds'])
            out['fastest_passing'][cid] = {'arm': f['arm'], 'median_seconds': f['median_seconds']}
    complete = result.get('status') == 'completed'
    prim = [c for c, v in cases.items() if v['primary']]
    h6_all = complete and all(out['h6_passed'].get(c, False) for c in prim)
    uses_modal8 = {c: ('modal8' in v['arm']) for c, v in out['fastest_passing'].items()}
    out.update(complete=complete,
               h6_decision='fp64_ranking_survives_in_multigrid' if h6_all else 'fp64_ranking_does_not_survive_or_incomplete',
               h7_fastest_uses_modal8=uses_modal8,
               performance_claim=False, novelty_claim=False,
               scope='RTX 4090 development session; warm contexts after charged, shared setup; fixed work '
                     'compares identical outer iterations. Not a complete-optimization claim.')
    return out
