"""Diagnostic planning decision; no claim of fully charged application speedup."""
import statistics


def analyze(result, protocol):
    rows = result['solves']
    expected = {(c['id'], path, mode == 'profile', rep)
                for c in protocol['cases'] for path in protocol['paths']
                for mode in ('plain', 'profile')
                for rep in range(1 if mode == 'profile' else c['plain_replicates'])}
    actual = [(r['case'], r['path'], r['profiled'], r['replicate']) for r in rows]
    complete = (result['status'] == 'completed' and len(actual) == len(expected)
                and set(actual) == expected and all(r['physical_passed'] for r in rows))
    summaries, comparisons = [], []
    for case in protocol['cases']:
        paths = {}
        for path in protocol['paths']:
            plain = [r for r in rows if r['case'] == case['id'] and r['path'] == path and not r['profiled']]
            profiled = [r for r in rows if r['case'] == case['id'] and r['path'] == path and r['profiled']]
            if len(plain) != case['plain_replicates'] or len(profiled) != 1:
                continue
            profile = profiled[0]
            median = statistics.median(r['solver_seconds'] for r in plain)
            same_work = all(r['iterations'] == profile['iterations'] and r['matvec_calls'] == profile['matvec_calls'] for r in plain)
            overhead = profile['solver_seconds'] / median - 1
            fraction = profile['product_event_seconds'] / profile['solver_seconds']
            valid_profile = same_work and abs(overhead) <= protocol['profile_overhead_limit'] and 0 <= fraction <= 1
            summary = {'case': case['id'], 'path': path, 'primary': case['primary'],
                       'plain_replicates': len(plain), 'median_plain_seconds': median,
                       'plain_iterations': [r['iterations'] for r in plain],
                       'profile_iterations': profile['iterations'],
                       'profile_product_seconds': profile['product_event_seconds'],
                       'profile_product_fraction': fraction,
                       'profile_per_product_seconds': profile['product_event_seconds'] / profile['matvec_calls'],
                       'profile_wall_over_plain_median_minus_one': overhead,
                       'profile_and_plain_work_identical': same_work,
                       'profile_qualified': valid_profile}
            paths[path] = summary
            summaries.append(summary)
        if protocol['candidate'] not in paths:
            continue
        candidate = paths[protocol['candidate']]
        for path, baseline in paths.items():
            if path == protocol['candidate']:
                continue
            r = baseline['profile_per_product_seconds'] / candidate['profile_per_product_seconds']
            f = baseline['profile_product_fraction']
            ratio = baseline['median_plain_seconds'] / candidate['median_plain_seconds']
            model_work_matches = baseline['profile_iterations'] == candidate['profile_iterations']
            comparisons.append({'case': case['id'], 'primary': case['primary'], 'baseline': path,
                'median_plain_solver_ratio': ratio, 'profiled_product_per_call_ratio': r,
                'baseline_profile_product_fraction': f,
                'conditional_fixed_work_solver_ratio': 1 / (1 - f + f / r),
                'same_profile_iterations_across_paths': model_work_matches,
                'profiles_qualified': baseline['profile_qualified'] and candidate['profile_qualified'],
                'passes_planning_ratio': ratio >= protocol['planning_ratio_target']})
    primaries = [r for r in comparisons if r['primary']]
    enough = len(primaries) == sum(c['primary'] for c in protocol['cases']) * (len(protocol['paths'])-1)
    gate = complete and enough and all(r['profiles_qualified'] and r['passes_planning_ratio'] for r in primaries)
    return {'complete_physical_diagnostic': complete, 'summaries': summaries, 'comparisons': comparisons,
            'advance_to_charged_campaign': bool(gate),
            'decision': 'premise_supports_fresh_charged_campaign' if gate else 'do_not_advance_on_this_diagnostic',
            'performance_claim': False, 'novelty_claim': False, 'publication_goal_achieved': False,
            'scope': 'Warm prepared-state diagnostic. Cold context/import/setup, profiler event allocation, independent replay and archive I/O are excluded from solve times and separately recorded. No original fully charged gate is evaluated.'}
