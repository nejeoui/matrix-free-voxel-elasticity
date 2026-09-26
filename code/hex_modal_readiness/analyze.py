"""Audit retained timings and compute conditional Amdahl requirements, without CUDA.

This is a retrospective audit. Separate product benchmarks cannot identify the
product fraction of a solve, and the conditional model is not a new speedup.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rescope.hex_modal_cuda.common import timing_decision


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def speedup(kernel_ratio, product_share, extra_cost=0.0):
    """Constant work; extra cost is candidate-minus-baseline / baseline total."""
    require(math.isfinite(kernel_ratio) and kernel_ratio > 0, 'Invalid kernel ratio')
    require(math.isfinite(product_share) and 0 <= product_share <= 1, 'Invalid product share')
    denominator = 1 - product_share + product_share / kernel_ratio + extra_cost
    require(math.isfinite(denominator) and denominator > 0, 'Invalid charged cost')
    return 1 / denominator


def required_share(kernel_ratio, target, extra_cost=0.0):
    require(kernel_ratio > 1 and target > 1, 'Requirements need improvements above one')
    return (1 - 1 / target + extra_cost) / (1 - 1 / kernel_ratio)


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_result(result, protocol, label):
    require(result['status'] == 'completed', 'Incomplete parent experiment')
    rebuilt = timing_decision(result['timings'], protocol)
    require(rebuilt['comparisons'] == result['comparisons'], 'Saved timing comparisons changed')
    require(rebuilt['timing_gate_passed'] == result['timing_gate_passed'], 'Saved timing gate changed')
    expected = {(case, path, replicate)
                for case in protocol['solver_check']['cases']
                for path in protocol['solver_check']['paths']
                for replicate in range(protocol['solver_check']['replicates'])}
    normal = [r for r in result['solves'] if not r['cap_one']]
    require(len(normal) == len(expected), 'Incomplete or duplicated normal solves')
    require({(r['id'], r['path'], r['replicate']) for r in normal} == expected,
            'Changed normal solve membership')
    require(all(r['passed'] and r['physical_passed'] and r['solver_seconds'] > 0
                and math.isfinite(r['solver_seconds']) for r in normal), 'Invalid normal solve')
    solver_rows = []
    for case in protocol['solver_check']['cases']:
        medians = {path: statistics.median(r['solver_seconds'] for r in normal
                                         if r['id'] == case and r['path'] == path)
                   for path in protocol['solver_check']['paths']}
        fastest = min(medians, key=medians.get)
        for path, seconds in medians.items():
            rows = [r for r in normal if r['id'] == case and r['path'] == path]
            solver_rows.append({'experiment': label, 'case': case, 'path': path,
                                'replicates': len(rows), 'median_seconds': seconds,
                                'fastest_path': fastest,
                                'baseline_over_modal_ratio': seconds / medians['modal8'],
                                'minimum_matvec_calls': min(r['matvec_calls'] for r in rows),
                                'maximum_matvec_calls': max(r['matvec_calls'] for r in rows),
                                'in_solve_product_time_recorded': False})
    return rebuilt['comparisons'], solver_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    declaration_path = Path(__file__).with_name('protocol.json')
    declaration = json.loads(declaration_path.read_text())
    sources = {}
    for name, digest in declaration['source_pins'].items():
        require(sha(ROOT / name) == digest, 'Source changed: ' + name)
        sources[name] = digest
    pending = json.loads((ROOT / declaration['pending_protocol']).read_text())
    require(pending['large_solves']['gate'].startswith('All normal physical checks pass.'),
            'Unexpected planned gate')
    kernel_rows, solver_rows, curves = [], [], []
    for item in declaration['experiments']:
        p = json.loads((ROOT / item['protocol']).read_text())
        result = json.loads((ROOT / item['result']).read_text())
        require(result['protocol_sha256'] == sha(ROOT / item['protocol']), 'Parent protocol mismatch')
        comparisons, solvers = audit_result(result, p, item['label'])
        solver_rows.extend(solvers)
        for shape in p['benchmarks']['primary_cases']:
            published = [row for row in comparisons if row['shape'] == shape
                         and row['baseline'] in declaration['published_baselines']]
            require(len(published) == len(declaration['published_baselines']), 'Missing baseline')
            best = min(published, key=lambda r: r['baseline_median_ms'])
            ratio = best['paired_median_ratio']
            row = {'experiment': item['label'], 'shape': 'x'.join(map(str, shape)),
                   'elements': math.prod(shape), 'fastest_published_path': best['baseline'],
                   'baseline_median_ms': best['baseline_median_ms'],
                   'candidate_median_ms': best['candidate_median_ms'],
                   'paired_median_kernel_ratio': ratio,
                   'kernel_ratio_bootstrap_lower_95': best['paired_bootstrap_lower_95'],
                   'kernel_elapsed_reduction_fraction': 1 - 1 / ratio,
                   'required_product_share_for_1_10': required_share(ratio, 1.10),
                   'required_product_share_for_1_30': required_share(ratio, 1.30),
                   'conditional_ratio_at_40pct_share': speedup(ratio, .4),
                   'product_share_measured': False}
            kernel_rows.append(row)
            for fraction in declaration['illustrative_product_shares']:
                curves.append({'experiment': item['label'], 'shape': row['shape'],
                               'assumed_product_share': fraction, 'assumed_extra_cost': 0,
                               'conditional_episode_ratio': speedup(ratio, fraction),
                               'measured_episode_ratio': False})
    write_csv(out / 'kernel-requirements.csv', kernel_rows)
    write_csv(out / 'solver-medians.csv', solver_rows)
    write_csv(out / 'conditional-sensitivity.csv', curves)
    decision = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_kind': 'retrospective saved-evidence and conditional-model audit',
        'protocol_sha256': sha(declaration_path), 'source_pins': sources,
        'saved_timing_comparisons_recomputed': 40,
        'normal_solver_records_checked': 144,
        'kernel_requirements': kernel_rows,
        'application_product_share': None,
        'application_product_share_identifiable_from_saved_records': False,
        'reason': ('Separate steady-state synthetic product batches and synchronized whole-solve times '
                   'do not supply same-state in-solve product durations. First-call setup timings '
                   'include cold preparation and cannot substitute for a product profile.'),
        'conditional_model': 'S = 1 / (1 - f + f / r + d)',
        'model_assumptions': [
            'Identical number of solver iterations and products.',
            'Synthetic complete-product ratio r transfers unchanged to the physical solve.',
            'Non-product work is unchanged; f uses the complete baseline charged episode.',
            'Tables set d=0; extra candidate setup d>0 raises the necessary product share.',
            'Kernel bootstrap limits are not confidence limits for an unmeasured solver speedup.'
        ],
        'three_times_ten_percent_ratio_target': 1.30,
        'three_times_margin_attainable_under_this_model': all(
            r['required_product_share_for_1_30'] <= 1 for r in kernel_rows),
        'decision': 'large_solver_campaign_not_justified_by_existing_share_evidence',
        'cuda_application_gate': 'not_executed',
        'large_solver_performance_gate': 'not_executed',
        'refutes_candidate': False, 'new_gpu_execution': False, 'new_gpu_spending_usd': 0,
        'performance_claim': False, 'novelty_claim': False, 'publication_goal_achieved': False,
    }
    (out / 'decision.json').write_text(json.dumps(decision, indent=2, allow_nan=False) + '\n')
    (out / 'protocol.json').write_bytes(declaration_path.read_bytes())
    (out / 'analyze.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: decision[k] for k in ('decision', 'saved_timing_comparisons_recomputed',
                                             'normal_solver_records_checked', 'new_gpu_spending_usd')}, indent=2))


if __name__ == '__main__':
    main()
