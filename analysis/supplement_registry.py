"""Regenerate provenance, audit units and hypothesis chronology from retained records.

python3 analysis/supplement_registry.py
No GPU execution. No historical protocol or evidence is modified.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/supplement'

def read(path):
    return json.loads((ROOT / path).read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def decision(session):
    return read(f'results/session-{session}-20260923/local-verification.json')['decision']

def audit_units():
    manifest = read('provenance/supplement/extractions.json')
    for row in manifest['files']:
        assert digest(ROOT / row['path']) == row['sha256'], row['path']
    inputs = ROOT / 'provenance/supplement/audit-inputs'
    state = np.load(inputs / 'maximum-state-error.npz')
    relative = float(np.linalg.norm(state['u'] - state['reference']) / np.linalg.norm(state['reference']))
    def vectors(prefix):
        return [np.fromfile(inputs / f'{prefix}-{s}.f64', dtype='<f8') for s in ['a-r', 'a-Br', 'b-r', 'b-Br']]
    a, ba, b, bb = vectors('bilinear')
    numerator = float(abs(a @ bb - b @ ba))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(bb) + np.linalg.norm(b) * np.linalg.norm(ba))
    a, ba, b, bb = vectors('leakage')
    fixed = state['fixed']
    assert np.all(a[fixed] == 0) and np.all(b[fixed] == 0)
    leakage = float(max(np.max(abs(ba[fixed])), np.max(abs(bb[fixed]))))
    base = 'evidence/companion/results/rescope/'
    recorded_state = max(read(base + 'hex-modal-mg-host-20260923/run-01/result.json')['states'], key=lambda r: r['state_relative_error'])
    pairs = read(base + 'hex-modal-mg-bc-audit-20260923/run-01/verification.json')['pairs']
    recorded_bilinear = max(abs(r['scaled_bilinear_symmetry']) for r in pairs if r['variant'] == 0)
    recorded_leakage = max(r['fixed_output_maximum'] for r in pairs if r['variant'] == 0)
    assert np.isclose(relative, recorded_state['state_relative_error'], rtol=1e-12)
    assert np.isclose(numerator / denominator, recorded_bilinear, rtol=1e-10)
    assert leakage == recorded_leakage
    return {
        'kind': 'CPU recomputation of selected saved array norms, not fresh operator/GPU execution',
        'source_manifest': 'provenance/supplement/extractions.json',
        'state': {'case': '24x12x8/modal/step-001', 'formula': 'norm(u-reference,2)/norm(reference,2), full DOFs', 'unit': 'dimensionless', 'relative': relative, 'percent': relative * 100, 'attribution': 'initial candidate used zero fixed rows instead of published identity rows; see host-diagnosis.json'},
        'bilinear': {'case': '16x8x8-l3/pattern-0/variant-0/pair-2', 'formula': 'abs(a.T@B(b)-b.T@B(a))/(norm(a)*norm(B(b))+norm(b)*norm(B(a)))', 'unit': 'dimensionless', 'numerator': numerator, 'denominator': denominator, 'value': numerator / denominator},
        'leakage': {'case': '24x12x8-l3/pattern-0/variant-0/pair-0', 'formula': 'max(abs(B(a)[fixed]),abs(B(b)[fixed]))', 'unit': 'absolute correction/displacement in benchmark units; no denominator', 'maximum': leakage, 'fixed_count': len(fixed), 'maximum_abs_probe': float(max(np.max(abs(a)), np.max(abs(b))))},
    }

def chronology(session):
    protocol = ROOT / f'experiments/session_{session}/protocol.json'
    p = json.loads(protocol.read_text())
    ph = digest(protocol)
    jobs = []
    for path in sorted(ROOT.glob(f'results/session-{session}-*/remote/job.json')):
        d = json.loads(path.read_text())
        matches = [str(q.relative_to(ROOT)) for q in path.parent.rglob('protocol*.json') if digest(q) == ph]
        verify = path.parent.parent / 'local-verification.json'
        if verify.exists() and json.loads(verify.read_text()).get('protocol_sha256') == ph:
            matches.append(str(verify.relative_to(ROOT)) + '#protocol_sha256')
        jobs.append({'record': str(path.relative_to(ROOT)), 'started_utc': d.get('started_utc'), 'finished_utc': d.get('finished_utc'), 'matching_protocol_identity_records': matches, 'chronology': 'declared timestamp precedes execution; self-recorded clock evidence' if matches and p['declared_utc'] < d.get('started_utc', '') else 'different amendment or no byte-identity link in inspected records; not used to prove final-protocol priority'})
    return {'protocol': str(protocol.relative_to(ROOT)), 'sha256': ph, 'declared_utc': p['declared_utc'], 'execution_records': jobs}

def main():
    OUT.mkdir(exist_ok=True)
    audit = audit_units()
    (OUT / 'audit-units.json').write_text(json.dumps(audit, indent=2) + '\n')
    outcomes = {
        'H2': ('a', 'hypothesis', 'supported', decision('a')['h2_decision']),
        'H3': ('b', 'hypothesis', 'not confirmed', decision('b')['h3_decision']),
        'H4': ('a', 'hypothesis', 'supported', decision('a')['h4_decision']),
        'H5': ('c', 'hypothesis', 'not supported', decision('c')['h5_decision']),
        'H6': ('d', 'hypothesis', 'supported', decision('d')['h6_decision']),
        'H7': ('d', 'descriptive question', 'descriptive', decision('d')['h7_fastest_uses_modal8']),
        'H8': ('e', 'hypothesis', 'supported on two hosts', read('paper/session_e_numbers.json')['replication']['h8_replicated']),
        'H9': ('e', 'hypothesis', 'supported on two hosts', read('paper/session_e_numbers.json')['replication']['h9_replicated']),
        'H10': ('e', 'descriptive question', 'descriptive', 'fastest passing route and outer-product effect'),
        'F-stack': ('f', 'packaging gate', 'failed as released; directory workaround passes', {k:v['smoke'] for k,v in read('paper/session_f_summary.json').items()}),
        'F-integration': ('f', 'integration gate', 'not passed', {k:v['integration_ok'] for k,v in read('paper/session_f_summary.json').items()}),
        'H11': ('g', 'hypothesis', 'failed on two hosts', read('paper/session_g_numbers.json')['H11_replicated']),
        'H12': ('g', 'descriptive expectation', 'not uniformly below 1.10; work-match qualification required', {k:v['H12_median_ratios'] for k,v in read('paper/session_g_numbers.json')['hosts'].items()}),
        'H11b': ('h', 'new hypothesis on changed pipeline', 'failed on two hosts', read('paper/session_h_numbers.json')['H11b_replicated']),
        'H12b': ('h', 'descriptive expectation', 'not uniformly below 1.10; work-match qualification required', {k:v['H12_median_ratios'] for k,v in read('paper/session_h_numbers.json')['hosts'].items()}),
        'H13': ('i', 'hypothesis', 'supported on fitting primary cases', {k:v['H13'] for k,v in read('paper/session_i_numbers.json').items() if 'H13' in v}),
        'H14': ('i', 'hypothesis', 'supported at largest fitting primary case', {k:v['H14'] for k,v in read('paper/session_i_numbers.json').items() if 'H14' in v}),
        'H15': ('j', 'hypothesis', 'failed count criterion; utilization part passed', read('paper/session_j_numbers.json')['H15']),
    }
    rows = []
    chronologies = {s: chronology(s) for s in 'abcdefghij'}
    for ident, (session, typ, status, outcome) in outcomes.items():
        p = read(f'experiments/session_{session}/protocol.json')
        qs = p['questions']
        if isinstance(qs, dict):
            question = next((v for k,v in qs.items() if ident in k), qs)
        else:
            question = next((v for v in qs if v.startswith(ident + ':')), qs)
        primary = [c for c in p.get('cases', []) if c.get('primary', False)]
        rules = p.get('decision', {})
        keys = [ident, ident.lower(), 'H11' if ident == 'H11b' else ident, 'H12' if ident == 'H12b' else ident, 'stack_ok' if ident == 'F-stack' else ident, 'integration_ok' if ident == 'F-integration' else ident]
        rule = next((rules[k] for k in keys if k in rules), rules)
        rows.append({'id': ident, 'type': typ, 'session': session.upper(), 'question': question, 'primary_cases': primary or p.get('cases', []), 'endpoint_and_threshold': rule, 'outcome': status, 'recorded_decision': outcome, 'chronology': chronologies[session]})
    data = {'convention': 'Five named hypotheses lack support: H3, H5, H11, H11b, H15. H5 retains its original not-supported label; H11b is a separate changed-pipeline hypothesis. Descriptive H12/H12b expectations and packaging/integration gates are separate. This count is for Sessions A-J, not every exploratory companion study.', 'failed_or_unsupported_ids': ['H3','H5','H11','H11b','H15'], 'chronology_limit': 'Hashes identify content, not temporal priority. These dates are author/system recorded timestamps, checked against byte-linked execution records where available, not independent preregistration timestamps. No local git history is present in this workspace.', 'H3_ratio_label_note': 'Frozen Session B h3.rule writes modal8/dense8, but the question and recorded analysis concern advantage dense8 time / modal8 time. The retained failed decision uses that latter ratio (about 2.24), not the literal reversed label. The historical protocol is preserved.', 'rows': rows}
    (OUT / 'hypothesis-registry.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = ['# Hypothesis and gate registry', '', data['convention'], '', data['chronology_limit'], '', data['H3_ratio_label_note'], '', 'The JSON companion preserves the exact primary cases, decision rules, hashes and every inspected execution record. The table abbreviates these fields.', '', '| ID | Type | Cases / endpoint and decision | Declaration UTC | Executed UTC (byte-linked final protocol) | Outcome |', '|---|---|---|---|---|---|']
    brief = {
        'H2':'optimized q64/q96; fixed Jacobi-PCG work, median ≥1.08 and one-sided 95% lower ≥1.05',
        'H3':'optimized q64/q96; FP64 dense8/parity product gain <1.10 expected',
        'H4':'optimized q64/q96; FP32 gain over fastest baseline <1.10 expected',
        'H5':'optimized q64/q96 at Emin=1e-9; at least one refinement failure while all FP64 PCG repetitions pass',
        'H6':'optimized q64/q96; fixed MG work, median ≥1.05, lower ≥1.02',
        'H7':'fastest physically accepted solve to tolerance; descriptive',
        'H8':'optimized q64/q96 on both hosts; product median ≥1.05, lower ≥1.02',
        'H9':'same cases; fixed MG work median ≥1.05, lower ≥1.02',
        'H10':'same cases; fastest solve and mixed outer-product effect, descriptive',
        'F-stack':'three GPUs; unchanged donor smoke-test exit code 0',
        'F-integration':'all grids, densities, schedules and arms; product ≤1e-12, convergence, residual ≤1e-10, cross-state ≤1e-8',
        'H11':'three primary cases, FP64 30-step trajectories; all medians ≥1.10, ≥2 repetitions, work/agreement gates',
        'H12':'same primary cases, mixed trajectories; descriptive expectation <1.10',
        'H11b':'same three primary cases, GPU optimizer, 100 steps; same ≥1.10 and qualification rules',
        'H12b':'GPU optimizer mixed trajectories; descriptive expectation <1.10',
        'H13':'fitting primary enlarged cases on each 80 GB card; parity/node fixed-work time ≥1/1.05',
        'H14':'largest fitting primary RTX case; fused/parity fixed-work median ≥1.18',
        'H15':'128×64×64 RTX4090 VM; FP64 active utilization ≥70%; executed count ratios within 5% of static 78:52:34',
    }
    for r in rows:
        c = r['chronology']; dates = [j['started_utc'] for j in c['execution_records'] if j['matching_protocol_identity_records']]
        dates_text = ', '.join(x.replace('T',' ')[:19] for x in dates) or 'no matching identity link found; see JSON'
        lines.append(f"| {r['id']} | {r['type']} | {brief[r['id']]} | {c['declared_utc'].replace('T',' ')[:19]} | {dates_text} | {r['outcome']} |")
    (OUT / 'hypothesis-registry.md').write_text('\n'.join(lines) + '\n')
    print('Wrote audit-units.json, hypothesis-registry.json and hypothesis-registry.md; saved-array checks passed.')

if __name__ == '__main__':
    main()
