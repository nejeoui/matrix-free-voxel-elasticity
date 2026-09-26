"""Session E tables and paper numbers from the verified evidence of both RTX 4090 hosts.

Inputs:
  results/session-e-host{1,2}-20260925/local-verification.json  (independent CPU replays + decisions)
  results/session-e-host{1,2}-20260925/rental.json               (host identity)
  Companion-project evidence (read-only):
    sm_89 SASS instruction analysis of the same butterflies (evidence/companion/.../hex-modal-signed-muladd-gpu-20260924)
    resident-action comparison with libCEED CUDA backends (evidence/companion/.../hex-modal-libceed-resident-20260924)
Outputs: paper/session_e_tables.md, paper/session_e_numbers.json
"""
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

V = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V / 'experiments/session_e'))
import analysis  # noqa: E402

COMPANION = V / 'evidence/companion/results/rescope'  # copies of the companion evidence
SASS = COMPANION / 'hex-modal-signed-muladd-gpu-20260924/rental-01/remote/run-01/build/instruction-analysis.json'
CEED = COMPANION / 'hex-modal-libceed-resident-20260924/figures/ratios.csv'
HOSTS = ['host1', 'host2']
REAL = ['optimized-q64', 'optimized-q96', 'uniform-q64']


def load(p):
    return json.loads(Path(p).read_text())


def main():
    out, md = {'hosts': {}}, ['# Session E: butterfly variants on two RTX 4090 hosts', '']
    decisions = []
    for h in HOSTS:
        d = V / f'results/session-e-{h}-20260925'
        if not (d / 'local-verification.json').exists():
            md.append(f'{h}: no verified evidence'); continue
        lv, rent = load(d / 'local-verification.json'), load(d / 'rental.json')
        dec = lv['decision']
        decisions.append(dec)
        o = rent['offer']
        host = {'machine_id': o['machine_id'], 'cpu': o['cpu_name'], 'geolocation': o['geolocation'],
                'instance_id': rent['instance_id'], 'evidence_valid': lv['evidence_valid'],
                'h8': dec['h8_decision'], 'h9': dec['h9_decision'], 'products': {}, 'mg_fixed': {},
                'to_tolerance': {}, 'outer_effect': {}, 'profiles': {}, 'variant_max_rel': 0.0}
        for e in dec['products']:
            host['products'].setdefault(e['case'], {})[e['numerator']] = {
                'ratio': e['median_ratio'], 'lower95': e['lower95'], 'ms': e['numerator_median_ms'],
                'candidate_ms': e['candidate_median_ms']}
        for e in dec['fixed_work']:
            host['mg_fixed'].setdefault(e['case'], {})[e['numerator']] = {
                'ratio': e['median_ratio'], 'lower95': e['lower95'], 'identical_work': e['identical_work']}
        for e in dec['to_tolerance']:
            host['to_tolerance'].setdefault(e['case'], {})[e['arm']] = {
                'seconds': e['median_seconds'], 'iterations': e['iterations'], 'passed': e['all_physical_passed']}
        for c, e in dec['outer_effect'].items():
            host['outer_effect'][c] = {'ratio': e['median_ratio'], 'lower95': e['lower95']}
        for p in dec['profiles']:
            host['profiles'].setdefault(p['case'], {})[p['arm']] = {
                k: p[k] for k in ('solver_seconds', 'vcycle_finest_product_seconds', 'outer_product_seconds')}
        host['variant_max_rel'] = max(v['relative_l2_vs_modal8'] for v in dec['variant_differences'])
        host['fastest_passing'] = dec['fastest_passing']
        out['hosts'][h] = host

        md += [f'## {h}: machine {o["machine_id"]} ({o["cpu_name"]}, {o["geolocation"]}), instance {rent["instance_id"]}',
               '', f'evidence valid {lv["evidence_valid"]}; H8 {dec["h8_decision"]}; H9 {dec["h9_decision"]}; '
               f'max variant-vs-modal8 relative L2 {host["variant_max_rel"]:.1e}', '',
               '| case | product: X / modal8_muladd | median | lower 95 | X ms | muladd ms |', '|---|---|---:|---:|---:|---:|']
        for c in REAL:
            for k, e in host['products'][c].items():
                md.append(f'| {c} | {k} | {e["ratio"]:.3f} | {e["lower95"]:.3f} | {e["ms"]:.3f} | {e["candidate_ms"]:.3f} |')
        md += ['', '| case | multigrid fixed work: arm / mg64-modal8_muladd | median | lower 95 | identical work |',
               '|---|---|---:|---:|---|']
        for c in REAL:
            for k, e in host['mg_fixed'][c].items():
                md.append(f'| {c} | {k} | {e["ratio"]:.3f} | {e["lower95"]:.3f} | {e["identical_work"]} |')
        md += ['', '| case | arm | time to 1e-8 (s) | iterations | all passed |', '|---|---|---:|---|---|']
        for c in REAL:
            for k, e in sorted(host['to_tolerance'][c].items(), key=lambda kv: kv[1]['seconds']):
                md.append(f'| {c} | {k} | {e["seconds"]:.4f} | {e["iterations"]} | {e["passed"]} |')
        md += ['', 'outer-kernel effect (mg32-fused_ai64-node32 / mg32-modal8_muladd-node32): ' +
               ', '.join(f'{c} {e["ratio"]:.3f} (lb {e["lower95"]:.3f})' for c, e in host['outer_effect'].items()
                         if c in REAL), '']
    out['replication'] = analysis.replicate(decisions)
    if len(out['hosts']) == 2:
        h1, h2 = (out['hosts'][h] for h in HOSTS)
        out['cross_host'] = {
            c: {'product_vs_fused_ai': [h1['products'][c]['fused_ai_fp64']['ratio'], h2['products'][c]['fused_ai_fp64']['ratio']],
                'product_vs_modal8': [h1['products'][c]['modal8']['ratio'], h2['products'][c]['modal8']['ratio']],
                'mg_vs_fused_ai': [h1['mg_fixed'][c]['mg64-fused_ai']['ratio'], h2['mg_fixed'][c]['mg64-fused_ai']['ratio']],
                'mg_vs_modal8': [h1['mg_fixed'][c]['mg64-modal8']['ratio'], h2['mg_fixed'][c]['mg64-modal8']['ratio']],
                'fastest': [h1['fastest_passing'][c]['arm'], h2['fastest_passing'][c]['arm']]}
            for c in REAL}
    # ---- companion evidence: static FP64 instruction sites (sm_89) and libCEED comparison
    sass = load(SASS)['action']
    sites = {}
    for k, v in sass.items():
        ops = Counter((i.split()[1] if i.startswith('@') else i.split()[0]).split('.')[0] for i in v['instructions'])
        sites[k] = {o: ops[o] for o in ('DADD', 'DMUL', 'DFMA')}
        sites[k]['total'] = sum(sites[k].values())
    out['companion_fp64_instruction_sites'] = sites
    ceed = {}
    with CEED.open() as f:
        for r in csv.DictReader(f):
            ceed.setdefault(r['control'], []).append(float(r['median']))
    out['companion_libceed'] = {k: {'min': min(v), 'max': max(v), 'cases': len(v)} for k, v in ceed.items()}
    md += ['## Companion evidence (different harness and memory layout)', '',
           '| kernel | DADD | DMUL | DFMA | FP64 sites |', '|---|---:|---:|---:|---:|']
    md += [f'| {k} | {v["DADD"]} | {v["DMUL"]} | {v["DFMA"]} | {v["total"]} |' for k, v in sites.items()]
    md += ['', '| control / modal_signed_muladd (evidence/companion/.../hex-modal-libceed-resident-20260924) | min | max |', '|---|---:|---:|']
    md += [f'| {k} | {v["min"]:.2f} | {v["max"]:.2f} |' for k, v in out['companion_libceed'].items()]
    md += ['', f'replication: {out["replication"]}']
    (V / 'paper/session_e_numbers.json').write_text(json.dumps(out, indent=1) + '\n')
    (V / 'paper/session_e_tables.md').write_text('\n'.join(md) + '\n')
    print(json.dumps({'replication': out['replication'], 'cross_host': out.get('cross_host')}, indent=1))


if __name__ == '__main__':
    main()
