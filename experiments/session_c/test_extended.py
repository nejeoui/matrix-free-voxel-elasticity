"""CPU tests of the extended decisions (H3, H5, fastest method) and runner patches. No GPU."""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'dependencies'))
import analysis  # noqa: E402

P = json.loads((HERE / 'protocol_template.json').read_text())
KIND = {m['id']: m['kind'] for m in P['to_tolerance']['methods']}


def products(ratio_fp64=1.05, ratio_fp32=1.0):
    blocks = []
    for c in P['cases']:
        if 'products' not in c.get('stages', ['products']):
            continue
        for prec, paths, r in (('fp64', P['paths_fp64'], ratio_fp64), ('fp32', P['paths_fp32'], ratio_fp32)):
            cand = P['candidate_fp64'] if prec == 'fp64' else P['candidate_fp32']
            rounds = [{'round': k, 'order': paths,
                       'ms_per_product': {p: (1.0 / r if p == cand else 1.0) for p in paths}}
                      for k in range(P['products']['rounds'])]
            blocks.append({'case': c['id'], 'precision': prec, 'rounds': rounds})
    return blocks


def tolerance(fail=()):
    rows = []
    for c in P['cases']:
        if 'to_tolerance' not in c.get('stages', ['to_tolerance']):
            continue
        for mid in c.get('methods', list(KIND)):
            for rep in range(P['to_tolerance']['repetitions']):
                rows.append({'case': c['id'], 'method': mid, 'rep': rep,
                             'solver_seconds': 1.0 if KIND[mid] == 'ir' else 2.0,
                             'physical_passed': not ((c['id'], mid) in fail and rep == 0)})
    return rows


def result(**kw):
    return {'status': 'completed', 'fixed_work': [], 'products': products(**{k: v for k, v in kw.items() if k.startswith('ratio')}),
            'to_tolerance': tolerance(kw.get('fail', ()))}


def test_runner_and_verifier_parse():
    for f in ('run_gpu.py', 'verify.py', 'analysis.py'):
        ast.parse((HERE / f).read_text())


def test_fastest_method_ignores_failed_methods():
    d = analysis.extended(result(), P)
    for c, e in d['fastest_passing_method'].items():
        assert KIND[e['method']] == 'ir'


if 'h3' in P['decision']:
    def test_h3_confirmed_when_datacentre_gain_is_small():
        assert analysis.extended(result(ratio_fp64=1.03), P)['h3_decision'] == 'predicted_no_datacentre_fp64_gain_confirmed'

    def test_h3_not_confirmed_when_gain_persists():
        assert analysis.extended(result(ratio_fp64=1.25), P)['h3_decision'] != 'predicted_no_datacentre_fp64_gain_confirmed'

if 'h5' in P['decision']:
    ROB = [c['id'] for c in P['cases'] if c.get('robustness')]

    def test_h5_supported_only_if_refinement_fails_and_fp64_pcg_passes():
        d = analysis.extended(result(fail={(ROB[0], 'ir-fused_ai64-fused_ai32')}), P)
        assert d['h5_decision'] == 'fp64_throughout_niche_supported'

    def test_h5_not_supported_when_everything_passes():
        assert analysis.extended(result(), P)['h5_decision'] == 'fp64_throughout_niche_not_supported'

    def test_h5_not_supported_when_fp64_pcg_also_fails():
        d = analysis.extended(result(fail={(ROB[0], 'ir-fused_ai64-fused_ai32'), (ROB[0], 'pcg64-modal8')}), P)
        assert d['h5_decision'] == 'fp64_throughout_niche_not_supported'

    def test_h5_ignores_failures_on_non_robustness_cases():
        other = next(c['id'] for c in P['cases'] if not c.get('robustness') and 'ir-fused_ai64-node32' in c.get('methods', []))
        assert analysis.extended(result(fail={(other, 'ir-fused_ai64-node32')}), P)['h5_decision'] == 'fp64_throughout_niche_not_supported'
