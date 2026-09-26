"""Replay every retained CUDA state independently on CPU and audit its profile."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dependencies'))
from common import Reference, now, read, relative, sha, write
from run_gpu import physical_check
from analysis import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    p, result = read(HERE / 'protocol.json'), read(args.run / 'result.json')
    assert result['protocol_sha256'] == sha(HERE / 'protocol.json')
    for filename, digest in p['package_source_pins'].items():
        assert sha(HERE / filename) == digest
    manifest = read(args.inputs / 'inputs.json')
    assert manifest['protocol_sha256'] == result['protocol_sha256']
    reference = Reference(args.reference)
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    reference.lib.element.argtypes = [ctypes.c_double, ctypes.c_double, array]
    reference.lib.element.restype = None
    states, maximum, cross = 0, 0., []
    for case in p['cases']:
        source = next(r for r in manifest['rows'] if r['id'] == case['id'])
        assert sha(args.inputs / source['file']) == source['sha256']
        with np.load(args.inputs / source['file']) as datafile:
            data = {name: np.ascontiguousarray(datafile[name]) if datafile[name].ndim else datafile[name].copy() for name in datafile.files}
        shape = tuple(map(int, data['shape']))
        assert shape == (2*case['q'], case['q'], case['q'])
        ke = np.empty((24,24))
        reference.lib.element(1/case['q'], p['material']['nu'], ke)
        assert relative(data['ke'], ke) < 1e-14
        assert np.isfinite(data['rho']).all() and np.all((data['rho'] >= 0) & (data['rho'] <= 1))
        data['floor'] = np.asarray(p['material']['Emin'])
        normal = [r for r in result['solves'] if r['case'] == case['id']]
        baselines = [r for r in normal if r['path'] == 'fused_ai_fp64' and not r['profiled']]
        base = None
        if baselines:
            baselines.sort(key=lambda r:r['replicate'])
            with np.load(args.run / baselines[0]['file']) as file:
                base = file['solution']
            base_compliance = baselines[0]['compliance']
        for record in [r for r in result['warmups'] if r['case'] == case['id']] + normal:
            filename = args.run / record['file']
            assert sha(filename) == record['sha256']
            with np.load(filename) as file:
                solution = file['solution']
            assert solution.shape == data['force'].shape and solution.dtype == np.float64
            check = physical_check(data, solution, reference, record['native_converged'], record['native_true_residual'], p['solver'])
            assert check['physical_passed'] == record['physical_passed']
            assert abs(check['independent_true_residual']-record['independent_true_residual']) <= 1e-12+1e-5*check['independent_true_residual']
            assert abs(check['compliance']/record['compliance']-1) <= 1e-11
            if record['cap_one']:
                assert record['iterations'] == 1 and not check['physical_passed']
            else:
                maximum = max(maximum, check['independent_true_residual'])
                if result['status'] == 'completed':
                    assert check['physical_passed']
                if base is not None:
                    state_error = relative(solution, base)
                    compliance_error = abs(check['compliance']/base_compliance-1)
                    valid = state_error <= p['solver']['cross_state_limit'] and compliance_error <= p['solver']['cross_compliance_limit']
                    cross.append({'file':record['file'],'state_relative_l2':state_error,'compliance_relative':compliance_error,'valid':valid})
                    if result['status'] == 'completed':
                        assert valid, cross[-1]
            if record['profiled']:
                assert len(record['product_event_ms']) == record['matvec_calls']
                assert all(np.isfinite(t) and t >= 0 for t in record['product_event_ms'])
                assert abs(sum(record['product_event_ms'])/1000-record['product_event_seconds']) < 1e-10
            states += 1
    decision = analyze(result,p)
    write(args.out, {'verified_utc':now(),'evidence_valid':True,'states_replayed':states,
         'normal_states':len(result['solves']),'capped_warmup_states':len(result['warmups']),
         'maximum_normal_true_residual':maximum,'cross_path_checks':cross,
         'result_sha256':sha(args.run/'result.json'),'protocol_sha256':sha(HERE/'protocol.json'),
         'diagnostic_decision':decision})
    print(json.dumps({'evidence_valid':True,'states_replayed':states,
                      'advance_to_charged_campaign':decision['advance_to_charged_campaign']}))


if __name__ == '__main__':
    main()
