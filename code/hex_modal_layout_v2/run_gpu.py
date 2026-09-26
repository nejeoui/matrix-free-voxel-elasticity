"""Fixed-work constrained products and independent physical compatibility solves."""
import argparse
import gc
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dependencies'))
from common import Reference, now, product_check, read, relative, sha, to_author, to_natural, write
from modal import Modal
from cuda import CudaModal
from variants import CudaVariants
from physical import physical_check, solve
from analysis import analyze


def load_data(path):
    with np.load(path) as file:
        return {name: np.ascontiguousarray(file[name]) if file[name].ndim else file[name].copy()
                for name in file.files}


def telemetry():
    command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw', '--format=csv,noheader']
    r = subprocess.run(command, capture_output=True, text=True, timeout=20)
    return {'utc': now(), 'returncode': r.returncode, 'output': r.stdout.strip()}


class Context:
    def __init__(self, data, cp, suite_type):
        self.cp, self.data = cp, data
        self.shape = tuple(map(int, data['shape']))
        self.suite = suite_type(*self.shape, data['ke'], build_edof=True)
        assert self.suite._edof32 is not None and self.suite._edof32.dtype == cp.int32
        assert self.suite._edof32.size == 24*int(np.prod(self.shape))
        blocks = Modal(data['ke']).blocks
        self.custom = CudaModal(self.shape, data['ke'], blocks)
        self.variants = CudaVariants(self.shape, data['ke'], blocks)
        self.mask = cp.asarray(to_author(data['mask'], self.shape, True))
        self.force = cp.asarray(to_author(data['force'], self.shape, True))
        self.fixed = 1-self.mask
        self.u = cp.asarray(to_author(data['u'], self.shape, True))
        self.path = 'modal8'
        self.material()

    def material(self):
        data, cp = self.data, self.cp
        self.modulus = cp.asarray(to_author(float(data['floor'])+(1-float(data['floor']))*data['rho']**3, self.shape))
        diagonal = self.suite.diagonal(self.modulus, path='node_fp64', scatter=False)
        self.diagonal = cp.where(self.mask != 0, diagonal, cp.float64(1))

    def action(self, u):
        free = u*self.mask
        if self.path in CudaVariants.NAMES:
            raw = self.variants.matvec_full(free, self.modulus, self.path)
        elif self.path in ('modal8', 'dense8'):
            raw = self.custom.matvec_full(free, self.modulus, self.path)
        else:
            raw = self.suite.matvec_full(free, self.modulus, path=self.path)
        return raw*self.mask+u*self.fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    out = a.out.resolve(); out.mkdir(exist_ok=False)
    p, inputs = read(HERE/'protocol.json'), read(a.inputs/'inputs.json')
    assert inputs['protocol_sha256'] == sha(HERE/'protocol.json')
    for name, digest in p['package_source_pins'].items():
        assert sha(HERE/name) == digest, name
    for row in inputs['rows']:
        assert sha(a.inputs/row['file']) == row['sha256'], row['file']
    shutil.copytree(HERE, out/'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(a.inputs/'inputs.json', out/'inputs.json')
    result = {'started_utc': now(), 'status': 'running', 'protocol_sha256': sha(HERE/'protocol.json'),
              'products': [], 'timings': [], 'solves': [], 'warmups': [], 'setups': [], 'telemetry': [],
              'gpu_executed': False, 'performance_claim': False, 'novelty_claim': False}
    write(out/'result.json', result)
    start = time.perf_counter()
    try:
        sys.path.insert(0, str(HERE/'donor/src'))
        import cupy as cp
        from gpu_fem.cuda_operators import OperatorSuite, PATH_SPEC
        from gpu_fem.simp_r2 import pcg
        assert cp.cuda.runtime.getDeviceCount() == 1
        device = cp.cuda.runtime.getDeviceProperties(0)
        assert device['name'].decode() == p['hardware']['gpu_full_name']
        result.update(gpu_executed=True, device={'name': device['name'].decode(),
            'total_memory': int(device['totalGlobalMem']), 'cupy': cp.__version__, 'numpy': np.__version__,
            'python': sys.version, 'platform': platform.platform(), 'driver_version': cp.cuda.runtime.driverGetVersion()})
        ref = Reference(a.reference)
        rng = np.random.default_rng(p['order_seed'])
        for case in p['cases']:
            assert time.perf_counter()-start < p['limits']['suite_seconds'], 'Suite wall-time limit'
            row = next(r for r in inputs['rows'] if r['id'] == case['id'])
            data = load_data(a.inputs/row['file'])
            target, base_energy, diagonal = ref.product(data, data['u'])
            modulus = float(data['floor'])+(1-float(data['floor']))*data['rho']**3
            base_compliance = float(modulus @ base_energy)
            begin = time.perf_counter(); ctx = Context(data, cp, OperatorSuite)
            cp.cuda.Stream.null.synchronize()
            error = relative(to_natural(cp.asnumpy(ctx.diagonal), ctx.shape, True), diagonal)
            assert error <= 1e-11, error
            setup = {'case': case['id'], 'seconds': time.perf_counter()-begin,
                     'diagonal_relative_error': error, 'kernel_attributes': {}}
            zero = cp.zeros_like(ctx.u)
            def products(phase):
                for path in p['paths']:
                    result['active_stage'] = {'case': case['id'], 'path': path, 'stage': phase+'-products'}
                    write(out/'result.json', result)
                    ctx.path = path
                    if path not in ('dense8', 'modal8', *p['new_paths']):
                        assert PATH_SPEC[path][:3] == ('fp64', 'fp64', 'fp64')
                    begin = time.perf_counter()
                    actual = to_natural(cp.asnumpy(ctx.action(ctx.u)), ctx.shape, True)
                    first_seconds = time.perf_counter()-begin
                    # Force a zero-input call directly after a nonzero one to catch stale accumulation.
                    zero_actual = to_natural(cp.asnumpy(ctx.action(zero)), ctx.shape, True)
                    check = product_check(actual, target, p)
                    filename = f"{case['id']}--{path}--{phase}-product.npz"
                    np.savez_compressed(out/filename, action=actual, zero=zero_actual)
                    record = {'case': case['id'], 'path': path, 'phase': phase, 'file': filename,
                        'sha256': sha(out/filename), 'action_check': check,
                        'zero_exact': bool(np.all(zero_actual == 0)),
                        'fixed_identity_exact': bool(np.array_equal(actual[data['mask'] == 0], data['u'][data['mask'] == 0])),
                        'first_or_audit_call_seconds_excluded': first_seconds}
                    result['products'].append(record); write(out/'result.json', result)
                    assert check['passed'] and record['zero_exact'] and record['fixed_identity_exact'], record
            products('before')
            for name, kernel in ctx.variants.kernels.items():
                setup['kernel_attributes'][name] = kernel.attributes
            result['setups'].append(setup)
            result['telemetry'].append(telemetry())
            if case['timed']:
                ctx.path = 'modal8'
                warm_start = time.perf_counter()
                while time.perf_counter()-warm_start < p['benchmarks']['thermal_seconds']:
                    for _ in range(50):
                        ctx.action(ctx.u)
                    cp.cuda.Stream.null.synchronize()
                for path in p['paths']:
                    ctx.path = path
                    for _ in range(p['benchmarks']['warm_calls']):
                        ctx.action(ctx.u)
                cp.cuda.Stream.null.synchronize()
                for rep in range(p['benchmarks']['rounds']):
                    order = list(rng.permutation(p['paths']))
                    for position, path in enumerate(order):
                        ctx.path = path
                        begin, end = cp.cuda.Event(), cp.cuda.Event()
                        cp.cuda.Stream.null.synchronize()
                        host_start = time.perf_counter(); begin.record()
                        for _ in range(p['benchmarks']['calls_per_round']):
                            ctx.action(ctx.u)
                        end.record(); end.synchronize()
                        host = time.perf_counter()-host_start
                        result['timings'].append({'case': case['id'], 'path': path, 'round': rep,
                            'position': position, 'calls': p['benchmarks']['calls_per_round'],
                            'gpu_ms_per_call': float(cp.cuda.get_elapsed_time(begin,end))/p['benchmarks']['calls_per_round'],
                            'host_ms_per_call': host*1000/p['benchmarks']['calls_per_round']})
                    write(out/'result.json', result)
                result['telemetry'].append(telemetry())
            products('after')
            if case['solve']:
                for path in rng.permutation(p['new_paths']):
                    ctx.path = path
                    for capped in (True, False):
                        result['active_stage'] = {'case': case['id'], 'path': path, 'stage': 'capped-solve' if capped else 'normal-solve'}
                        write(out/'result.json', result)
                        solution, record = solve(ctx, pcg, p['solver'], cap_one=capped)
                        record.update(physical_check(data, solution, ref, record['native_converged'], record['native_true_residual'], p['solver']))
                        state_error = relative(solution, data['u'])
                        compliance_error = abs(record['compliance']/base_compliance-1)
                        record.update(case=case['id'], path=path, cross_state_relative=state_error,
                            cross_compliance_relative=compliance_error,
                            cross_passed=bool(state_error <= p['solver']['cross_state_limit'] and compliance_error <= p['solver']['cross_compliance_limit']))
                        filename = f"{case['id']}--{path}--{'capped' if capped else 'normal'}-state.npz"
                        np.savez_compressed(out/filename, solution=solution)
                        record.update(file=filename, sha256=sha(out/filename))
                        result['warmups' if capped else 'solves'].append(record)
                        write(out/'result.json', result)
                        assert (record['iterations'] == 1 and not record['physical_passed']) if capped else (record['physical_passed'] and record['cross_passed']), record
            del ctx, data, target, diagonal, base_energy, zero
            gc.collect(); cp.get_default_memory_pool().free_all_blocks()
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    result.update(finished_utc=now(), elapsed_seconds=time.perf_counter()-start)
    write(out/'result.json', result)
    decision = analyze(result, p)
    write(out/'decision.json', decision)
    print(json.dumps({'status': result['status'], 'products': len(result['products']),
        'timings': len(result['timings']), 'solves': len(result['solves']), 'decision': decision['development_gates']}))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
