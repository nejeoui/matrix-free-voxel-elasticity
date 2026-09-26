"""Run every declared CUDA path with raw outputs and synchronized timings."""
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

from common import Reference, now, product_check, read, relative, sha, solver_check, timing_decision, to_author, to_natural, write
from cuda import CudaModal

HERE = Path(__file__).resolve().parent


def telemetry():
    command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw', '--format=csv,noheader']
    p = subprocess.run(command, capture_output=True, text=True, timeout=20)
    return {'utc': now(), 'returncode': p.returncode, 'output': p.stdout.strip(), 'stderr': p.stderr.strip()}


class Context:
    def __init__(self, data, cp, suite_type):
        start = time.perf_counter()
        self.cp, self.data = cp, data
        self.shape = tuple(map(int, data['shape']))
        self.suite = suite_type(*self.shape, data['ke'])
        self.custom = CudaModal(self.shape, data['ke'], data['blocks'])
        self.E = cp.asarray(to_author(float(data['floor']) + (1 - float(data['floor'])) * data['rho'] ** 3, self.shape))
        self.mask = cp.asarray(to_author(data['mask'], self.shape, True))
        self.fixed = 1 - self.mask
        self.u = cp.asarray(to_author(data['u'], self.shape, True))
        self.force = cp.asarray(to_author(data['force'], self.shape, True))
        self.diagonal = cp.asarray(to_author(data['diagonal'], self.shape, True))
        cp.cuda.Stream.null.synchronize()
        self.setup_seconds = time.perf_counter() - start

    def action(self, u, path):
        free = u * self.mask
        if path in ('modal8', 'dense8'):
            raw = self.custom.matvec_full(free, self.E, path)
        else:
            raw = self.suite.matvec_full(free, self.E, path=path)
        return raw * self.mask + u * self.fixed

    def natural(self, x):
        return to_natural(self.cp.asnumpy(x), self.shape, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = args.inputs.resolve()
    manifest = read(inputs / 'inputs.json')
    p = read(HERE / 'protocol.json')
    assert sha(HERE / 'protocol.json') == manifest['protocol_sha256']
    for name, row in manifest['files'].items():
        assert sha(inputs / name) == row['sha256'] and (inputs / name).stat().st_size == row['bytes'], name
    assert sha(HERE / 'cuda.py') == sha(inputs / 'cuda.py'), 'Prototype differs from frozen inputs'
    for name, digest in p['donor_files'].items():
        assert sha(inputs / 'donor' / name) == digest
    sys.path.insert(0, str(inputs / 'donor/src'))
    for name in ('run_gpu.py', 'common.py', 'cuda.py', 'protocol.json'):
        shutil.copyfile(HERE / name, out / name)
    result = {'started_utc': now(), 'status': 'running', 'protocol_sha256': sha(HERE / 'protocol.json'),
              'inputs_manifest_sha256': sha(inputs / 'inputs.json'), 'reference_sha256': sha(args.reference),
              'source_hashes': {n: sha(HERE / n) for n in ('run_gpu.py', 'common.py', 'cuda.py')},
              'products': [], 'timings': [], 'solves': [], 'setups': [], 'telemetry': [],
              'gpu_executed': False, 'publication_goal_achieved': False}
    started = time.monotonic()

    def save():
        write(out / 'result.json', result)

    def guard():
        assert time.monotonic() - started < p['limits']['maximum_job_seconds'], 'Study time limit'

    save()
    try:
        import cupy as cp
        from gpu_fem.cuda_operators import OperatorSuite, PATH_SPEC
        from gpu_fem.simp_r2 import pcg
        assert cp.cuda.runtime.getDeviceCount() == 1, 'Exclusive single visible GPU required'
        assert CudaModal.BLOCK == p['cuda_block_threads']
        for name in p['paths']:
            if name not in ('modal8', 'dense8'):
                assert PATH_SPEC[name][:3] == ('fp64', 'fp64', 'fp64')
        ref = Reference(args.reference)
        props = cp.cuda.runtime.getDeviceProperties(0)
        result.update(gpu_executed=True, device={'name': props['name'].decode(), 'total_memory': int(props['totalGlobalMem']),
            'compute_capability': [int(props['major']), int(props['minor'])], 'cupy': cp.__version__, 'numpy': np.__version__,
            'python': sys.version, 'platform': platform.platform(), 'driver_version': cp.cuda.runtime.driverGetVersion(),
            'runtime_version': cp.cuda.runtime.runtimeGetVersion()})
        result['telemetry'].append(telemetry())
        rng = np.random.default_rng(p['benchmarks']['seed'])
        for row in manifest['rows']:
            guard()
            with np.load(inputs / row['file']) as file:
                data = {k: np.ascontiguousarray(file[k]) if file[k].ndim else file[k].copy() for k in file.files}
            ctx = Context(data, cp, OperatorSuite)
            setup = {'id': row['id'], 'construction_seconds': ctx.setup_seconds, 'first_call_seconds': {}}
            target, _, diagonal = ref.product(data, data['u'])
            assert product_check(target, data['target'], p)['passed'], 'Independent reference disagrees with pinned input'
            assert relative(diagonal, data['diagonal']) <= 1e-11
            for path in p['paths']:
                guard()
                cp.cuda.Stream.null.synchronize()
                begin = time.perf_counter()
                actual = ctx.natural(ctx.action(ctx.u, path))
                setup['first_call_seconds'][path] = time.perf_counter() - begin
                check = product_check(actual, target, p)
                zero = ctx.natural(ctx.action(cp.zeros_like(ctx.u), path))
                check.update(id=row['id'], path=path, zero_exact=bool(np.count_nonzero(zero) == 0),
                             fixed_identity_exact=bool(np.array_equal(actual[data['mask'] == 0], data['u'][data['mask'] == 0])))
                if row['kind'] == 'saved_state':
                    check['true_residual'] = relative(actual, data['force'])
                check['passed'] = bool(check['passed'] and check['zero_exact'] and check['fixed_identity_exact'] and
                    check.get('true_residual', 0.) <= p['correctness']['state_true_residual_limit'])
                name = row['id'] + '--' + path + '--product.npz'
                np.savez_compressed(out / name, action=actual, zero=zero)
                check.update(file=name, sha256=sha(out / name))
                result['products'].append(check)
                save()
            result['setups'].append(setup)
            assert all(c['passed'] for c in result['products'] if c['id'] == row['id']), 'Product control failed'
            if row['kind'] == 'benchmark':
                for path in p['paths']:
                    for _ in range(p['benchmarks']['warmup_calls_per_path']):
                        ctx.action(ctx.u, path)
                cp.cuda.Stream.null.synchronize()
                result['telemetry'].append(telemetry())
                for round_number in range(p['benchmarks']['rounds']):
                    order = [str(x) for x in rng.permutation(p['paths'])]
                    for position, path in enumerate(order):
                        guard()
                        start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
                        cp.cuda.Stream.null.synchronize()
                        begin = time.perf_counter()
                        start_event.record()
                        for _ in range(p['benchmarks']['calls_per_batch']):
                            ctx.action(ctx.u, path)
                        end_event.record()
                        end_event.synchronize()
                        wall = time.perf_counter() - begin
                        result['timings'].append({'id': row['id'], 'round': round_number, 'position': position,
                            'path': path, 'calls': p['benchmarks']['calls_per_batch'],
                            'gpu_ms_per_call': float(cp.cuda.get_elapsed_time(start_event, end_event)) / p['benchmarks']['calls_per_batch'],
                            'synchronized_host_seconds_per_call': wall / p['benchmarks']['calls_per_batch']})
                    save()
                result['telemetry'].append(telemetry())
                print(json.dumps({'benchmark_completed': row['id']}), flush=True)
            if row['id'] in p['solver_check']['cases']:
                for replicate in range(p['solver_check']['replicates']):
                    for path in map(str, rng.permutation(p['paths'])):
                        guard()
                        history, calls = [], [0]

                        def action(x):
                            calls[0] += 1
                            return ctx.action(x, path)

                        for cap_one in ([False, True] if replicate == 0 and row['id'] == 'cantilever-q8-final' else [False]):
                            history.clear()
                            calls[0] = 0
                            cp.cuda.Stream.null.synchronize()
                            begin = time.perf_counter()
                            u, iterations, ok, native = pcg(action, ctx.force, 1 / ctx.diagonal,
                                tol=p['correctness']['solver_requested_rtol'],
                                maxiter=1 if cap_one else p['correctness']['solver_max_iterations'],
                                mask=ctx.mask, check_every=p['correctness']['solver_check_every'], residual_history=history)
                            cp.cuda.Stream.null.synchronize()
                            elapsed = time.perf_counter() - begin
                            solution = ctx.natural(u)
                            check, independent, energy = solver_check(solution, data, ref, p, ok, native)
                            check.update(id=row['id'], path=path, replicate=replicate, cap_one=cap_one,
                                         iterations=int(iterations), matvec_calls=calls[0], solver_seconds=elapsed,
                                         residual_history=[float(v) for v in history])
                            check['passed'] = not check['physical_passed'] if cap_one else check['physical_passed']
                            name = row['id'] + '--' + path + f'--solve-{replicate}' + ('-cap1' if cap_one else '') + '.npz'
                            np.savez_compressed(out / name, solution=solution, action=independent, energy=energy)
                            check.update(file=name, sha256=sha(out / name))
                            result['solves'].append(check)
                            save()
                            print(json.dumps({k: check[k] for k in ('id', 'path', 'replicate', 'cap_one', 'passed', 'iterations', 'solver_seconds')}), flush=True)
                assert all(c['passed'] for c in result['solves'] if c['id'] == row['id']), 'Solver control failed'
            cp.cuda.Stream.null.synchronize()
            del ctx
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        assert len(result['products']) == len(manifest['rows']) * len(p['paths']) == 126
        assert len(result['solves']) == len(p['solver_check']['cases']) * len(p['paths']) * p['solver_check']['replicates'] + len(p['paths']) == 78
        decision = timing_decision(result['timings'], p)
        result.update(decision)
        correctness = all(r['passed'] for r in result['products'] + result['solves'])
        result.update(status='completed', correctness_gate_passed=correctness,
                      development_gate_passed=correctness and decision['timing_gate_passed'])
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                      correctness_gate_passed=False, development_gate_passed=False)
        save()
        raise
    finally:
        result.update(finished_utc=now(), operational_seconds=time.monotonic() - started)
        save()
    print(json.dumps({k: result[k] for k in ('status', 'correctness_gate_passed', 'timing_gate_passed', 'development_gate_passed', 'operational_seconds')}), flush=True)


if __name__ == '__main__':
    main()
