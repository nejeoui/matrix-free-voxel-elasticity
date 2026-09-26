"""Physical-state CUDA solve profiling; all mathematical kernels remain pinned."""
import time
PROCESS_START = time.perf_counter()
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dependencies'))
from common import Reference, now, read, relative, sha, to_author, to_natural, write
from modal import Modal
from cuda import CudaModal


def telemetry():
    command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw', '--format=csv,noheader']
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    return {'utc': now(), 'returncode': result.returncode, 'output': result.stdout.strip()}


class Context:
    def __init__(self, data, path, cp, suite_type):
        self.cp, self.path, self.data = cp, path, data
        self.shape = tuple(map(int, data['shape']))
        self.suite = suite_type(*self.shape, data['ke'], build_edof=(path == 'fused_fp64'))
        if path == 'fused_fp64':
            table = self.suite._edof32
            assert table is not None and table.dtype == cp.int32
            assert table.size == 24 * int(np.prod(self.shape))
        self.custom = CudaModal(self.shape, data['ke'], Modal(data['ke']).blocks) if path in ('modal8', 'dense8') else None
        self.mask = cp.asarray(to_author(data['mask'], self.shape, True))
        self.force = cp.asarray(to_author(data['force'], self.shape, True))
        self.fixed = 1 - self.mask

    def material(self):
        data, cp = self.data, self.cp
        self.modulus = cp.asarray(to_author(float(data['floor']) + (1-float(data['floor'])) * data['rho']**3, self.shape))
        diagonal = self.suite.diagonal(self.modulus, path='node_fp64', scatter=False)
        self.diagonal = cp.where(self.mask != 0, diagonal, cp.float64(1))

    def action(self, u):
        free = u * self.mask
        raw = (self.custom.matvec_full(free, self.modulus, self.path) if self.custom is not None
               else self.suite.matvec_full(free, self.modulus, path=self.path))
        return raw * self.mask + u * self.fixed


def physical_check(data, solution, ref, native_ok, native, config):
    action, energy, _ = ref.product(data, solution)
    residual = relative(action, data['force'])
    modulus = float(data['floor']) + (1-float(data['floor'])) * data['rho']**3
    compliance, work = float(modulus @ energy), float(data['force'] @ solution)
    energy_error = abs(compliance / work - 1) if work != 0 else float('inf')
    valid = bool(native_ok and native <= config['requested_rtol'] and
                 residual <= config['true_residual_limit'] and
                 energy_error <= config['energy_force_relative_limit'] and
                 np.isfinite(solution).all() and np.all(solution[data['mask'] == 0] == 0))
    return {'native_converged': bool(native_ok), 'native_true_residual': float(native),
            'independent_true_residual': residual, 'compliance': compliance,
            'force_work': work, 'energy_force_relative_error': energy_error,
            'physical_passed': valid}


def solve(ctx, pcg, config, profiled=False, cap_one=False):
    cp = ctx.cp
    maximum = 1 if cap_one else config['maximum_iterations']
    events = []
    event_setup = time.perf_counter()
    if profiled:
        capacity = maximum + maximum // config['check_every'] + 4
        events = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(capacity)]
    event_setup = time.perf_counter() - event_setup
    calls, history = 0, []
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    ctx.material()

    def action(u):
        nonlocal calls
        if profiled:
            begin, end = events[calls]
            begin.record()
            result = ctx.action(u)
            end.record()
        else:
            result = ctx.action(u)
        calls += 1
        if time.perf_counter() - start > config['maximum_state_seconds']:
            raise TimeoutError('Declared state wall-time limit')
        return result

    u, iterations, ok, native = pcg(action, ctx.force, 1 / ctx.diagonal,
        tol=config['requested_rtol'], maxiter=maximum, mask=ctx.mask,
        check_every=config['check_every'], residual_history=history)
    cp.cuda.Stream.null.synchronize()
    solution = to_natural(cp.asnumpy(u), ctx.shape, True)
    elapsed = time.perf_counter() - start
    event_ms = [float(cp.cuda.get_elapsed_time(a, b)) for a, b in events[:calls]] if profiled else []
    return solution, {'iterations': int(iterations), 'matvec_calls': calls,
        'solver_seconds': elapsed, 'profiled': profiled, 'cap_one': cap_one,
        'event_setup_seconds_excluded': event_setup,
        'product_event_ms': event_ms,
        'product_event_seconds': sum(event_ms) / 1000 if profiled else None,
        'residual_history': [float(v) for v in history],
        'native_converged': bool(ok), 'native_true_residual': float(native)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    p = read(HERE / 'protocol.json')
    for filename, digest in p['package_source_pins'].items():
        assert sha(HERE / filename) == digest, filename
    manifest = read(args.inputs / 'inputs.json')
    assert manifest['protocol_sha256'] == sha(HERE / 'protocol.json')
    for row in manifest['rows']:
        assert sha(args.inputs / row['file']) == row['sha256']
    shutil.copytree(HERE, out / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(args.inputs / 'inputs.json', out / 'inputs.json')
    result = {'started_utc': now(), 'status': 'running', 'protocol_sha256': sha(HERE / 'protocol.json'),
              'reference_sha256': sha(args.reference), 'solves': [], 'warmups': [], 'setups': [],
              'telemetry': [], 'gpu_executed': False, 'performance_claim': False,
              'novelty_claim': False, 'publication_goal_achieved': False}
    write(out / 'result.json', result)
    suite_start = time.perf_counter()
    try:
        sys.path.insert(0, str(HERE / 'donor/src'))
        import cupy as cp
        from gpu_fem.cuda_operators import OperatorSuite, PATH_SPEC
        from gpu_fem.simp_r2 import pcg
        assert cp.cuda.runtime.getDeviceCount() == 1
        device = cp.cuda.runtime.getDeviceProperties(0)
        assert device['name'].decode() == p['hardware']['gpu_full_name']
        result.update(gpu_executed=True, device={'name': device['name'].decode(),
            'total_memory': int(device['totalGlobalMem']), 'cupy': cp.__version__,
            'numpy': np.__version__, 'python': sys.version, 'platform': platform.platform(),
            'driver_version': cp.cuda.runtime.driverGetVersion()})
        result['process_start_to_cuda_ready_seconds'] = time.perf_counter() - PROCESS_START
        ref = Reference(args.reference)
        rng = np.random.default_rng(p['order_seed'])
        result['telemetry'].append(telemetry())
        for case in p['cases']:
            assert time.perf_counter() - suite_start < p['limits']['suite_seconds'], 'Suite wall-time limit'
            row = next(r for r in manifest['rows'] if r['id'] == case['id'])
            with np.load(args.inputs / row['file']) as saved:
                data = {name: np.ascontiguousarray(saved[name]) if saved[name].ndim else saved[name].copy() for name in saved.files}
            data['floor'] = np.asarray(p['material']['Emin'])
            _, _, independent_diagonal = ref.product(data, np.zeros_like(data['force']))
            contexts = {}
            for path in p['paths']:
                result['active_stage'] = {'case':case['id'], 'path':path, 'stage':'setup_and_capped_warmup'}
                write(out / 'result.json', result)
                if path not in ('modal8', 'dense8'):
                    assert PATH_SPEC[path][:3] == ('fp64', 'fp64', 'fp64')
                begin = time.perf_counter()
                ctx = Context(data, path, cp, OperatorSuite)
                ctx.material()
                diagonal = to_natural(cp.asnumpy(ctx.diagonal), ctx.shape, True)
                error = relative(diagonal, independent_diagonal)
                assert error <= 1e-11
                contexts[path] = ctx
                # A declared capped warmup compiles the actual solver operations.
                # Its nonconverged state is retained and never used as a timing sample.
                solution, control = solve(ctx, pcg, p['solver'], cap_one=True)
                control.update(physical_check(data, solution, ref, control['native_converged'], control['native_true_residual'], p['solver']))
                filename = f"{case['id']}--{path}--warmup.npz"
                np.savez_compressed(out / filename, solution=solution)
                control.update(case=case['id'], path=path, file=filename, sha256=sha(out / filename),
                               expected_rejection=True, diagonal_relative_l2=error)
                assert control['iterations'] == 1 and not control['physical_passed'] and not control['native_converged']
                result['warmups'].append(control)
                result['setups'].append({'case': case['id'], 'path': path,
                                         'setup_and_warmup_seconds': time.perf_counter()-begin})
                write(out / 'result.json', result)
            if case.get('thermal_preamble'):
                begin = time.perf_counter()
                probe = cp.zeros_like(contexts['modal8'].force)
                while time.perf_counter() - begin < p['thermal_preamble_seconds']:
                    for _ in range(20):
                        contexts['modal8'].action(probe)
                    cp.cuda.Stream.null.synchronize()
                result['thermal_preamble_actual_seconds'] = time.perf_counter() - begin
            modes = ['plain'] * case['plain_replicates'] + ['profile']
            sequence = [(mode, path) for mode in modes for path in p['paths']]
            # Shuffle a labelled block schedule, preserving all repetitions.
            occurrences = {}
            labelled = []
            for mode, path in sequence:
                replicate = occurrences.get((mode, path), 0)
                occurrences[(mode, path)] = replicate + 1
                labelled.append((mode, path, replicate))
            order = rng.permutation(len(labelled))
            for position, index in enumerate(order):
                assert time.perf_counter() - suite_start < p['limits']['suite_seconds'], 'Suite wall-time limit'
                mode, path, replicate = labelled[int(index)]
                result['active_stage'] = {'case':case['id'], 'path':path, 'stage':mode, 'replicate':replicate}
                write(out / 'result.json', result)
                solution, record = solve(contexts[path], pcg, p['solver'], profiled=(mode == 'profile'))
                audit_start = time.perf_counter()
                record.update(physical_check(data, solution, ref, record['native_converged'], record['native_true_residual'], p['solver']))
                filename = f"{case['id']}--{path}--{mode}-{replicate:02d}.npz"
                np.savez_compressed(out / filename, solution=solution)
                record.update(case=case['id'], path=path, replicate=replicate, position=position,
                              file=filename, sha256=sha(out / filename),
                              audit_and_retention_seconds=time.perf_counter()-audit_start)
                result['solves'].append(record)
                write(out / 'result.json', result)
                print(json.dumps({k: record[k] for k in ('case','path','profiled','replicate','iterations','solver_seconds','product_event_seconds','physical_passed')}), flush=True)
                assert record['physical_passed'], 'Normal state failed physical acceptance'
            result['telemetry'].append(telemetry())
            del contexts, ctx, solution, data, independent_diagonal
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        result['status'] = 'completed'
        result.pop('active_stage', None)
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_utc=now(), elapsed_seconds=time.perf_counter()-suite_start)
        write(out / 'result.json', result)


if __name__ == '__main__':
    main()
