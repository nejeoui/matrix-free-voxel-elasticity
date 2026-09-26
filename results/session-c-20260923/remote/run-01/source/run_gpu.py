"""Voxel session_c: fixed-work PCG, time-to-tolerance and FP64/FP32 products (generalised session A runner).

Generalisations over session A: a list of accepted GPU names, a per-case material floor (Emin), and
per-case stages. Everything else, including every kernel and solver, is unchanged.

All mathematical kernels are pinned: the donor operators (BSD-3, Yang/Wang/Wang), the measured FP64
modal8/dense8 prototype, and FP32 variants derived from it by type substitution. The donor `pcg`
and `pcg_ir` are used unchanged.

E1 fixes the flaw that failed JPDC's profile gate: every fixed-work solve runs PCG with tol=0 and
an identical `maxiter`, so all paths perform exactly the same iterations and matvec calls. It runs
in paired rounds; each round executes every path once in random order.
"""
import argparse
import gc
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROCESS_START = time.perf_counter()
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dependencies'))
from common import Reference, now, read, relative, sha, to_author, to_natural, write
from modal import Modal
from cuda import CudaModal
from cuda32 import CudaModal32


def telemetry():
    """Same nvidia-smi query that ran successfully in JPDC profile v2 on an RTX 4090."""
    command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,'
               'temperature.gpu,clocks.sm,clocks.mem,power.draw', '--format=csv,noheader']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {'utc': now(), 'returncode': result.returncode, 'output': result.stdout.strip()}
    except Exception as exc:  # telemetry must never stop the experiment
        return {'utc': now(), 'error': repr(exc)}


class Context:
    """One operator path at one precision on one physical state."""

    def __init__(self, data, path, cp, suite_type, needs_edof):
        self.cp, self.path, self.data = cp, path, data
        self.fp32 = path.endswith('_fp32')
        self.dtype = cp.float32 if self.fp32 else cp.float64
        self.shape = tuple(map(int, data['shape']))
        blocks = Modal(data['ke']).blocks
        self.custom = None
        if path in ('modal8', 'dense8'):
            self.custom = CudaModal(self.shape, data['ke'], blocks)
        elif path in ('modal8_fp32', 'dense8_fp32'):
            self.custom = CudaModal32(self.shape, data['ke'], blocks)
        else:
            self.suite = suite_type(*self.shape, data['ke'], build_edof=(path in needs_edof))
        self.mask = cp.asarray(to_author(data['mask'], self.shape, True), dtype=self.dtype)
        self.force = cp.asarray(to_author(data['force'], self.shape, True), dtype=self.dtype)
        self.fixed = 1 - self.mask

    def material(self, diag_suite):
        data, cp = self.data, self.cp
        modulus64 = cp.asarray(to_author(float(data['floor']) + (1 - float(data['floor'])) * data['rho'] ** 3,
                                         self.shape))
        diagonal = diag_suite.diagonal(modulus64, path='node_fp64', scatter=False)
        self.diagonal64 = cp.where(cp.asarray(to_author(data['mask'], self.shape, True)) != 0,
                                   diagonal, cp.float64(1))
        self.modulus = modulus64.astype(self.dtype)
        self.inverse_diagonal = (1 / self.diagonal64).astype(self.dtype)

    def action(self, u):
        free = u * self.mask
        raw = (self.custom.matvec_full(free, self.modulus, self.path) if self.custom is not None
               else self.suite.matvec_full(free, self.modulus, path=self.path))
        return raw * self.mask + u * self.fixed


def physical_check(data, solution, ref, native_ok, native, config):
    action, energy, _ = ref.product(data, solution)
    residual = relative(action, data['force'])
    modulus = float(data['floor']) + (1 - float(data['floor'])) * data['rho'] ** 3
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


def counted(ctx, limit_seconds, start):
    calls = [0]

    def action(u):
        calls[0] += 1
        if time.perf_counter() - start > limit_seconds:
            raise TimeoutError('Declared state wall-time limit')
        return ctx.action(u)
    return action, calls


def run_pcg(ctx, pcg, solver, maxiter, tol):
    cp = ctx.cp
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    action, calls = counted(ctx, solver['maximum_state_seconds'], start)
    u, iterations, ok, native = pcg(action, ctx.force, ctx.inverse_diagonal, tol=tol, maxiter=maxiter,
                                    mask=ctx.mask, check_every=solver['check_every'])
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    return u, {'iterations': int(iterations), 'matvec_calls': calls[0], 'solver_seconds': elapsed,
               'native_converged': bool(ok), 'native_true_residual': float(native)}


def run_ir(outer, inner, pcg_ir, solver, ir):
    cp = outer.cp
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    act64, calls64 = counted(outer, solver['maximum_state_seconds'], start)
    act32, calls32 = counted(inner, solver['maximum_state_seconds'], start)
    trace = []
    u, outer_it, inner_it, ok, native = pcg_ir(
        act64, outer.force, act32, inner.inverse_diagonal, tol=solver['requested_rtol'],
        mask32=inner.mask, inner_tol=ir['inner_tol'], max_outer=ir['max_outer'],
        inner_maxiter=ir['inner_maxiter'], check_every=solver['check_every'], trace=trace)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    return u, {'outer_iterations': int(outer_it), 'inner_iterations': int(inner_it),
               'matvec64_calls': calls64[0], 'matvec32_calls': calls32[0], 'solver_seconds': elapsed,
               'native_converged': bool(ok), 'native_true_residual': float(native),
               'outer_trace': [float(v) for v in trace]}


def product_benchmark(contexts, paths, vector, bench, cp, rng):
    """Paired rounds of `products` calls per path, randomized path order within each round."""
    inputs = {p: (vector.astype(contexts[p].dtype) * contexts[p].mask) for p in paths}
    rounds = []
    for r in range(bench['rounds']):
        order = [paths[i] for i in rng.permutation(len(paths))]
        row = {'round': r, 'order': order, 'ms_per_product': {}}
        for p in order:
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            cp.cuda.Stream.null.synchronize()
            begin.record()
            for _ in range(bench['products']):
                contexts[p].action(inputs[p])
            end.record()
            end.synchronize()
            row['ms_per_product'][p] = float(cp.cuda.get_elapsed_time(begin, end)) / bench['products']
        rounds.append(row)
    return rounds


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
    result = {'started_utc': now(), 'status': 'running', 'protocol_sha256': sha(HERE / 'protocol.json'),
              'reference_sha256': sha(args.reference), 'setups': [], 'fixed_work': [],
              'to_tolerance': [], 'products': [], 'product_checks': [], 'telemetry': [],
              'gpu_executed': False, 'performance_claim': False, 'novelty_claim': False}
    write(out / 'result.json', result)
    suite_start = time.perf_counter()
    try:
        sys.path.insert(0, str(HERE / 'donor/src'))
        import cupy as cp
        from gpu_fem.cuda_operators import NEEDS_EDOF, OperatorSuite, PATH_SPEC
        from gpu_fem.simp_r2 import pcg, pcg_ir
        assert cp.cuda.runtime.getDeviceCount() == 1
        device = cp.cuda.runtime.getDeviceProperties(0)
        accepted = p['hardware'].get('gpu_full_names', [p['hardware'].get('gpu_full_name')])
        assert device['name'].decode() in accepted, device['name'].decode()
        result.update(gpu_executed=True, device={
            'name': device['name'].decode(), 'total_memory': int(device['totalGlobalMem']),
            'cupy': cp.__version__, 'numpy': np.__version__, 'python': sys.version,
            'platform': platform.platform(), 'driver_version': cp.cuda.runtime.driverGetVersion()})
        result['process_start_to_cuda_ready_seconds'] = time.perf_counter() - PROCESS_START
        for path in p['paths_fp64'] + p['paths_fp32']:
            if path in PATH_SPEC:
                expected = 'fp32' if path.endswith('_fp32') else 'fp64'
                assert PATH_SPEC[path][:3] == (expected,) * 3, path
        ref = Reference(args.reference)
        rng = np.random.default_rng(p['order_seed'])
        solver = p['solver']
        result['telemetry'].append(telemetry())
        for case in p['cases']:
            assert time.perf_counter() - suite_start < p['limits']['suite_seconds'], 'Suite wall-time limit'
            row = next(r for r in manifest['rows'] if r['id'] == case.get('input', case['id']))
            with np.load(args.inputs / row['file']) as saved:
                data = {k: np.ascontiguousarray(saved[k]) if saved[k].ndim else saved[k].copy()
                        for k in saved.files}
            data['floor'] = np.asarray(case.get('Emin', p['material']['Emin']))
            stages = case.get('stages', ['fixed_work', 'to_tolerance', 'products'])
            _, _, independent_diagonal = ref.product(data, np.zeros_like(data['force']))
            diag_suite = OperatorSuite(*map(int, data['shape']), data['ke'], build_edof=False)
            contexts = {}
            # Setup is timed and CHARGED per path: construction, kernel compilation, material and
            # diagonal, and one capped PCG iteration that compiles the solver operations.
            for path in p['paths_fp64'] + p['paths_fp32']:
                result['active_stage'] = {'case': case['id'], 'path': path, 'stage': 'setup'}
                write(out / 'result.json', result)
                cp.cuda.Stream.null.synchronize()
                begin = time.perf_counter()
                ctx = Context(data, path, cp, OperatorSuite, NEEDS_EDOF)
                ctx.material(diag_suite)
                if not ctx.fp32:
                    _, warm = run_pcg(ctx, pcg, solver, 1, solver['requested_rtol'])
                else:
                    ctx.action(ctx.force)
                cp.cuda.Stream.null.synchronize()
                seconds = time.perf_counter() - begin
                diagonal_error = relative(to_natural(cp.asnumpy(ctx.diagonal64), ctx.shape, True),
                                          independent_diagonal)
                assert diagonal_error <= 1e-11
                contexts[path] = ctx
                result['setups'].append({'case': case['id'], 'path': path, 'setup_seconds': seconds,
                                         'diagonal_relative_l2': diagonal_error})
            write(out / 'result.json', result)
            # ---- E1: fixed-work PCG, paired rounds
            fixed_states = {}
            for rnd in range(p['fixed_work']['rounds'] if 'fixed_work' in stages else 0):
                order = [p['paths_fp64'][i] for i in rng.permutation(len(p['paths_fp64']))]
                for position, path in enumerate(order):
                    assert time.perf_counter() - suite_start < p['limits']['suite_seconds'], 'Suite wall-time limit'
                    result['active_stage'] = {'case': case['id'], 'path': path, 'stage': 'fixed_work', 'round': rnd}
                    u, rec = run_pcg(contexts[path], pcg, solver, case['fixed_iterations'], 0.0)
                    solution = to_natural(cp.asnumpy(u), contexts[path].shape, True)
                    rec.update(case=case['id'], path=path, round=rnd, position=position)
                    if rnd == 0:
                        filename = f"{case['id']}--{path}--fixed-00.npz"
                        np.savez_compressed(out / filename, solution=solution)
                        rec.update(file=filename, sha256=sha(out / filename))
                        fixed_states[path] = solution
                        action, _, _ = ref.product(data, solution)
                        rec['independent_true_residual'] = relative(action, data['force'])
                    result['fixed_work'].append(rec)
                    write(out / 'result.json', result)
                    print(json.dumps({k: rec[k] for k in ('case', 'path', 'round', 'iterations',
                                                          'matvec_calls', 'solver_seconds')}), flush=True)
            result['telemetry'].append(telemetry())
            # ---- Time to tolerance: FP64 PCG and the donor's mixed-precision refinement
            for rep in range(p['to_tolerance']['repetitions'] if 'to_tolerance' in stages else 0):
                methods = [m for m in p['to_tolerance']['methods'] if m['id'] in case.get('methods', [x['id'] for x in p['to_tolerance']['methods']])]
                for i in rng.permutation(len(methods)):
                    m = methods[int(i)]
                    assert time.perf_counter() - suite_start < p['limits']['suite_seconds'], 'Suite wall-time limit'
                    result['active_stage'] = {'case': case['id'], 'method': m['id'], 'stage': 'to_tolerance', 'rep': rep}
                    began = time.perf_counter()
                    try:
                        if m['kind'] == 'pcg64':
                            u, rec = run_pcg(contexts[m['operator']], pcg, solver, solver['maximum_iterations'],
                                             solver['requested_rtol'])
                            shape = contexts[m['operator']].shape
                        else:
                            u, rec = run_ir(contexts[m['outer']], contexts[m['inner']], pcg_ir, solver,
                                            p['to_tolerance']['ir'])
                            shape = contexts[m['outer']].shape
                    except TimeoutError:
                        # A declared per-state time limit reached is a RESULT for a robustness test (the
                        # method did not reach tolerance in time), recorded as a physical failure.
                        cp.cuda.Stream.null.synchronize()
                        rec = {'timeout': True, 'solver_seconds': time.perf_counter() - began,
                               'native_converged': False, 'native_true_residual': float('inf'),
                               'physical_passed': False, 'case': case['id'], 'method': m['id'], 'rep': rep}
                        result['to_tolerance'].append(rec)
                        write(out / 'result.json', result)
                        print(json.dumps({k: rec.get(k) for k in ('case', 'method', 'rep', 'solver_seconds', 'timeout')}), flush=True)
                        continue
                    solution = to_natural(cp.asnumpy(u), shape, True)
                    rec.update(physical_check(data, solution, ref, rec['native_converged'],
                                              rec['native_true_residual'], solver))
                    filename = f"{case['id']}--{m['id']}--tol-{rep:02d}.npz"
                    np.savez_compressed(out / filename, solution=solution)
                    rec.update(case=case['id'], method=m['id'], rep=rep, file=filename, sha256=sha(out / filename))
                    result['to_tolerance'].append(rec)
                    write(out / 'result.json', result)
                    print(json.dumps({k: rec.get(k) for k in ('case', 'method', 'rep', 'solver_seconds',
                                                              'physical_passed')}), flush=True)
            # ---- E3 + FP64 product timing and accuracy, same input for every path
            # The product input is a seeded random vector with constrained DOFs zeroed. A converged
            # state is NOT used: K u ~ f there, so the output is dominated by cancellation and an
            # FP32 relative-error check would measure conditioning, not the kernel. Accuracy is
            # judged on free DOFs only, so no convention at constrained DOFs enters the check.
            state = (np.random.default_rng(p['order_seed'] + int(case['q'])).standard_normal(data['force'].shape)
                     * data['mask'])
            vector64 = cp.asarray(to_author(state, contexts['modal8'].shape, True))
            reference_action, _, _ = ref.product(data, state)
            free = data['mask'] != 0
            for path in (p['paths_fp64'] + p['paths_fp32']) if 'products' in stages else []:
                ctx = contexts[path]
                y = ctx.action(vector64.astype(ctx.dtype) * ctx.mask)
                natural = to_natural(cp.asnumpy(y).astype(np.float64), ctx.shape, True)
                err = relative(natural[free], reference_action[free])
                limit = p['products']['fp32_relative_limit'] if ctx.fp32 else p['products']['fp64_relative_limit']
                result['product_checks'].append({'case': case['id'], 'path': path, 'relative_l2': err,
                                                 'limit': limit, 'passed': bool(err <= limit)})
                assert err <= limit, (path, err)
            for precision, paths in ((('fp64', p['paths_fp64']), ('fp32', p['paths_fp32'])) if 'products' in stages else ()):
                rounds = product_benchmark(contexts, paths, vector64, p['products'], cp, rng)
                result['products'].append({'case': case['id'], 'precision': precision, 'rounds': rounds})
            result['telemetry'].append(telemetry())
            write(out / 'result.json', result)
            del contexts, ctx, data, independent_diagonal, diag_suite
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        result['status'] = 'completed'
        result.pop('active_stage', None)
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_utc=now(), elapsed_seconds=time.perf_counter() - suite_start)
        write(out / 'result.json', result)


if __name__ == '__main__':
    main()
