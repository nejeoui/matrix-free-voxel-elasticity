"""Voxel session I (combined paper W2b): the FP64 finest kernels inside Voxel's rediscretised multigrid at
scale (upsampled optimized designs up to tens of millions of DOFs) on RTX 4090, A100 80 GB and H100.
Derived from session E's runner; changes: generated inputs, no saved solution files (sha256 only),
and a size ladder that stops at the first out-of-memory case.

Stages per case: (1) FP64 product accuracy against the independent C reference and variant-vs-modal8
differences on one seeded vector; (2) paired product timing rounds (CUDA events); (3) multigrid
time-to-tolerance; (4) multigrid fixed work (identical outer iterations by construction); (5) the
finest-product share of one profiled solve.
`--cpu-dry-run` executes the whole schedule with NumPy products (no CUDA) to check orchestration.
"""
import argparse
import gc
import hashlib
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
import voxel_mg as mg


def telemetry():
    command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,'
               'temperature.gpu,clocks.sm,clocks.mem,power.draw', '--format=csv,noheader']
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {'utc': now(), 'returncode': r.returncode, 'output': r.stdout.strip()}
    except Exception as exc:
        return {'utc': now(), 'error': repr(exc)}


class DonorProduct:
    """Donor OperatorSuite path as a voxel_mg product; the diagonal is always built in FP64."""

    def __init__(self, suite, path, dtype, cp):
        self.suite, self.path, self.dtype, self.cp = suite, path, dtype, cp

    def raw(self, u, E):
        return self.suite.matvec_full(u, E, path=self.path)

    def diagonal(self, E):
        return self.suite.diagonal(E.astype(self.cp.float64), path='node_fp64', scatter=False).astype(self.dtype)


class CustomProduct:
    """modal8/dense8 (FP64) or their FP32 variants as a voxel_mg product."""

    def __init__(self, module, path, diag_suite, dtype, cp):
        self.module, self.path, self.diag_suite, self.dtype, self.cp = module, path, diag_suite, dtype, cp

    def raw(self, u, E):
        return self.module.matvec_full(u, E, self.path)

    def diagonal(self, E):
        return self.diag_suite.diagonal(E.astype(self.cp.float64), path='node_fp64', scatter=False).astype(self.dtype)


class Profiled:
    """Wraps a finest-level product with CUDA events (no per-call synchronisation)."""

    def __init__(self, inner, cp):
        self.inner, self.cp, self.events = inner, cp, []

    def raw(self, u, E):
        b, e = self.cp.cuda.Event(), self.cp.cuda.Event()
        b.record()
        y = self.inner.raw(u, E)
        e.record()
        self.events.append((b, e))
        return y

    def diagonal(self, E):
        return self.inner.diagonal(E)

    def seconds(self):
        return sum(float(self.cp.cuda.get_elapsed_time(b, e)) for b, e in self.events) / 1000.0


def physical_check(data, solution, ref, native_ok, native, solver):
    action, energy, _ = ref.product(data, solution)
    residual = relative(action, data['force'])
    modulus = float(data['floor']) + (1 - float(data['floor'])) * data['rho'] ** 3
    compliance, work = float(modulus @ energy), float(data['force'] @ solution)
    energy_error = abs(compliance / work - 1) if work != 0 else float('inf')
    valid = bool(native_ok and native <= solver['requested_rtol'] and residual <= solver['true_residual_limit']
                 and energy_error <= solver['energy_force_relative_limit'] and np.isfinite(solution).all()
                 and np.all(solution[data['mask'] == 0] == 0))
    return {'native_converged': bool(native_ok), 'native_true_residual': float(native),
            'independent_true_residual': residual, 'compliance': compliance, 'force_work': work,
            'energy_force_relative_error': energy_error, 'physical_passed': valid}


class NumpyRef:
    """Dry-run stand-in for the C reference: same interface, NumPy element-by-element product."""

    def __init__(self, ke):
        self.ke = ke

    def product(self, data, u):
        shape = tuple(int(s) for s in data['shape'])
        E = 1e-6 + (1 - 1e-6) * data['rho'] ** 3
        prod = mg.NumpyProduct(shape, self.ke)
        ua = to_author(u, shape, True)
        Ea = to_author(E, shape)
        y = prod.raw(ua * to_author(data['mask'], shape, True), Ea)
        m = to_author(data['mask'], shape, True)
        y = y * m + ua * (1 - m)
        edof = prod.edof
        uu = ua * m
        energy = np.einsum('ei,ij,ej->e', uu[edof], self.ke, uu[edof])
        return to_natural(y, shape, True), to_natural(energy, shape), None


def build_case(xp, cp, data, p, arms, shapes, dry):
    """Shared hierarchy pieces plus one finest product per kernel name."""
    ke = data['ke']
    E = xp.asarray(to_author(float(data['floor']) + (1 - float(data['floor'])) * data['rho'] ** 3, shapes[0]))
    mask = xp.asarray(to_author(data['mask'], shapes[0], True))
    force = xp.asarray(to_author(data['force'], shapes[0], True))
    kernels64 = sorted({a['outer'] for a in arms} | {a['vcycle'] for a in arms if a['precision'] == 'fp64'})
    kernels32 = sorted({a['vcycle'] for a in arms if a['precision'] == 'fp32'})
    if dry:
        coarse64 = [mg.NumpyProduct(s, ke * 2 ** i, np.float64) for i, s in enumerate(shapes)]
        coarse32 = [mg.NumpyProduct(s, ke * 2 ** i, np.float32) for i, s in enumerate(shapes)]
        finest = {k: mg.NumpyProduct(shapes[0], ke, np.float64) for k in kernels64}
        finest.update({k: mg.NumpyProduct(shapes[0], ke, np.float32) for k in kernels32})
    else:
        from gpu_fem.cuda_operators import NEEDS_EDOF, OperatorSuite
        from cuda import CudaModal
        from cuda32 import CudaModal32
        from modal import Modal
        suites = [OperatorSuite(*s, ke * 2 ** i, build_edof=False) for i, s in enumerate(shapes)]
        coarse64 = [DonorProduct(suites[i], 'node_fp64', cp.float64, cp) for i in range(len(shapes))]
        coarse32 = [DonorProduct(suites[i], 'node_fp32', cp.float32, cp) for i in range(len(shapes))]
        blocks = Modal(ke).blocks
        custom64 = CudaModal(shapes[0], ke, blocks)
        custom32 = CudaModal32(shapes[0], ke, blocks)
        finest = {}
        for k in kernels64 + kernels32:
            dt = cp.float32 if k.endswith('_fp32') else cp.float64
            if k in ('modal8', 'modal8_muladd', 'modal8_signbit', 'dense8'):
                finest[k] = CustomProduct(custom64, k, suites[0], dt, cp)
            elif k in ('modal8_fp32', 'dense8_fp32'):
                finest[k] = CustomProduct(custom32, k, suites[0], dt, cp)
            else:
                s = OperatorSuite(*shapes[0], ke, build_edof=(k in NEEDS_EDOF))
                finest[k] = DonorProduct(s, k, dt, cp)
    return E, mask, force, coarse64, coarse32, finest


def product_stage(result, case, data, shape, shapes, finest, E, ref, p, rng, cp, dry):
    """FP64 product accuracy (free DOFs, vs the C reference) and paired product timing rounds.

    The input is a seeded standard-normal vector with constrained DOFs zeroed (not a converged state,
    whose product is dominated by cancellation). The raw operator product is timed: no masking, no
    Dirichlet rows, the same call every multigrid outer iteration makes before masking."""
    pp = p['products']
    xp = np if dry else cp
    state = np.random.default_rng(p['order_seed'] + int(case['q'])).standard_normal(data['force'].shape) * data['mask']
    u = xp.asarray(to_author(state, shapes[0], True))
    reference_action, _, _ = ref.product(data, state)
    free = data['mask'] != 0
    outputs = {}
    for path in pp['paths']:
        y = finest[path].raw(u, E)
        natural = to_natural(np.asarray(y.get() if hasattr(y, 'get') else y).astype(np.float64), shape, True)
        outputs[path] = natural
        err = relative(natural[free], reference_action[free])
        result['product_checks'].append({'case': case['id'], 'path': path, 'relative_l2': err,
                                         'limit': pp['fp64_relative_limit'], 'passed': bool(err <= pp['fp64_relative_limit'])})
        assert err <= pp['fp64_relative_limit'], (path, err)
    for path in pp['variants']:
        d = outputs[path] - outputs['modal8']
        result['variant_differences'].append({
            'case': case['id'], 'path': path, 'relative_l2_vs_modal8': relative(outputs[path], outputs['modal8']),
            'max_abs_vs_modal8': float(np.max(np.abs(d))), 'bitwise_equal_outputs': bool(np.array_equal(
                outputs[path].view(np.uint64), outputs['modal8'].view(np.uint64))),
            'note': 'atomic accumulation order is not deterministic, so outputs may differ in the last bits'})
    paths = list(pp['paths'])
    rounds = []
    for r in range(pp['rounds']):
        order = [paths[i] for i in rng.permutation(len(paths))]
        row = {'round': r, 'order': order, 'ms_per_product': {}}
        for path in order:
            if dry:
                t = time.perf_counter()
                for _ in range(pp['products']):
                    finest[path].raw(u, E)
                row['ms_per_product'][path] = 1e3 * (time.perf_counter() - t) / pp['products']
                continue
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            cp.cuda.Stream.null.synchronize()
            begin.record()
            for _ in range(pp['products']):
                finest[path].raw(u, E)
            end.record()
            end.synchronize()
            row['ms_per_product'][path] = float(cp.cuda.get_elapsed_time(begin, end)) / pp['products']
        rounds.append(row)
    result['products'].append({'case': case['id'], 'rounds': rounds})
    print(json.dumps({'case': case['id'], 'products_ms_median': {
        q: float(np.median([row['ms_per_product'][q] for row in rounds])) for q in paths}}), flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--inputs', type=Path, required=True)
    a.add_argument('--reference', type=Path)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--cpu-dry-run', action='store_true')
    a.add_argument('--dry-max-coarse-x', type=int, default=None)
    a.add_argument('--dry-cases', nargs='*', default=None)
    args = a.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    p = read(HERE / 'protocol.json') if (HERE / 'protocol.json').exists() else read(HERE / 'protocol_template.json')
    if not args.cpu_dry_run:
        for filename, digest in p['package_source_pins'].items():
            assert sha(HERE / filename) == digest, filename
        manifest = read(args.inputs / 'generated.json')
        for row in manifest['rows']:
            assert sha(args.inputs / row['file']) == row['sha256']
        shutil.copytree(HERE, out / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    dry = args.cpu_dry_run
    result = {'started_utc': now(), 'status': 'running', 'dry_run': dry,
              'protocol_sha256': sha(HERE / 'protocol.json') if (HERE / 'protocol.json').exists() else None,
              'cases': [], 'to_tolerance': [], 'fixed_work': [], 'profiles': [], 'telemetry': [], 'ladder': [],
              'products': [], 'product_checks': [], 'variant_differences': [],
              'gpu_executed': False, 'performance_claim': False, 'novelty_claim': False}
    write(out / 'result.json', result)
    suite_start = time.perf_counter()
    try:
        if dry:
            cp, xp = None, np
            sync = lambda: None
        else:
            sys.path.insert(0, str(HERE / 'donor/src'))
            import cupy as cp
            xp = cp
            sync = cp.cuda.Stream.null.synchronize
            assert cp.cuda.runtime.getDeviceCount() == 1
            dev = cp.cuda.runtime.getDeviceProperties(0)
            accepted = p['hardware'].get('gpu_full_names', [p['hardware'].get('gpu_full_name')])
            assert dev['name'].decode() in accepted, dev['name'].decode()
            result.update(gpu_executed=True, device={
                'name': dev['name'].decode(), 'total_memory': int(dev['totalGlobalMem']), 'cupy': cp.__version__,
                'numpy': np.__version__, 'python': sys.version, 'platform': platform.platform(),
                'driver_version': cp.cuda.runtime.driverGetVersion()})
            result['telemetry'].append(telemetry())
        solver, mgp = p['solver'], p['multigrid']
        rng = np.random.default_rng(p['order_seed'])
        arms = p['arms']
        cases = [c for c in p['cases'] if not args.dry_cases or c['id'] in args.dry_cases]
        for case in cases:
          try:
            if time.perf_counter() - suite_start > p['limits']['suite_seconds']:
                result['ladder'].append({'case': case['id'], 'status': 'skipped_suite_deadline'})
                break
            src = case.get('input', case['id'])
            with np.load(args.inputs / f'{src}.npz') as f:
                data = {k: np.ascontiguousarray(f[k]) if f[k].ndim else f[k].copy() for k in f.files}
            data['floor'] = np.asarray(case.get('Emin', p['material']['Emin']))
            ref = NumpyRef(data['ke']) if dry else Reference(args.reference)
            shape = tuple(int(s) for s in data['shape'])
            shapes = mg.hierarchy_shapes(shape, args.dry_max_coarse_x if dry and args.dry_max_coarse_x
                                         else mgp['max_coarse_x'])
            sync(); t0 = time.perf_counter()
            E, mask, force, coarse64, coarse32, finest = build_case(xp, cp, data, p, arms, shapes, dry)
            # Shared coarsest factor and omegas (reference: FP64 hierarchy with node-owned finest).
            Ec, mc = to_author(float(data['floor']) + (1 - float(data['floor'])) * data['rho'] ** 3, shapes[0]), \
                     to_author(data['mask'], shapes[0], True)
            for i in range(1, len(shapes)):
                Ec, mc = mg.coarsen_modulus(np, Ec, shapes[i - 1]), mg.coarsen_mask(np, mc, shapes[i - 1])
            factor_host = np.linalg.cholesky(mg.assemble_dense(shapes[-1], data['ke'] * 2 ** (len(shapes) - 1), Ec, mc))
            factor = xp.asarray(factor_host)
            ref_kernel = mgp['omega_reference_kernel']
            ref_mg = mg.Multigrid(xp, shapes, [finest.get(ref_kernel, coarse64[0])] + coarse64[1:], E, mask,
                                  factor, xp.float64 if not dry else np.float64, pre=mgp['pre'], post=mgp['post'])
            omegas, lambdas = mg.spectral_omegas(xp, ref_mg.levels, mgp['power_iterations'], p['order_seed'])
            sync()
            case_row = {'case': case['id'], 'shapes': shapes, 'omegas': omegas, 'lambdas': lambdas,
                        'shared_setup_seconds': time.perf_counter() - t0, 'arm_setup_seconds': {}}
            solvers = {}
            for arm in arms:
                sync(); t1 = time.perf_counter()
                if arm['precision'] == 'fp64':
                    prods, dt = [finest[arm['vcycle']]] + coarse64[1:], (np.float64 if dry else cp.float64)
                else:
                    prods, dt = [finest[arm['vcycle']]] + coarse32[1:], (np.float32 if dry else cp.float32)
                M = mg.Multigrid(xp, shapes, prods, E, mask, factor, dt, pre=mgp['pre'], post=mgp['post'],
                                 omegas=omegas)
                outer = finest[arm['outer']]
                A = (lambda o: (lambda u: o.raw(u * mask, E) * mask + u * (1 - mask)))(outer)
                solvers[arm['id']] = (A, M)
                sync()
                case_row['arm_setup_seconds'][arm['id']] = time.perf_counter() - t1
            result['cases'].append(case_row)
            write(out / 'result.json', result)
            product_stage(result, case, data, shape, shapes, finest, E, ref, p, rng, cp, dry)
            write(out / 'result.json', result)

            def solve(arm_id, fixed=None):
                A, M = solvers[arm_id]
                sync(); t = time.perf_counter()
                x, it, ok, rel, mv = mg.fcg(xp, A, force, M, solver['requested_rtol'], solver['maximum_iterations'],
                                            check_every=solver['check_every'], fixed_iterations=fixed)
                sync()
                return x, {'iterations': it, 'native_converged': ok, 'native_true_residual': rel,
                           'outer_matvecs': mv, 'solver_seconds': time.perf_counter() - t}

            # warm-up: one short solve per arm (compiles, allocates); never timed
            for arm in arms:
                solve(arm['id'], fixed=2)
            # ---- time to tolerance, randomized order
            converged_iterations = {}
            for rep in range(p['to_tolerance']['repetitions']):
                for i in rng.permutation(len(arms)):
                    arm = arms[int(i)]
                    assert time.perf_counter() - suite_start < p['limits']['suite_seconds'], 'Suite wall-time limit'
                    x, rec = solve(arm['id'])
                    sol = to_natural(np.asarray(x.get() if hasattr(x, 'get') else x), shape, True)
                    rec.update(physical_check(data, sol, ref, rec['native_converged'], rec['native_true_residual'], solver))
                    rec.update(case=case['id'], arm=arm['id'], rep=rep)
                    rec.update(solution_sha256=hashlib.sha256(np.ascontiguousarray(sol).tobytes()).hexdigest(),
                               solution_norm=float(np.linalg.norm(sol)))
                    converged_iterations.setdefault(arm['id'], []).append(rec['iterations'])
                    result['to_tolerance'].append(rec)
                    write(out / 'result.json', result)
                    print(json.dumps({k: rec[k] for k in ('case', 'arm', 'rep', 'iterations', 'solver_seconds',
                                                          'physical_passed')}), flush=True)
            # ---- fixed work: every FP64 arm runs K outer iterations, K = max over reps of the reference arm
            K = int(max(converged_iterations[p['fixed_work']['iterations_from_arm']]))
            fixed_arms = [a for a in arms if a['id'] in p['fixed_work']['arms']]
            for rnd in range(p['fixed_work']['rounds']):
                for pos, i in enumerate(rng.permutation(len(fixed_arms))):
                    arm = fixed_arms[int(i)]
                    _, rec = solve(arm['id'], fixed=K)
                    rec.update(case=case['id'], arm=arm['id'], round=rnd, position=pos, fixed_iterations=K)
                    result['fixed_work'].append(rec)
                write(out / 'result.json', result)
            # ---- profile: finest-product share of one fixed-work solve per profiled arm
            if not dry:
                for arm_id in p['profile']['arms']:
                    arm = next(a for a in arms if a['id'] == arm_id)
                    A, M = solvers[arm_id]
                    prof_v = Profiled(M.levels[0].product, cp)
                    prof_o = Profiled(finest[arm['outer']], cp)
                    M.levels[0].product = prof_v
                    Ap = lambda u: prof_o.raw(u * mask, E) * mask + u * (1 - mask)
                    sync(); t = time.perf_counter()
                    mg.fcg(xp, Ap, force, M, solver['requested_rtol'], solver['maximum_iterations'],
                           check_every=solver['check_every'], fixed_iterations=K)
                    sync(); total = time.perf_counter() - t
                    M.levels[0].product = prof_v.inner
                    result['profiles'].append({'case': case['id'], 'arm': arm_id, 'fixed_iterations': K,
                                               'solver_seconds': total,
                                               'vcycle_finest_product_seconds': prof_v.seconds(),
                                               'outer_product_seconds': prof_o.seconds(),
                                               'vcycle_finest_calls': len(prof_v.events),
                                               'outer_calls': len(prof_o.events)})
                result['telemetry'].append(telemetry())
            write(out / 'result.json', result)
            del solvers, finest, coarse64, coarse32, E, mask, force, factor
            result['ladder'].append({'case': case['id'], 'status': 'ok'})
          except Exception as exc:
            oom = 'OutOfMemory' in type(exc).__name__ or 'out of memory' in str(exc).lower()
            result['ladder'].append({'case': case['id'], 'status': 'out_of_memory' if oom else 'error',
                                     'error': repr(exc)[:2000], 'traceback': traceback.format_exc()[-3000:]})
            write(out / 'result.json', result)
            if not oom:
                raise
            break                                         # stop the ladder at the first out-of-memory case
          finally:
            gc.collect()
            if not dry:
                cp.get_default_memory_pool().free_all_blocks()
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_utc=now(), elapsed_seconds=time.perf_counter() - suite_start)
        write(out / 'result.json', result)


if __name__ == '__main__':
    main()
