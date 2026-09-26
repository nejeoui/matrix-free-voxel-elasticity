"""Physical checks and unmodified solve wrapper copied from the sealed v2 diagnostic."""
import time
import numpy as np
from common import relative, to_natural

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

