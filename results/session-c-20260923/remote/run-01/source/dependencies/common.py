"""Portable data layouts, independent assembly and declared gate calculations."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def to_author(a, shape, nodal=False):
    x, y, z = map(int, shape)
    sizes = (z + 1, y + 1, x + 1, 3) if nodal else (z, y, x)
    axes = (2, 1, 0, 3) if nodal else (2, 1, 0)
    return np.ascontiguousarray(a.reshape(sizes).transpose(axes)).ravel()


def to_natural(a, shape, nodal=False):
    x, y, z = map(int, shape)
    sizes = (x + 1, y + 1, z + 1, 3) if nodal else (x, y, z)
    axes = (2, 1, 0, 3) if nodal else (2, 1, 0)
    return np.ascontiguousarray(a.reshape(sizes).transpose(axes)).ravel()


def product_check(actual, expected, protocol):
    c = protocol['correctness']
    error = float(np.linalg.norm(actual - expected))
    limit = c['relative_l2'] * np.linalg.norm(expected) + c['absolute_per_sqrt_dof'] * np.sqrt(expected.size)
    return {'relative_l2': relative(actual, expected), 'absolute_l2': error,
            'limit_l2': float(limit), 'passed': bool(np.isfinite(actual).all() and error <= limit)}


class Reference:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        self.lib.product.argtypes = [ctypes.c_int] * 3 + [array] * 4 + [ctypes.c_double] * 3 + [array] * 3
        self.lib.product.restype = None

    def product(self, data, u):
        u = np.ascontiguousarray(u, dtype=np.float64)
        action, energy, diagonal = np.empty_like(u), np.empty_like(data['rho']), np.empty_like(u)
        self.lib.product(*map(int, data['shape']), data['rho'], u, data['ke'].ravel(), data['mask'],
                         float(data['floor']), 1., 3., action, energy, diagonal)
        return action, energy, diagonal


def solver_check(solution, data, ref, protocol, native_ok, native_residual):
    c = protocol['correctness']
    action, energy, _ = ref.product(data, solution)
    modulus = float(data['floor']) + (1 - float(data['floor'])) * data['rho'] ** 3
    compliance, reference_compliance = float(modulus @ energy), float(modulus @ data['energy'])
    residual = relative(action, data['force'])
    state_error = relative(solution, data['u'])
    quality_error = abs(compliance / reference_compliance - 1)
    result = {'native_converged': bool(native_ok), 'native_true_residual': float(native_residual),
              'independent_true_residual': residual, 'state_relative_error': state_error,
              'compliance': compliance, 'reference_compliance': reference_compliance,
              'compliance_relative_error': quality_error}
    result['physical_passed'] = bool(native_ok and native_residual <= c['solver_requested_rtol'] and
        residual <= c['state_true_residual_limit'] and state_error <= c['solver_state_relative_limit'] and
        quality_error <= c['solver_compliance_relative_limit'])
    return result, action, energy


def timing_decision(timings, protocol):
    config = protocol['benchmarks']
    rng = np.random.default_rng(config['bootstrap_seed'])
    comparisons = []
    for shape in config['shapes']:
        identifier = 'benchmark-' + 'x'.join(map(str, shape))
        arrays = {}
        for path in protocol['paths']:
            rows = sorted((r for r in timings if r['id'] == identifier and r['path'] == path), key=lambda r: r['round'])
            assert [r['round'] for r in rows] == list(range(config['rounds']))
            arrays[path] = np.array([r['gpu_ms_per_call'] for r in rows])
            assert (arrays[path] > 0).all() and np.isfinite(arrays[path]).all()
        candidate = arrays[protocol['candidate']]
        for baseline in protocol['paths']:
            if baseline == protocol['candidate']:
                continue
            ratios = arrays[baseline] / candidate
            samples = rng.integers(0, len(ratios), size=(config['bootstrap_resamples'], len(ratios)))
            medians = np.median(ratios[samples], axis=1)
            median, lower = float(np.median(ratios)), float(np.quantile(medians, .05))
            comparisons.append({'id': identifier, 'shape': shape, 'baseline': baseline,
                                'candidate_median_ms': float(np.median(candidate)),
                                'baseline_median_ms': float(np.median(arrays[baseline])),
                                'paired_median_ratio': median, 'paired_bootstrap_lower_95': lower,
                                'primary': shape in config['primary_cases'],
                                'passed': median >= 1.15 and lower > 1.05})
    return {'comparisons': comparisons,
            'timing_gate_passed': all(r['passed'] for r in comparisons if r['primary'])}
