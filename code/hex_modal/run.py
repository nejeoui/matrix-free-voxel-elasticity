"""Execute the prospective CPU Hex8 modal-action screen in a fresh directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from rescope.floor_work.common import Reference, arrays, fixture, relative
from rescope.hex_modal.modal import Modal, T, XYZ, BINARY_TO_NATIVE, butterfly, geometry, product, quadrature


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def passed_product(a, b, limits):
    return bool(np.linalg.norm(a - b) <= limits['global_product_relative_tolerance'] * np.linalg.norm(b)
                + limits['global_product_absolute_per_sqrt_dof'] * np.sqrt(len(b)))


def controls(p, ref, out):
    rng = np.random.default_rng(p['controls']['seed'])
    lim = p['acceptance']
    rows = []
    for lengths in p['controls']['cells']:
        for nu in p['controls']['poisson_ratios']:
            ke = quadrature(lengths, nu)
            op = Modal(ke, lim['matrix_relative_tolerance'])
            vectors = rng.standard_normal((p['controls']['random_cell_vectors'], 24))
            coords = (XYZ - .5) * lengths
            rigid = np.column_stack([np.tile(axis, (8, 1)).ravel() for axis in np.eye(3)] +
                                     [np.cross(axis, coords).ravel() for axis in np.eye(3)])
            transformed = butterfly(vectors.reshape(-1, 8, 3)[:, BINARY_TO_NATIVE]).reshape(-1, 24)
            row = {'lengths': lengths, 'nu': nu, 'discarded_relative': op.discarded_relative,
                   'reconstruction_relative': relative(op.reconstructed, ke),
                   'symmetry_relative': relative(op.reconstructed.T, op.reconstructed),
                   'minimum_eigenvalue_normalized': float(np.linalg.eigvalsh(op.reconstructed).min() / np.linalg.norm(ke)),
                   'rigid_relative': float(np.linalg.norm(op.apply(rigid.T)) / (np.linalg.norm(ke) * np.linalg.norm(rigid))),
                   'butterfly_relative': relative(transformed, vectors @ T.T),
                   'action_relative': relative(op.apply(vectors), vectors @ ke.T),
                   'energy_relative': relative(np.sum(vectors * op.apply(vectors), axis=1), np.sum(vectors * (vectors @ ke.T), axis=1))}
            if lengths[0] == lengths[1] == lengths[2]:
                cpp = np.empty(576)
                ref.lib.element(lengths[0], nu, cpp)
                row['independent_quadrature_relative'] = relative(ke, cpp.reshape(24, 24))
            row['passed'] = bool(all(row[k] <= lim['matrix_relative_tolerance'] for k in
                                    ('discarded_relative', 'reconstruction_relative', 'symmetry_relative', 'rigid_relative', 'butterfly_relative'))
                                 and row.get('independent_quadrature_relative', 0) <= lim['matrix_relative_tolerance']
                                 and row['minimum_eigenvalue_normalized'] >= -lim['matrix_relative_tolerance']
                                 and max(row['action_relative'], row['energy_relative']) <= lim['local_action_relative_tolerance'])
            index = len(rows)
            np.savez(out / f'cell-control-{index:02d}.npz', ke=ke, reconstructed=op.reconstructed,
                     blocks=op.blocks, full_modal=op.full_modal, sparse_modal=op.sparse_modal,
                     vectors=vectors, candidate=op.apply(vectors), reference=vectors @ ke.T, rigid=rigid)
            rows.append(row)
    shape = tuple(p['controls']['asymmetric_grid'])
    nx, ny, nz = shape
    n = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    mask = np.ones((nz + 1, ny + 1, nx + 1, 3))
    mask[:, :, 0] = 0
    mask = np.ascontiguousarray(mask.ravel())
    rho = rng.random(nx * ny * nz)
    ke = ref.ke(4)
    op = Modal(ke)
    global_rows = []
    for floor in p['controls']['material_floors']:
        for scale in p['controls']['action_probe_scales']:
            u = rng.standard_normal(n) * scale
            target, energy, diagonal = ref.product(shape, rho, u, ke, mask, floor)
            actual, ae, ag = product(shape, rho, u, mask, floor, op)
            row = {'floor': floor, 'scale': scale, 'product_relative': relative(actual, target),
                   'energy_relative': relative(ae, energy), 'fixed_identity_exact': bool(np.array_equal(actual[mask == 0], u[mask == 0]))}
            row['passed'] = bool(passed_product(actual, target, lim) and row['fixed_identity_exact'] and row['energy_relative'] <= lim['local_action_relative_tolerance'])
            np.savez(out / f'global-control-{len(global_rows):02d}.npz', rho=rho, u=u, mask=mask,
                     target=target, candidate=actual, reference_energy=energy, candidate_energy=ae, gradient=ag)
            global_rows.append(row)
    bad = op.full_modal.copy()
    bad[0, 1] += .001 * np.linalg.norm(bad)
    bad[1, 0] = bad[0, 1]
    rejected = False
    try:
        Modal(T.T @ bad @ T)
    except ValueError:
        rejected = True
    local = rng.standard_normal((17, 24))
    wrong = local.reshape(-1, 8, 3).copy()
    wrong[:, [1, 2]] = wrong[:, [2, 1]]
    wrong_error = relative(op.apply(wrong), local @ ke.reshape(24, 24).T)
    omitted = actual.copy()
    omitted[mask == 0] = 0
    negative = {'off_block_perturbation_rejected': rejected,
                'wrong_order_relative_error': wrong_error,
                'wrong_order_rejected': bool(wrong_error > lim['local_action_relative_tolerance']),
                'missing_identity_rejected': not passed_product(omitted, target, lim)}
    result = {'cells': rows, 'global_products': global_rows, 'negative_controls': negative,
              'passed': all(r['passed'] for r in rows + global_rows) and rejected and negative['wrong_order_rejected'] and negative['missing_identity_rejected']}
    write(out / 'controls.json', result)
    return result


def load_case(case):
    path, static_path = ROOT / case['state'], ROOT / case['static_inputs']
    assert sha(path) == case['sha256'] and sha(static_path) == case['static_sha256']
    shape, mask, expected_force = fixture(case['fixture'], case['q'])
    assert list(shape) == case['shape']
    if case['format'] == 'npz':
        with np.load(path) as a:
            rho, u = a['density'].copy(), a['solution'].copy()
        with np.load(static_path) as a:
            ke, force = a['ke'].copy().ravel(), a['force'].copy()
            assert np.array_equal(a['shape'], shape)
    else:
        _, rho, u, _ = arrays(path)
        saved_mask, force, ke = arrays(static_path)
        assert np.array_equal(mask, saved_mask)
    assert np.array_equal(force, expected_force)
    return shape, mask, np.ascontiguousarray(force), np.ascontiguousarray(rho), np.ascontiguousarray(u), np.ascontiguousarray(ke)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads((HERE / 'protocol.json').read_text())
    for name, digest in p['dependencies'].items():
        assert sha(ROOT / name) == digest, name
    for name in ('protocol.json', 'modal.py', 'run.py'):
        shutil.copyfile(HERE / name, out / name)
    result = {'started_utc': now(), 'status': 'running', 'protocol_sha256': sha(HERE / 'protocol.json'),
              'source_hashes': {n: sha(HERE / n) for n in ('modal.py', 'run.py')}, 'cases': [],
              'performance_claim': False, 'novelty_claim': False, 'gpu_executed': False}
    write(out / 'result.json', result)
    try:
        ref = Reference(ROOT / p['reference_build'])
        c = controls(p, ref, out)
        result['controls_passed'] = c['passed']
        assert c['passed'], 'Controls failed; no saved-state advancement'
        limits = p['acceptance']
        for case in p['cases']:
            shape, mask, force, rho, u, ke = load_case(case)
            op = Modal(ke, limits['matrix_relative_tolerance'])
            target, energy, diagonal = ref.product(shape, rho, u, ke, mask, case['floor'])
            actual, ae, gradient = product(shape, rho, u, mask, case['floor'], op)
            baseline_gradient = -3 * (1 - case['floor']) * rho ** 2 * energy
            E = case['floor'] + (1 - case['floor']) * rho ** 3
            row = {'id': case['id'], 'discarded_relative': op.discarded_relative,
                   'product_relative': relative(actual, target), 'reference_true_residual': relative(target, force),
                   'candidate_true_residual': relative(actual, force),
                   'reference_compliance': float(E @ energy), 'candidate_compliance': float(E @ ae),
                   'energy_relative': relative(ae, energy), 'gradient_relative': relative(gradient, baseline_gradient)}
            row['compliance_relative'] = abs(row['candidate_compliance'] / row['reference_compliance'] - 1)
            row['passed'] = bool(passed_product(actual, target, limits) and
                max(row['reference_true_residual'], row['candidate_true_residual']) <= limits['saved_true_residual_limit'] and
                row['compliance_relative'] <= limits['saved_compliance_relative_tolerance'] and
                row['gradient_relative'] <= limits['saved_gradient_relative_tolerance'])
            file = out / (case['id'] + '.npz')
            np.savez(file, reference_action=target, candidate_action=actual, reference_energy=energy,
                     candidate_energy=ae, reference_gradient=baseline_gradient, candidate_gradient=gradient,
                     blocks=op.blocks, reconstructed=op.reconstructed, diagonal=diagonal)
            row.update(file=file.name, sha256=sha(file))
            result['cases'].append(row)
            write(out / 'result.json', result)
            print(json.dumps(row), flush=True)
        work = {'dense_fma': 576, 'candidate_transform_adds': 144, 'candidate_block_fma': 72,
                'dense_counted_arithmetic': 1152, 'candidate_counted_arithmetic': 288, 'ratio': .25}
        work['passed'] = work['ratio'] <= limits['maximum_algebraic_work_ratio']
        result.update(status='completed', finished_utc=now(), work_model=work,
                      correctness_gate_passed=all(r['passed'] for r in result['cases']) and c['passed'],
                      development_gate_passed=all(r['passed'] for r in result['cases']) and c['passed'] and work['passed'])
    except Exception as exc:
        result.update(status='failed', finished_utc=now(), error=repr(exc), traceback=traceback.format_exc(),
                      correctness_gate_passed=False, development_gate_passed=False)
        write(out / 'result.json', result)
        raise
    result['output_hashes'] = {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != 'result.json'}
    write(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'output_hashes')}, indent=2))


if __name__ == '__main__':
    main()
