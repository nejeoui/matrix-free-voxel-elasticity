"""Freeze Voxel session E: pin sources, write protocol.json and inputs/inputs.json, build the package.

Inputs are the exact session D inputs (themselves the JPDC profile-v2 physical states), verified
against session D's recorded SHA-256 before use. Refuses to overwrite a frozen protocol unless
--redeclare is given (which must be recorded as an amendment in scientific_report.md).

    python3 experiments/session_e/build_package.py --jpdc-inputs experiments/session_d/inputs
"""
import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_FILES = ['analysis.py', 'bootstrap.sh', 'collect.py', 'job.py', 'run_gpu.py', 'verify.py',
                 'dependencies/common.py', 'dependencies/cuda.py', 'dependencies/cuda32.py',
                 'dependencies/modal.py', 'dependencies/voxel_mg.py',
                 'dependencies/ATTRIBUTION.md', 'dependencies/DAO-LICENSE']
INPUT_FILES = ['smoke-q4.npz', 'optimized-q64.npz', 'optimized-q96.npz', 'uniform-q64.npz',
               'reference.cc', 'petsc_element.inc']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--jpdc-inputs', type=Path, required=True)
    a.add_argument('--redeclare', action='store_true')
    args = a.parse_args()
    if (HERE / 'protocol.json').exists() and not args.redeclare:
        raise SystemExit('protocol.json already frozen; use --redeclare and record an amendment')
    donor = sorted(str(f.relative_to(HERE)) for f in (HERE / 'donor').rglob('*')
                   if f.is_file() and '__pycache__' not in f.parts)
    protocol = json.loads((HERE / 'protocol_template.json').read_text())
    protocol['declared_utc'] = datetime.now(timezone.utc).isoformat()
    protocol['package_source_pins'] = {f: sha(HERE / f) for f in PACKAGE_FILES + donor}
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=1) + '\n')
    psha = sha(HERE / 'protocol.json')

    recorded = {r['file']: r['sha256'] for r in json.loads((args.jpdc_inputs / 'inputs.json').read_text())['rows']}
    inputs = HERE / 'inputs'
    if inputs.exists():
        shutil.rmtree(inputs)
    inputs.mkdir()
    rows = []
    for name in INPUT_FILES:
        shutil.copy2(args.jpdc_inputs / name, inputs / name)
        if name in recorded:
            assert sha(inputs / name) == recorded[name], f'{name} differs from JPDC profile-v2 record'
    for input_id in sorted({c.get('input', c['id']) for c in protocol['cases']}):
        case = next(c for c in protocol['cases'] if c.get('input', c['id']) == input_id)
        f = f"{input_id}.npz"
        rows.append({'id': input_id, 'q': case['q'], 'primary': case['primary'], 'file': f,
                     'bytes': (inputs / f).stat().st_size, 'sha256': sha(inputs / f)})
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': psha, 'rows': rows,
                'support_files': {n: sha(inputs / n) for n in ('reference.cc', 'petsc_element.inc')},
                'origin': str(args.jpdc_inputs)}
    (inputs / 'inputs.json').write_text(json.dumps(manifest, indent=1) + '\n')

    package = HERE / 'package.tar'
    with tarfile.open(package, 'w') as tar:
        for f in ['protocol.json'] + PACKAGE_FILES + donor:
            tar.add(HERE / f, arcname=f'source/{f}')
        for f in sorted(inputs.iterdir()):
            tar.add(f, arcname=f'inputs/{f.name}')
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': psha,
               'package_sha256': sha(package), 'bytes': package.stat().st_size}
    (HERE / 'package-receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
