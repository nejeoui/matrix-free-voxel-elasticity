"""Rent one vast.ai instance for a frozen Voxel session, run it, collect, verify, destroy.

Money-first rules:
- Acts ONLY on the contract id this script creates; other instances on the account are recorded
  and never touched.
- An independent deadline watchdog (vast_control.py) destroys the instance at the protocol's
  maximum rental time even if this process dies.
- The instance is destroyed immediately after the evidence archive's SHA-256 matches its receipt,
  and on ANY failure after creation (after best-effort collection).
- Refuses to start the job on a GPU that is already busy (co-tenancy).
The API key is read from a file and never written to any output.

    python3 rent_run_destroy.py --session ../session_c --offer-id N --out ../../results/session-c-YYYYMMDD
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vast_control import api, destroy  # noqa: E402

KEY = Path.home() / '.config/voxel/vast_api_key'
IDENTITY = Path.home() / '.ssh/id_ed25519'
CREATE_TEMPLATE = Path('/Users/nejeoui/JPDC/results/rescope/hex-modal-cuda-20260923/rental-01/create-request.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(msg):
    print(f'{now()} {msg}', flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--session', type=Path, required=True)
    a.add_argument('--gpu', action='append', required=True, help='provider short GPU name(s), e.g. "RTX 4090"')
    a.add_argument('--max-attempts', type=int, default=6)
    a.add_argument('--min-gpu-ram-mb', type=int, default=0, help='operational filter, only ever STRICTER than the protocol')
    a.add_argument('--min-inet-down', type=float, default=0.0,
                   help='operational filter (Mbit/s), only ever STRICTER; avoids slow package installs')
    a.add_argument('--exclude-machine-id', type=int, action='append', default=[],
                   help='replication: never rent on these provider machine ids')
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    session, out = args.session.resolve(), args.out.resolve()
    name = session.name                                  # e.g. session_c
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads((session / 'protocol.json').read_text())
    receipt = json.loads((session / 'package-receipt.json').read_text())
    package = session / 'package.tar'
    assert sha(package) == receipt['package_sha256'], 'package hash mismatch'
    assert sha(session / 'protocol.json') == receipt['protocol_sha256'], 'protocol hash mismatch'
    hw = p['hardware']

    # ---- offer re-check (fresh search, strict local filter)
    query = json.loads(Path('/Users/nejeoui/JPDC/results/rescope/hex-modal-profile-v2-20260923/offers-query-01.json').read_text())
    query.pop('gpu_name', None)
    query['dph_total'] = {'lte': hw['maximum_total_usd_per_hour']}
    query['cpu_ram'] = {'gte': hw['minimum_cpu_ram_mb']}
    query['gpu_ram'] = {'gte': hw['minimum_gpu_ram_mb']}
    offers = api(KEY, 'POST', 'bundles/', query).get('offers', [])
    write(out / 'offers.json', {'utc': now(), 'query': query, 'offers': offers})
    matches = sorted((o for o in offers if o['gpu_name'] in args.gpu and o['num_gpus'] == 1
                      and o['dph_total'] <= hw['maximum_total_usd_per_hour']
                      and o['cpu_ram'] >= hw['minimum_cpu_ram_mb']
                      and o['gpu_ram'] >= max(hw['minimum_gpu_ram_mb'], args.min_gpu_ram_mb)
                      and o.get('inet_down', 0) >= args.min_inet_down
                      and o['machine_id'] not in args.exclude_machine_id),
                     key=lambda o: o['dph_total'])
    assert matches, 'no offer within the declared limits'
    before = api(KEY, 'GET', 'instances/')
    write(out / 'pre-create-instances.json', {'ids': [i['id'] for i in before.get('instances', [])]})
    request = json.loads(CREATE_TEMPLATE.read_text())
    request.update(label=f'voxel-{name}', disk=hw['disk_gb'])
    write(out / 'create-request.json', request)
    import urllib.error
    attempts, response, offer = [], None, None
    for offer in matches[:args.max_attempts]:
        started = time.time()
        try:
            response = api(KEY, 'PUT', f"asks/{offer['id']}/", request)
            attempts.append({'offer_id': offer['id'], 'dph_total': offer['dph_total'], 'result': 'created'})
            break
        except urllib.error.HTTPError as exc:
            body = exc.read(4000).decode('utf-8', 'replace').replace(KEY.read_text().strip(), '[REDACTED]')
            attempts.append({'offer_id': offer['id'], 'dph_total': offer['dph_total'], 'status': exc.code, 'body': body})
            if 'no_such_ask' not in body:
                write(out / 'create-attempts.json', attempts); raise
    write(out / 'create-attempts.json', attempts)
    write(out / 'create-response.json', response or {})
    assert response and response.get('success') and response.get('new_contract'), 'no offer could be created'
    ident = response['new_contract']
    write(out / 'rental.json', {'instance_id': ident, 'offer': offer, 'created_utc': now(),
                                'created_epoch': started,
                                'deadline_epoch': started + p['limits']['maximum_rental_seconds'],
                                'protocol_sha256': receipt['protocol_sha256'],
                                'package_sha256': receipt['package_sha256']})
    log(f'created instance {ident} at USD {offer["dph_total"]:.4f}/h')
    with (out / 'watchdog.log').open('x') as wl:
        guard = subprocess.Popen([sys.executable, str(HERE / 'vast_control.py'), 'watchdog', '--key-file', str(KEY),
                                  '--out', str(out)], stdin=subprocess.DEVNULL, stdout=wl, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    subprocess.Popen(['caffeinate', '-dimsu', '-w', str(guard.pid)], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(0.5)
    assert guard.poll() is None, 'watchdog failed to start'

    remote_root = f'/root/voxel-{name}'
    evidence = f'/root/voxel-{name}-evidence.tar.gz'
    evidence_receipt = f'/root/voxel-{name}-evidence.tar.receipt.json'
    collection = f'/root/voxel-{name}-collection.json'
    ssh = scp_opts = None
    verified = False
    try:
        for attempt in range(60):
            inst = next(x for x in api(KEY, 'GET', 'instances/')['instances'] if x['id'] == ident)
            ports = inst.get('ports') or {}
            if ports.get('2200/tcp') and inst.get('public_ipaddr'):
                break
            if attempt % 4 == 0:
                log(f"provisioning: {inst.get('actual_status')} {inst.get('status_msg') or ''}"[:200])
            time.sleep(15)
        else:
            raise RuntimeError('provisioning exceeded 15 min')
        host, port = inst['public_ipaddr'], int(ports['2200/tcp'][0]['HostPort'])
        opts = ['-i', str(IDENTITY), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                '-o', f'UserKnownHostsFile={out / "known_hosts"}', '-o', 'StrictHostKeyChecking=accept-new']
        ssh = ['ssh', *opts, '-p', str(port), f'root@{host}']
        scp_opts = ['scp', '-q', *opts, '-P', str(port)]
        write(out / 'ssh.json', {'host': host, 'port': port})
        for attempt in range(20):
            if subprocess.run([*ssh, 'true'], capture_output=True, timeout=40).returncode == 0:
                break
            time.sleep(10)
        else:
            raise RuntimeError('ssh never became ready')
        pre = subprocess.run([*ssh, 'nvidia-smi --query-gpu=name,utilization.gpu,power.draw,memory.used,driver_version '
                                    '--format=csv,noheader; nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l'],
                             capture_output=True, text=True, timeout=60)
        write(out / 'preflight.json', {'utc': now(), 'stdout': pre.stdout, 'stderr': pre.stderr[-2000:]})
        first, apps = pre.stdout.strip().splitlines()[0], int(pre.stdout.strip().splitlines()[-1])
        gpu_name, util = [x.strip() for x in first.split(',')][:2]
        accepted = hw.get('gpu_full_names', [hw.get('gpu_full_name')])
        assert gpu_name in accepted, f'unexpected GPU {gpu_name}'
        assert int(util.split()[0]) <= 5 and apps == 0, f'GPU busy (co-tenancy?): {first} apps={apps}'
        log(f'preflight ok: {first}')
        subprocess.run([*scp_opts, str(package), f'root@{host}:/root/package.tar'], check=True, timeout=600)
        code = (f"set -e; test $(sha256sum /root/package.tar | cut -d' ' -f1) = {receipt['package_sha256']}; "
                f"mkdir {remote_root}; tar -xf /root/package.tar -C {remote_root}; "
                f"nohup bash {remote_root}/source/bootstrap.sh > /root/voxel-{name}-bootstrap.log 2>&1 < /dev/null & "
                f"echo launched")
        launched = subprocess.run([*ssh, code], capture_output=True, text=True, timeout=120)
        assert 'launched' in launched.stdout, launched.stderr
        log('job launched')
        write(out / 'remote-launch.json', {'utc': now()})
        # ---- poll
        deadline = json.loads((out / 'rental.json').read_text())['deadline_epoch'] - 600
        while time.time() < deadline:
            r = subprocess.run([*ssh, f'test -f {collection} && echo DONE || tail -1 {remote_root}/gpu-run.log 2>/dev/null'],
                               capture_output=True, text=True, timeout=60)
            line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
            log(line[:180])
            if line == 'DONE':
                break
            time.sleep(60)
        else:
            raise RuntimeError('job did not finish before the rental deadline margin')
        subprocess.run([*scp_opts, f'root@{host}:{evidence}', f'root@{host}:{evidence_receipt}',
                        f'root@{host}:{collection}', str(out)], check=True, timeout=1800)
        want = json.loads((out / Path(evidence_receipt).name).read_text())['sha256']
        got = sha(out / Path(evidence).name)
        write(out / 'archive-check.json', {'utc': now(), 'receipt_sha256': want, 'local_sha256': got,
                                           'match': want == got})
        assert want == got, 'archive hash mismatch'
        verified = True
        log('archive verified')
    except Exception as exc:
        write(out / 'failure.json', {'utc': now(), 'error': repr(exc)})
        log(f'FAILURE: {exc!r}')
        if ssh is not None and not verified:
            # Best-effort: keep whatever the job wrote before destroying.
            subprocess.run([*ssh, f'cd {remote_root} 2>/dev/null && tar -czf /root/partial.tar.gz job.json gpu-run.log '
                                  f'run-01/result.json installation 2>/dev/null; true'], capture_output=True, timeout=120)
            subprocess.run([*scp_opts, f'root@{host}:/root/partial.tar.gz', str(out)], capture_output=True, timeout=300)
    finally:
        try:
            receipt_d = destroy(KEY, out, 'Tasks finished; evidence verified' if verified else 'Failure path')
            log(f"destroyed: {receipt_d['success']}")
        except Exception as exc:
            log(f'DESTROY FAILED, watchdog remains armed: {exc!r}')
        time.sleep(20)
        remaining = [i['id'] for i in api(KEY, 'GET', 'instances/').get('instances', [])]
        write(out / 'post-destruction-instances.json', {'utc': now(), 'ids': remaining,
                                                        'ours_still_listed': ident in remaining})
        log(f'post-destruction: ours still listed = {ident in remaining}')


if __name__ == '__main__':
    main()
