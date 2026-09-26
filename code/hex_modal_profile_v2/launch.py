"""Rent once, deploy the unchanged sealed package, and start collection cleanup.

The separate watchdog is started immediately after the provider creates the
instance. This script never retries rental creation and never changes GPU code.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rescope/hex_modal_cuda'))
from common import now, read, sha, write
sys.path.insert(0, str(ROOT / 'rescope/matched_convergence'))
from vast_control import api, destroy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--offer-id', type=int, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    a = parser.parse_args()
    base = a.experiment.resolve()
    out = base / 'rental-01'
    out.mkdir(mode=0o700)
    p = read(base / 'protocol.json')
    previous = ROOT / 'results/rescope/hex-modal-cuda-20260923'
    package = base / 'package-01.tar'
    assert sha(package) == read(base / 'package-receipt.json')['sha256']
    offer = next(o for o in read(base / 'offers-01.json')['offers'] if o['id'] == a.offer_id)
    assert offer['gpu_name'] == p['hardware']['gpu'] and offer['num_gpus'] == 1
    assert offer['dph_total'] <= p['hardware']['maximum_total_usd_per_hour']
    inventory = api(a.key_file, 'GET', 'instances/')
    write(out / 'pre-create-instances.json', inventory)
    assert not inventory.get('instances'), 'Inventory changed; inspect before creating a rental'
    request = read(previous / 'rental-01/create-request.json')
    request.update(label='jpdc-hex-profile-v2-20260923', disk=p['hardware']['disk_gb'])
    write(out / 'create-request.json', request)
    started = time.time()
    try:
        response = api(a.key_file, 'PUT', f"asks/{offer['id']}/", request)
    except urllib.error.HTTPError as exc:
        body = exc.read(16384).decode('utf-8', errors='replace')
        body = body.replace(a.key_file.read_text().strip(), '[REDACTED]')
        write(out / 'create-failure.json', {'utc': now(), 'status': exc.code,
            'offer_id': offer['id'], 'body': body, 'created_contract_receipt_present': False})
        try:
            write(out / 'post-error-instances.json', api(a.key_file, 'GET', 'instances/'))
        except Exception as check_error:
            write(out / 'post-error-check-failure.json', {'utc': now(), 'type': type(check_error).__name__})
        raise
    write(out / 'create-response.json', response)
    assert response.get('success') and response.get('new_contract'), response
    identifier = response['new_contract']
    write(out / 'rental.json', {'instance_id': identifier, 'offer': offer,
        'created_utc': now(), 'created_epoch': started,
        'deadline_epoch': started + p['limits']['maximum_rental_seconds'],
        'purpose': p['question'], 'diagnostic_protocol_sha256': sha(base / 'protocol.json'),
        'package_sha256': sha(package)})
    with (out / 'watchdog.log').open('x') as log:
        guard = subprocess.Popen([sys.executable, str(ROOT / 'rescope/matched_convergence/vast_control.py'),
            'watchdog', '--key-file', str(a.key_file), '--out', str(out)], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(.2)
    assert guard.poll() is None
    awake = subprocess.Popen(['caffeinate', '-dimsu', '-w', str(guard.pid)], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    write(out / 'watchdog-launch.json', {'pid': guard.pid, 'caffeinate_pid': awake.pid, 'utc': now(), 'live_after_launch': True})
    print(json.dumps({'instance_id': identifier, 'watchdog_pid': guard.pid, 'state': 'provisioning'}), flush=True)
    gpu_started = False
    try:
        for attempt in range(40):
            inventory = api(a.key_file, 'GET', 'instances/')
            write(out / f'status-{attempt:02d}.json', inventory)
            instance = next(x for x in inventory['instances'] if x['id'] == identifier)
            ports = instance.get('ports') or {}
            if ports.get('2200/tcp'):
                break
            if attempt % 4 == 0:
                print(json.dumps({'utc': now(), 'state': instance.get('actual_status'), 'detail': instance.get('status_msg')}), flush=True)
            time.sleep(15)
        else:
            raise RuntimeError('Provisioning exceeded bounded wait')
        endpoint = {'host': instance['public_ipaddr'], 'port': int(ports['2200/tcp'][0]['HostPort']),
            'identity': '/Users/nejeoui/.ssh/id_ed25519', 'known_hosts': str(out / 'known_hosts')}
        write(out / 'ssh.json', endpoint)
        options = ['-i', endpoint['identity'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'UserKnownHostsFile=' + endpoint['known_hosts']]
        target = 'root@' + endpoint['host']
        ssh = ['ssh', *options, '-o', 'StrictHostKeyChecking=accept-new', '-p', str(endpoint['port']), target]
        for attempt in range(12):
            check = subprocess.run([*ssh, 'true'], capture_output=True, text=True, timeout=30)
            if check.returncode == 0:
                break
            time.sleep(10)
        else:
            raise RuntimeError('SSH failed to become ready')
        subprocess.run(['scp', *options, '-o', 'StrictHostKeyChecking=yes', '-P', str(endpoint['port']),
            str(package), target + ':/root/hex-profile-v2-package.tar'], check=True, timeout=300)
        code = """import pathlib, hashlib, tarfile, subprocess, json, datetime
archive = pathlib.Path('/root/hex-profile-v2-package.tar')
assert hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest() == DIGEST
root = pathlib.Path('/root/jpdc-hex-profile-v2-20260923')
root.mkdir()
with tarfile.open(archive) as tar:
    tar.extractall(root, filter='data')
log = open('/root/hex-profile-v2-bootstrap.log', 'x')
child = subprocess.Popen(['bash', str(root / 'source/bootstrap.sh')], stdin=subprocess.DEVNULL,
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {'pid': child.pid, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'package_sha256': DIGEST}
(root / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\\n')
print(json.dumps(receipt))
""".replace('DIGEST', repr(read(base / 'package-receipt.json')['sha256']))
        launched = subprocess.run([*ssh, 'python3 -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=90)
        assert launched.returncode == 0, launched.stderr
        gpu_started = True
        write(out / 'remote-launch.json', json.loads(launched.stdout))
        with (out / 'monitor.log').open('x') as log:
            monitor = subprocess.Popen([sys.executable, str(ROOT / 'rescope/hex_modal_profile_v2/monitor.py'),
                '--rental', str(out), '--key-file', str(a.key_file), '--reference',
                str(ROOT / 'results/rescope/floor-work-20260923/build-02/reference.dylib')], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(.2)
        assert monitor.poll() is None
        write(out / 'monitor-launch.json', {'pid': monitor.pid, 'utc': now(), 'live_after_launch': True})
        print(json.dumps({'instance_id': identifier, 'remote_launch': json.loads(launched.stdout), 'monitor_pid': monitor.pid}), flush=True)
    except Exception as exc:
        write(out / 'launch-failure.json', {'utc': now(), 'error': repr(exc), 'gpu_started': gpu_started})
        if not gpu_started:
            destroy(a.key_file, out, 'Bounded setup failed before GPU job launch; provider status evidence retained')
        raise


if __name__ == '__main__':
    main()
