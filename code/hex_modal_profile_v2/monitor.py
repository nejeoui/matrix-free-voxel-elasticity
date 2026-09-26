"""Collect this rental's evidence, verify it locally, and destroy its instance.

Credentials are used only by the local provider client. A separate absolute
deadline watchdog remains active if collection or verification fails.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'dependencies'))
from common import now, read, sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rental', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    a = p.parse_args()
    out = a.rental.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'matched_convergence'))
    from vast_control import api, destroy
    rental, endpoint = read(out / 'rental.json'), read(out / 'ssh.json')
    options = ['-i', endpoint['identity'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + endpoint['known_hosts']]
    target = 'root@' + endpoint['host']
    ssh = ['ssh', *options, '-p', str(endpoint['port']), target]
    scp = ['scp', *options, '-P', str(endpoint['port'])]
    status = {'started_utc': now(), 'instance_id': rental['instance_id'], 'state': 'waiting_for_collection'}
    write(out / 'monitor.json', status)
    receipt_path = '/root/hex-profile-v2-evidence.tar.receipt.json'
    while time.time() < rental['deadline_epoch']:
        command = 'test -f ' + shlex.quote(receipt_path) + ' && cat ' + shlex.quote(receipt_path)
        probe = subprocess.run([*ssh, command], capture_output=True, text=True, timeout=100)
        if probe.returncode == 0:
            remote_receipt = json.loads(probe.stdout)
            break
        time.sleep(20)
    else:
        raise RuntimeError('Deadline reached before complete collection; watchdog handles destruction')
    write(out / 'archive-receipt.json', remote_receipt)
    archive = out / 'evidence.tar.gz'
    assert not archive.exists()
    subprocess.run([*scp, target + ':/root/hex-profile-v2-evidence.tar.gz', str(archive)], check=True, timeout=1200)
    assert archive.stat().st_size == remote_receipt['bytes']
    assert sha(archive) == remote_receipt['sha256']
    destination = out / 'remote'
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        manifest = json.load(tar.extractfile('REMOTE_MANIFEST.json'))
        members = tar.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert {m.name for m in members} == set(manifest['files']) | {'REMOTE_MANIFEST.json'}
        for member in members:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            payload = tar.extractfile(member).read()
            if member.name != 'REMOTE_MANIFEST.json':
                expected = manifest['files'][member.name]
                assert len(payload) == expected['bytes']
                assert hashlib.sha256(payload).hexdigest() == expected['sha256']
            output = destination / path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
    write(out / 'collection-verification.json', {'checked_utc': now(), 'verified': True,
        'files': len(manifest['files']), 'archive_sha256': sha(archive)})
    status['destruction'] = destroy(a.key_file, out, 'Complete archive downloaded and hash-verified; GPU no longer needed for local numerical replay')
    inventory = api(a.key_file, 'GET', 'instances/')
    write(out / 'post-destruction-instances.json', inventory)
    assert all(x['id'] != rental['instance_id'] for x in inventory.get('instances', []))
    account = api(a.key_file, 'GET', 'users/current/')
    write(out / 'post-destruction-account.json', {k: account.get(k) for k in ('credit', 'balance')})
    status.update(state='collected_and_destroyed_local_replay_pending', gpu_released_utc=now())
    write(out / 'monitor.json', status)
    result = destination / 'run-01/result.json'
    if result.exists() and read(result)['status'] in ('completed', 'failed'):
        command = [sys.executable, str(destination / 'source/verify.py'), '--inputs', str(destination / 'inputs'),
                   '--run', str(destination / 'run-01'), '--reference', str(a.reference.resolve()),
                   '--out', str(out / 'local-verification.json')]
        with (out / 'local-verification.log').open('x') as log:
            verified = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        status['numerical_verification_returncode'] = verified.returncode
        # All bytes have been verified and retained even if independent replay
        # detects a scientific defect. Local diagnosis no longer needs the GPU.
    else:
        status['numerical_verification'] = 'No terminal numerical result; complete failure evidence retained'
    status.update(state='collected_and_destroyed', finished_utc=now())
    write(out / 'monitor.json', status)


if __name__ == '__main__':
    main()
