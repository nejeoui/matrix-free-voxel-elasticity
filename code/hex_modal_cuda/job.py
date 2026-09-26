"""Bounded remote build/run/verification with collection even after failure."""
import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import now, read, sha, write


def run(command, log, cwd, timeout):
    begin = time.monotonic()
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        expired = False
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired = True
            __import__('os').killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                __import__('os').killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    return {'command': command, 'pid': child.pid, 'returncode': code, 'timeout': expired,
            'elapsed_seconds': time.monotonic() - begin, 'log': log.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source, inputs = root / 'source', root / 'inputs'
    result_file = root / 'job.json'
    assert not result_file.exists()
    result = {'started_utc': now(), 'state': 'running', 'pid': __import__('os').getpid(), 'stages': []}
    write(result_file, result)
    p = read(source / 'protocol.json')
    try:
        build = root / 'build'
        build.mkdir()
        command = ['g++', '-O3', '-std=c++17', '-shared', '-fPIC', str(inputs / 'reference.cc'), '-o', str(build / 'reference.so')]
        result['stages'].append(run(command, root / 'reference-build.log', root, 180))
        write(result_file, result)
        assert result['stages'][-1]['returncode'] == 0
        write(build / 'build.json', {'created_utc': now(), 'library_sha256': sha(build / 'reference.so'),
              'reference_source_sha256': sha(inputs / 'reference.cc'), 'element_source_sha256': sha(inputs / 'petsc_element.inc'),
              'compiler': subprocess.check_output(['g++', '--version'], text=True), 'command': command})
        command = [sys.executable, str(source / 'run_gpu.py'), '--inputs', str(inputs), '--reference', str(build / 'reference.so'), '--out', str(root / 'run-01')]
        result['stages'].append(run(command, root / 'gpu-run.log', root, p['limits']['maximum_job_seconds']))
        write(result_file, result)
        if (root / 'run-01/result.json').exists() and read(root / 'run-01/result.json')['status'] in ('completed', 'failed'):
            command = [sys.executable, str(source / 'verify.py'), '--inputs', str(inputs), '--reference', str(build / 'reference.so'), '--run', str(root / 'run-01'), '--out', str(root / 'remote-verification.json')]
            result['stages'].append(run(command, root / 'remote-verification.log', root, 600))
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        result.update(state='finished', finished_utc=now())
        write(result_file, result)
        command = [sys.executable, str(source / 'collect.py'), '--root', str(root), '--out', str(root.parent / 'hex-modal-evidence.tar.gz')]
        receipt = run(command, root.parent / 'hex-modal-collection.log', root, 900)
        write(root.parent / 'hex-modal-collection.json', receipt)


if __name__ == '__main__':
    main()
