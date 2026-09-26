"""Bounded remote compilation, diagnostic and replay; collect failed attempts too."""
import argparse
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'dependencies'))
from common import now,read,sha,write


def execute(command, log, timeout):
    start = time.monotonic()
    with log.open('x') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        expired = False
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired = True
            os.killpg(child.pid,signal.SIGTERM)
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                code = child.wait()
    return {'command':command,'returncode':code,'timeout':expired,
            'elapsed_seconds':time.monotonic()-start,'log':log.name}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();root=a.root.resolve();p=read(HERE/'protocol.json')
    result={'started_utc':now(),'state':'running','stages':[]}
    write(root/'job.json',result)
    deadline=time.monotonic()+p['limits']['maximum_job_seconds']
    def stage(command,log,limit):
        remaining=deadline-time.monotonic()
        assert remaining>0,'Job deadline'
        receipt=execute(command,root/log,min(limit,remaining));result['stages'].append(receipt)
        write(root/'job.json',result);return receipt
    try:
        build=root/'build';build.mkdir()
        command=['g++','-O3','-std=c++17','-shared','-fPIC',str(root/'inputs/reference.cc'),'-o',str(build/'reference.so')]
        receipt=stage(command,'reference-build.log',180)
        assert receipt['returncode']==0
        write(build/'build.json',{'utc':now(),'command':command,'binary_sha256':sha(build/'reference.so'),
              'reference_sha256':sha(root/'inputs/reference.cc'),'element_sha256':sha(root/'inputs/petsc_element.inc'),
              'compiler':subprocess.check_output(['g++','--version'],text=True)})
        command=[sys.executable,str(HERE/'run_gpu.py'),'--inputs',str(root/'inputs'),
                 '--reference',str(build/'reference.so'),'--out',str(root/'run-01')]
        stage(command,'gpu-run.log',p['limits']['suite_seconds']+p['solver']['maximum_state_seconds'])
        if (root/'run-01/result.json').exists() and read(root/'run-01/result.json')['status'] in ('completed','failed'):
            command=[sys.executable,str(HERE/'verify.py'),'--inputs',str(root/'inputs'),
                     '--reference',str(build/'reference.so'),'--run',str(root/'run-01'),
                     '--out',str(root/'remote-verification.json')]
            stage(command,'remote-verification.log',600)
    except Exception as exc:
        result['error']=repr(exc)
    finally:
        result.update(state='finished',finished_utc=now());write(root/'job.json',result)
        receipt=execute([sys.executable,str(HERE/'collect.py'),'--root',str(root),
                         '--out','/root/hex-profile-v2-evidence.tar.gz'],
                         root.parent/'hex-profile-v2-collection.log',900)
        write(root.parent/'hex-profile-v2-collection.json',receipt)


if __name__=='__main__':main()
