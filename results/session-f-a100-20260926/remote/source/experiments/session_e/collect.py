"""Complete experiment archive excluding downloaded installers and Python wheels."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def collect(root,out):
    root=root.resolve();out=out.resolve();assert not out.exists()
    binaries=sorted(root.glob('application-build*/topopt-published'))
    if binaries:
        binary=binaries[-1]
        text=subprocess.check_output(['ldd',str(binary)],text=True)
        libraries=root/'runtime-libraries';libraries.mkdir(exist_ok=False)
        entries={}
        for line in text.splitlines():
            pieces=line.strip().split()
            path=next((Path(v) for v in pieces if v.startswith('/') and Path(v).is_file()),None)
            if path is None:continue
            path=path.resolve();destination=libraries/path.name
            if not destination.exists():shutil.copyfile(path,destination)
            entries[str(path)]={'sha256':sha(path),'archived':str(destination.relative_to(root))}
        write(root/'runtime-libraries.json',{'ldd':text,'libraries':entries})
    paths=[]
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        relative=path.relative_to(root)
        if '.git' in relative.parts or '__pycache__' in relative.parts:continue
        if relative.parts[0]=='installation':
            if len(relative.parts)!=2 or path.suffix not in ('.json','.log','.txt'):continue
        if path.name.startswith('core.') or path.name=='core':continue
        if path==out:continue
        paths.append(path)
    paths.sort()
    entries={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'files':entries}
    with tarfile.open(out,'w:gz') as archive:
        for path in paths:archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
        data=(json.dumps(manifest,indent=2)+'\n').encode();entry=tarfile.TarInfo('REMOTE_MANIFEST.json');entry.size=len(data)
        archive.addfile(entry,io.BytesIO(data))
    with tarfile.open(out) as archive:
        for name,row in entries.items():
            data=archive.extractfile(name).read();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    receipt={'sha256':sha(out),'bytes':out.stat().st_size,'files':len(entries)}
    write(out.with_suffix('.receipt.json'),receipt);print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();collect(a.root,a.out)
