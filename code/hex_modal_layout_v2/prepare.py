"""Freeze sources and assemble fixed input vectors before renting a GPU."""
import ctypes
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(HERE/'dependencies'))
from common import Reference, now, read, relative, sha, write


def main():
    out=ROOT/'results/rescope/hex-modal-layout-v2-20260923'
    assert not (out/'package-receipt.json').exists()
    p=read(HERE/'protocol.json')
    p['declared_utc']=now()
    p['package_source_pins']={str(f.relative_to(HERE)):sha(f) for f in sorted(HERE.rglob('*'))
        if f.is_file() and '__pycache__' not in f.parts and f.name!='protocol.json'}
    write(HERE/'protocol.json',p);shutil.copyfile(HERE/'protocol.json',out/'protocol.json')
    inputs=out/'inputs';inputs.mkdir()
    old=ROOT/'results/rescope/hex-modal-profile-v2-20260923'
    run=old/'rental-01/remote/run-01'
    previous=read(run/'result.json');manifest=read(old/'inputs/inputs.json')
    ref=Reference(ROOT/'results/rescope/floor-work-20260923/build-02/reference.dylib')
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    ref.lib.element.argtypes=[ctypes.c_double,ctypes.c_double,array]
    rng=np.random.default_rng(2026092317)
    rows=[]
    for case in p['cases']:
        provenance={}
        if case['source']=='physical-v2':
            oldrow=next(r for r in manifest['rows'] if r['id']==case['id'])
            assert sha(old/'inputs'/oldrow['file'])==oldrow['sha256']
            with np.load(old/'inputs'/oldrow['file']) as f:
                data={k:f[k] for k in f.files}
            state=next(r for r in previous['solves'] if r['case']==case['id'] and r['path']=='fused_ai_fp64' and r['replicate']==0 and not r['profiled'])
            assert state['physical_passed'] and sha(run/state['file'])==state['sha256']
            with np.load(run/state['file']) as f:
                data['u']=f['solution']
            data.update(floor=np.array(p['material']['Emin']),h=np.array(1/case['q']),nu=np.array(.3))
            provenance={'input':str((old/'inputs'/oldrow['file']).relative_to(ROOT)),
                'input_sha256':oldrow['sha256'],'state':str((run/state['file']).relative_to(ROOT)),
                'state_sha256':state['sha256'],'source_result_sha256':sha(run/'result.json')}
            target,_,_=ref.product(data,data['u'])
            assert relative(target,data['force'])<=p['solver']['true_residual_limit']
            provenance['independent_baseline_true_residual']=relative(target,data['force'])
        else:
            shape=np.array(case['shape']);nx,ny,nz=shape;ndof=3*int(np.prod(shape+1))
            mask=np.ones((nz+1,ny+1,nx+1,3));mask[:,:,0,:]=0
            ke=np.empty((24,24));ref.lib.element(.25,case['nu'],ke)
            data={'shape':shape,'ke':ke,'rho':rng.uniform(.02,1,int(np.prod(shape))),
                'mask':mask.ravel(),'u':rng.standard_normal(ndof),'force':np.zeros(ndof),
                'floor':np.array(p['material']['Emin']),'h':np.array(.25),'nu':np.array(case['nu'])}
            assert np.count_nonzero(data['u'][data['mask']==0])>0
            provenance={'kind':'seeded random vector with nonzero constrained entries','seed':2026092317}
        assert tuple(data['shape'])==tuple(case['shape'])
        filename=case['id']+'.npz';np.savez_compressed(inputs/filename,**data)
        rows.append({**case,'file':filename,'bytes':(inputs/filename).stat().st_size,
                     'sha256':sha(inputs/filename),'provenance':provenance})
    refs={}
    for name,digest in manifest['reference_source_pins'].items():
        assert sha(old/'inputs'/name)==digest
        shutil.copyfile(old/'inputs'/name,inputs/name);refs[name]=digest
    write(inputs/'inputs.json',{'created_utc':now(),'protocol_sha256':sha(HERE/'protocol.json'),
        'rows':rows,'reference_source_pins':refs})
    package=out/'package-01.tar'
    with tarfile.open(package,'w') as tar:
        for folder,name in ((HERE,'source'),(inputs,'inputs')):
            for f in sorted(folder.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts:
                    tar.add(f,arcname=str(Path(name)/f.relative_to(folder)),recursive=False)
    write(out/'package-receipt.json',{'created_utc':now(),'sha256':sha(package),'bytes':package.stat().st_size,
        'protocol_sha256':sha(HERE/'protocol.json'),'gpu_data_seen':False})
    print({'source_files':len(p['package_source_pins']),'cases':len(rows),'bytes':package.stat().st_size,
           'protocol_sha256':sha(HERE/'protocol.json')})


if __name__=='__main__':main()
