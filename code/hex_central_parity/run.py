"""Prospective CPU test of central parity; save every sampled cell and tensor."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    p = json.loads((HERE/'protocol.json').read_text())
    for name,digest in p['source_dependency'].items():
        assert sha(ROOT/name)==digest
    spec=importlib.util.spec_from_file_location('basis',ROOT/'rescope/hex_modal_basis/check_basis.py')
    basis=importlib.util.module_from_spec(spec);spec.loader.exec_module(basis)
    N,H,T=basis.NODES,basis.H,basis.T
    total=np.repeat([0,1,1,1,0,0,0,1],3)
    off=total[:,None]!=total[None,:]
    remove_translation=np.ones((24,24),dtype=bool)
    remove_translation[:3,:]=False;remove_translation[:,:3]=False
    allowed=(~off)&remove_translation
    assert int(allowed.sum())==225
    nu=.3;lam=nu/((1+nu)*(1-2*nu));mu=1/(2*(1+nu))
    isotropic=np.zeros((6,6));isotropic[:3,:3]=lam
    isotropic[range(3),range(3)]+=2*mu;isotropic[3:,3:]=mu*np.eye(3)
    gauss=list(itertools.product([-1/np.sqrt(3),1/np.sqrt(3)],repeat=3))

    def stiffness(X,D):
        K=np.zeros((24,24));dets=[]
        for x,y,z in gauss:
            dN=np.array([[n[0]*(1+n[1]*y)*(1+n[2]*z),
                n[1]*(1+n[0]*x)*(1+n[2]*z),n[2]*(1+n[0]*x)*(1+n[1]*y)] for n in N])/8
            J=dN.T@X;det=float(np.linalg.det(J));assert det>0;dets.append(det)
            grad=dN@np.linalg.inv(J).T;B=np.zeros((6,24))
            for i,(a,b,c) in enumerate(grad):
                B[:,3*i:3*i+3]=[[a,0,0],[0,b,0],[0,0,c],[b,a,0],[0,c,b],[c,0,a]]
            K+=B.T@D@B*det
        return K,min(dets)

    def rotation(rng):
        Q,_=np.linalg.qr(rng.standard_normal((3,3)))
        if np.linalg.det(Q)<0:Q[:,0]*=-1
        return Q

    rng=np.random.default_rng(p['seed']);rows=[];arrays={}
    for group,count in p['counts'].items():
        for i in range(count):
            R=rotation(rng);scales=rng.uniform(.3,.8,3)
            shear=np.eye(3);shear[np.triu_indices(3,1)]=rng.uniform(-.4,.4,3)
            X=N@np.diag(scales)@shear.T@R.T
            D=isotropic.copy()
            if group=='random_affine_anisotropic':
                A=rng.standard_normal((6,6));D=A.T@A+np.eye(6)
            elif group=='centrally_symmetric_nonaffine':
                X+=H[:,7,None]*rng.uniform(-1,1,3)*p['nonaffine_jitter_fraction']
            elif group=='unpaired_distortion_controls':
                X+=rng.uniform(-1,1,(8,3))*p['nonaffine_jitter_fraction']
            elif group=='rotated_rectangular_controls':
                X=N@np.diag(scales)@R.T
            K,mindet=stiffness(X,D);Km=T.T@K@T
            reconstructed=T@np.where(allowed,Km,0)@T.T
            u=rng.standard_normal(24);energy=float(u@K@u)
            reduced_energy=float(u@reconstructed@u)
            errors={'central_offblock_relative':float(np.linalg.norm(Km[off])/np.linalg.norm(Km)),
                    'action_matrix_relative':float(np.linalg.norm(reconstructed-K)/np.linalg.norm(K)),
                    'energy_relative':abs(reduced_energy/energy-1),
                    'eight_class_offblock_global':float(np.linalg.norm(Km[basis.OFF])/np.linalg.norm(Km))}
            if group=='rotated_rectangular_controls':
                rotated=np.kron(H/np.sqrt(8),R);Kr=rotated.T@K@rotated
                errors['eight_class_offblock_corotated']=float(np.linalg.norm(Kr[basis.OFF])/np.linalg.norm(Kr))
            key=f'{group}-{i:02d}';arrays[key+'-nodes']=X;arrays[key+'-material']=D
            arrays[key+'-vector']=u;arrays[key+'-stiffness']=K
            rows.append({'id':key,'group':group,'minimum_gauss_jacobian_determinant':mindet,**errors})
    filename=args.out/'samples.npz';np.savez_compressed(filename,**arrays)
    exact=[r for r in rows if r['group']!='unpaired_distortion_controls']
    checks={'all_exact_groups_pass':all(r['action_matrix_relative']<=p['exact_relative_tolerance'] for r in exact),
            'co_rotated_rectangles_pass':all(r['eight_class_offblock_corotated']<=p['exact_relative_tolerance'] for r in rows if r['group']=='rotated_rectangular_controls'),
            'all_gauss_jacobians_positive':all(r['minimum_gauss_jacobian_determinant']>0 for r in rows)}
    result={'utc':datetime.now(timezone.utc).isoformat(),'question':p['question'],
            'protocol_sha256':sha(HERE/'protocol.json'),'script_sha256':sha(Path(__file__)),
            'samples_sha256':sha(filename),'checks':checks,'records':rows,
            'environment':{'numpy':np.__version__,'python':platform.python_version()},
            'gpu_executed':False,'performance_claim':False,'novelty_claim':False}
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checks':checks,'groups':{group:{'count':sum(r['group']==group for r in rows),
        'max_two_block_error':max(r['action_matrix_relative'] for r in rows if r['group']==group),
        'median_eight_class_error':float(np.median([r['eight_class_offblock_global'] for r in rows if r['group']==group]))}
        for group in p['counts']}},indent=2))


if __name__=='__main__':main()
