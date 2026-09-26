"""Replay parity algebra and rigid motions from retained matrices and geometry."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    result=json.loads((args.run/'results.json').read_text())
    p=json.loads((args.run.parent/'protocol.json').read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(args.run/'samples.npz')==result['samples_sha256']
    assert sha(args.run.parent/'protocol.json')==result['protocol_sha256']
    spec=importlib.util.spec_from_file_location('basis',ROOT/'rescope/hex_modal_basis/check_basis.py')
    basis=importlib.util.module_from_spec(spec);spec.loader.exec_module(basis)
    perm=np.array([3*n+c for n in [6,7,4,5,2,3,0,1] for c in range(3)])
    data=np.load(args.run/'samples.npz');exact=[];negative=[];max_rigid=0.;max_gauss=0.
    assert len(result['records'])==sum(p['counts'].values())
    for row in result['records']:
        key=row['id'];K=data[key+'-stiffness'];X=data[key+'-nodes'];D=data[key+'-material']
        assert np.isfinite(K).all() and np.linalg.eigvalsh(D)[0]>0
        norm=np.linalg.norm(K)
        assert np.linalg.norm(K-K.T)/norm<1e-12
        assert np.linalg.eigvalsh(K)[0]>=-1e-12*norm
        for c in range(3):
            translate=np.tile(np.eye(3)[c],8)
            rotate=np.cross(np.eye(3)[c],X).ravel()
            for u in [translate,rotate]:
                error=float(np.linalg.norm(K@u)/(norm*np.linalg.norm(u)))
                max_rigid=max(max_rigid,error);assert error<1e-12
        if row['group']!='random_affine_anisotropic':
            error=float(np.linalg.norm(K-basis.stiffness(X))/norm)
            max_gauss=max(max_gauss,error);assert error<1e-12
        commutator=float(np.linalg.norm(K-K[np.ix_(perm,perm)])/norm)
        if row['group']=='unpaired_distortion_controls':negative.append(commutator)
        else:exact.append(commutator);assert commutator<1e-12
    assert all(x>1e-4 for x in negative)
    receipt={'utc':datetime.now(timezone.utc).isoformat(),'evidence_valid':True,
             'samples_checked':len(result['records']),'central_symmetry_samples':len(exact),
             'distortion_controls_rejected':len(negative),
             'maximum_central_inversion_commutator':max(exact),
             'maximum_rigid_motion_relative_residual':max_rigid,
             'maximum_isotropic_independent_assembly_discrepancy':max_gauss,
             'gpu_executed':False,'speedup_claim':False,'novelty_claim':False}
    args.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
