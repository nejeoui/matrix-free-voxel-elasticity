"""Physical-rejection and diagnostic-decision controls runnable without a GPU."""
import copy
import ctypes
from pathlib import Path
import unittest
import numpy as np

from run_gpu import Reference, physical_check
from analysis import analyze

ROOT=Path(__file__).resolve().parents[2]


class PhysicalControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref=Reference(ROOT/'results/rescope/hex-modal-profile-20260923/build-local/reference.dylib')
        cls.ref.lib.element.argtypes=[ctypes.c_double,ctypes.c_double,np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
        cls.ref.lib.element.restype=None
        shape=(4,2,2);ke=np.empty((24,24));cls.ref.lib.element(.5,.3,ke)
        rng=np.random.default_rng(21);mask=np.ones((3,3,5,3));mask[:,:,0,:]=0
        cls.data={'shape':np.array(shape),'rho':rng.uniform(.2,1,16),'ke':ke,'mask':mask.ravel(),'floor':np.array(1e-6)}
        cls.solution=rng.standard_normal(mask.size)*mask.ravel()
        cls.data['force']=cls.ref.product(cls.data,cls.solution)[0]
        cls.config={'requested_rtol':1e-8,'true_residual_limit':1e-7,'energy_force_relative_limit':1e-6}

    def check(self,u,ok=True):
        return physical_check(self.data,u,self.ref,ok,0.,self.config)['physical_passed']

    def test_manufactured_equilibrium_passes(self):
        self.assertTrue(self.check(self.solution))

    def test_perturbed_equilibrium_rejected(self):
        self.assertFalse(self.check(1.01*self.solution))

    def test_constraint_violation_rejected(self):
        u=self.solution.copy();u[0]=1
        self.assertFalse(self.check(u))

    def test_native_nonconvergence_cannot_be_overridden(self):
        self.assertFalse(self.check(self.solution,False))


class DecisionControls(unittest.TestCase):
    def setUp(self):
        self.p={'cases':[{'id':'case','plain_replicates':3,'primary':True}],
                'paths':['baseline','modal8'],'candidate':'modal8',
                'profile_overhead_limit':.05,'planning_ratio_target':1.10}
        self.result={'status':'completed','solves':[]}
        for path,seconds in [('baseline',2.),('modal8',1.)]:
            for profiled,reps in [(False,3),(True,1)]:
                for rep in range(reps):
                    self.result['solves'].append({'case':'case','path':path,'profiled':profiled,'replicate':rep,
                        'physical_passed':True,'solver_seconds':seconds,'iterations':100,'matvec_calls':103,
                        'product_event_seconds':seconds*.5 if profiled else None})

    def test_complete_control_advances(self):
        self.assertTrue(analyze(self.result,self.p)['advance_to_charged_campaign'])

    def test_missing_replicate_does_not_advance(self):
        self.result['solves'].pop()
        self.assertFalse(analyze(self.result,self.p)['advance_to_charged_campaign'])

    def test_profiler_overhead_does_not_advance(self):
        self.result['solves'][-1]['solver_seconds']*=1.06
        self.assertFalse(analyze(self.result,self.p)['advance_to_charged_campaign'])

    def test_changed_work_prevents_overhead_attribution(self):
        self.result['solves'][-1]['iterations']=101
        self.assertFalse(analyze(self.result,self.p)['advance_to_charged_campaign'])

    def test_failed_physics_does_not_advance(self):
        self.result['solves'][0]['physical_passed']=False
        self.assertFalse(analyze(self.result,self.p)['advance_to_charged_campaign'])


if __name__=='__main__':unittest.main()
