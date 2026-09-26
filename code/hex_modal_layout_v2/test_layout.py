"""CPU semantic and decision controls. These do not claim CUDA compatibility."""
import ctypes
from pathlib import Path
import sys
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'dependencies'))
from modal import Modal, quadrature
from common import read
from variants import pack_symmetric, symmetric_index, emulate_modal32
from analysis import analyze


class LayoutTests(unittest.TestCase):
    def test_triangle_bijection_and_products(self):
        indices = [symmetric_index(i,j) for i in range(24) for j in range(i+1)]
        self.assertEqual(sorted(indices), list(range(300)))
        rng = np.random.default_rng(230916)
        for _ in range(8):
            a = rng.standard_normal((24,24)); k = a.T@a
            packed = pack_symmetric(k)
            recovered = np.array([[packed[symmetric_index(i,j)] for j in range(24)] for i in range(24)])
            np.testing.assert_array_equal(recovered, k)
            v = rng.standard_normal(24)
            np.testing.assert_array_equal(recovered@v, k@v)

    def test_asymmetric_rejected(self):
        a = np.eye(24); a[2,4] = .1
        with self.assertRaises(ValueError):
            pack_symmetric(a)

    def test_modal_lane_schedule_against_independent_element(self):
        root = HERE.parents[1]
        lib = ctypes.CDLL(str(root/'results/rescope/floor-work-20260923/build-02/reference.dylib'))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        lib.element.argtypes = [ctypes.c_double, ctypes.c_double, array]
        rng = np.random.default_rng(230917)
        for nu in (.2,.3,.45):
            for h in (.25,1/64):
                ke = np.empty((24,24)); lib.element(h,nu,ke)
                np.testing.assert_allclose(ke, quadrature([h]*3,nu), rtol=2e-13,atol=2e-16)
                blocks = Modal(ke).blocks
                for v in np.r_[np.eye(24),rng.standard_normal((8,24))]:
                    np.testing.assert_allclose(emulate_modal32(v,blocks),ke@v,rtol=2e-12,atol=2e-15)

    def test_shuffle_sources_all_active(self):
        for lane in range(24):
            component,mode = divmod(lane,8)
            for d in range(3):
                source = 8*d+(mode^(1<<component)^(1<<d))
                self.assertTrue(0 <= source < 24)
            for bit in (1,2,4):
                self.assertEqual((lane^bit)//8,component)
        # Partial grid blocks always retain complete 8-lane element groups or whole warps.
        for ne in (1,3,24,45,128):
            for threads in (8,32):
                blocks = (threads*ne+127)//128
                elements = [t//threads for t in range(blocks*128) if t//threads < ne]
                self.assertEqual(len(elements),threads*ne)

    def test_decision_complete_failure_and_no_winner_selection(self):
        p = read(HERE/'protocol.json')
        result = {'status':'completed','products':[],'timings':[],'solves':[],'warmups':[]}
        for c in p['cases']:
            for path in p['paths']:
                for phase in ('before','after'):
                    result['products'].append({'case':c['id'],'path':path,'phase':phase,
                        'action_check':{'passed':True},'zero_exact':True,'fixed_identity_exact':True})
                if c['timed']:
                    value = 1 if path=='modal32' else 1.2 if path=='modal8' else 1.6
                    for rep in range(p['benchmarks']['rounds']):
                        result['timings'].append({'case':c['id'],'path':path,'round':rep,'gpu_ms_per_call':value})
            if c['solve']:
                for path in p['new_paths']:
                    result['solves'].append({'case':c['id'],'path':path,'physical_passed':True,'cross_passed':True})
                    result['warmups'].append({'case':c['id'],'path':path,'physical_passed':False,'iterations':1})
        self.assertTrue(all(analyze(result,p)['development_gates'].values()))
        for row in result['timings']:
            if row['path']=='modal32': row['gpu_ms_per_call']=1.3
        gates=analyze(result,p)['development_gates']
        self.assertTrue(gates['modal8']); self.assertFalse(gates['modal32'])
        result['products'][0]['zero_exact']=False
        self.assertFalse(any(analyze(result,p)['development_gates'].values()))
        result['products'][0]['zero_exact']=True
        result['products'].pop()
        self.assertFalse(analyze(result,p)['complete'])


if __name__ == '__main__':
    unittest.main()
