"""Boundary and corruption controls for the saved-evidence readiness audit."""
import copy
import importlib.util
import json
import unittest
from unittest.mock import patch
from pathlib import Path

from analyze import audit_result, required_share, speedup

ROOT = Path(__file__).resolve().parents[2]


class AmdahlControls(unittest.TestCase):
    def test_no_product_and_all_product(self):
        self.assertEqual(speedup(1.25, 0), 1)
        self.assertEqual(speedup(1.25, 1), 1.25)

    def test_hand_calculated_share_and_setup_charge(self):
        # A 20% product-time saving needs 50% of the episode for 10% total
        # time reduction: 1 / .9 = 1.111..., not a 1.10 speedup.
        self.assertAlmostEqual(speedup(1.25, .5), 1 / .9)
        self.assertAlmostEqual(required_share(1.25, 1.10), 5 / 11)
        self.assertAlmostEqual(required_share(1.25, 1.10, .02), 5 / 11 + .1)

    def test_unattainable_target(self):
        self.assertGreater(required_share(1.25, 1.30), 1)

    def test_invalid_share_rejected(self):
        for value in [-.01, 1.01, float('nan')]:
            with self.assertRaises(ValueError):
                speedup(1.25, value)


class ReceiptControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT / 'results/rescope/hex-modal-cuda-20260923/rental-01/remote/run-01'
        cls.result = json.loads((base / 'result.json').read_text())
        cls.protocol = json.loads((base / 'protocol.json').read_text())

    def test_altered_timing_comparison_rejected(self):
        data = copy.deepcopy(self.result)
        data['comparisons'][0]['paired_median_ratio'] += .1
        with self.assertRaisesRegex(ValueError, 'timing comparisons'):
            audit_result(data, self.protocol, 'tampered')

    def test_duplicate_solve_cannot_replace_missing_replicate(self):
        data = copy.deepcopy(self.result)
        indices = [i for i, row in enumerate(data['solves']) if not row['cap_one']]
        data['solves'][indices[-1]] = copy.deepcopy(data['solves'][indices[0]])
        with self.assertRaisesRegex(ValueError, 'membership'):
            audit_result(data, self.protocol, 'duplicated')


class PendingSourcePinControls(unittest.TestCase):
    def test_changed_modal_sources_are_rejected_before_execution(self):
        spec = importlib.util.spec_from_file_location(
            'pending_gpu_common', ROOT / 'rescope/hex_modal_gpu_application/common.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_sha = module.sha
        for name in ('rescope/hex_modal/modal.py', 'rescope/hex_modal_cuda/cuda.py'):
            with self.subTest(name=name):
                def changed(path):
                    return '0' * 64 if Path(path) == ROOT / name else original_sha(path)
                with patch.object(module, 'sha', side_effect=changed):
                    with self.assertRaisesRegex(ValueError, 'Changed dependency'):
                        module.protocol()


if __name__ == '__main__':
    unittest.main()
