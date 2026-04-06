import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import comparative_fastinfo


class ComparativeFastInfoTests(unittest.TestCase):
    def test_equivalence_inputs_have_expected_shapes(self):
        inputs = comparative_fastinfo.build_equivalence_inputs("quick")
        cfg = comparative_fastinfo.mode_config("quick")

        self.assertEqual(inputs["mi_none"]["x_slice"].shape, (cfg["equiv_ntrl"], cfg["equiv_nx"]))
        self.assertEqual(inputs["mi_effect"]["x_matched"].shape, (cfg["equiv_ntrl"], cfg["equiv_nmatched"]))
        self.assertEqual(inputs["cmi_none"]["z"].shape, (cfg["equiv_ntrl"],))
        self.assertTrue(np.all(inputs["mi_none"]["x_scalar"] >= 0))
        self.assertTrue(np.all(inputs["mi_none"]["x_scalar"] < comparative_fastinfo.XB))

    def test_python_fast_layout_is_trial_last_and_c_contiguous(self):
        x = np.arange(12, dtype=np.int64).reshape(3, 4)
        fast = comparative_fastinfo._py_fast_layout(x)

        self.assertEqual(fast.shape, (4, 3))
        self.assertTrue(fast.flags.c_contiguous)
        np.testing.assert_array_equal(fast, x.T)

    def test_python_equivalence_cases_match(self):
        cases = comparative_fastinfo.run_python_equivalence("quick")

        self.assertGreater(len(cases), 0)
        for case in cases:
            self.assertLess(case["max_abs_diff"], 1e-10, msg=f"{case['scenario']} {case['operation']}")


if __name__ == "__main__":
    unittest.main()
