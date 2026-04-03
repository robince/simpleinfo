import unittest

import numpy as np

import simpleinfo


def entropy_from_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    nonzero = probabilities > 0
    return float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))


def mutual_information_from_counts(counts, beta=0.0):
    counts = np.asarray(counts, dtype=float)
    pxy = (counts + beta) / (counts.sum() + beta * counts.size)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    return entropy_from_probabilities(px) + entropy_from_probabilities(py) - entropy_from_probabilities(pxy)


def binary_entropy(probability):
    probability = float(probability)
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * np.log2(probability) - (1.0 - probability) * np.log2(1.0 - probability)


class SimpleInfoTests(unittest.TestCase):
    def test_entropy_matches_analytic_value(self):
        probabilities = np.array([0.5, 0.25, 0.25, 0.0])
        self.assertAlmostEqual(simpleinfo.entropy(probabilities), 1.5)

    def test_entropy_of_deterministic_distribution_is_zero(self):
        probabilities = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(simpleinfo.entropy(probabilities), 0.0)

    def test_calcinfo_known_binary_case(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 0, 1, 1])

        info = simpleinfo.calcinfo(x, 2, y, 2, bias=False)
        pmi_info, pmi = simpleinfo.calcpmi(x, 2, y, 2, beta=0.0)
        smi_info, smi = simpleinfo.calcsmi(x, 2, y, 2, beta=0.0)

        self.assertAlmostEqual(info, 1.0)
        self.assertAlmostEqual(pmi_info, 1.0)
        self.assertAlmostEqual(smi_info, 1.0)
        self.assertAlmostEqual(float(np.mean(smi)), 1.0)
        np.testing.assert_allclose(pmi, np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_calcinfo_with_smoothing_matches_reference_formula(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 0, 1, 1])
        expected = mutual_information_from_counts(np.array([[2, 0], [0, 2]]), beta=0.5)

        info = simpleinfo.calcinfo(x, 2, y, 2, bias=False, beta=0.5)
        pmi_info = simpleinfo.calcpmi(x, 2, y, 2, beta=0.5)[0]

        self.assertAlmostEqual(info, expected)
        self.assertAlmostEqual(pmi_info, expected)

    def test_calcinfo_matches_binary_symmetric_channel_analytic_value(self):
        counts = np.array([[3, 1], [1, 3]])
        x = np.repeat([0, 0, 1, 1], [3, 1, 1, 3])
        y = np.repeat([0, 1, 0, 1], [3, 1, 1, 3])
        expected = 1.0 - binary_entropy(0.25)

        info = simpleinfo.calcinfo(x, 2, y, 2, bias=False)
        pmi_info, pmi = simpleinfo.calcpmi(x, 2, y, 2)
        smi_info, smi = simpleinfo.calcsmi(x, 2, y, 2)

        self.assertAlmostEqual(info, expected)
        self.assertAlmostEqual(pmi_info, expected)
        self.assertAlmostEqual(smi_info, expected)
        self.assertAlmostEqual(float(np.mean(smi)), expected)
        np.testing.assert_allclose(pmi, np.log2(4.0 * counts / counts.sum()))

    def test_calcinfo_is_zero_for_independent_distribution(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])

        info = simpleinfo.calcinfo(x, 2, y, 2, bias=False)
        pmi_info = simpleinfo.calcpmi(x, 2, y, 2)[0]

        self.assertAlmostEqual(info, 0.0)
        self.assertAlmostEqual(pmi_info, 0.0)

    def test_sparse_declared_bins_do_not_error(self):
        x = np.array([0, 0, 0, 0])
        y = np.array([0, 0, 0, 0])
        z = np.array([0, 0, 0, 0])

        self.assertAlmostEqual(simpleinfo.calcinfo(x, 2, y, 2, bias=False), 0.0)
        self.assertAlmostEqual(simpleinfo.calccmi(x, 2, y, 2, z, 2, bias=False), 0.0)

    def test_calccmi_matches_manual_conditioning(self):
        x = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        info = simpleinfo.calccmi(x, 2, y, 2, z, 2, bias=False)
        self.assertAlmostEqual(info, 0.5)

    def test_calccmi_matches_analytic_weighted_condition_value(self):
        x = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

        info = simpleinfo.calccmi(x, 2, y, 2, z, 2, bias=False)
        expected = (2.0 / 3.0) * 0.0 + (1.0 / 3.0) * 1.0

        self.assertAlmostEqual(info, expected)

    def test_out_of_range_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"x must take values in \[0, 1\]"):
            simpleinfo.calcinfo(np.array([0, 2]), 2, np.array([0, 1]), 2, bias=False)

    def test_eqpopbin_matches_matlab_tie_behavior(self):
        x = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        xb, edges = simpleinfo.eqpopbin(x, 4, return_edges=True)

        np.testing.assert_array_equal(xb, np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
        np.testing.assert_allclose(edges, np.array([0.0, 0.0, 1.0, 2.0, 3.0]))

    def test_rebin_preserves_shape(self):
        x = np.array([[0, 1, 2], [2, 3, 3]])
        rebinned = simpleinfo.rebin(x, 2)

        self.assertEqual(rebinned.shape, x.shape)
        self.assertGreaterEqual(rebinned.min(), 0)
        self.assertLessEqual(rebinned.max(), 1)


if __name__ == "__main__":
    unittest.main()
