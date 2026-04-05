import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

PYTHON_SRC = Path(__file__).resolve().parents[1] / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import simpleinfo

try:
    from simpleinfo.fastinfo import _api as fastinfo_api
    from simpleinfo.fastinfo import _fallback as fastinfo_fallback
    from simpleinfo.fastinfo import _numba as fastinfo_numba
except Exception:  # pragma: no cover - optional backend import
    fastinfo_api = None
    fastinfo_fallback = None
    fastinfo_numba = None


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

    def test_calccondcmi_weighted_contributions_sum_to_total(self):
        x = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        k = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        total, contributions = simpleinfo.calccondcmi(x, 2, y, 2, z, 2, k, 2)

        self.assertAlmostEqual(total, 0.5)
        self.assertAlmostEqual(float(np.sum(contributions)), total)

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

    def test_numbase2dec_and_numdec2base_roundtrip(self):
        words = np.array([[2, 0, 1], [1, 2, 0], [2, 1, 1]])
        decimals = simpleinfo.numbase2dec(words, 3)
        roundtrip = simpleinfo.numdec2base(decimals, 3)

        np.testing.assert_array_equal(decimals, np.array([23, 7, 10]))
        np.testing.assert_array_equal(roundtrip, words)

    def test_numdec2base_handles_exact_powers_of_base(self):
        digits = simpleinfo.numdec2base(np.array([8, 9]), 2)
        np.testing.assert_array_equal(
            digits,
            np.array([[1, 1], [0, 0], [0, 0], [0, 1]]),
        )

    def test_numdec2base_rejects_explicit_width_that_is_too_small(self):
        with self.assertRaisesRegex(ValueError, "m is too small"):
            simpleinfo.numdec2base(np.array([8]), 2, m=3)

    def test_fastinfo_calcinfo_matches_reference_unbiased(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 0, 1, 1])

        actual = simpleinfo.fastinfo.calcinfo(x, 2, y, 2)
        expected = simpleinfo.calcinfo(x, 2, y, 2, bias=False)
        self.assertAlmostEqual(actual, expected)

    def test_fastinfo_calccmi_matches_reference_unbiased(self):
        x = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        actual = simpleinfo.fastinfo.calccmi(x, 2, y, 2, z, 2)
        expected = simpleinfo.calccmi(x, 2, y, 2, z, 2, bias=False)
        self.assertAlmostEqual(actual, expected)

    def test_fastinfo_calcinfo_slice_matches_columnwise_reference(self):
        x = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ])
        y = np.array([0, 0, 1, 1])

        actual = simpleinfo.fastinfo.calcinfo_slice(x, 2, y, 2)
        expected = np.array([
            simpleinfo.calcinfo(x[0], 2, y, 2, bias=False),
            simpleinfo.calcinfo(x[1], 2, y, 2, bias=False),
            simpleinfo.calcinfo(x[2], 2, y, 2, bias=False),
        ])
        np.testing.assert_allclose(actual, expected)

    def test_fastinfo_calcinfomatched_matches_reference(self):
        x = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]])
        y = np.array([[0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0]])

        actual = simpleinfo.fastinfo.calcinfomatched(x, 2, y, 2)
        expected = simpleinfo.calcinfomatched(x.T, 2, y.T, 2, bias=False)
        np.testing.assert_allclose(actual, expected)

    def test_fastinfo_calccondcmi_matches_reference(self):
        x = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        k = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        actual_total, actual_contrib = simpleinfo.fastinfo.calccondcmi(x, 2, y, 2, z, 2, k, 2)
        expected_total, expected_contrib = simpleinfo.calccondcmi(x, 2, y, 2, z, 2, k, 2)

        self.assertAlmostEqual(actual_total, expected_total)
        np.testing.assert_allclose(actual_contrib, expected_contrib)

    def test_fastinfo_calccmi_slice_matches_columnwise_reference(self):
        x = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]])
        y = np.array([0, 0, 1, 1])
        z = np.array([0, 1, 0, 1])

        actual = simpleinfo.fastinfo.calccmi_slice(x, 2, y, 2, z, 2)
        expected = simpleinfo.calccmi_slice(x.T, 2, y, 2, z, 2, bias=False)
        np.testing.assert_allclose(actual, expected)

    def test_fastinfo_eqpop_sorted_preserves_ties(self):
        x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xb = simpleinfo.fastinfo.eqpop_sorted(x, 3)
        np.testing.assert_array_equal(xb, np.array([0, 0, 1, 1, 2, 2]))

    def test_fastinfo_eqpop_errors_when_ties_make_requested_bins_impossible(self):
        x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "Cannot form the requested number of equal-population bins"):
                simpleinfo.fastinfo.eqpop(x, 4)

    def test_fastinfo_eqpop_warns_on_heavily_quantized_input(self):
        x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simpleinfo.fastinfo.eqpop_sorted(x, 3)
        self.assertTrue(any("heavily repeated values" in str(w.message) for w in caught))

    def test_fastinfo_eqpop_slice_returns_nan_for_failed_pages(self):
        x = np.array([
            [10.0, 0.0, 20.0, 30.0, 11.0, 1.0, 21.0, 31.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = simpleinfo.fastinfo.eqpop_slice(x, 4)
        np.testing.assert_array_equal(out[0], np.array([1, 0, 2, 3, 1, 0, 2, 3], dtype=float))
        np.testing.assert_array_equal(out[1], np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float))
        self.assertTrue(np.all(np.isnan(out[2])))

    def test_fastinfo_eqpop_sorted_slice_returns_nan_for_failed_pages(self):
        x = np.array([
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = simpleinfo.fastinfo.eqpop_sorted_slice(x, 4)
        np.testing.assert_array_equal(out[0], np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float))
        np.testing.assert_array_equal(out[1], np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float))
        self.assertTrue(np.all(np.isnan(out[2])))

    def test_fastinfo_calcinfoperm_seed_is_reproducible(self):
        x = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        a = simpleinfo.fastinfo.calcinfoperm(x, 2, y, 2, 8, seed=123)
        b = simpleinfo.fastinfo.calcinfoperm(x, 2, y, 2, 8, seed=123)
        np.testing.assert_allclose(a, b)

    def test_fastinfo_calcinfoperm_slice_seed_is_reproducible(self):
        x = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]])
        y = np.array([0, 1, 0, 1])
        a = simpleinfo.fastinfo.calcinfoperm_slice(x, 2, y, 2, 8, seed=123)
        b = simpleinfo.fastinfo.calcinfoperm_slice(x, 2, y, 2, 8, seed=123)
        np.testing.assert_allclose(a, b)

    def test_fastinfo_slice_apis_require_c_contiguous_trial_last_matrices(self):
        x = np.asfortranarray(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]))
        y = np.array([0, 0, 1, 1])

        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            simpleinfo.fastinfo.calcinfo_slice(x, 2, y, 2)

    def test_fastinfo_calcpairwiseinfo_matches_reference(self):
        x = np.array([0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        actual = simpleinfo.fastinfo.calcpairwiseinfo(x, 3, y, 3)
        expected = simpleinfo.calcpairwiseinfo(x, 3, y, 3)
        np.testing.assert_allclose(actual, expected)

    def test_fastinfo_calcinfoperm_mean_is_near_zero_for_independent_inputs(self):
        x = np.repeat([0, 1], 128)
        y = np.tile([0, 1], 128)

        observed = simpleinfo.fastinfo.calcinfo(x, 2, y, 2)
        null = simpleinfo.fastinfo.calcinfoperm(x, 2, y, 2, 128, seed=123)

        self.assertAlmostEqual(observed, 0.0)
        self.assertEqual(null.shape, (128,))
        self.assertTrue(np.all(np.isfinite(null)))
        self.assertTrue(np.all(null >= -1e-12))
        self.assertLess(float(np.mean(null)), 0.02)

    def test_fastinfo_discrete_validators_preserve_integer_dtype_without_copy(self):
        x = np.array([0, 1, 1, 0], dtype=np.int16)
        X = np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.int32)

        vector = fastinfo_fallback._as_fastinfo_discrete_vector(x, 2, "x")
        matrix = fastinfo_fallback._as_fastinfo_discrete_matrix(X, 2, "X")

        self.assertEqual(vector.dtype, np.int16)
        self.assertEqual(matrix.dtype, np.int32)
        self.assertTrue(np.shares_memory(vector, x))
        self.assertTrue(np.shares_memory(matrix, X))

    def test_fastinfo_public_api_accepts_low_width_integer_inputs(self):
        x = np.array([0, 0, 1, 1], dtype=np.int16)
        y = np.array([0, 0, 1, 1], dtype=np.int16)
        X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32)

        self.assertAlmostEqual(simpleinfo.fastinfo.calcinfo(x, 2, y, 2), 1.0)
        np.testing.assert_allclose(
            simpleinfo.fastinfo.calcinfo_slice(X, 2, y.astype(np.int32), 2),
            np.array([1.0, 0.0]),
        )

    @unittest.skipUnless(fastinfo_api is not None and fastinfo_api.BACKEND == "numba", "Numba backend not active")
    def test_numba_backend_matches_numpy_fallback(self):
        x = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X = np.vstack((x, y, z, x))
        cont = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1])

        self.assertAlmostEqual(
            fastinfo_numba.calcinfo(x, 2, y, 2),
            fastinfo_fallback.calcinfo(x, 2, y, 2),
        )
        self.assertAlmostEqual(
            fastinfo_numba.calccmi(x, 2, y, 2, z, 2),
            fastinfo_fallback.calccmi(x, 2, y, 2, z, 2),
        )
        np.testing.assert_allclose(
            fastinfo_numba.calcinfomatched(X, 2, X, 2),
            fastinfo_fallback.calcinfomatched(X, 2, X, 2),
        )
        np.testing.assert_allclose(
            fastinfo_numba.calcinfo_slice(X, 2, y, 2),
            fastinfo_fallback.calcinfo_slice(X, 2, y, 2),
        )
        np.testing.assert_allclose(
            fastinfo_numba.calccmi_slice(X, 2, y, 2, z, 2),
            fastinfo_fallback.calccmi_slice(X, 2, y, 2, z, 2),
        )
        np.testing.assert_allclose(
            fastinfo_numba.calcinfoperm(x, 2, y, 2, 12, seed=123),
            fastinfo_fallback.calcinfoperm(x, 2, y, 2, 12, seed=123),
        )
        np.testing.assert_allclose(
            fastinfo_numba.calcinfoperm_slice(X, 2, y, 2, 12, seed=123),
            fastinfo_fallback.calcinfoperm_slice(X, 2, y, 2, 12, seed=123),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_array_equal(
                fastinfo_numba.eqpop(cont, 4),
                fastinfo_fallback.eqpop(cont, 4),
            )
            np.testing.assert_array_equal(
                fastinfo_numba.eqpop_sorted(cont, 4),
                fastinfo_fallback.eqpop_sorted(cont, 4),
            )
            slice_cont = np.vstack((
                cont,
                np.arange(cont.size, dtype=float),
                np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=float),
            ))
            np.testing.assert_allclose(
                fastinfo_numba.eqpop_slice(slice_cont, 4),
                fastinfo_fallback.eqpop_slice(slice_cont, 4),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                fastinfo_numba.eqpop_sorted_slice(slice_cont, 4),
                fastinfo_fallback.eqpop_sorted_slice(slice_cont, 4),
                equal_nan=True,
            )

    @unittest.skipUnless(fastinfo_api is not None and fastinfo_api.BACKEND == "numba", "Numba backend not active")
    def test_numba_backend_rejects_per_call_threads(self):
        with self.assertRaisesRegex(ValueError, "Per-call threads control is unsupported"):
            simpleinfo.fastinfo.calcinfo(np.array([0, 1]), 2, np.array([0, 1]), 2, threads=2)


if __name__ == "__main__":
    unittest.main()
