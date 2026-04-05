"""Optional Numba-accelerated kernels for simpleinfo.fastinfo."""

from __future__ import annotations

import numpy as np

from ..core import _validate_bin_count, mmbiascmi, mmbiasinfo
from . import _fallback

from numba import get_num_threads, njit, prange, set_num_threads


def _reject_per_call_threads(threads):
    if threads is not None:
        raise ValueError(
            "Per-call threads control is unsupported for the Numba backend because "
            "Numba thread count is global state. Use simpleinfo.fastinfo.set_threads(...) instead."
        )


@njit(cache=True)
def _entropy_from_counts_numba(counts, n_trials):
    if n_trials == 0:
        return 0.0
    acc = 0.0
    for i in range(counts.size):
        count = counts[i]
        if count > 0:
            acc += count * np.log2(count)
    return np.log2(n_trials) - acc / n_trials


@njit(cache=True)
def _calcinfo_numba(x, xb, y, yb):
    px = np.zeros(xb, dtype=np.int64)
    py = np.zeros(yb, dtype=np.int64)
    pxy = np.zeros((xb, yb), dtype=np.int64)
    return _calcinfo_with_scratch_numba(x, y, px, py, pxy)


@njit(cache=True)
def _calcinfo_with_scratch_numba(x, y, px, py, pxy):
    px.fill(0)
    py.fill(0)
    pxy.fill(0)
    for i in range(x.size):
        xi = x[i]
        yi = y[i]
        px[xi] += 1
        py[yi] += 1
        pxy[xi, yi] += 1
    return (
        _entropy_from_counts_numba(px, x.size)
        + _entropy_from_counts_numba(py, x.size)
        - _entropy_from_counts_numba(pxy.ravel(), x.size)
    )


@njit(cache=True)
def _calccmi_numba(x, xb, y, yb, z, zb):
    pz = np.zeros(zb, dtype=np.int64)
    pxz = np.zeros((xb, zb), dtype=np.int64)
    pyz = np.zeros((yb, zb), dtype=np.int64)
    pxyz = np.zeros((xb, yb, zb), dtype=np.int64)
    return _calccmi_with_scratch_numba(x, y, z, pz, pxz, pyz, pxyz)


@njit(cache=True)
def _calccmi_with_scratch_numba(x, y, z, pz, pxz, pyz, pxyz):
    pz.fill(0)
    pxz.fill(0)
    pyz.fill(0)
    pxyz.fill(0)
    for i in range(x.size):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        pz[zi] += 1
        pxz[xi, zi] += 1
        pyz[yi, zi] += 1
        pxyz[xi, yi, zi] += 1
    return (
        _entropy_from_counts_numba(pxz.ravel(), x.size)
        + _entropy_from_counts_numba(pyz.ravel(), x.size)
        - _entropy_from_counts_numba(pxyz.ravel(), x.size)
        - _entropy_from_counts_numba(pz, x.size)
    )


@njit(cache=True, parallel=True)
def _calcinfo_slice_numba(x, y, px, py, pxy, n_threads):
    out = np.zeros(x.shape[0], dtype=np.float64)
    for tid in prange(n_threads):
        for col in range(tid, x.shape[0], n_threads):
            out[col] = _calcinfo_with_scratch_numba(x[col], y, px[tid], py[tid], pxy[tid])
    return out


@njit(cache=True, parallel=True)
def _calcinfomatched_numba(x, y, px, py, pxy, n_threads):
    out = np.zeros(x.shape[0], dtype=np.float64)
    for tid in prange(n_threads):
        for col in range(tid, x.shape[0], n_threads):
            out[col] = _calcinfo_with_scratch_numba(x[col], y[col], px[tid], py[tid], pxy[tid])
    return out


@njit(cache=True)
def _splitmix64(x):
    z = np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _fisher_yates_shuffle_numba(values, seed):
    state = np.uint64(seed)
    for i in range(values.size - 1, 0, -1):
        state = _splitmix64(state + np.uint64(i))
        j = int(state % np.uint64(i + 1))
        tmp = values[i]
        values[i] = values[j]
        values[j] = tmp


@njit(cache=True)
def _slice_seed_numba(seed, col, perm):
    seed_col = _splitmix64(_splitmix64(np.uint64(seed)) + np.uint64(col))
    return _splitmix64(seed_col + np.uint64(perm))


@njit(cache=True, parallel=True)
def _calcinfoperm_numba(x, y, nperm, seed, xsh, px, py, pxy, n_threads):
    out = np.zeros(nperm, dtype=np.float64)
    for tid in prange(n_threads):
        xsh_tid = xsh[tid]
        for perm in range(tid, nperm, n_threads):
            for i in range(x.size):
                xsh_tid[i] = x[i]
            _fisher_yates_shuffle_numba(xsh_tid, np.uint64(seed + perm))
            out[perm] = _calcinfo_with_scratch_numba(xsh_tid, y, px[tid], py[tid], pxy[tid])
    return out


@njit(cache=True, parallel=True)
def _calcinfoperm_slice_numba(x, y, nperm, seed, xsh, px, py, pxy, n_threads):
    out = np.zeros((nperm, x.shape[0]), dtype=np.float64)
    for tid in prange(n_threads):
        xsh_tid = xsh[tid]
        for col in range(tid, x.shape[0], n_threads):
            for perm in range(nperm):
                for i in range(x.shape[1]):
                    xsh_tid[i] = x[col, i]
                _fisher_yates_shuffle_numba(xsh_tid, _slice_seed_numba(seed, col, perm))
                out[perm, col] = _calcinfo_with_scratch_numba(xsh_tid, y, px[tid], py[tid], pxy[tid])
    return out


@njit(cache=True)
def _group_starts_into_numba(sorted_values, group_starts):
    n_groups = 1
    group_starts[0] = 0
    for i in range(1, sorted_values.size):
        if sorted_values[i] != sorted_values[i - 1]:
            group_starts[n_groups] = i
            n_groups += 1
    group_starts[n_groups] = sorted_values.size
    return n_groups


@njit(cache=True)
def _eqpop_sorted_labels_with_scratch_numba(sorted_values, nb, labels, group_starts, dp, parent, group_cuts):
    n_groups = _group_starts_into_numba(sorted_values, group_starts)
    if n_groups < nb:
        return False

    ideal = sorted_values.size / float(nb)
    for b in range(nb + 1):
        for g in range(n_groups + 1):
            dp[b, g] = np.inf
            parent[b, g] = -1
    dp[0, 0] = 0.0

    for b in range(1, nb + 1):
        min_groups_used = b
        max_groups_used = n_groups - (nb - b)
        for g in range(min_groups_used, max_groups_used + 1):
            best_cost = np.inf
            best_parent = -1
            for prev in range(b - 1, g):
                prefix_cost = dp[b - 1, prev]
                if not np.isfinite(prefix_cost):
                    continue
                count = group_starts[g] - group_starts[prev]
                deviation = count - ideal
                cost = prefix_cost + deviation * deviation
                if cost < best_cost:
                    best_cost = cost
                    best_parent = prev
            dp[b, g] = best_cost
            parent[b, g] = best_parent

    if not np.isfinite(dp[nb, n_groups]):
        return False

    group_cuts[nb] = n_groups
    g = n_groups
    for b in range(nb, 0, -1):
        prev = parent[b, g]
        if prev < 0:
            return False
        group_cuts[b - 1] = prev
        g = prev

    for bin_index in range(nb):
        start = group_starts[group_cuts[bin_index]]
        stop = group_starts[group_cuts[bin_index + 1]]
        labels[start:stop] = bin_index
    return True


@njit(cache=True)
def _eqpop_sorted_labels_ok_numba(sorted_values, nb, labels):
    group_starts = np.empty(sorted_values.size + 1, dtype=np.int64)
    dp = np.empty((nb + 1, sorted_values.size + 1), dtype=np.float64)
    parent = np.empty((nb + 1, sorted_values.size + 1), dtype=np.int64)
    group_cuts = np.empty(nb + 1, dtype=np.int64)
    return _eqpop_sorted_labels_with_scratch_numba(sorted_values, nb, labels, group_starts, dp, parent, group_cuts)


@njit(cache=True)
def _eqpop_sorted_labels_numba(sorted_values, nb):
    labels = np.empty(sorted_values.size, dtype=np.int64)
    ok = _eqpop_sorted_labels_ok_numba(sorted_values, nb, labels)
    return ok, labels


@njit(cache=True)
def _eqpop_numba(values, nb):
    order = np.argsort(values)
    sorted_values = np.empty_like(values)
    for i in range(order.size):
        sorted_values[i] = values[order[i]]
    labels_sorted = np.empty(values.size, dtype=np.int64)
    ok = _eqpop_sorted_labels_ok_numba(sorted_values, nb, labels_sorted)
    if not ok:
        return False, labels_sorted
    labels = np.empty(values.size, dtype=np.int64)
    for i in range(order.size):
        labels[order[i]] = labels_sorted[i]
    return True, labels


@njit(cache=True, parallel=True)
def _calccmi_slice_numba(x, y, z, pz, pxz, pyz, pxyz, n_threads):
    out = np.zeros(x.shape[0], dtype=np.float64)
    for tid in prange(n_threads):
        for col in range(tid, x.shape[0], n_threads):
            out[col] = _calccmi_with_scratch_numba(x[col], y, z, pz[tid], pxz[tid], pyz[tid], pxyz[tid])
    return out


def calcinfo(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    _reject_per_call_threads(threads)
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    x = _fallback._as_fastinfo_discrete_vector(x, xb, "x") if validate else _fallback._require_fastinfo_vector_layout(x, "x")
    y = _fallback._as_fastinfo_discrete_vector(y, yb, "y") if validate else _fallback._require_fastinfo_vector_layout(y, "y")
    if x.size != y.size:
        raise ValueError("calcinfo: Number of trials must match.")
    out = float(_calcinfo_numba(x, xb, y, yb))
    if bias:
        out -= mmbiasinfo(xb, yb, x.size)
    return out


def calcinfomatched(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    _reject_per_call_threads(threads)
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    x = _fallback._as_fastinfo_discrete_matrix(x, xb, "x") if validate else _fallback._require_fastinfo_matrix_layout(x, "x")
    y = _fallback._as_fastinfo_discrete_matrix(y, yb, "y") if validate else _fallback._require_fastinfo_matrix_layout(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    n_threads = get_num_threads()
    px = np.zeros((n_threads, xb), dtype=np.int64)
    py = np.zeros((n_threads, yb), dtype=np.int64)
    pxy = np.zeros((n_threads, xb, yb), dtype=np.int64)
    out = _calcinfomatched_numba(x, y, px, py, pxy, n_threads)
    if bias:
        out = out - mmbiasinfo(xb, yb, x.shape[1])
    return out


def calccmi(x, xb, y, yb, z, zb, *, bias=False, validate=True, threads=None):
    _reject_per_call_threads(threads)
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    zb = _validate_bin_count(zb, "zb")
    x = _fallback._as_fastinfo_discrete_vector(x, xb, "x") if validate else _fallback._require_fastinfo_vector_layout(x, "x")
    y = _fallback._as_fastinfo_discrete_vector(y, yb, "y") if validate else _fallback._require_fastinfo_vector_layout(y, "y")
    z = _fallback._as_fastinfo_discrete_vector(z, zb, "z") if validate else _fallback._require_fastinfo_vector_layout(z, "z")
    if x.size != y.size or x.size != z.size:
        raise ValueError("calccmi: Number of trials must match.")
    out = float(_calccmi_numba(x, xb, y, yb, z, zb))
    if bias:
        out -= mmbiascmi(xb, yb, zb, x.size)
    return out


def calccondcmi(x, xb, y, yb, z, zb, k, kb, *, validate=True, threads=None):
    return _fallback.calccondcmi(x, xb, y, yb, z, zb, k, kb, validate=validate, threads=threads)


def calcinfo_slice(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    _reject_per_call_threads(threads)
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    x = _fallback._as_fastinfo_discrete_matrix(x, xb, "x") if validate else _fallback._require_fastinfo_matrix_layout(x, "x")
    y = _fallback._as_fastinfo_discrete_vector(y, yb, "y") if validate else _fallback._require_fastinfo_vector_layout(y, "y")
    if x.shape[1] != y.size:
        raise ValueError("x must have shape [Nx, Ntrl] and y must have length Ntrl.")
    n_threads = get_num_threads()
    px = np.zeros((n_threads, xb), dtype=np.int64)
    py = np.zeros((n_threads, yb), dtype=np.int64)
    pxy = np.zeros((n_threads, xb, yb), dtype=np.int64)
    out = _calcinfo_slice_numba(x, y, px, py, pxy, n_threads)
    if bias:
        out = out - mmbiasinfo(xb, yb, x.shape[1])
    return out


def calccmi_slice(x, xb, y, yb, z, zb, *, bias=False, validate=True, threads=None):
    _reject_per_call_threads(threads)
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    zb = _validate_bin_count(zb, "zb")
    x = _fallback._as_fastinfo_discrete_matrix(x, xb, "x") if validate else _fallback._require_fastinfo_matrix_layout(x, "x")
    y = _fallback._as_fastinfo_discrete_vector(y, yb, "y") if validate else _fallback._require_fastinfo_vector_layout(y, "y")
    z = _fallback._as_fastinfo_discrete_vector(z, zb, "z") if validate else _fallback._require_fastinfo_vector_layout(z, "z")
    if x.shape[1] != y.size or x.shape[1] != z.size:
        raise ValueError("x must have shape [Nx, Ntrl], y and z must have length Ntrl.")
    n_threads = get_num_threads()
    pz = np.zeros((n_threads, zb), dtype=np.int64)
    pxz = np.zeros((n_threads, xb, zb), dtype=np.int64)
    pyz = np.zeros((n_threads, yb, zb), dtype=np.int64)
    pxyz = np.zeros((n_threads, xb, yb, zb), dtype=np.int64)
    out = _calccmi_slice_numba(x, y, z, pz, pxz, pyz, pxyz, n_threads)
    if bias:
        out = out - mmbiascmi(xb, yb, zb, x.shape[1])
    return out


def calcinfoperm(x, xb, y, yb, nperm, *, bias=False, validate=True, threads=None, rng=None, seed=None):
    if rng is not None:
        return _fallback.calcinfoperm(
            x, xb, y, yb, nperm, bias=bias, validate=validate, threads=threads, rng=rng, seed=seed
        )

    _reject_per_call_threads(threads)
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    if not isinstance(nperm, (int, np.integer)) or int(nperm) < 0:
        raise ValueError("nperm must be a non-negative integer.")
    nperm = int(nperm)
    x = _fallback._as_fastinfo_discrete_vector(x, xb, "x") if validate else _fallback._require_fastinfo_vector_layout(x, "x")
    y = _fallback._as_fastinfo_discrete_vector(y, yb, "y") if validate else _fallback._require_fastinfo_vector_layout(y, "y")
    if x.size != y.size:
        raise ValueError("calcinfoperm: Number of trials must match.")
    if seed is None:
        seed = 5489
    n_threads = get_num_threads()
    xsh = np.empty((n_threads, x.size), dtype=x.dtype)
    px = np.zeros((n_threads, xb), dtype=np.int64)
    py = np.zeros((n_threads, yb), dtype=np.int64)
    pxy = np.zeros((n_threads, xb, yb), dtype=np.int64)
    out = _calcinfoperm_numba(x, y, nperm, int(seed), xsh, px, py, pxy, n_threads)
    if bias:
        out = out - mmbiasinfo(xb, yb, x.size)
    return out


def calcinfoperm_slice(x, xb, y, yb, nperm, *, bias=False, validate=True, threads=None, seed=None):
    _reject_per_call_threads(threads)
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    if not isinstance(nperm, (int, np.integer)) or int(nperm) < 0:
        raise ValueError("nperm must be a non-negative integer.")
    nperm = int(nperm)
    x = _fallback._as_fastinfo_discrete_matrix(x, xb, "x") if validate else _fallback._require_fastinfo_matrix_layout(x, "x")
    y = _fallback._as_fastinfo_discrete_vector(y, yb, "y") if validate else _fallback._require_fastinfo_vector_layout(y, "y")
    if x.shape[1] != y.size:
        raise ValueError("x must have shape [Nx, Ntrl] and y must have length Ntrl.")
    if seed is None:
        seed = 5489
    n_threads = get_num_threads()
    xsh = np.empty((n_threads, x.shape[1]), dtype=x.dtype)
    px = np.zeros((n_threads, xb), dtype=np.int64)
    py = np.zeros((n_threads, yb), dtype=np.int64)
    pxy = np.zeros((n_threads, xb, yb), dtype=np.int64)
    out = _calcinfoperm_slice_numba(x, y, nperm, int(seed), xsh, px, py, pxy, n_threads)
    if bias:
        out = out - mmbiasinfo(xb, yb, x.shape[1])
    return out


def eqpop(x, nb, *, validate=True, warn_on_ties=True):
    values = _fallback._validate_continuous_vector(x, "x") if validate else np.asarray(x, dtype=float).reshape(-1)
    nb = _validate_bin_count(nb, "nb")
    _fallback._warn_if_quantized(values, nb, warn_on_ties, "fastinfo.eqpop")
    ok, labels = _eqpop_numba(values, nb)
    if not ok:
        raise ValueError(
            "Cannot form the requested number of equal-population bins without splitting tied values. "
            "If the input is discrete or strongly quantized, use rebin instead."
        )
    return labels


@njit(cache=True, parallel=True)
def _eqpop_sorted_slice_numba(values, nb, labels, group_starts, dp, parent, group_cuts, n_threads):
    out = np.full(values.shape, np.nan)
    failed = np.zeros(values.shape[0], dtype=np.uint8)
    for tid in prange(n_threads):
        for col in range(tid, values.shape[0], n_threads):
            valid = True
            for row in range(values.shape[1]):
                if not np.isfinite(values[col, row]):
                    valid = False
                    break
                if row > 0 and values[col, row] < values[col, row - 1]:
                    valid = False
                    break
            if not valid:
                failed[col] = 1
                continue
            ok = _eqpop_sorted_labels_with_scratch_numba(
                values[col], nb, labels[tid], group_starts[tid], dp[tid], parent[tid], group_cuts[tid]
            )
            if ok:
                for row in range(values.shape[1]):
                    out[col, row] = labels[tid, row]
            else:
                failed[col] = 1
    return out, failed


@njit(cache=True, parallel=True)
def _eqpop_slice_numba(values, nb, labels, labels_sorted, sorted_values, group_starts, dp, parent, group_cuts, n_threads):
    out = np.full(values.shape, np.nan)
    failed = np.zeros(values.shape[0], dtype=np.uint8)
    for tid in prange(n_threads):
        for col in range(tid, values.shape[0], n_threads):
            valid = True
            for row in range(values.shape[1]):
                if not np.isfinite(values[col, row]):
                    valid = False
                    break
            if not valid:
                failed[col] = 1
                continue
            order = np.argsort(values[col])
            for row in range(values.shape[1]):
                sorted_values[tid, row] = values[col, order[row]]
            ok = _eqpop_sorted_labels_with_scratch_numba(
                sorted_values[tid],
                nb,
                labels_sorted[tid],
                group_starts[tid],
                dp[tid],
                parent[tid],
                group_cuts[tid],
            )
            if ok:
                for row in range(values.shape[1]):
                    labels[tid, order[row]] = labels_sorted[tid, row]
                for row in range(values.shape[1]):
                    out[col, row] = labels[tid, row]
            else:
                failed[col] = 1
    return out, failed


def eqpop_sorted(x_sorted, nb, *, validate=True, warn_on_ties=True):
    values = _fallback._validate_continuous_vector(x_sorted, "x_sorted") if validate else np.asarray(x_sorted, dtype=float).reshape(-1)
    nb = _validate_bin_count(nb, "nb")
    if validate and np.any(np.diff(values) < 0):
        raise ValueError("x_sorted must be sorted in nondecreasing order.")
    _fallback._warn_if_quantized(values, nb, warn_on_ties, "fastinfo.eqpop_sorted")
    ok, labels = _eqpop_sorted_labels_numba(values, nb)
    if not ok:
        raise ValueError(
            "Cannot form the requested number of equal-population bins without splitting tied values. "
            "If the input is discrete or strongly quantized, use rebin instead."
        )
    return labels


def eqpop_slice(x, nb, *, validate=True, warn_on_ties=True, threads=None):
    _reject_per_call_threads(threads)
    values = _fallback._validate_fastinfo_continuous_matrix(x, "x") if validate else np.asarray(
        _fallback._require_fastinfo_matrix_layout(x, "x"), dtype=float
    )
    nb = _validate_bin_count(nb, "nb")
    n_threads = get_num_threads()
    labels = np.empty((n_threads, values.shape[1]), dtype=np.int64)
    labels_sorted = np.empty((n_threads, values.shape[1]), dtype=np.int64)
    sorted_values = np.empty((n_threads, values.shape[1]), dtype=np.float64)
    group_starts = np.empty((n_threads, values.shape[1] + 1), dtype=np.int64)
    dp = np.empty((n_threads, nb + 1, values.shape[1] + 1), dtype=np.float64)
    parent = np.empty((n_threads, nb + 1, values.shape[1] + 1), dtype=np.int64)
    group_cuts = np.empty((n_threads, nb + 1), dtype=np.int64)
    out, failed = _eqpop_slice_numba(
        values, nb, labels, labels_sorted, sorted_values, group_starts, dp, parent, group_cuts, n_threads
    )
    failed_count = int(np.sum(failed))
    if warn_on_ties and failed_count > 0:
        import warnings
        warnings.warn(
            f"fastinfo.eqpop_slice could not form tie-consistent bins for {failed_count} page(s). "
            "Those page outputs were set to NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def eqpop_sorted_slice(x_sorted, nb, *, validate=True, warn_on_ties=True, threads=None):
    _reject_per_call_threads(threads)
    values = _fallback._validate_fastinfo_continuous_matrix(x_sorted, "x_sorted") if validate else np.asarray(
        _fallback._require_fastinfo_matrix_layout(x_sorted, "x_sorted"), dtype=float
    )
    nb = _validate_bin_count(nb, "nb")
    n_threads = get_num_threads()
    labels = np.empty((n_threads, values.shape[1]), dtype=np.int64)
    group_starts = np.empty((n_threads, values.shape[1] + 1), dtype=np.int64)
    dp = np.empty((n_threads, nb + 1, values.shape[1] + 1), dtype=np.float64)
    parent = np.empty((n_threads, nb + 1, values.shape[1] + 1), dtype=np.int64)
    group_cuts = np.empty((n_threads, nb + 1), dtype=np.int64)
    out, failed = _eqpop_sorted_slice_numba(values, nb, labels, group_starts, dp, parent, group_cuts, n_threads)
    failed_count = int(np.sum(failed))
    if warn_on_ties and failed_count > 0:
        import warnings
        warnings.warn(
            f"fastinfo.eqpop_sorted_slice could not form tie-consistent bins for {failed_count} page(s). "
            "Those page outputs were set to NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def calcpairwiseinfo(x, xb, y, yb, *, bias=False, validate=True):
    return _fallback.calcpairwiseinfo(x, xb, y, yb, bias=bias, validate=validate)


def calcpairwiseinfo_slice(x, xb, y, yb, *, bias=False, validate=True):
    return _fallback.calcpairwiseinfo_slice(x, xb, y, yb, bias=bias, validate=validate)


def set_threads(threads):
    threads = max(1, int(threads))
    set_num_threads(threads)
    return threads
