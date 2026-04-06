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
def _entropy_from_counts_numba(counts, count_log2, log2_n, inv_n):
    if count_log2.size == 0:
        return 0.0
    acc = 0.0
    for i in range(counts.size):
        acc += count_log2[counts[i]]
    return log2_n - acc * inv_n


@njit(cache=True)
def _derive_marginals_from_joint_numba(pxy, px, py):
    px.fill(0)
    py.fill(0)
    for xi in range(pxy.shape[0]):
        for yi in range(pxy.shape[1]):
            count = pxy[xi, yi]
            px[xi] += count
            py[yi] += count


@njit(cache=True)
def _derive_x_marginal_from_joint_numba(pxy, px):
    px.fill(0)
    for xi in range(pxy.shape[0]):
        for yi in range(pxy.shape[1]):
            px[xi] += pxy[xi, yi]


@njit(cache=True)
def _derive_cmi_marginals_from_joint_numba(pxyz, pz, pxz, pyz):
    pz.fill(0)
    pxz.fill(0)
    pyz.fill(0)
    for xi in range(pxyz.shape[0]):
        for yi in range(pxyz.shape[1]):
            for zi in range(pxyz.shape[2]):
                count = pxyz[xi, yi, zi]
                pz[zi] += count
                pxz[xi, zi] += count
                pyz[yi, zi] += count
    

@njit(cache=True)
def _calcinfo_small_mode_numba(x, y, xm, ym, count_log2, log2_n, inv_n, mode, constant_term):
    if xm == 2 and ym == 2:
        joint = np.zeros(4, dtype=np.int64)
        for i in range(x.size):
            joint[int(x[i]) * 2 + int(y[i])] += 1
        if mode == 2:
            return constant_term - _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        px = np.empty(2, dtype=np.int64)
        px[0] = joint[0] + joint[1]
        px[1] = joint[2] + joint[3]
        hx = _entropy_from_counts_numba(px, count_log2, log2_n, inv_n)
        hxy = _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        if mode == 1:
            return hx + constant_term - hxy
        py = np.empty(2, dtype=np.int64)
        py[0] = joint[0] + joint[2]
        py[1] = joint[1] + joint[3]
        return hx + _entropy_from_counts_numba(py, count_log2, log2_n, inv_n) - hxy
    if xm == 3 and ym == 2:
        joint = np.zeros(6, dtype=np.int64)
        for i in range(x.size):
            joint[int(x[i]) * 2 + int(y[i])] += 1
        if mode == 2:
            return constant_term - _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        px = np.empty(3, dtype=np.int64)
        px[0] = joint[0] + joint[1]
        px[1] = joint[2] + joint[3]
        px[2] = joint[4] + joint[5]
        hx = _entropy_from_counts_numba(px, count_log2, log2_n, inv_n)
        hxy = _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        if mode == 1:
            return hx + constant_term - hxy
        py = np.empty(2, dtype=np.int64)
        py[0] = joint[0] + joint[2] + joint[4]
        py[1] = joint[1] + joint[3] + joint[5]
        return hx + _entropy_from_counts_numba(py, count_log2, log2_n, inv_n) - hxy
    if xm == 2 and ym == 3:
        joint = np.zeros(6, dtype=np.int64)
        for i in range(x.size):
            joint[int(x[i]) * 3 + int(y[i])] += 1
        if mode == 2:
            return constant_term - _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        px = np.empty(2, dtype=np.int64)
        px[0] = joint[0] + joint[1] + joint[2]
        px[1] = joint[3] + joint[4] + joint[5]
        hx = _entropy_from_counts_numba(px, count_log2, log2_n, inv_n)
        hxy = _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        if mode == 1:
            return hx + constant_term - hxy
        py = np.empty(3, dtype=np.int64)
        py[0] = joint[0] + joint[3]
        py[1] = joint[1] + joint[4]
        py[2] = joint[2] + joint[5]
        return hx + _entropy_from_counts_numba(py, count_log2, log2_n, inv_n) - hxy
    if xm == 3 and ym == 3:
        joint = np.zeros(9, dtype=np.int64)
        for i in range(x.size):
            joint[int(x[i]) * 3 + int(y[i])] += 1
        if mode == 2:
            return constant_term - _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        px = np.empty(3, dtype=np.int64)
        py = np.empty(3, dtype=np.int64)
        for xi in range(3):
            px[xi] = joint[xi * 3] + joint[xi * 3 + 1] + joint[xi * 3 + 2]
        for yi in range(3):
            py[yi] = joint[yi] + joint[3 + yi] + joint[6 + yi]
        hx = _entropy_from_counts_numba(px, count_log2, log2_n, inv_n)
        hxy = _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        if mode == 1:
            return hx + constant_term - hxy
        return hx + _entropy_from_counts_numba(py, count_log2, log2_n, inv_n) - hxy
    if xm == 5 and ym == 2:
        joint = np.zeros(10, dtype=np.int64)
        for i in range(x.size):
            joint[int(x[i]) * 2 + int(y[i])] += 1
        if mode == 2:
            return constant_term - _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        px = np.empty(5, dtype=np.int64)
        for xi in range(5):
            px[xi] = joint[xi * 2] + joint[xi * 2 + 1]
        hx = _entropy_from_counts_numba(px, count_log2, log2_n, inv_n)
        hxy = _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        if mode == 1:
            return hx + constant_term - hxy
        py = np.empty(2, dtype=np.int64)
        py[0] = joint[0] + joint[2] + joint[4] + joint[6] + joint[8]
        py[1] = joint[1] + joint[3] + joint[5] + joint[7] + joint[9]
        return hx + _entropy_from_counts_numba(py, count_log2, log2_n, inv_n) - hxy
    if xm == 2 and ym == 5:
        joint = np.zeros(10, dtype=np.int64)
        for i in range(x.size):
            joint[int(x[i]) * 5 + int(y[i])] += 1
        if mode == 2:
            return constant_term - _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        px = np.empty(2, dtype=np.int64)
        px[0] = joint[0] + joint[1] + joint[2] + joint[3] + joint[4]
        px[1] = joint[5] + joint[6] + joint[7] + joint[8] + joint[9]
        hx = _entropy_from_counts_numba(px, count_log2, log2_n, inv_n)
        hxy = _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        if mode == 1:
            return hx + constant_term - hxy
        py = np.empty(5, dtype=np.int64)
        for yi in range(5):
            py[yi] = joint[yi] + joint[5 + yi]
        return hx + _entropy_from_counts_numba(py, count_log2, log2_n, inv_n) - hxy
    if xm == 5 and ym == 3:
        joint = np.zeros(15, dtype=np.int64)
        for i in range(x.size):
            joint[int(x[i]) * 3 + int(y[i])] += 1
        if mode == 2:
            return constant_term - _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        px = np.empty(5, dtype=np.int64)
        py = np.empty(3, dtype=np.int64)
        for xi in range(5):
            px[xi] = joint[xi * 3] + joint[xi * 3 + 1] + joint[xi * 3 + 2]
        for yi in range(3):
            py[yi] = joint[yi] + joint[3 + yi] + joint[6 + yi] + joint[9 + yi] + joint[12 + yi]
        hx = _entropy_from_counts_numba(px, count_log2, log2_n, inv_n)
        hxy = _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        if mode == 1:
            return hx + constant_term - hxy
        return hx + _entropy_from_counts_numba(py, count_log2, log2_n, inv_n) - hxy
    if xm == 3 and ym == 5:
        joint = np.zeros(15, dtype=np.int64)
        for i in range(x.size):
            joint[int(x[i]) * 5 + int(y[i])] += 1
        if mode == 2:
            return constant_term - _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        px = np.empty(3, dtype=np.int64)
        py = np.empty(5, dtype=np.int64)
        for xi in range(3):
            offset = xi * 5
            px[xi] = joint[offset] + joint[offset + 1] + joint[offset + 2] + joint[offset + 3] + joint[offset + 4]
        for yi in range(5):
            py[yi] = joint[yi] + joint[5 + yi] + joint[10 + yi]
        hx = _entropy_from_counts_numba(px, count_log2, log2_n, inv_n)
        hxy = _entropy_from_counts_numba(joint, count_log2, log2_n, inv_n)
        if mode == 1:
            return hx + constant_term - hxy
        return hx + _entropy_from_counts_numba(py, count_log2, log2_n, inv_n) - hxy
    return np.nan


def _build_entropy_lookup(n_trials):
    if n_trials <= 0:
        return np.empty(0, dtype=np.float64), 0.0, 0.0
    counts = np.arange(n_trials + 1, dtype=np.float64)
    count_log2 = np.zeros(n_trials + 1, dtype=np.float64)
    count_log2[2:] = counts[2:] * np.log2(counts[2:])
    return count_log2, float(np.log2(n_trials)), 1.0 / float(n_trials)


@njit(cache=True)
def _calcinfo_numba(x, xb, y, yb, count_log2, log2_n, inv_n):
    px = np.zeros(xb, dtype=np.int64)
    py = np.zeros(yb, dtype=np.int64)
    pxy = np.zeros((xb, yb), dtype=np.int64)
    return _calcinfo_with_scratch_numba(x, y, px, py, pxy, count_log2, log2_n, inv_n)


@njit(cache=True)
def _calcinfo_with_scratch_numba(x, y, px, py, pxy, count_log2, log2_n, inv_n):
    pxy.fill(0)
    special = _calcinfo_small_mode_numba(x, y, pxy.shape[0], pxy.shape[1], count_log2, log2_n, inv_n, 0, 0.0)
    if not np.isnan(special):
        return special
    for i in range(x.size):
        pxy[x[i], y[i]] += 1
    _derive_marginals_from_joint_numba(pxy, px, py)
    return (
        _entropy_from_counts_numba(px, count_log2, log2_n, inv_n)
        + _entropy_from_counts_numba(py, count_log2, log2_n, inv_n)
        - _entropy_from_counts_numba(pxy.ravel(), count_log2, log2_n, inv_n)
    )


@njit(cache=True)
def _calcinfo_with_precomputed_hy_numba(x, y, px, pxy, count_log2, log2_n, inv_n, hy):
    pxy.fill(0)
    special = _calcinfo_small_mode_numba(x, y, pxy.shape[0], pxy.shape[1], count_log2, log2_n, inv_n, 1, hy)
    if not np.isnan(special):
        return special
    for i in range(x.size):
        pxy[x[i], y[i]] += 1
    _derive_x_marginal_from_joint_numba(pxy, px)
    return _entropy_from_counts_numba(px, count_log2, log2_n, inv_n) + hy - _entropy_from_counts_numba(pxy.ravel(), count_log2, log2_n, inv_n)


@njit(cache=True)
def _calcinfo_with_precomputed_sum_numba(x, y, pxy, count_log2, log2_n, inv_n, hx_plus_hy):
    pxy.fill(0)
    special = _calcinfo_small_mode_numba(x, y, pxy.shape[0], pxy.shape[1], count_log2, log2_n, inv_n, 2, hx_plus_hy)
    if not np.isnan(special):
        return special
    for i in range(x.size):
        pxy[x[i], y[i]] += 1
    return hx_plus_hy - _entropy_from_counts_numba(pxy.ravel(), count_log2, log2_n, inv_n)


@njit(cache=True)
def _calccmi_numba(x, xb, y, yb, z, zb, count_log2, log2_n, inv_n):
    pz = np.zeros(zb, dtype=np.int64)
    pxz = np.zeros((xb, zb), dtype=np.int64)
    pyz = np.zeros((yb, zb), dtype=np.int64)
    pxyz = np.zeros((xb, yb, zb), dtype=np.int64)
    return _calccmi_with_scratch_numba(x, y, z, pz, pxz, pyz, pxyz, count_log2, log2_n, inv_n)


@njit(cache=True)
def _calccmi_with_scratch_numba(x, y, z, pz, pxz, pyz, pxyz, count_log2, log2_n, inv_n):
    pxyz.fill(0)
    for i in range(x.size):
        pxyz[x[i], y[i], z[i]] += 1
    _derive_cmi_marginals_from_joint_numba(pxyz, pz, pxz, pyz)
    return (
        _entropy_from_counts_numba(pxz.ravel(), count_log2, log2_n, inv_n)
        + _entropy_from_counts_numba(pyz.ravel(), count_log2, log2_n, inv_n)
        - _entropy_from_counts_numba(pxyz.ravel(), count_log2, log2_n, inv_n)
        - _entropy_from_counts_numba(pz, count_log2, log2_n, inv_n)
    )


@njit(cache=True)
def _calc_cmi_with_precomputed_yz_terms_numba(x, y, z, pxz, pxyz, count_log2, log2_n, inv_n, hyz_minus_hz):
    pxyz.fill(0)
    for i in range(x.size):
        pxyz[x[i], y[i], z[i]] += 1
    pxz.fill(0)
    for xi in range(pxyz.shape[0]):
        for yi in range(pxyz.shape[1]):
            for zi in range(pxyz.shape[2]):
                pxz[xi, zi] += pxyz[xi, yi, zi]
    return _entropy_from_counts_numba(pxz.ravel(), count_log2, log2_n, inv_n) + hyz_minus_hz - _entropy_from_counts_numba(pxyz.ravel(), count_log2, log2_n, inv_n)


@njit(cache=True, parallel=True)
def _calcinfo_slice_numba(x, y, px, pxy, count_log2, log2_n, inv_n, hy, n_threads):
    out = np.zeros(x.shape[0], dtype=np.float64)
    for tid in prange(n_threads):
        for col in range(tid, x.shape[0], n_threads):
            out[col] = _calcinfo_with_precomputed_hy_numba(x[col], y, px[tid], pxy[tid], count_log2, log2_n, inv_n, hy)
    return out


@njit(cache=True, parallel=True)
def _calcinfomatched_numba(x, y, px, py, pxy, count_log2, log2_n, inv_n, n_threads):
    out = np.zeros(x.shape[0], dtype=np.float64)
    for tid in prange(n_threads):
        for col in range(tid, x.shape[0], n_threads):
            out[col] = _calcinfo_with_scratch_numba(x[col], y[col], px[tid], py[tid], pxy[tid], count_log2, log2_n, inv_n)
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
def _calcinfoperm_numba(x, y, nperm, seed, xsh, pxy, count_log2, log2_n, inv_n, hx_plus_hy, n_threads):
    out = np.zeros(nperm, dtype=np.float64)
    for tid in prange(n_threads):
        xsh_tid = xsh[tid]
        for perm in range(tid, nperm, n_threads):
            for i in range(x.size):
                xsh_tid[i] = x[i]
            _fisher_yates_shuffle_numba(xsh_tid, np.uint64(seed + perm))
            out[perm] = _calcinfo_with_precomputed_sum_numba(xsh_tid, y, pxy[tid], count_log2, log2_n, inv_n, hx_plus_hy)
    return out


@njit(cache=True, parallel=True)
def _calcinfoperm_slice_numba(x, y, nperm, seed, xsh, pxy, count_log2, log2_n, inv_n, hx_plus_hy_cols, n_threads):
    out = np.zeros((nperm, x.shape[0]), dtype=np.float64)
    for tid in prange(n_threads):
        xsh_tid = xsh[tid]
        for col in range(tid, x.shape[0], n_threads):
            for perm in range(nperm):
                for i in range(x.shape[1]):
                    xsh_tid[i] = x[col, i]
                _fisher_yates_shuffle_numba(xsh_tid, _slice_seed_numba(seed, col, perm))
                out[perm, col] = _calcinfo_with_precomputed_sum_numba(
                    xsh_tid, y, pxy[tid], count_log2, log2_n, inv_n, hx_plus_hy_cols[col]
                )
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
def _calccmi_slice_numba(x, y, z, pxz, pxyz, count_log2, log2_n, inv_n, hyz_minus_hz, n_threads):
    out = np.zeros(x.shape[0], dtype=np.float64)
    for tid in prange(n_threads):
        for col in range(tid, x.shape[0], n_threads):
            out[col] = _calc_cmi_with_precomputed_yz_terms_numba(
                x[col], y, z, pxz[tid], pxyz[tid], count_log2, log2_n, inv_n, hyz_minus_hz
            )
    return out


def calcinfo(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    _reject_per_call_threads(threads)
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    x = _fallback._as_fastinfo_discrete_vector(x, xb, "x") if validate else _fallback._require_fastinfo_vector_layout(x, "x")
    y = _fallback._as_fastinfo_discrete_vector(y, yb, "y") if validate else _fallback._require_fastinfo_vector_layout(y, "y")
    if x.size != y.size:
        raise ValueError("calcinfo: Number of trials must match.")
    count_log2, log2_n, inv_n = _build_entropy_lookup(x.size)
    out = float(_calcinfo_numba(x, xb, y, yb, count_log2, log2_n, inv_n))
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
    count_log2, log2_n, inv_n = _build_entropy_lookup(x.shape[1])
    px = np.zeros((n_threads, xb), dtype=np.int64)
    py = np.zeros((n_threads, yb), dtype=np.int64)
    pxy = np.zeros((n_threads, xb, yb), dtype=np.int64)
    out = _calcinfomatched_numba(x, y, px, py, pxy, count_log2, log2_n, inv_n, n_threads)
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
    count_log2, log2_n, inv_n = _build_entropy_lookup(x.size)
    out = float(_calccmi_numba(x, xb, y, yb, z, zb, count_log2, log2_n, inv_n))
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
    count_log2, log2_n, inv_n = _build_entropy_lookup(x.shape[1])
    y_counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=yb).astype(np.int64, copy=False)
    hy = _entropy_from_counts_numba(y_counts, count_log2, log2_n, inv_n)
    px = np.zeros((n_threads, xb), dtype=np.int64)
    pxy = np.zeros((n_threads, xb, yb), dtype=np.int64)
    out = _calcinfo_slice_numba(x, y, px, pxy, count_log2, log2_n, inv_n, hy, n_threads)
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
    count_log2, log2_n, inv_n = _build_entropy_lookup(x.shape[1])
    yz_index = np.asarray(y, dtype=np.int64) * zb + np.asarray(z, dtype=np.int64)
    pyz_counts = np.bincount(yz_index, minlength=yb * zb).astype(np.int64, copy=False)
    z_counts = np.bincount(np.asarray(z, dtype=np.int64), minlength=zb).astype(np.int64, copy=False)
    hyz_minus_hz = _entropy_from_counts_numba(pyz_counts, count_log2, log2_n, inv_n) - _entropy_from_counts_numba(
        z_counts, count_log2, log2_n, inv_n
    )
    pxz = np.zeros((n_threads, xb, zb), dtype=np.int64)
    pxyz = np.zeros((n_threads, xb, yb, zb), dtype=np.int64)
    out = _calccmi_slice_numba(x, y, z, pxz, pxyz, count_log2, log2_n, inv_n, hyz_minus_hz, n_threads)
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
    count_log2, log2_n, inv_n = _build_entropy_lookup(x.size)
    x_counts = np.bincount(np.asarray(x, dtype=np.int64), minlength=xb).astype(np.int64, copy=False)
    y_counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=yb).astype(np.int64, copy=False)
    hx_plus_hy = _entropy_from_counts_numba(x_counts, count_log2, log2_n, inv_n) + _entropy_from_counts_numba(
        y_counts, count_log2, log2_n, inv_n
    )
    xsh = np.empty((n_threads, x.size), dtype=x.dtype)
    pxy = np.zeros((n_threads, xb, yb), dtype=np.int64)
    out = _calcinfoperm_numba(x, y, nperm, int(seed), xsh, pxy, count_log2, log2_n, inv_n, hx_plus_hy, n_threads)
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
    count_log2, log2_n, inv_n = _build_entropy_lookup(x.shape[1])
    y_counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=yb).astype(np.int64, copy=False)
    hy = _entropy_from_counts_numba(y_counts, count_log2, log2_n, inv_n)
    hx_plus_hy_cols = np.empty(x.shape[0], dtype=np.float64)
    for col in range(x.shape[0]):
        x_counts = np.bincount(np.asarray(x[col], dtype=np.int64), minlength=xb).astype(np.int64, copy=False)
        hx_plus_hy_cols[col] = hy + _entropy_from_counts_numba(x_counts, count_log2, log2_n, inv_n)
    xsh = np.empty((n_threads, x.shape[1]), dtype=x.dtype)
    pxy = np.zeros((n_threads, xb, yb), dtype=np.int64)
    out = _calcinfoperm_slice_numba(
        x, y, nperm, int(seed), xsh, pxy, count_log2, log2_n, inv_n, hx_plus_hy_cols, n_threads
    )
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
