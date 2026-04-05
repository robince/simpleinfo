"""NumPy fallback implementation for the fastinfo namespace."""

from __future__ import annotations

import warnings

import numpy as np

from ..core import (
    _joint_counts_2d,
    _joint_counts_3d,
    _validate_bin_count,
    calccmi_slice as calccmi_slice_reference,
    calccondcmi as calccondcmi_reference,
    calcinfomatched as calcinfomatched_reference,
    calcinfoperm_slice as calcinfoperm_slice_reference,
    calcpairwiseinfo as calcpairwiseinfo_reference,
    calcpairwiseinfo_slice as calcpairwiseinfo_slice_reference,
    mmbiascmi,
    mmbiasinfo,
)


def _splitmix64_py(x):
    x = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF


def _fisher_yates_shuffle_seeded(values, seed):
    shuffled = np.array(values, copy=True)
    state = int(seed) & 0xFFFFFFFFFFFFFFFF
    for i in range(shuffled.size - 1, 0, -1):
        state = _splitmix64_py(state + i)
        j = state % (i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled


def _warn_if_quantized(values, nb, warn_on_ties, func_name):
    if not warn_on_ties:
        return
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return
    n_unique = np.unique(flat).size
    if n_unique < flat.size and n_unique <= 2 * nb:
        warnings.warn(
            f"{func_name} received heavily repeated values ({n_unique} unique values for {nb} requested bins). "
            "If the input is effectively discrete, use rebin instead.",
            RuntimeWarning,
            stacklevel=2,
        )


def _validate_integer_dtype(array, name):
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype.")
    return array


def _validate_discrete_bounds(array, nbins, name):
    if np.any(array < 0) or np.any(array >= nbins):
        raise ValueError(f"{name} must take values in [0, {nbins - 1}].")
    return array


def _as_fastinfo_discrete_vector(values, nbins, name):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    array = _validate_integer_dtype(array, name)
    return _validate_discrete_bounds(array, nbins, name)


def _require_fastinfo_vector_layout(values, name):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return array


def _as_discrete_matrix(values, nbins, name):
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    array = _validate_integer_dtype(array, name)
    return _validate_discrete_bounds(array, nbins, name)


def _require_fastinfo_matrix_layout(values, name):
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not array.flags.c_contiguous:
        raise ValueError(
            f"{name} must be a C-contiguous 2-D array with trials on the last axis."
        )
    return array


def _as_fastinfo_discrete_matrix(values, nbins, name):
    array = _require_fastinfo_matrix_layout(values, name)
    array = _validate_integer_dtype(array, name)
    return _validate_discrete_bounds(array, nbins, name)


def _validate_continuous_vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_continuous_matrix(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_fastinfo_continuous_matrix(values, name):
    _require_fastinfo_matrix_layout(values, name)
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _eqpop_sorted_labels(sorted_values, nb):
    sorted_values = _validate_continuous_vector(sorted_values, "x_sorted")
    nb = _validate_bin_count(nb, "nb")
    if sorted_values.size < nb:
        raise ValueError("nb cannot exceed the number of samples.")
    if np.any(np.diff(sorted_values) < 0):
        raise ValueError("x_sorted must be sorted in nondecreasing order.")

    group_starts = np.concatenate((
        np.array([0], dtype=np.int64),
        np.nonzero(np.diff(sorted_values) != 0)[0].astype(np.int64) + 1,
        np.array([sorted_values.size], dtype=np.int64),
    ))
    n_groups = group_starts.size - 1
    if n_groups < nb:
        raise ValueError(
            "Cannot form the requested number of equal-population bins without splitting tied values. "
            "If the input is discrete or strongly quantized, use rebin instead."
        )

    ideal = sorted_values.size / float(nb)
    dp = np.full((nb + 1, n_groups + 1), np.inf, dtype=float)
    parent = np.full((nb + 1, n_groups + 1), -1, dtype=np.int64)
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
        raise ValueError("Failed to construct a tie-consistent equal-population partition.")

    group_cuts = np.zeros(nb + 1, dtype=np.int64)
    group_cuts[nb] = n_groups
    g = n_groups
    for b in range(nb, 0, -1):
        prev = parent[b, g]
        if prev < 0:
            raise ValueError("Failed to reconstruct the tie-consistent equal-population partition.")
        group_cuts[b - 1] = prev
        g = prev

    labels = np.zeros(sorted_values.size, dtype=np.int64)
    for bin_index in range(nb):
        start = group_starts[group_cuts[bin_index]]
        stop = group_starts[group_cuts[bin_index + 1]]
        labels[start:stop] = bin_index
    return labels


def calcinfo(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    del threads
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    if validate:
        x = _as_fastinfo_discrete_vector(x, xb, "x")
        y = _as_fastinfo_discrete_vector(y, yb, "y")
    else:
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
    if x.size != y.size:
        raise ValueError("calcinfo: Number of trials must match.")
    counts = _joint_counts_2d(x, xb, y, yb)
    px = counts.sum(axis=1)
    py = counts.sum(axis=0)
    n_trials = x.size
    info = (
        np.log2(n_trials) - np.sum(px[px > 0] * np.log2(px[px > 0])) / n_trials
        + np.log2(n_trials) - np.sum(py[py > 0] * np.log2(py[py > 0])) / n_trials
        - (np.log2(n_trials) - np.sum(counts[counts > 0] * np.log2(counts[counts > 0])) / n_trials)
    )
    if bias:
        info -= mmbiasinfo(xb, yb, x.size)
    return info


def calcinfomatched(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    del threads
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    x = _as_fastinfo_discrete_matrix(x, xb, "x") if validate else _require_fastinfo_matrix_layout(x, "x")
    y = _as_fastinfo_discrete_matrix(y, yb, "y") if validate else _require_fastinfo_matrix_layout(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    return calcinfomatched_reference(x.T, xb, y.T, yb, bias=bias, beta=0.0)


def calccmi(x, xb, y, yb, z, zb, *, bias=False, validate=True, threads=None):
    del threads
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    zb = _validate_bin_count(zb, "zb")
    if validate:
        x = _as_fastinfo_discrete_vector(x, xb, "x")
        y = _as_fastinfo_discrete_vector(y, yb, "y")
        z = _as_fastinfo_discrete_vector(z, zb, "z")
    else:
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        z = np.asarray(z).reshape(-1)
    if x.size != y.size or x.size != z.size:
        raise ValueError("calccmi: Number of trials must match.")
    counts = _joint_counts_3d(x, xb, y, yb, z, zb)
    n_trials = x.size
    pxz = np.sum(counts, axis=1)
    pyz = np.sum(counts, axis=0)
    pz = np.sum(pyz, axis=0)
    info = (
        np.log2(n_trials) - np.sum(pxz[pxz > 0] * np.log2(pxz[pxz > 0])) / n_trials
        + np.log2(n_trials) - np.sum(pyz[pyz > 0] * np.log2(pyz[pyz > 0])) / n_trials
        - (np.log2(n_trials) - np.sum(counts[counts > 0] * np.log2(counts[counts > 0])) / n_trials)
        - (np.log2(n_trials) - np.sum(pz[pz > 0] * np.log2(pz[pz > 0])) / n_trials)
    )
    if bias:
        info -= mmbiascmi(xb, yb, zb, x.size)
    return info


def calccondcmi(x, xb, y, yb, z, zb, k, kb, *, validate=True, threads=None):
    del threads
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    zb = _validate_bin_count(zb, "zb")
    kb = _validate_bin_count(kb, "kb")
    if validate:
        x = _as_fastinfo_discrete_vector(x, xb, "x")
        y = _as_fastinfo_discrete_vector(y, yb, "y")
        z = _as_fastinfo_discrete_vector(z, zb, "z")
        k = _as_fastinfo_discrete_vector(k, kb, "k")
    else:
        x = _require_fastinfo_vector_layout(x, "x")
        y = _require_fastinfo_vector_layout(y, "y")
        z = _require_fastinfo_vector_layout(z, "z")
        k = _require_fastinfo_vector_layout(k, "k")
    return calccondcmi_reference(x, xb, y, yb, z, zb, k, kb)


def calcinfoperm(x, xb, y, yb, nperm, *, bias=False, validate=True, threads=None, rng=None, seed=None):
    del threads
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    if not isinstance(nperm, (int, np.integer)) or int(nperm) < 0:
        raise ValueError("nperm must be a non-negative integer.")
    nperm = int(nperm)

    if validate:
        x = _as_fastinfo_discrete_vector(x, xb, "x")
        y = _as_fastinfo_discrete_vector(y, yb, "y")
    else:
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
    if x.size != y.size:
        raise ValueError("calcinfoperm: Number of trials must match.")

    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    out = np.zeros(nperm, dtype=float)
    if rng is not None:
        xsh = x.copy()
        generator = rng
        for i in range(nperm):
            generator.shuffle(xsh)
            out[i] = calcinfo(xsh, xb, y, yb, bias=False, validate=False)
    else:
        if seed is None:
            seed = 5489
        for i in range(nperm):
            xsh = _fisher_yates_shuffle_seeded(x, int(seed) + i)
            out[i] = calcinfo(xsh, xb, y, yb, bias=False, validate=False)

    if bias:
        out -= mmbiasinfo(xb, yb, x.size)
    return out


def calcinfoperm_slice(x, xb, y, yb, nperm, *, bias=False, validate=True, threads=None, seed=None):
    del threads
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    if not isinstance(nperm, (int, np.integer)) or int(nperm) < 0:
        raise ValueError("nperm must be a non-negative integer.")
    nperm = int(nperm)
    x = _as_fastinfo_discrete_matrix(x, xb, "x") if validate else _require_fastinfo_matrix_layout(x, "x")
    y = _as_fastinfo_discrete_vector(y, yb, "y") if validate else _require_fastinfo_vector_layout(y, "y")
    if x.shape[1] != y.size:
        raise ValueError("x must have shape [Nx, Ntrl] and y must have length Ntrl.")
    if seed is None:
        out = calcinfoperm_slice_reference(x.T, xb, y, yb, nperm, bias=bias, beta=0.0).T
        return out

    out = np.zeros((nperm, x.shape[0]), dtype=float)
    for col in range(x.shape[0]):
        for perm in range(nperm):
            seed_col = _splitmix64_py(_splitmix64_py(int(seed)) + col)
            xsh = _fisher_yates_shuffle_seeded(x[col], _splitmix64_py(seed_col + perm))
            out[perm, col] = calcinfo(xsh, xb, y, yb, bias=False, validate=False)
    if bias:
        out -= mmbiasinfo(xb, yb, x.shape[1])
    return out


def calcinfo_slice(x, xb, y, yb, *, bias=False, validate=True, threads=None):
    del threads
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    x = _as_fastinfo_discrete_matrix(x, xb, "x") if validate else _require_fastinfo_matrix_layout(x, "x")
    y = _as_fastinfo_discrete_vector(y, yb, "y") if validate else _require_fastinfo_vector_layout(y, "y")
    if x.shape[1] != y.size:
        raise ValueError("x must have shape [Nx, Ntrl] and y must have length Ntrl.")

    out = np.zeros(x.shape[0], dtype=float)
    for col in range(x.shape[0]):
        counts = _joint_counts_2d(x[col], xb, y, yb)
        px = counts.sum(axis=1)
        py = counts.sum(axis=0)
        nonzero_x = px > 0
        nonzero_y = py > 0
        nonzero_xy = counts > 0
        n_trials = x.shape[1]
        hx = np.log2(n_trials) - np.sum(px[nonzero_x] * np.log2(px[nonzero_x])) / n_trials
        hy = np.log2(n_trials) - np.sum(py[nonzero_y] * np.log2(py[nonzero_y])) / n_trials
        hxy = np.log2(n_trials) - np.sum(counts[nonzero_xy] * np.log2(counts[nonzero_xy])) / n_trials
        out[col] = hx + hy - hxy

    if bias:
        out -= mmbiasinfo(xb, yb, x.shape[1])
    return out


def calccmi_slice(x, xb, y, yb, z, zb, *, bias=False, validate=True, threads=None):
    del threads
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    zb = _validate_bin_count(zb, "zb")
    x = _as_fastinfo_discrete_matrix(x, xb, "x") if validate else _require_fastinfo_matrix_layout(x, "x")
    y = _as_fastinfo_discrete_vector(y, yb, "y") if validate else _require_fastinfo_vector_layout(y, "y")
    z = _as_fastinfo_discrete_vector(z, zb, "z") if validate else _require_fastinfo_vector_layout(z, "z")
    if x.shape[1] != y.size or x.shape[1] != z.size:
        raise ValueError("x must have shape [Nx, Ntrl], y and z must have length Ntrl.")
    return calccmi_slice_reference(x.T, xb, y, yb, z, zb, bias=bias, beta=0.0)


def eqpop(x, nb, *, validate=True, warn_on_ties=True):
    values = _validate_continuous_vector(x, "x") if validate else np.asarray(x, dtype=float).reshape(-1)
    nb = _validate_bin_count(nb, "nb")
    _warn_if_quantized(values, nb, warn_on_ties, "fastinfo.eqpop")
    order = np.argsort(values, kind="stable")
    labels_sorted = _eqpop_sorted_labels(values[order], nb)
    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels


def eqpop_sorted(x_sorted, nb, *, validate=True, warn_on_ties=True):
    values = _validate_continuous_vector(x_sorted, "x_sorted") if validate else np.asarray(x_sorted, dtype=float).reshape(-1)
    nb = _validate_bin_count(nb, "nb")
    _warn_if_quantized(values, nb, warn_on_ties, "fastinfo.eqpop_sorted")
    return _eqpop_sorted_labels(values, nb)


def eqpop_slice(x, nb, *, validate=True, warn_on_ties=True, threads=None):
    del threads
    values = _validate_fastinfo_continuous_matrix(x, "x") if validate else np.asarray(
        _require_fastinfo_matrix_layout(x, "x"), dtype=float
    )
    nb = _validate_bin_count(nb, "nb")
    out = np.full(values.shape, np.nan, dtype=float)
    failed = 0
    for col in range(values.shape[0]):
        try:
            out[col] = eqpop(values[col], nb, validate=False, warn_on_ties=False)
        except ValueError:
            failed += 1
    if warn_on_ties and failed > 0:
        warnings.warn(
            f"fastinfo.eqpop_slice could not form tie-consistent bins for {failed} page(s). "
            "Those page outputs were set to NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def eqpop_sorted_slice(x_sorted, nb, *, validate=True, warn_on_ties=True, threads=None):
    del threads
    values = _validate_fastinfo_continuous_matrix(x_sorted, "x_sorted") if validate else np.asarray(
        _require_fastinfo_matrix_layout(x_sorted, "x_sorted"), dtype=float
    )
    nb = _validate_bin_count(nb, "nb")
    out = np.full(values.shape, np.nan, dtype=float)
    failed = 0
    for col in range(values.shape[0]):
        try:
            out[col] = eqpop_sorted(values[col], nb, validate=validate, warn_on_ties=False)
        except ValueError:
            failed += 1
    if warn_on_ties and failed > 0:
        warnings.warn(
            f"fastinfo.eqpop_sorted_slice could not form tie-consistent bins for {failed} page(s). "
            "Those page outputs were set to NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def calcpairwiseinfo(x, xb, y, yb, *, bias=False, validate=True):
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    if validate:
        x = _validate_continuous_vector(x, "x")
        y = _as_fastinfo_discrete_vector(y, yb, "y")
    else:
        y = _require_fastinfo_vector_layout(y, "y")
    return calcpairwiseinfo_reference(x, xb, y, yb, bias=bias)


def calcpairwiseinfo_slice(x, xb, y, yb, *, bias=False, validate=True):
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    x = _validate_fastinfo_continuous_matrix(x, "x") if validate else np.asarray(
        _require_fastinfo_matrix_layout(x, "x"), dtype=float
    )
    y = _as_fastinfo_discrete_vector(y, yb, "y") if validate else _require_fastinfo_vector_layout(y, "y")
    if x.shape[1] != y.size:
        raise ValueError("x must have shape [Nx, Ntrl] and y must have length Ntrl.")
    return calcpairwiseinfo_slice_reference(x.T, xb, y, yb, bias=bias)


def set_threads(threads):
    del threads
    return None
