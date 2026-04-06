"""Core estimators for simple discrete information-theoretic quantities."""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from scipy.stats import chi2

rng = default_rng()


def entropy(p):
    """Entropy of a probability distribution."""
    probabilities = np.asarray(p, dtype=float)
    nonzero = probabilities > 0
    return float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))


def numbase2dec(x, b):
    """Convert base-b words arranged as [M, N] into decimal integers."""
    b = _validate_bin_count(b, "b")
    if b < 2:
        raise ValueError("b must be an integer greater than or equal to 2.")
    digits = np.asarray(x)
    if digits.size == 0:
        raise ValueError("x must contain at least one digit.")
    if digits.ndim == 1:
        digits = digits.reshape(-1, 1)
    elif digits.ndim != 2:
        raise ValueError("x must be a one- or two-dimensional array.")

    rounded = np.rint(digits)
    if not np.allclose(digits, rounded) or np.any(rounded < 0) or np.any(rounded >= b):
        raise ValueError("x must contain integer digits in the range 0:(b-1).")

    digits = rounded.astype(np.int64)
    powers = np.power(int(b), np.arange(digits.shape[0] - 1, -1, -1, dtype=np.int64), dtype=np.int64)
    return powers @ digits


def numdec2base(d, b, m=None):
    """Convert decimal integers into base-b words arranged as [M, N]."""
    b = _validate_bin_count(b, "b")
    if b < 2:
        raise ValueError("b must be an integer greater than or equal to 2.")
    values = np.asarray(d)
    if values.size == 0:
        raise ValueError("d must contain at least one integer.")

    flat = np.rint(values.reshape(-1))
    if not np.allclose(values.reshape(-1), flat) or np.any(flat < 0):
        raise ValueError("d must contain non-negative integers.")

    flat = flat.astype(np.int64)
    required_digits = _required_base_digits(int(np.max(flat)), int(b))
    if m is None:
        m = required_digits
    else:
        m = _validate_bin_count(m, "m")
        if m < required_digits:
            raise ValueError("m is too small to represent the largest value in d.")

    powers = np.power(int(b), np.arange(m - 1, -1, -1, dtype=np.int64), dtype=np.int64)
    return ((flat[np.newaxis, :] // powers[:, np.newaxis]) % int(b)).astype(np.int64, copy=False)


def _validate_bin_count(nbins, name):
    if not isinstance(nbins, (int, np.integer)) or nbins < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(nbins)


def _required_base_digits(max_value, base):
    max_value = int(max_value)
    if max_value < 0:
        raise ValueError("max_value must be non-negative.")
    if int(base) < 2:
        raise ValueError("base must be an integer greater than or equal to 2.")
    digits = 1
    while max_value >= base:
        max_value //= base
        digits += 1
    return digits


def _as_discrete_samples(values, nbins, name):
    samples = np.asarray(values).reshape(-1)
    if samples.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.int64, copy=False)
    else:
        rounded = np.rint(samples)
        if not np.allclose(samples, rounded):
            raise ValueError(f"{name} must contain integer-valued samples.")
        samples = rounded.astype(np.int64)

    if np.any(samples < 0) or np.any(samples >= nbins):
        raise ValueError(f"{name} must take values in [0, {nbins - 1}].")

    return samples


def _validate_beta(beta):
    beta = float(beta)
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    return beta


def _joint_counts_2d(x, xb, y, yb):
    indices = x * yb + y
    return np.bincount(indices, minlength=xb * yb).reshape(xb, yb)


def _joint_counts_3d(x, xb, y, yb, z, zb):
    indices = (x * yb + y) * zb + z
    return np.bincount(indices, minlength=xb * yb * zb).reshape(xb, yb, zb)


def _joint_probability(counts, n_trials, beta):
    return (counts + beta) / (float(n_trials) + beta * counts.size)


def calcinfo(x, xb, y, yb, bias=True, calc_p=False, beta=0.0):
    """Calculate mutual information between discrete data sets x and y."""
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    beta = _validate_beta(beta)

    x = _as_discrete_samples(x, xb, "x")
    y = _as_discrete_samples(y, yb, "y")
    if x.size != y.size:
        raise ValueError("calcinfo: Number of trials must match.")

    counts = _joint_counts_2d(x, xb, y, yb)
    pxy = _joint_probability(counts, x.size, beta)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    inobc = entropy(px) + entropy(py) - entropy(pxy)
    info = inobc - mmbiasinfo(xb, yb, x.size) if bias else inobc

    if calc_p:
        p_value = chi2.sf(2 * x.size * np.log(2) * inobc, (xb - 1) * (yb - 1))
        return info, p_value
    return info


def mmbiasinfo(nx, ny, ntrl):
    """Miller-Madow bias estimate for binned mutual information."""
    return (nx - 1) * (ny - 1) / (2 * ntrl * np.log(2))


def calcpmi(x, xb, y, yb, weighted=False, calc_p=False, beta=0.0):
    """Calculate pointwise mutual information between discrete x and y."""
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    beta = _validate_beta(beta)

    x = _as_discrete_samples(x, xb, "x")
    y = _as_discrete_samples(y, yb, "y")
    if x.size != y.size:
        raise ValueError("calcpmi: Number of trials must match.")

    counts = _joint_counts_2d(x, xb, y, yb)
    pxy = _joint_probability(counts, x.size, beta)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    pxy_ind = px * py

    pmi = np.zeros_like(pxy)
    observed = pxy > 0
    pmi[observed] = np.log2(pxy[observed] / pxy_ind[observed])

    weighted_pmi = pxy * pmi
    info = float(np.sum(weighted_pmi))

    if weighted:
        pmi = weighted_pmi

    if calc_p:
        p_value = chi2.sf(2 * x.size * np.log(2) * info, (xb - 1) * (yb - 1))
        return info, pmi, p_value
    return info, pmi


def calcsmi(x, xb, y, yb, weighted=False, beta=0.0):
    """Calculate sample-wise mutual information between discrete x and y."""
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")

    x = _as_discrete_samples(x, xb, "x")
    y = _as_discrete_samples(y, yb, "y")
    if x.size != y.size:
        raise ValueError("calcsmi: Number of trials must match.")

    info, pmi = calcpmi(x, xb, y, yb, weighted=weighted, beta=beta)
    smi = pmi[x, y]
    return info, smi


def calccmi(x, xb, y, yb, z, zb, bias=True, calc_p=False, beta=0.0):
    """Calculate conditional mutual information I(X;Y|Z)."""
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    zb = _validate_bin_count(zb, "zb")
    beta = _validate_beta(beta)

    x = _as_discrete_samples(x, xb, "x")
    y = _as_discrete_samples(y, yb, "y")
    z = _as_discrete_samples(z, zb, "z")
    if x.size != y.size or x.size != z.size:
        raise ValueError("calccmi: Number of trials must match.")

    counts = _joint_counts_3d(x, xb, y, yb, z, zb)
    pxyz = _joint_probability(counts, x.size, beta)
    pxz = np.sum(pxyz, axis=1)
    pyz = np.sum(pxyz, axis=0)
    pz = np.sum(pxyz, axis=(0, 1))

    inobc = entropy(pxz) + entropy(pyz) - entropy(pxyz) - entropy(pz)
    info = inobc - mmbiascmi(xb, yb, zb, x.size) if bias else inobc

    if calc_p:
        p_value = chi2.sf(2 * x.size * np.log(2) * inobc, zb * (xb - 1) * (yb - 1))
        return info, p_value
    return info


def mmbiascmi(nx, ny, nz, ntrl):
    """Miller-Madow bias estimate for binned conditional mutual information."""
    return nz * (nx - 1) * (ny - 1) / (2 * ntrl * np.log(2))


def calcinfoperm(x, xb, y, yb, nperm, bias=True, beta=0.0):
    """Permutation null samples for the hypothesis that X and Y are independent."""
    if not isinstance(nperm, (int, np.integer)) or nperm < 0:
        raise ValueError("nperm must be a non-negative integer.")

    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")

    x = _as_discrete_samples(x, xb, "x")
    y = _as_discrete_samples(y, yb, "y")
    if x.size != y.size:
        raise ValueError("calcinfoperm: Number of trials must match.")

    iperm = np.zeros(int(nperm), dtype=float)
    for permutation_index in range(int(nperm)):
        idx = rng.permutation(x.size)
        iperm[permutation_index] = calcinfo(x[idx], xb, y, yb, bias=False, beta=beta)

    if bias:
        iperm = iperm - mmbiasinfo(xb, yb, x.size)

    return iperm


def calcinfomatched(x, xb, y, yb, bias=True, beta=0.0):
    """Column-wise MI for matched X/Y pages."""
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be two-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    out = np.zeros(x.shape[1], dtype=float)
    for col in range(x.shape[1]):
        out[col] = calcinfo(x[:, col], xb, y[:, col], yb, bias=bias, beta=beta)
    return out


def calccondcmi(x, xb, y, yb, z, zb, k, kb):
    """Conditional MI plus weighted per-K contributions under global normalization."""
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    zb = _validate_bin_count(zb, "zb")
    kb = _validate_bin_count(kb, "kb")

    x = _as_discrete_samples(x, xb, "x")
    y = _as_discrete_samples(y, yb, "y")
    z = _as_discrete_samples(z, zb, "z")
    k = _as_discrete_samples(k, kb, "k")
    if x.size != y.size or x.size != z.size or x.size != k.size:
        raise ValueError("calccondcmi: Number of trials must match.")

    ntrl = x.size
    counts = _joint_counts_3d(x, xb, y, yb, z, zb) / float(ntrl)
    total = (
        entropy(np.sum(counts, axis=1))
        + entropy(np.sum(counts, axis=0))
        - entropy(counts)
        - entropy(np.sum(np.sum(counts, axis=0), axis=0))
    )

    contributions = np.zeros(kb, dtype=float)
    for ki in range(kb):
        mask = k == ki
        if not np.any(mask):
            continue
        counts_k = _joint_counts_3d(x[mask], xb, y[mask], yb, z[mask], zb) / float(ntrl)
        contributions[ki] = (
            entropy(np.sum(counts_k, axis=1))
            + entropy(np.sum(counts_k, axis=0))
            - entropy(counts_k)
            - entropy(np.sum(np.sum(counts_k, axis=0), axis=0))
        )
    return total, contributions


def calccmi_slice(x, xb, y, yb, z, zb, bias=True, beta=0.0):
    """Column-wise conditional MI for matched X pages."""
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    zb = _validate_bin_count(zb, "zb")
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")
    y = np.asarray(y).reshape(-1)
    z = np.asarray(z).reshape(-1)
    if x.shape[0] != y.size or x.shape[0] != z.size:
        raise ValueError("x must have shape [Ntrl, Nx], y and z must have length Ntrl.")

    out = np.zeros(x.shape[1], dtype=float)
    for col in range(x.shape[1]):
        out[col] = calccmi(x[:, col], xb, y, yb, z, zb, bias=bias, beta=beta)
    return out


def calcinfoperm_slice(x, xb, y, yb, nperm, bias=True, beta=0.0):
    """Permutation MI null samples for each X page against a fixed Y."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")
    y = np.asarray(y).reshape(-1)
    if x.shape[0] != y.size:
        raise ValueError("x must have shape [Ntrl, Nx] and y must have length Ntrl.")

    out = np.zeros((int(nperm), x.shape[1]), dtype=float)
    for col in range(x.shape[1]):
        out[:, col] = calcinfoperm(x[:, col], xb, y, yb, nperm, bias=bias, beta=beta)
    return out


def calcpairwiseinfo(x, xb, y, yb, bias=False):
    """Pairwise binary MI after pair-specific equal-population binning."""
    xb = _validate_bin_count(xb, "xb")
    yb = _validate_bin_count(yb, "yb")
    x = np.asarray(x, dtype=float).reshape(-1)
    y = _as_discrete_samples(y, yb, "y")
    if x.size != y.size:
        raise ValueError("calcpairwiseinfo: Number of trials must match.")

    order = np.argsort(x, kind="stable")
    xs = x[order]
    ys = y[order]
    pair_values = []
    for yi in range(yb - 1):
        iidx = ys == yi
        for yj in range(yi + 1, yb):
            mask = iidx | (ys == yj)
            px = xs[mask]
            py = np.where(ys[mask] == yi, 0, 1)
            qpx = _eqpop_sorted_labels_reference(px, xb)
            pair_values.append(calcinfo(qpx, xb, py, 2, bias=bias, beta=0.0))
    return np.asarray(pair_values, dtype=float)


def calcpairwiseinfo_slice(x, xb, y, yb, bias=False):
    """Pairwise binary MI for each X page."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")
    y = np.asarray(y).reshape(-1)
    if x.shape[0] != y.size:
        raise ValueError("x must have shape [Ntrl, Nx] and y must have length Ntrl.")

    out = np.zeros((yb * (yb - 1) // 2, x.shape[1]), dtype=float)
    for col in range(x.shape[1]):
        out[:, col] = calcpairwiseinfo(x[:, col], xb, y, yb, bias=bias)
    return out


def _eqpop_sorted_labels_reference(sorted_values, nb):
    sorted_values = np.asarray(sorted_values, dtype=float).reshape(-1)
    nb = _validate_bin_count(nb, "nb")
    if sorted_values.size < nb:
        raise ValueError("nb cannot exceed the number of samples.")
    if np.any(np.diff(sorted_values) < 0):
        raise ValueError("sorted_values must be sorted in nondecreasing order.")

    group_starts = np.concatenate((
        np.array([0], dtype=np.int64),
        np.nonzero(np.diff(sorted_values) != 0)[0].astype(np.int64) + 1,
        np.array([sorted_values.size], dtype=np.int64),
    ))
    n_groups = group_starts.size - 1
    if n_groups < nb:
        raise ValueError(
            "Cannot form the requested number of equal-population bins without splitting tied values."
        )

    ideal = sorted_values.size / float(nb)
    dp = np.full((nb + 1, n_groups + 1), np.inf, dtype=float)
    parent = np.full((nb + 1, n_groups + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for b in range(1, nb + 1):
        min_groups_used = b
        max_groups_used = n_groups - (nb - b)
        for g in range(min_groups_used, max_groups_used + 1):
            for prev in range(b - 1, g):
                prefix = dp[b - 1, prev]
                if not np.isfinite(prefix):
                    continue
                count = group_starts[g] - group_starts[prev]
                deviation = count - ideal
                cost = prefix + deviation * deviation
                if cost < dp[b, g]:
                    dp[b, g] = cost
                    parent[b, g] = prev

    cuts = np.zeros(nb + 1, dtype=np.int64)
    cuts[nb] = n_groups
    g = n_groups
    for b in range(nb, 0, -1):
        prev = parent[b, g]
        if prev < 0:
            raise ValueError("Failed to reconstruct equal-population partition.")
        cuts[b - 1] = prev
        g = prev

    labels = np.zeros(sorted_values.size, dtype=np.int64)
    for bin_index in range(nb):
        start = group_starts[cuts[bin_index]]
        stop = group_starts[cuts[bin_index + 1]]
        labels[start:stop] = bin_index
    return labels


def eqpopbin(x, nb, return_edges=False):
    """Approximate equal-population binning that matches the MATLAB implementation."""
    nb = _validate_bin_count(nb, "nb")

    values = np.asarray(x)
    if values.size == 0:
        raise ValueError("x must contain at least one sample.")

    flat_values = values.reshape(-1)
    sorted_values = np.sort(flat_values)
    n_samples = sorted_values.size
    numel_bin = n_samples // nb
    if numel_bin == 0:
        raise ValueError("nb cannot exceed the number of samples.")

    remainder = n_samples - (numel_bin * nb)
    indices = np.arange(0, numel_bin * nb, numel_bin, dtype=int)
    indices[:remainder] += np.arange(remainder)
    indices[remainder:] += remainder

    edges = np.empty(nb + 1, dtype=np.result_type(flat_values, np.float64))
    edges[:nb] = sorted_values[indices]
    edges[nb] = sorted_values[-1] + 1

    xb = np.searchsorted(edges, flat_values, side="right") - 1
    xb = xb.reshape(values.shape)

    if return_edges:
        return xb, edges
    return xb


def rebin(x, nb):
    """Rebin an integer sequence by iteratively merging the smallest neighboring bins."""
    nb = _validate_bin_count(nb, "nb")

    values = np.asarray(x)
    if values.size == 0:
        raise ValueError("x must contain at least one sample.")

    flat_values = values.reshape(-1)
    rounded = np.rint(flat_values)
    if not np.allclose(flat_values, rounded) or np.any(rounded < 0):
        raise ValueError("Input must be positive integers")

    flat_values = rounded.astype(np.int64)
    if flat_values.max() < nb:
        return flat_values.reshape(values.shape)

    counts = list(np.bincount(flat_values))
    labels = list(np.arange(len(counts)))
    nbins = len(counts)
    rebinned = flat_values.copy()

    def merge_bins(source, target):
        nonlocal nbins
        counts[target] += counts[source]
        rebinned[rebinned == labels[source]] = labels[target]
        del labels[source]
        del counts[source]
        nbins -= 1

    while nbins > nb:
        sorted_indices = np.argsort(np.asarray(counts))
        smallest = int(sorted_indices[0])
        if smallest == 0:
            merge_bins(smallest, 1)
        elif smallest == nbins - 1:
            merge_bins(smallest, smallest - 1)
        else:
            neighbors = np.array([smallest - 1, smallest + 1])
            target = neighbors[np.argmin([counts[smallest - 1], counts[smallest + 1]])]
            merge_bins(smallest, int(target))

    for new_label, old_label in enumerate(labels):
        if new_label != old_label:
            rebinned[rebinned == old_label] = new_label

    return rebinned.reshape(values.shape)
