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


def _validate_bin_count(nbins, name):
    if not isinstance(nbins, (int, np.integer)) or nbins < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(nbins)


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
