#include "fastinfo_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastinfo {

namespace {

constexpr double kLn2 = 0.693147180559945309417232121458176568;

struct EntropyLookup {
    double log2_n = 0.0;
    double inv_n = 0.0;
    std::vector<double> count_log2;
};

mwSize clamp_threads(mwSize requested) {
#ifdef _OPENMP
    if (requested == 0) {
        return static_cast<mwSize>(omp_get_max_threads());
    }
    return std::max<mwSize>(1, requested);
#else
    (void)requested;
    return 1;
#endif
}

EntropyLookup build_entropy_lookup(mwSize nTrials) {
    EntropyLookup lookup;
    if (nTrials == 0) {
        return lookup;
    }
    lookup.log2_n = std::log(static_cast<double>(nTrials)) / kLn2;
    lookup.inv_n = 1.0 / static_cast<double>(nTrials);
    lookup.count_log2.assign(nTrials + 1, 0.0);
    for (mwSize count = 2; count <= nTrials; ++count) {
        lookup.count_log2[count] = static_cast<double>(count) * std::log(static_cast<double>(count)) / kLn2;
    }
    return lookup;
}

double entropy_from_counts(const mwSize* counts, mwSize nCounts, const EntropyLookup& lookup) {
    if (lookup.count_log2.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (mwSize i = 0; i < nCounts; ++i) {
        const mwSize count = counts[i];
        sum += lookup.count_log2[count];
    }
    return lookup.log2_n - sum * lookup.inv_n;
}

double entropy_from_counts(const std::vector<mwSize>& counts, const EntropyLookup& lookup) {
    return entropy_from_counts(counts.data(), counts.size(), lookup);
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

std::uint64_t splitmix64_next(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void fisher_yates_shuffle_seeded(std::vector<mwSize>& values, std::uint64_t seed) {
    if (values.size() < 2) {
        return;
    }
    std::uint64_t state = seed;
    for (mwSize i = values.size() - 1; i > 0; --i) {
        const mwSize j = static_cast<mwSize>(splitmix64_next(state) % static_cast<std::uint64_t>(i + 1));
        std::swap(values[i], values[j]);
    }
}

double calc_info_buffered(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    const EntropyLookup& lookup,
    mwSize* px,
    mwSize* py,
    mwSize* pxy) {
    std::fill(px, px + xm, 0);
    std::fill(py, py + ym, 0);
    std::fill(pxy, pxy + xm * ym, 0);

    const mwSize nTrials = static_cast<mwSize>(lookup.count_log2.size() - 1);
    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = x[i];
        const mwSize yi = y[i];
        ++px[xi];
        ++py[yi];
        ++pxy[xi * ym + yi];
    }

    return entropy_from_counts(px, xm, lookup)
        + entropy_from_counts(py, ym, lookup)
        - entropy_from_counts(pxy, xm * ym, lookup);
}

double calc_info_with_precomputed_hy(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    const EntropyLookup& lookup,
    double hy,
    mwSize* px,
    mwSize* pxy) {
    std::fill(px, px + xm, 0);
    std::fill(pxy, pxy + xm * ym, 0);
    const mwSize nTrials = static_cast<mwSize>(lookup.count_log2.size() - 1);
    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = x[i];
        const mwSize yi = y[i];
        ++px[xi];
        ++pxy[xi * ym + yi];
    }
    return entropy_from_counts(px, xm, lookup) + hy - entropy_from_counts(pxy, xm * ym, lookup);
}

double calc_cmi_buffered(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    const mwSize* z,
    mwSize zm,
    const EntropyLookup& lookup,
    mwSize* pz,
    mwSize* pxz,
    mwSize* pyz,
    mwSize* pxyz) {
    std::fill(pz, pz + zm, 0);
    std::fill(pxz, pxz + xm * zm, 0);
    std::fill(pyz, pyz + ym * zm, 0);
    std::fill(pxyz, pxyz + xm * ym * zm, 0);

    const mwSize nTrials = static_cast<mwSize>(lookup.count_log2.size() - 1);
    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = x[i];
        const mwSize yi = y[i];
        const mwSize zi = z[i];
        const mwSize yz = yi * zm + zi;
        ++pz[zi];
        ++pxz[xi * zm + zi];
        ++pyz[yz];
        ++pxyz[xi * (ym * zm) + yz];
    }

    return entropy_from_counts(pxz, xm * zm, lookup)
        + entropy_from_counts(pyz, ym * zm, lookup)
        - entropy_from_counts(pxyz, xm * ym * zm, lookup)
        - entropy_from_counts(pz, zm, lookup);
}

void plan_eqpop_bins_from_sorted(
    const double* xSorted,
    mwSize nSamples,
    mwSize nBins,
    std::vector<mwSize>& sortedLabels) {
    require(nBins > 0, "Number of bins must be positive");
    require(nSamples > 0, "Input must contain at least one sample");

    std::vector<mwSize> groupOffsets;
    groupOffsets.reserve(nSamples + 1);
    groupOffsets.push_back(0);
    for (mwSize i = 0; i < nSamples; ++i) {
        require(std::isfinite(xSorted[i]), "Input must contain only finite values");
        if (i > 0) {
            require(xSorted[i] >= xSorted[i - 1], "Input to eqpop_sorted must be sorted in nondecreasing order");
            if (xSorted[i] != xSorted[i - 1]) {
                groupOffsets.push_back(i);
            }
        }
    }
    groupOffsets.push_back(nSamples);

    const mwSize nGroups = groupOffsets.size() - 1;
    if (nGroups < nBins) {
        throw std::runtime_error(
            "Cannot form the requested number of equal-population bins without splitting tied values. "
            "If the input is discrete or strongly quantized, use rebin instead.");
    }

    const double idealSize = static_cast<double>(nSamples) / static_cast<double>(nBins);
    const double inf = std::numeric_limits<double>::infinity();
    const mwSize stride = nGroups + 1;
    std::vector<double> dp((nBins + 1) * stride, inf);
    std::vector<mwSize> parent((nBins + 1) * stride, std::numeric_limits<mwSize>::max());
    auto index = [stride](mwSize b, mwSize g) { return b * stride + g; };

    dp[index(0, 0)] = 0.0;
    parent[index(0, 0)] = 0;

    for (mwSize b = 1; b <= nBins; ++b) {
        const mwSize minGroupsUsed = b;
        const mwSize maxGroupsUsed = nGroups - (nBins - b);
        for (mwSize g = minGroupsUsed; g <= maxGroupsUsed; ++g) {
            double bestCost = inf;
            mwSize bestParent = std::numeric_limits<mwSize>::max();
            for (mwSize prev = b - 1; prev < g; ++prev) {
                const double prefixCost = dp[index(b - 1, prev)];
                if (!std::isfinite(prefixCost)) {
                    continue;
                }
                const mwSize count = groupOffsets[g] - groupOffsets[prev];
                const double deviation = static_cast<double>(count) - idealSize;
                const double cost = prefixCost + deviation * deviation;
                if (cost < bestCost) {
                    bestCost = cost;
                    bestParent = prev;
                }
            }
            dp[index(b, g)] = bestCost;
            parent[index(b, g)] = bestParent;
        }
    }

    require(
        std::isfinite(dp[index(nBins, nGroups)]),
        "Failed to construct a tie-consistent equal-population partition");

    std::vector<mwSize> groupCuts(nBins + 1, 0);
    groupCuts[nBins] = nGroups;
    mwSize g = nGroups;
    for (mwSize b = nBins; b > 0; --b) {
        const mwSize prev = parent[index(b, g)];
        require(prev != std::numeric_limits<mwSize>::max(), "Failed to reconstruct equal-population partition");
        groupCuts[b - 1] = prev;
        g = prev;
    }

    sortedLabels.assign(nSamples, 0);
    for (mwSize bin = 0; bin < nBins; ++bin) {
        const mwSize start = groupOffsets[groupCuts[bin]];
        const mwSize stop = groupOffsets[groupCuts[bin + 1]];
        std::fill(sortedLabels.begin() + static_cast<mwSignedIndex>(start),
                  sortedLabels.begin() + static_cast<mwSignedIndex>(stop),
                  bin);
    }
}

}  // namespace

double calc_info(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    mwSize nTrials) {
    const EntropyLookup lookup = build_entropy_lookup(nTrials);
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> py(ym, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    return calc_info_buffered(x, xm, y, ym, lookup, px.data(), py.data(), pxy.data());
}

void calc_info_matched(
    const mwSize* x,
    const mwSize* y,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    mwSize ym,
    mwSize threadCount,
    double* output) {
    const mwSize nThreads = clamp_threads(threadCount);
    const EntropyLookup lookup = build_entropy_lookup(nTrials);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> px(xm, 0);
        std::vector<mwSize> py(ym, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const mwSize offset = static_cast<mwSize>(col) * nTrials;
            output[col] = calc_info_buffered(
                x + offset, xm, y + offset, ym, lookup, px.data(), py.data(), pxy.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> py(ym, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const mwSize offset = col * nTrials;
        output[col] = calc_info_buffered(
            x + offset, xm, y + offset, ym, lookup, px.data(), py.data(), pxy.data());
    }
#endif
}

double calc_cmi(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    const mwSize* z,
    mwSize zm,
    mwSize nTrials) {
    const EntropyLookup lookup = build_entropy_lookup(nTrials);
    std::vector<mwSize> pz(zm, 0);
    std::vector<mwSize> pxz(xm * zm, 0);
    std::vector<mwSize> pyz(ym * zm, 0);
    std::vector<mwSize> pxyz(xm * ym * zm, 0);
    return calc_cmi_buffered(
        x, xm, y, ym, z, zm, lookup, pz.data(), pxz.data(), pyz.data(), pxyz.data());
}

void calc_cond_cmi(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    const mwSize* z,
    mwSize zm,
    const mwSize* k,
    mwSize km,
    mwSize nTrials,
    double* total,
    double* contributions) {
    std::vector<mwSize> pz(zm, 0);
    std::vector<mwSize> pxz(xm * zm, 0);
    std::vector<mwSize> pyz(ym * zm, 0);
    std::vector<mwSize> pxyz(xm * ym * zm, 0);
    std::vector<mwSize> pzk(km * zm, 0);
    std::vector<mwSize> pxzk(km * xm * zm, 0);
    std::vector<mwSize> pyzk(km * ym * zm, 0);
    std::vector<mwSize> pxyzk(km * xm * ym * zm, 0);
    const EntropyLookup lookup = build_entropy_lookup(nTrials);

    const mwSize yzStride = ym * zm;
    const mwSize xzStride = xm * zm;
    const mwSize xyzStride = xm * ym * zm;

    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = x[i];
        const mwSize yi = y[i];
        const mwSize zi = z[i];
        const mwSize ki = k[i];
        const mwSize yz = yi * zm + zi;
        const mwSize xz = xi * zm + zi;
        const mwSize xyz = xi * yzStride + yz;
        ++pz[zi];
        ++pxz[xz];
        ++pyz[yz];
        ++pxyz[xyz];

        ++pzk[ki * zm + zi];
        ++pxzk[ki * xzStride + xz];
        ++pyzk[ki * yzStride + yz];
        ++pxyzk[ki * xyzStride + xyz];
    }

    *total = entropy_from_counts(pxz.data(), pxz.size(), lookup)
        + entropy_from_counts(pyz.data(), pyz.size(), lookup)
        - entropy_from_counts(pxyz.data(), pxyz.size(), lookup)
        - entropy_from_counts(pz.data(), pz.size(), lookup);

    for (mwSize ki = 0; ki < km; ++ki) {
        const mwSize pzOffset = ki * zm;
        const mwSize xzOffset = ki * xzStride;
        const mwSize yzOffset = ki * yzStride;
        const mwSize xyzOffset = ki * xyzStride;
        contributions[ki] = entropy_from_counts(pxzk.data() + xzOffset, xzStride, lookup)
            + entropy_from_counts(pyzk.data() + yzOffset, yzStride, lookup)
            - entropy_from_counts(pxyzk.data() + xyzOffset, xyzStride, lookup)
            - entropy_from_counts(pzk.data() + pzOffset, zm, lookup);
    }
}

void calc_cmi_slice(
    const mwSize* x,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    const mwSize* z,
    mwSize zm,
    mwSize threadCount,
    double* output) {
    const mwSize nThreads = clamp_threads(threadCount);
    const EntropyLookup lookup = build_entropy_lookup(nTrials);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> pz(zm, 0);
        std::vector<mwSize> pxz(xm * zm, 0);
        std::vector<mwSize> pyz(ym * zm, 0);
        std::vector<mwSize> pxyz(xm * ym * zm, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const mwSize* xCol = x + static_cast<mwSize>(col) * nTrials;
            output[col] = calc_cmi_buffered(
                xCol, xm, y, ym, z, zm, lookup, pz.data(), pxz.data(), pyz.data(), pxyz.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> pz(zm, 0);
    std::vector<mwSize> pxz(xm * zm, 0);
    std::vector<mwSize> pyz(ym * zm, 0);
    std::vector<mwSize> pxyz(xm * ym * zm, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const mwSize* xCol = x + col * nTrials;
        output[col] = calc_cmi_buffered(
            xCol, xm, y, ym, z, zm, lookup, pz.data(), pxz.data(), pyz.data(), pxyz.data());
    }
#endif
}

void calc_info_slice(
    const mwSize* x,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    mwSize threadCount,
    double* output) {
    const mwSize nThreads = clamp_threads(threadCount);
    const EntropyLookup lookup = build_entropy_lookup(nTrials);
    std::vector<mwSize> py(ym, 0);
    for (mwSize i = 0; i < nTrials; ++i) {
        ++py[y[i]];
    }
    const double hy = entropy_from_counts(py.data(), py.size(), lookup);

#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> px(xm, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const mwSize* xCol = x + static_cast<mwSize>(col) * nTrials;
            output[col] = calc_info_with_precomputed_hy(
                xCol, xm, y, ym, lookup, hy, px.data(), pxy.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const mwSize* xCol = x + col * nTrials;
        output[col] = calc_info_with_precomputed_hy(
            xCol, xm, y, ym, lookup, hy, px.data(), pxy.data());
    }
#endif
}

void calc_info_perm(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    mwSize nTrials,
    mwSize nPerm,
    mwSize threadCount,
    std::uint64_t seed,
    double* output) {
    const mwSize nThreads = clamp_threads(threadCount);
    const EntropyLookup lookup = build_entropy_lookup(nTrials);

#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> xsh(nTrials, 0);
        std::vector<mwSize> px(xm, 0);
        std::vector<mwSize> py(ym, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex perm = 0; perm < static_cast<mwSignedIndex>(nPerm); ++perm) {
            std::copy(x, x + nTrials, xsh.begin());
            fisher_yates_shuffle_seeded(xsh, splitmix64(seed + static_cast<std::uint64_t>(perm)));
            output[perm] = calc_info_buffered(
                xsh.data(), xm, y, ym, lookup, px.data(), py.data(), pxy.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> xsh(nTrials, 0);
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> py(ym, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize perm = 0; perm < nPerm; ++perm) {
        std::copy(x, x + nTrials, xsh.begin());
        fisher_yates_shuffle_seeded(xsh, splitmix64(seed + static_cast<std::uint64_t>(perm)));
        output[perm] = calc_info_buffered(
            xsh.data(), xm, y, ym, lookup, px.data(), py.data(), pxy.data());
    }
#endif
}

void calc_info_perm_slice(
    const mwSize* x,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    mwSize nPerm,
    mwSize threadCount,
    std::uint64_t seed,
    double* output) {
    const mwSize nThreads = clamp_threads(threadCount);
    const EntropyLookup lookup = build_entropy_lookup(nTrials);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> xsh(nTrials, 0);
        std::vector<mwSize> px(xm, 0);
        std::vector<mwSize> py(ym, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const mwSize* xCol = x + static_cast<mwSize>(col) * nTrials;
            double* outCol = output + static_cast<mwSize>(col) * nPerm;
            const std::uint64_t colSeed = splitmix64(seed + static_cast<std::uint64_t>(col));
            for (mwSize perm = 0; perm < nPerm; ++perm) {
                std::copy(xCol, xCol + nTrials, xsh.begin());
                const std::uint64_t permSeed = splitmix64(colSeed + static_cast<std::uint64_t>(perm));
                fisher_yates_shuffle_seeded(xsh, permSeed);
                outCol[perm] = calc_info_buffered(
                    xsh.data(), xm, y, ym, lookup, px.data(), py.data(), pxy.data());
            }
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> xsh(nTrials, 0);
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> py(ym, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const mwSize* xCol = x + col * nTrials;
        double* outCol = output + col * nPerm;
        const std::uint64_t colSeed = splitmix64(seed + static_cast<std::uint64_t>(col));
        for (mwSize perm = 0; perm < nPerm; ++perm) {
            std::copy(xCol, xCol + nTrials, xsh.begin());
            const std::uint64_t permSeed = splitmix64(colSeed + static_cast<std::uint64_t>(perm));
            fisher_yates_shuffle_seeded(xsh, permSeed);
            outCol[perm] = calc_info_buffered(
                xsh.data(), xm, y, ym, lookup, px.data(), py.data(), pxy.data());
        }
    }
#endif
}

void eqpop_sorted(
    const double* xSorted,
    mwSize nSamples,
    mwSize nBins,
    int32_t* output) {
    std::vector<mwSize> labels;
    plan_eqpop_bins_from_sorted(xSorted, nSamples, nBins, labels);
    for (mwSize i = 0; i < nSamples; ++i) {
        output[i] = static_cast<int32_t>(labels[i]);
    }
}

void eqpop(
    const double* x,
    mwSize nSamples,
    mwSize nBins,
    int32_t* output) {
    std::vector<std::pair<double, mwSize>> ordered(nSamples);
    for (mwSize i = 0; i < nSamples; ++i) {
        require(std::isfinite(x[i]), "Input must contain only finite values");
        ordered[i] = std::make_pair(x[i], i);
    }

    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first < rhs.first) {
            return true;
        }
        if (lhs.first > rhs.first) {
            return false;
        }
        return lhs.second < rhs.second;
    });

    std::vector<double> sortedValues(nSamples, 0.0);
    for (mwSize i = 0; i < nSamples; ++i) {
        sortedValues[i] = ordered[i].first;
    }

    std::vector<mwSize> sortedLabels;
    plan_eqpop_bins_from_sorted(sortedValues.data(), nSamples, nBins, sortedLabels);

    for (mwSize i = 0; i < nSamples; ++i) {
        output[ordered[i].second] = static_cast<int32_t>(sortedLabels[i]);
    }
}

void eqpop_sorted_slice(
    const double* xSorted,
    mwSize nRows,
    mwSize nCols,
    mwSize nBins,
    mwSize threadCount,
    double* output) {
    const mwSize nThreads = clamp_threads(threadCount);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> labels;
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const double* xCol = xSorted + static_cast<mwSize>(col) * nRows;
            double* outCol = output + static_cast<mwSize>(col) * nRows;
            try {
                plan_eqpop_bins_from_sorted(xCol, nRows, nBins, labels);
                for (mwSize i = 0; i < nRows; ++i) {
                    outCol[i] = static_cast<double>(labels[i]);
                }
            } catch (...) {
                std::fill(outCol, outCol + nRows, nan_value());
            }
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> labels;
    for (mwSize col = 0; col < nCols; ++col) {
        const double* xCol = xSorted + col * nRows;
        double* outCol = output + col * nRows;
        try {
            plan_eqpop_bins_from_sorted(xCol, nRows, nBins, labels);
            for (mwSize i = 0; i < nRows; ++i) {
                outCol[i] = static_cast<double>(labels[i]);
            }
        } catch (...) {
            std::fill(outCol, outCol + nRows, nan_value());
        }
    }
#endif
}

void eqpop_slice(
    const double* x,
    mwSize nRows,
    mwSize nCols,
    mwSize nBins,
    mwSize threadCount,
    double* output) {
    const mwSize nThreads = clamp_threads(threadCount);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<std::pair<double, mwSize>> ordered(nRows);
        std::vector<double> sortedValues(nRows, 0.0);
        std::vector<mwSize> sortedLabels;
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const double* xCol = x + static_cast<mwSize>(col) * nRows;
            double* outCol = output + static_cast<mwSize>(col) * nRows;
            try {
                for (mwSize i = 0; i < nRows; ++i) {
                    require(std::isfinite(xCol[i]), "Input must contain only finite values");
                    ordered[i] = std::make_pair(xCol[i], i);
                }
                std::stable_sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
                    if (lhs.first < rhs.first) {
                        return true;
                    }
                    if (lhs.first > rhs.first) {
                        return false;
                    }
                    return lhs.second < rhs.second;
                });
                for (mwSize i = 0; i < nRows; ++i) {
                    sortedValues[i] = ordered[i].first;
                }
                plan_eqpop_bins_from_sorted(sortedValues.data(), nRows, nBins, sortedLabels);
                for (mwSize i = 0; i < nRows; ++i) {
                    outCol[ordered[i].second] = static_cast<double>(sortedLabels[i]);
                }
            } catch (...) {
                std::fill(outCol, outCol + nRows, nan_value());
            }
        }
    }
#else
    (void)nThreads;
    std::vector<std::pair<double, mwSize>> ordered(nRows);
    std::vector<double> sortedValues(nRows, 0.0);
    std::vector<mwSize> sortedLabels;
    for (mwSize col = 0; col < nCols; ++col) {
        const double* xCol = x + col * nRows;
        double* outCol = output + col * nRows;
        try {
            for (mwSize i = 0; i < nRows; ++i) {
                require(std::isfinite(xCol[i]), "Input must contain only finite values");
                ordered[i] = std::make_pair(xCol[i], i);
            }
            std::stable_sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.first < rhs.first) {
                    return true;
                }
                if (lhs.first > rhs.first) {
                    return false;
                }
                return lhs.second < rhs.second;
            });
            for (mwSize i = 0; i < nRows; ++i) {
                sortedValues[i] = ordered[i].first;
            }
            plan_eqpop_bins_from_sorted(sortedValues.data(), nRows, nBins, sortedLabels);
            for (mwSize i = 0; i < nRows; ++i) {
                outCol[ordered[i].second] = static_cast<double>(sortedLabels[i]);
            }
        } catch (...) {
            std::fill(outCol, outCol + nRows, nan_value());
        }
    }
#endif
}

}  // namespace fastinfo
