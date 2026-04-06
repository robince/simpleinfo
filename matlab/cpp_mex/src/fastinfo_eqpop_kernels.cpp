#include "fastinfo_eqpop_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastinfo {

namespace {

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
