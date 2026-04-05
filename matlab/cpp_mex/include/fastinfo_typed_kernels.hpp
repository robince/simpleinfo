#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "fastinfo_mex_utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastinfo::typed {

namespace detail {

constexpr double kLn2 = 0.693147180559945309417232121458176568;

inline mwSize clamp_threads(mwSize requested) {
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

inline double entropy_from_counts(const mwSize* counts, mwSize nCounts, mwSize nTrials) {
    if (nTrials == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (mwSize i = 0; i < nCounts; ++i) {
        const mwSize count = counts[i];
        if (count > 0) {
            sum += static_cast<double>(count) * std::log(static_cast<double>(count));
        }
    }
    return (std::log(static_cast<double>(nTrials)) - sum / static_cast<double>(nTrials)) / kLn2;
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline std::uint64_t splitmix64_next(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

template <typename TValue>
inline mwSize label_index(TValue value) {
    return static_cast<mwSize>(value);
}

template <typename TX, typename TY>
inline double calc_info_buffered(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    mwSize nTrials,
    mwSize* px,
    mwSize* py,
    mwSize* pxy) {
    std::fill(px, px + xm, 0);
    std::fill(py, py + ym, 0);
    std::fill(pxy, pxy + xm * ym, 0);

    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = label_index(x[i]);
        const mwSize yi = label_index(y[i]);
        ++px[xi];
        ++py[yi];
        ++pxy[xi * ym + yi];
    }

    return entropy_from_counts(px, xm, nTrials)
        + entropy_from_counts(py, ym, nTrials)
        - entropy_from_counts(pxy, xm * ym, nTrials);
}

template <typename TX, typename TY>
inline double calc_info_with_precomputed_hy(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    mwSize nTrials,
    double hy,
    mwSize* px,
    mwSize* pxy) {
    std::fill(px, px + xm, 0);
    std::fill(pxy, pxy + xm * ym, 0);
    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = label_index(x[i]);
        const mwSize yi = label_index(y[i]);
        ++px[xi];
        ++pxy[xi * ym + yi];
    }
    return entropy_from_counts(px, xm, nTrials) + hy - entropy_from_counts(pxy, xm * ym, nTrials);
}

template <typename TX, typename TY, typename TZ>
inline double calc_cmi_buffered(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    const TZ* z,
    mwSize zm,
    mwSize nTrials,
    mwSize* pz,
    mwSize* pxz,
    mwSize* pyz,
    mwSize* pxyz) {
    std::fill(pz, pz + zm, 0);
    std::fill(pxz, pxz + xm * zm, 0);
    std::fill(pyz, pyz + ym * zm, 0);
    std::fill(pxyz, pxyz + xm * ym * zm, 0);

    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = label_index(x[i]);
        const mwSize yi = label_index(y[i]);
        const mwSize zi = label_index(z[i]);
        const mwSize yz = yi * zm + zi;
        ++pz[zi];
        ++pxz[xi * zm + zi];
        ++pyz[yz];
        ++pxyz[xi * (ym * zm) + yz];
    }

    return entropy_from_counts(pxz, xm * zm, nTrials)
        + entropy_from_counts(pyz, ym * zm, nTrials)
        - entropy_from_counts(pxyz, xm * ym * zm, nTrials)
        - entropy_from_counts(pz, zm, nTrials);
}

inline void fisher_yates_shuffle_seeded(std::vector<mwSize>& values, std::uint64_t seed) {
    if (values.size() < 2) {
        return;
    }
    std::uint64_t state = seed;
    for (mwSize i = values.size() - 1; i > 0; --i) {
        const mwSize j = static_cast<mwSize>(splitmix64_next(state) % static_cast<std::uint64_t>(i + 1));
        std::swap(values[i], values[j]);
    }
}

}  // namespace detail

template <typename TX, typename TY>
inline double calc_info(const TX* x, mwSize xm, const TY* y, mwSize ym, mwSize nTrials) {
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> py(ym, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    return detail::calc_info_buffered(x, xm, y, ym, nTrials, px.data(), py.data(), pxy.data());
}

template <typename TX, typename TY>
inline void calc_info_matched(
    const TX* x,
    const TY* y,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    mwSize ym,
    mwSize threadCount,
    double* output) {
    const mwSize nThreads = detail::clamp_threads(threadCount);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> px(xm, 0);
        std::vector<mwSize> py(ym, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const mwSize offset = static_cast<mwSize>(col) * nTrials;
            output[col] = detail::calc_info_buffered(
                x + offset, xm, y + offset, ym, nTrials, px.data(), py.data(), pxy.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> py(ym, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const mwSize offset = col * nTrials;
        output[col] = detail::calc_info_buffered(
            x + offset, xm, y + offset, ym, nTrials, px.data(), py.data(), pxy.data());
    }
#endif
}

template <typename TX, typename TY, typename TZ>
inline double calc_cmi(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    const TZ* z,
    mwSize zm,
    mwSize nTrials) {
    std::vector<mwSize> pz(zm, 0);
    std::vector<mwSize> pxz(xm * zm, 0);
    std::vector<mwSize> pyz(ym * zm, 0);
    std::vector<mwSize> pxyz(xm * ym * zm, 0);
    return detail::calc_cmi_buffered(
        x, xm, y, ym, z, zm, nTrials, pz.data(), pxz.data(), pyz.data(), pxyz.data());
}

template <typename TX, typename TY, typename TZ, typename TK>
inline void calc_cond_cmi(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    const TZ* z,
    mwSize zm,
    const TK* k,
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

    const mwSize yzStride = ym * zm;
    const mwSize xzStride = xm * zm;
    const mwSize xyzStride = xm * ym * zm;

    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = detail::label_index(x[i]);
        const mwSize yi = detail::label_index(y[i]);
        const mwSize zi = detail::label_index(z[i]);
        const mwSize ki = detail::label_index(k[i]);
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

    *total = detail::entropy_from_counts(pxz.data(), pxz.size(), nTrials)
        + detail::entropy_from_counts(pyz.data(), pyz.size(), nTrials)
        - detail::entropy_from_counts(pxyz.data(), pxyz.size(), nTrials)
        - detail::entropy_from_counts(pz.data(), pz.size(), nTrials);

    for (mwSize ki = 0; ki < km; ++ki) {
        const mwSize pzOffset = ki * zm;
        const mwSize xzOffset = ki * xzStride;
        const mwSize yzOffset = ki * yzStride;
        const mwSize xyzOffset = ki * xyzStride;
        contributions[ki] = detail::entropy_from_counts(pxzk.data() + xzOffset, xzStride, nTrials)
            + detail::entropy_from_counts(pyzk.data() + yzOffset, yzStride, nTrials)
            - detail::entropy_from_counts(pxyzk.data() + xyzOffset, xyzStride, nTrials)
            - detail::entropy_from_counts(pzk.data() + pzOffset, zm, nTrials);
    }
}

template <typename TX, typename TY>
inline void calc_info_slice(
    const TX* x,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    const TY* y,
    mwSize ym,
    mwSize threadCount,
    double* output) {
    const mwSize nThreads = detail::clamp_threads(threadCount);
    std::vector<mwSize> py(ym, 0);
    for (mwSize i = 0; i < nTrials; ++i) {
        ++py[detail::label_index(y[i])];
    }
    const double hy = detail::entropy_from_counts(py.data(), py.size(), nTrials);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> px(xm, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const TX* xCol = x + static_cast<mwSize>(col) * nTrials;
            output[col] = detail::calc_info_with_precomputed_hy(
                xCol, xm, y, ym, nTrials, hy, px.data(), pxy.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const TX* xCol = x + col * nTrials;
        output[col] = detail::calc_info_with_precomputed_hy(
            xCol, xm, y, ym, nTrials, hy, px.data(), pxy.data());
    }
#endif
}

template <typename TX, typename TY, typename TZ>
inline void calc_cmi_slice(
    const TX* x,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    const TY* y,
    mwSize ym,
    const TZ* z,
    mwSize zm,
    mwSize threadCount,
    double* output) {
    const mwSize nThreads = detail::clamp_threads(threadCount);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> pz(zm, 0);
        std::vector<mwSize> pxz(xm * zm, 0);
        std::vector<mwSize> pyz(ym * zm, 0);
        std::vector<mwSize> pxyz(xm * ym * zm, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const TX* xCol = x + static_cast<mwSize>(col) * nTrials;
            output[col] = detail::calc_cmi_buffered(
                xCol, xm, y, ym, z, zm, nTrials, pz.data(), pxz.data(), pyz.data(), pxyz.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> pz(zm, 0);
    std::vector<mwSize> pxz(xm * zm, 0);
    std::vector<mwSize> pyz(ym * zm, 0);
    std::vector<mwSize> pxyz(xm * ym * zm, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const TX* xCol = x + col * nTrials;
        output[col] = detail::calc_cmi_buffered(
            xCol, xm, y, ym, z, zm, nTrials, pz.data(), pxz.data(), pyz.data(), pxyz.data());
    }
#endif
}

template <typename TX, typename TY>
inline void calc_info_perm(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    mwSize nTrials,
    mwSize nPerm,
    mwSize threadCount,
    std::uint64_t seed,
    double* output) {
    const mwSize nThreads = detail::clamp_threads(threadCount);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> xsh(nTrials, 0);
        std::vector<mwSize> px(xm, 0);
        std::vector<mwSize> py(ym, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex perm = 0; perm < static_cast<mwSignedIndex>(nPerm); ++perm) {
            for (mwSize i = 0; i < nTrials; ++i) {
                xsh[i] = detail::label_index(x[i]);
            }
            detail::fisher_yates_shuffle_seeded(xsh, detail::splitmix64(seed + static_cast<std::uint64_t>(perm)));
            output[perm] = detail::calc_info_buffered(
                xsh.data(), xm, y, ym, nTrials, px.data(), py.data(), pxy.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> xsh(nTrials, 0);
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> py(ym, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize perm = 0; perm < nPerm; ++perm) {
        for (mwSize i = 0; i < nTrials; ++i) {
            xsh[i] = detail::label_index(x[i]);
        }
        detail::fisher_yates_shuffle_seeded(xsh, detail::splitmix64(seed + static_cast<std::uint64_t>(perm)));
        output[perm] = detail::calc_info_buffered(
            xsh.data(), xm, y, ym, nTrials, px.data(), py.data(), pxy.data());
    }
#endif
}

template <typename TX, typename TY>
inline void calc_info_perm_slice(
    const TX* x,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    const TY* y,
    mwSize ym,
    mwSize nPerm,
    mwSize threadCount,
    std::uint64_t seed,
    double* output) {
    const mwSize nThreads = detail::clamp_threads(threadCount);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> xsh(nTrials, 0);
        std::vector<mwSize> px(xm, 0);
        std::vector<mwSize> py(ym, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const TX* xCol = x + static_cast<mwSize>(col) * nTrials;
            double* outCol = output + static_cast<mwSize>(col) * nPerm;
            const std::uint64_t colSeed = detail::splitmix64(seed + static_cast<std::uint64_t>(col));
            for (mwSize perm = 0; perm < nPerm; ++perm) {
                for (mwSize i = 0; i < nTrials; ++i) {
                    xsh[i] = detail::label_index(xCol[i]);
                }
                detail::fisher_yates_shuffle_seeded(xsh, detail::splitmix64(colSeed + static_cast<std::uint64_t>(perm)));
                outCol[perm] = detail::calc_info_buffered(
                    xsh.data(), xm, y, ym, nTrials, px.data(), py.data(), pxy.data());
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
        const TX* xCol = x + col * nTrials;
        double* outCol = output + col * nPerm;
        const std::uint64_t colSeed = detail::splitmix64(seed + static_cast<std::uint64_t>(col));
        for (mwSize perm = 0; perm < nPerm; ++perm) {
            for (mwSize i = 0; i < nTrials; ++i) {
                xsh[i] = detail::label_index(xCol[i]);
            }
            detail::fisher_yates_shuffle_seeded(xsh, detail::splitmix64(colSeed + static_cast<std::uint64_t>(perm)));
            outCol[perm] = detail::calc_info_buffered(
                xsh.data(), xm, y, ym, nTrials, px.data(), py.data(), pxy.data());
        }
    }
#endif
}

}  // namespace fastinfo::typed
