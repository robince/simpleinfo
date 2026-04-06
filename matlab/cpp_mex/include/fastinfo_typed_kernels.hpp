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

struct EntropyLookup {
    double log2_n = 0.0;
    double inv_n = 0.0;
    std::vector<double> count_log2;
};

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

inline EntropyLookup build_entropy_lookup(mwSize nTrials) {
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

inline double entropy_from_counts(const mwSize* counts, mwSize nCounts, const EntropyLookup& lookup) {
    if (lookup.count_log2.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (mwSize i = 0; i < nCounts; ++i) {
        sum += lookup.count_log2[counts[i]];
    }
    return lookup.log2_n - sum * lookup.inv_n;
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

inline void zero_counts(mwSize* counts, mwSize nCounts) {
    std::fill(counts, counts + nCounts, 0);
}

template <typename TX, typename TY>
inline void count_joint_only(
    const TX* x,
    const TY* y,
    mwSize ym,
    mwSize nTrials,
    mwSize* pxy) {
    for (mwSize i = 0; i < nTrials; ++i) {
        ++pxy[label_index(x[i]) * ym + label_index(y[i])];
    }
}

inline void derive_marginals_from_joint(
    const mwSize* pxy,
    mwSize xm,
    mwSize ym,
    mwSize* px,
    mwSize* py) {
    zero_counts(px, xm);
    zero_counts(py, ym);
    for (mwSize xi = 0; xi < xm; ++xi) {
        const mwSize rowOffset = xi * ym;
        for (mwSize yi = 0; yi < ym; ++yi) {
            const mwSize count = pxy[rowOffset + yi];
            px[xi] += count;
            py[yi] += count;
        }
    }
}

inline void derive_x_marginal_from_joint(
    const mwSize* pxy,
    mwSize xm,
    mwSize ym,
    mwSize* px) {
    zero_counts(px, xm);
    for (mwSize xi = 0; xi < xm; ++xi) {
        const mwSize rowOffset = xi * ym;
        for (mwSize yi = 0; yi < ym; ++yi) {
            px[xi] += pxy[rowOffset + yi];
        }
    }
}

template <int XM, int YM, typename TX, typename TY>
inline double calc_info_small(
    const TX* x,
    const TY* y,
    mwSize nTrials,
    const EntropyLookup& lookup) {
    mwSize joint[XM * YM] = {};
    mwSize px[XM] = {};
    mwSize py[YM] = {};
    count_joint_only(x, y, YM, nTrials, joint);
    derive_marginals_from_joint(joint, XM, YM, px, py);
    return entropy_from_counts(px, XM, lookup)
        + entropy_from_counts(py, YM, lookup)
        - entropy_from_counts(joint, XM * YM, lookup);
}

template <int XM, int YM, typename TX, typename TY>
inline double calc_info_small_with_precomputed_hy(
    const TX* x,
    const TY* y,
    mwSize nTrials,
    double hy,
    const EntropyLookup& lookup) {
    mwSize joint[XM * YM] = {};
    mwSize px[XM] = {};
    count_joint_only(x, y, YM, nTrials, joint);
    derive_x_marginal_from_joint(joint, XM, YM, px);
    return entropy_from_counts(px, XM, lookup)
        + hy
        - entropy_from_counts(joint, XM * YM, lookup);
}

template <int XM, int YM, typename TX, typename TY>
inline double calc_info_small_with_precomputed_sum(
    const TX* x,
    const TY* y,
    mwSize nTrials,
    double hx_plus_hy,
    const EntropyLookup& lookup) {
    mwSize joint[XM * YM] = {};
    count_joint_only(x, y, YM, nTrials, joint);
    return hx_plus_hy - entropy_from_counts(joint, XM * YM, lookup);
}

template <typename TX, typename TY>
inline bool dispatch_calc_info_small(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    mwSize nTrials,
    const EntropyLookup& lookup,
    double* out) {
    if (xm == 2 && ym == 2) { *out = calc_info_small<2, 2>(x, y, nTrials, lookup); return true; }
    if (xm == 3 && ym == 2) { *out = calc_info_small<3, 2>(x, y, nTrials, lookup); return true; }
    if (xm == 2 && ym == 3) { *out = calc_info_small<2, 3>(x, y, nTrials, lookup); return true; }
    if (xm == 3 && ym == 3) { *out = calc_info_small<3, 3>(x, y, nTrials, lookup); return true; }
    if (xm == 5 && ym == 2) { *out = calc_info_small<5, 2>(x, y, nTrials, lookup); return true; }
    if (xm == 2 && ym == 5) { *out = calc_info_small<2, 5>(x, y, nTrials, lookup); return true; }
    if (xm == 5 && ym == 3) { *out = calc_info_small<5, 3>(x, y, nTrials, lookup); return true; }
    if (xm == 3 && ym == 5) { *out = calc_info_small<3, 5>(x, y, nTrials, lookup); return true; }
    return false;
}

template <typename TX, typename TY>
inline bool dispatch_calc_info_small_with_precomputed_hy(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    mwSize nTrials,
    double hy,
    const EntropyLookup& lookup,
    double* out) {
    if (xm == 2 && ym == 2) { *out = calc_info_small_with_precomputed_hy<2, 2>(x, y, nTrials, hy, lookup); return true; }
    if (xm == 3 && ym == 2) { *out = calc_info_small_with_precomputed_hy<3, 2>(x, y, nTrials, hy, lookup); return true; }
    if (xm == 2 && ym == 3) { *out = calc_info_small_with_precomputed_hy<2, 3>(x, y, nTrials, hy, lookup); return true; }
    if (xm == 3 && ym == 3) { *out = calc_info_small_with_precomputed_hy<3, 3>(x, y, nTrials, hy, lookup); return true; }
    if (xm == 5 && ym == 2) { *out = calc_info_small_with_precomputed_hy<5, 2>(x, y, nTrials, hy, lookup); return true; }
    if (xm == 2 && ym == 5) { *out = calc_info_small_with_precomputed_hy<2, 5>(x, y, nTrials, hy, lookup); return true; }
    if (xm == 5 && ym == 3) { *out = calc_info_small_with_precomputed_hy<5, 3>(x, y, nTrials, hy, lookup); return true; }
    if (xm == 3 && ym == 5) { *out = calc_info_small_with_precomputed_hy<3, 5>(x, y, nTrials, hy, lookup); return true; }
    return false;
}

template <typename TX, typename TY>
inline bool dispatch_calc_info_small_with_precomputed_sum(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    mwSize nTrials,
    double hx_plus_hy,
    const EntropyLookup& lookup,
    double* out) {
    if (xm == 2 && ym == 2) { *out = calc_info_small_with_precomputed_sum<2, 2>(x, y, nTrials, hx_plus_hy, lookup); return true; }
    if (xm == 3 && ym == 2) { *out = calc_info_small_with_precomputed_sum<3, 2>(x, y, nTrials, hx_plus_hy, lookup); return true; }
    if (xm == 2 && ym == 3) { *out = calc_info_small_with_precomputed_sum<2, 3>(x, y, nTrials, hx_plus_hy, lookup); return true; }
    if (xm == 3 && ym == 3) { *out = calc_info_small_with_precomputed_sum<3, 3>(x, y, nTrials, hx_plus_hy, lookup); return true; }
    if (xm == 5 && ym == 2) { *out = calc_info_small_with_precomputed_sum<5, 2>(x, y, nTrials, hx_plus_hy, lookup); return true; }
    if (xm == 2 && ym == 5) { *out = calc_info_small_with_precomputed_sum<2, 5>(x, y, nTrials, hx_plus_hy, lookup); return true; }
    if (xm == 5 && ym == 3) { *out = calc_info_small_with_precomputed_sum<5, 3>(x, y, nTrials, hx_plus_hy, lookup); return true; }
    if (xm == 3 && ym == 5) { *out = calc_info_small_with_precomputed_sum<3, 5>(x, y, nTrials, hx_plus_hy, lookup); return true; }
    return false;
}

template <typename TX, typename TY>
inline double calc_info_buffered(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    const EntropyLookup& lookup,
    mwSize* px,
    mwSize* py,
    mwSize* pxy) {
    const mwSize nTrials = static_cast<mwSize>(lookup.count_log2.size() - 1);
    double specialized = 0.0;
    if (dispatch_calc_info_small(x, xm, y, ym, nTrials, lookup, &specialized)) {
        return specialized;
    }
    zero_counts(pxy, xm * ym);
    count_joint_only(x, y, ym, nTrials, pxy);
    derive_marginals_from_joint(pxy, xm, ym, px, py);
    return entropy_from_counts(px, xm, lookup)
        + entropy_from_counts(py, ym, lookup)
        - entropy_from_counts(pxy, xm * ym, lookup);
}

template <typename TX, typename TY>
inline double calc_info_with_precomputed_hy(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    const EntropyLookup& lookup,
    double hy,
    mwSize* px,
    mwSize* pxy) {
    const mwSize nTrials = static_cast<mwSize>(lookup.count_log2.size() - 1);
    double specialized = 0.0;
    if (dispatch_calc_info_small_with_precomputed_hy(x, xm, y, ym, nTrials, hy, lookup, &specialized)) {
        return specialized;
    }
    zero_counts(pxy, xm * ym);
    count_joint_only(x, y, ym, nTrials, pxy);
    derive_x_marginal_from_joint(pxy, xm, ym, px);
    return entropy_from_counts(px, xm, lookup) + hy - entropy_from_counts(pxy, xm * ym, lookup);
}

template <typename TX, typename TY>
inline double calc_info_with_precomputed_sum(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    const EntropyLookup& lookup,
    double hx_plus_hy,
    mwSize* pxy) {
    const mwSize nTrials = static_cast<mwSize>(lookup.count_log2.size() - 1);
    double specialized = 0.0;
    if (dispatch_calc_info_small_with_precomputed_sum(x, xm, y, ym, nTrials, hx_plus_hy, lookup, &specialized)) {
        return specialized;
    }
    zero_counts(pxy, xm * ym);
    count_joint_only(x, y, ym, nTrials, pxy);
    return hx_plus_hy - entropy_from_counts(pxy, xm * ym, lookup);
}

inline void derive_cmi_marginals_from_joint(
    const mwSize* pxyz,
    mwSize xm,
    mwSize ym,
    mwSize zm,
    mwSize* pz,
    mwSize* pxz,
    mwSize* pyz) {
    zero_counts(pz, zm);
    zero_counts(pxz, xm * zm);
    zero_counts(pyz, ym * zm);
    const mwSize yzStride = ym * zm;
    for (mwSize xi = 0; xi < xm; ++xi) {
        const mwSize xOffset = xi * yzStride;
        for (mwSize yi = 0; yi < ym; ++yi) {
            const mwSize yzOffset = yi * zm;
            for (mwSize zi = 0; zi < zm; ++zi) {
                const mwSize count = pxyz[xOffset + yzOffset + zi];
                pz[zi] += count;
                pxz[xi * zm + zi] += count;
                pyz[yi * zm + zi] += count;
            }
        }
    }
}

template <typename TX, typename TY, typename TZ>
inline double calc_cmi_buffered(
    const TX* x,
    mwSize xm,
    const TY* y,
    mwSize ym,
    const TZ* z,
    mwSize zm,
    const EntropyLookup& lookup,
    mwSize* pz,
    mwSize* pxz,
    mwSize* pyz,
    mwSize* pxyz) {
    const mwSize nTrials = static_cast<mwSize>(lookup.count_log2.size() - 1);
    zero_counts(pxyz, xm * ym * zm);
    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = label_index(x[i]);
        const mwSize yi = label_index(y[i]);
        const mwSize zi = label_index(z[i]);
        ++pxyz[xi * (ym * zm) + yi * zm + zi];
    }

    derive_cmi_marginals_from_joint(pxyz, xm, ym, zm, pz, pxz, pyz);
    return entropy_from_counts(pxz, xm * zm, lookup)
        + entropy_from_counts(pyz, ym * zm, lookup)
        - entropy_from_counts(pxyz, xm * ym * zm, lookup)
        - entropy_from_counts(pz, zm, lookup);
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
    const detail::EntropyLookup lookup = detail::build_entropy_lookup(nTrials);
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> py(ym, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    return detail::calc_info_buffered(x, xm, y, ym, lookup, px.data(), py.data(), pxy.data());
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
    const detail::EntropyLookup lookup = detail::build_entropy_lookup(nTrials);
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
        output[col] = detail::calc_info_buffered(
            x + offset, xm, y + offset, ym, lookup, px.data(), py.data(), pxy.data());
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
    const detail::EntropyLookup lookup = detail::build_entropy_lookup(nTrials);
    std::vector<mwSize> pz(zm, 0);
    std::vector<mwSize> pxz(xm * zm, 0);
    std::vector<mwSize> pyz(ym * zm, 0);
    std::vector<mwSize> pxyz(xm * ym * zm, 0);
    return detail::calc_cmi_buffered(
        x, xm, y, ym, z, zm, lookup, pz.data(), pxz.data(), pyz.data(), pxyz.data());
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
    std::vector<mwSize> pxyzk(km * xm * ym * zm, 0);
    std::vector<mwSize> pz(zm, 0);
    std::vector<mwSize> pxz(xm * zm, 0);
    std::vector<mwSize> pyz(ym * zm, 0);
    const detail::EntropyLookup lookup = detail::build_entropy_lookup(nTrials);

    const mwSize yzStride = ym * zm;
    const mwSize xyzStride = xm * ym * zm;

    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize xi = detail::label_index(x[i]);
        const mwSize yi = detail::label_index(y[i]);
        const mwSize zi = detail::label_index(z[i]);
        const mwSize ki = detail::label_index(k[i]);
        const mwSize yz = yi * zm + zi;
        const mwSize xyz = xi * yzStride + yz;
        ++pxyzk[ki * xyzStride + xyz];
    }

    double totalValue = 0.0;
    for (mwSize ki = 0; ki < km; ++ki) {
        const mwSize xyzOffset = ki * xyzStride;
        detail::derive_cmi_marginals_from_joint(
            pxyzk.data() + xyzOffset, xm, ym, zm, pz.data(), pxz.data(), pyz.data());
        contributions[ki] = detail::entropy_from_counts(pxz.data(), pxz.size(), lookup)
            + detail::entropy_from_counts(pyz.data(), pyz.size(), lookup)
            - detail::entropy_from_counts(pxyzk.data() + xyzOffset, xyzStride, lookup)
            - detail::entropy_from_counts(pz.data(), zm, lookup);
        totalValue += contributions[ki];
    }
    *total = totalValue;
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
    const detail::EntropyLookup lookup = detail::build_entropy_lookup(nTrials);
    std::vector<mwSize> py(ym, 0);
    for (mwSize i = 0; i < nTrials; ++i) {
        ++py[detail::label_index(y[i])];
    }
    const double hy = detail::entropy_from_counts(py.data(), py.size(), lookup);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> px(xm, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const TX* xCol = x + static_cast<mwSize>(col) * nTrials;
            output[col] = detail::calc_info_with_precomputed_hy(
                xCol, xm, y, ym, lookup, hy, px.data(), pxy.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> px(xm, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const TX* xCol = x + col * nTrials;
        output[col] = detail::calc_info_with_precomputed_hy(
            xCol, xm, y, ym, lookup, hy, px.data(), pxy.data());
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
    const detail::EntropyLookup lookup = detail::build_entropy_lookup(nTrials);
    std::vector<mwSize> pyz_const(ym * zm, 0);
    std::vector<mwSize> pz_const(zm, 0);
    for (mwSize i = 0; i < nTrials; ++i) {
        const mwSize yi = detail::label_index(y[i]);
        const mwSize zi = detail::label_index(z[i]);
        ++pyz_const[yi * zm + zi];
        ++pz_const[zi];
    }
    const double hyz_minus_hz = detail::entropy_from_counts(pyz_const.data(), pyz_const.size(), lookup)
        - detail::entropy_from_counts(pz_const.data(), pz_const.size(), lookup);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> pxz(xm * zm, 0);
        std::vector<mwSize> pxyz(xm * ym * zm, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const TX* xCol = x + static_cast<mwSize>(col) * nTrials;
            detail::zero_counts(pxyz.data(), xm * ym * zm);
            for (mwSize i = 0; i < nTrials; ++i) {
                ++pxyz[detail::label_index(xCol[i]) * (ym * zm) + detail::label_index(y[i]) * zm + detail::label_index(z[i])];
            }
            detail::zero_counts(pxz.data(), xm * zm);
            for (mwSize xi = 0; xi < xm; ++xi) {
                for (mwSize yi = 0; yi < ym; ++yi) {
                    for (mwSize zi = 0; zi < zm; ++zi) {
                        pxz[xi * zm + zi] += pxyz[xi * (ym * zm) + yi * zm + zi];
                    }
                }
            }
            output[col] = detail::entropy_from_counts(pxz.data(), pxz.size(), lookup)
                + hyz_minus_hz
                - detail::entropy_from_counts(pxyz.data(), pxyz.size(), lookup);
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> pxz(xm * zm, 0);
    std::vector<mwSize> pxyz(xm * ym * zm, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const TX* xCol = x + col * nTrials;
        detail::zero_counts(pxyz.data(), xm * ym * zm);
        for (mwSize i = 0; i < nTrials; ++i) {
            ++pxyz[detail::label_index(xCol[i]) * (ym * zm) + detail::label_index(y[i]) * zm + detail::label_index(z[i])];
        }
        detail::zero_counts(pxz.data(), xm * zm);
        for (mwSize xi = 0; xi < xm; ++xi) {
            for (mwSize yi = 0; yi < ym; ++yi) {
                for (mwSize zi = 0; zi < zm; ++zi) {
                    pxz[xi * zm + zi] += pxyz[xi * (ym * zm) + yi * zm + zi];
                }
            }
        }
        output[col] = detail::entropy_from_counts(pxz.data(), pxz.size(), lookup)
            + hyz_minus_hz
            - detail::entropy_from_counts(pxyz.data(), pxyz.size(), lookup);
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
    const detail::EntropyLookup lookup = detail::build_entropy_lookup(nTrials);
    std::vector<mwSize> px_const(xm, 0);
    std::vector<mwSize> py_const(ym, 0);
    for (mwSize i = 0; i < nTrials; ++i) {
        ++px_const[detail::label_index(x[i])];
        ++py_const[detail::label_index(y[i])];
    }
    const double hx_plus_hy = detail::entropy_from_counts(px_const.data(), xm, lookup)
        + detail::entropy_from_counts(py_const.data(), ym, lookup);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> xsh(nTrials, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex perm = 0; perm < static_cast<mwSignedIndex>(nPerm); ++perm) {
            for (mwSize i = 0; i < nTrials; ++i) {
                xsh[i] = detail::label_index(x[i]);
            }
            detail::fisher_yates_shuffle_seeded(xsh, detail::splitmix64(seed + static_cast<std::uint64_t>(perm)));
            output[perm] = detail::calc_info_with_precomputed_sum(
                xsh.data(), xm, y, ym, lookup, hx_plus_hy, pxy.data());
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> xsh(nTrials, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize perm = 0; perm < nPerm; ++perm) {
        for (mwSize i = 0; i < nTrials; ++i) {
            xsh[i] = detail::label_index(x[i]);
        }
        detail::fisher_yates_shuffle_seeded(xsh, detail::splitmix64(seed + static_cast<std::uint64_t>(perm)));
        output[perm] = detail::calc_info_with_precomputed_sum(
            xsh.data(), xm, y, ym, lookup, hx_plus_hy, pxy.data());
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
    const detail::EntropyLookup lookup = detail::build_entropy_lookup(nTrials);
    std::vector<mwSize> py_const(ym, 0);
    for (mwSize i = 0; i < nTrials; ++i) {
        ++py_const[detail::label_index(y[i])];
    }
    const double hy = detail::entropy_from_counts(py_const.data(), ym, lookup);
    std::vector<double> hx_plus_hy_cols(nCols, hy);
    std::vector<mwSize> px_const(xm, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        detail::zero_counts(px_const.data(), xm);
        const TX* xCol = x + col * nTrials;
        for (mwSize i = 0; i < nTrials; ++i) {
            ++px_const[detail::label_index(xCol[i])];
        }
        hx_plus_hy_cols[col] += detail::entropy_from_counts(px_const.data(), xm, lookup);
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(nThreads)) default(shared)
    {
        std::vector<mwSize> xsh(nTrials, 0);
        std::vector<mwSize> pxy(xm * ym, 0);
#pragma omp for schedule(static)
        for (mwSignedIndex col = 0; col < static_cast<mwSignedIndex>(nCols); ++col) {
            const TX* xCol = x + static_cast<mwSize>(col) * nTrials;
            double* outCol = output + static_cast<mwSize>(col) * nPerm;
            const std::uint64_t colSeed = detail::splitmix64(seed + static_cast<std::uint64_t>(col));
            const double hx_plus_hy = hx_plus_hy_cols[static_cast<mwSize>(col)];
            for (mwSize perm = 0; perm < nPerm; ++perm) {
                for (mwSize i = 0; i < nTrials; ++i) {
                    xsh[i] = detail::label_index(xCol[i]);
                }
                detail::fisher_yates_shuffle_seeded(xsh, detail::splitmix64(colSeed + static_cast<std::uint64_t>(perm)));
                outCol[perm] = detail::calc_info_with_precomputed_sum(
                    xsh.data(), xm, y, ym, lookup, hx_plus_hy, pxy.data());
            }
        }
    }
#else
    (void)nThreads;
    std::vector<mwSize> xsh(nTrials, 0);
    std::vector<mwSize> pxy(xm * ym, 0);
    for (mwSize col = 0; col < nCols; ++col) {
        const TX* xCol = x + col * nTrials;
        double* outCol = output + col * nPerm;
        const std::uint64_t colSeed = detail::splitmix64(seed + static_cast<std::uint64_t>(col));
        const double hx_plus_hy = hx_plus_hy_cols[col];
        for (mwSize perm = 0; perm < nPerm; ++perm) {
            for (mwSize i = 0; i < nTrials; ++i) {
                xsh[i] = detail::label_index(xCol[i]);
            }
            detail::fisher_yates_shuffle_seeded(xsh, detail::splitmix64(colSeed + static_cast<std::uint64_t>(perm)));
            outCol[perm] = detail::calc_info_with_precomputed_sum(
                xsh.data(), xm, y, ym, lookup, hx_plus_hy, pxy.data());
        }
    }
#endif
}

}  // namespace fastinfo::typed
