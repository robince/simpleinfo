#pragma once

#include <cstdint>
#include <vector>

#include "fastinfo_mex_utils.hpp"

namespace fastinfo {

// Contract: all discrete-label inputs must already be validated to lie in
// 0:(M-1) for their corresponding state counts. These kernels do not perform
// bounds checks on every sample.

double calc_info(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    mwSize nTrials);

void calc_info_matched(
    const mwSize* x,
    const mwSize* y,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    mwSize ym,
    mwSize threadCount,
    double* output);

double calc_cmi(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    const mwSize* z,
    mwSize zm,
    mwSize nTrials);

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
    double* contributions);

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
    double* output);

void calc_info_slice(
    const mwSize* x,
    mwSize nTrials,
    mwSize nCols,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    mwSize threadCount,
    double* output);

void calc_info_perm(
    const mwSize* x,
    mwSize xm,
    const mwSize* y,
    mwSize ym,
    mwSize nTrials,
    mwSize nPerm,
    mwSize threadCount,
    std::uint64_t seed,
    double* output);

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
    double* output);

void eqpop_sorted(
    const double* xSorted,
    mwSize nSamples,
    mwSize nBins,
    int32_t* output);

void eqpop(
    const double* x,
    mwSize nSamples,
    mwSize nBins,
    int32_t* output);

void eqpop_sorted_slice(
    const double* xSorted,
    mwSize nRows,
    mwSize nCols,
    mwSize nBins,
    mwSize threadCount,
    double* output);

void eqpop_slice(
    const double* x,
    mwSize nRows,
    mwSize nCols,
    mwSize nBins,
    mwSize threadCount,
    double* output);

}  // namespace fastinfo
