#pragma once

#include "fastinfo_mex_utils.hpp"

namespace fastinfo {

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
