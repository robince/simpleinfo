#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace fastinfo {

using mwSize = std::size_t;
using mwSignedIndex = std::ptrdiff_t;

inline void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline double nan_value() {
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace fastinfo
