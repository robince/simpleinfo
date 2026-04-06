#pragma once

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "fastinfo_mex_utils.hpp"

namespace fastinfo {

struct ParsedDiscreteArray {
    std::vector<mwSize> values;
    std::vector<mwSize> dims;
};

class MexAdapterBase : public matlab::mex::Function {
protected:
    matlab::data::ArrayFactory factory_;

    [[noreturn]] void fail(const char* id, const char* message) {
        auto engine = getEngine();
        engine->feval(u"error", 0, std::vector<matlab::data::Array>({
            factory_.createCharArray(id),
            factory_.createCharArray(message)}));
        throw std::runtime_error(message);
    }

    [[noreturn]] void fail(const char* id, const std::string& message) {
        fail(id, message.c_str());
    }

    mwSize scalar_to_size(const matlab::data::Array& array, const char* name) {
        if (array.getNumberOfElements() != 1 || array.getType() != matlab::data::ArrayType::DOUBLE) {
            fail("fastinfo_cpp:scalar", std::string(name) + " must be a scalar double");
        }
        const auto typed = matlab::data::TypedArray<double>(array);
        const double value = typed[0];
        if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value) {
            fail("fastinfo_cpp:scalar", std::string(name) + " must be a non-negative integer scalar");
        }
        return static_cast<mwSize>(value);
    }

    std::uint64_t scalar_to_uint64(const matlab::data::Array& array, const char* name) {
        if (array.getNumberOfElements() != 1 || array.getType() != matlab::data::ArrayType::DOUBLE) {
            fail("fastinfo_cpp:scalar", std::string(name) + " must be a scalar double");
        }
        const auto typed = matlab::data::TypedArray<double>(array);
        const double value = typed[0];
        if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value) {
            fail("fastinfo_cpp:scalar", std::string(name) + " must be a non-negative integer scalar");
        }
        if (value > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
            fail("fastinfo_cpp:scalar", std::string(name) + " is too large for uint64");
        }
        return static_cast<std::uint64_t>(value);
    }

    matlab::data::TypedArray<double> require_double_array(const matlab::data::Array& array, const char* name) {
        if (array.getType() != matlab::data::ArrayType::DOUBLE) {
            fail("fastinfo_cpp:type", std::string(name) + " must have class double");
        }
        return array;
    }

    template <typename T>
    const T* raw_data(const matlab::data::TypedArray<T>& array) const {
        return array.getNumberOfElements() == 0 ? nullptr : &(*array.cbegin());
    }

    template <typename T>
    T* raw_data(matlab::data::TypedArray<T>& array) const {
        return array.getNumberOfElements() == 0 ? nullptr : &(*array.begin());
    }

    ParsedDiscreteArray parse_discrete_array(const matlab::data::Array& array, mwSize nStates, const char* name) {
        ParsedDiscreteArray parsed;
        parsed.dims = array.getDimensions();
        parsed.values.reserve(array.getNumberOfElements());

        switch (array.getType()) {
            case matlab::data::ArrayType::DOUBLE:
                append_discrete_values(matlab::data::TypedArray<double>(array), nStates, name, parsed.values);
                break;
            case matlab::data::ArrayType::INT8:
                append_discrete_values(matlab::data::TypedArray<int8_t>(array), nStates, name, parsed.values);
                break;
            case matlab::data::ArrayType::INT16:
                append_discrete_values(matlab::data::TypedArray<int16_t>(array), nStates, name, parsed.values);
                break;
            case matlab::data::ArrayType::INT32:
                append_discrete_values(matlab::data::TypedArray<int32_t>(array), nStates, name, parsed.values);
                break;
            case matlab::data::ArrayType::INT64:
                append_discrete_values(matlab::data::TypedArray<int64_t>(array), nStates, name, parsed.values);
                break;
            case matlab::data::ArrayType::UINT8:
                append_discrete_values(matlab::data::TypedArray<uint8_t>(array), nStates, name, parsed.values);
                break;
            case matlab::data::ArrayType::UINT16:
                append_discrete_values(matlab::data::TypedArray<uint16_t>(array), nStates, name, parsed.values);
                break;
            case matlab::data::ArrayType::UINT32:
                append_discrete_values(matlab::data::TypedArray<uint32_t>(array), nStates, name, parsed.values);
                break;
            case matlab::data::ArrayType::UINT64:
                append_discrete_values(matlab::data::TypedArray<uint64_t>(array), nStates, name, parsed.values);
                break;
            default:
                fail("fastinfo_cpp:type", std::string(name) + " must be a numeric integer-valued array");
        }

        return parsed;
    }

    template <typename Fn>
    decltype(auto) dispatch_integer_array(const matlab::data::Array& array, const char* name, Fn&& fn) {
        switch (array.getType()) {
            case matlab::data::ArrayType::INT16:
                return fn(matlab::data::TypedArray<int16_t>(array));
            case matlab::data::ArrayType::INT32:
                return fn(matlab::data::TypedArray<int32_t>(array));
            case matlab::data::ArrayType::INT64:
                return fn(matlab::data::TypedArray<int64_t>(array));
            default:
                fail("fastinfo_cpp:type", std::string(name) + " must have class int16, int32, or int64");
        }
    }

    template <typename ValueType>
    void validate_discrete_array_view(
        const matlab::data::TypedArray<ValueType>& typed,
        mwSize nStates,
        const char* name) {
        for (ValueType value : typed) {
            if constexpr (std::is_signed_v<ValueType>) {
                if (value < 0) {
                    fail("fastinfo_cpp:labels", std::string(name) + " must contain non-negative labels");
                }
            }
            const auto asIndex = static_cast<std::uint64_t>(value);
            if (asIndex >= static_cast<std::uint64_t>(nStates)) {
                fail("fastinfo_cpp:labels", std::string(name) + " must take values in the range 0..M-1");
            }
        }
    }

private:
    template <typename ValueType>
    void append_discrete_values(
        const matlab::data::TypedArray<ValueType>& typed,
        mwSize nStates,
        const char* name,
        std::vector<mwSize>& output) {
        for (ValueType value : typed) {
            double asDouble = 0.0;
            if constexpr (std::is_floating_point_v<ValueType>) {
                asDouble = static_cast<double>(value);
                if (!std::isfinite(asDouble) || std::floor(asDouble) != asDouble || asDouble < 0.0) {
                    fail("fastinfo_cpp:labels", std::string(name) + " must contain non-negative integer-valued labels");
                }
            } else {
                if constexpr (std::is_signed_v<ValueType>) {
                    if (value < 0) {
                        fail("fastinfo_cpp:labels", std::string(name) + " must contain non-negative labels");
                    }
                }
                asDouble = static_cast<double>(value);
            }

            if (asDouble >= static_cast<double>(nStates)) {
                fail("fastinfo_cpp:labels", std::string(name) + " must take values in the range 0..M-1");
            }
            output.push_back(static_cast<mwSize>(asDouble));
        }
    }
};

}  // namespace fastinfo
