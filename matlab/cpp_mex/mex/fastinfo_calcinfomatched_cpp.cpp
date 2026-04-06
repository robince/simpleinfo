#include "fastinfo_mex_adapter_utils.hpp"

#include "fastinfo_typed_kernels.hpp"

class MexFunction : public fastinfo::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 5 || outputs.size() > 1) {
            fail("fastinfo_calcinfomatched_cpp:usage",
                "fastinfo_calcinfomatched_cpp(X, Xm, Y, Ym, Nthread) expects five inputs.");
        }

        const auto xm = scalar_to_size(inputs[1], "Xm");
        const auto ym = scalar_to_size(inputs[3], "Ym");
        const auto nThreads = scalar_to_size(inputs[4], "Nthread");
        dispatch_integer_array(inputs[0], "X", [&](const auto& x) {
            validate_discrete_array_view(x, xm, "X");
            dispatch_integer_array(inputs[2], "Y", [&](const auto& y) {
                validate_discrete_array_view(y, ym, "Y");
                const auto xDims = x.getDimensions();
                const auto yDims = y.getDimensions();
                if (xDims.size() != 2 || yDims.size() != 2) {
                    fail("fastinfo_calcinfomatched_cpp:shape", "X and Y must be 2-D matrices of shape [Ntrl, Nx].");
                }
                if (xDims[0] != yDims[0] || xDims[1] != yDims[1]) {
                    fail("fastinfo_calcinfomatched_cpp:shape", "X and Y must have the same shape.");
                }
                const auto nTrials = xDims[0];
                const auto nCols = xDims[1];
                auto out = factory_.createArray<double>({nCols, 1});
                fastinfo::typed::calc_info_matched(
                    raw_data(x), raw_data(y), nTrials, nCols, xm, ym, nThreads, raw_data(out));
                outputs[0] = out;
            });
        });
    }
};
