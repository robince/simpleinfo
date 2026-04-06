#include "fastinfo_mex_adapter_utils.hpp"

#include "fastinfo_typed_kernels.hpp"

class MexFunction : public fastinfo::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 6 && inputs.size() != 7) {
            fail("fastinfo_calcinfoperm_slice_cpp:usage",
                "fastinfo_calcinfoperm_slice_cpp(X, Xm, Y, Ym, Nperm, Nthread, Seed) expects six or seven inputs.");
        }
        if (outputs.size() > 1) {
            fail("fastinfo_calcinfoperm_slice_cpp:usage", "Too many output arguments.");
        }

        const auto xm = scalar_to_size(inputs[1], "Xm");
        const auto ym = scalar_to_size(inputs[3], "Ym");
        const auto nPerm = scalar_to_size(inputs[4], "Nperm");
        const auto nThreads = scalar_to_size(inputs[5], "Nthread");
        const auto seed = inputs.size() == 7 ? scalar_to_uint64(inputs[6], "Seed") : 5489ull;

        dispatch_integer_array(inputs[0], "X", [&](const auto& x) {
            validate_discrete_array_view(x, xm, "X");
            const auto dims = x.getDimensions();
            if (dims.size() != 2) {
                fail("fastinfo_calcinfoperm_slice_cpp:shape", "X must be a 2-D matrix of shape [Ntrl, Nx].");
            }
            const auto nTrials = dims[0];
            const auto nCols = dims[1];
            dispatch_integer_array(inputs[2], "Y", [&](const auto& y) {
                validate_discrete_array_view(y, ym, "Y");
                if (y.getNumberOfElements() != nTrials) {
                    fail("fastinfo_calcinfoperm_slice_cpp:shape", "Y must contain exactly Ntrl samples.");
                }

                auto out = factory_.createArray<double>({nPerm, nCols});
                fastinfo::typed::calc_info_perm_slice(
                    raw_data(x), nTrials, nCols, xm, raw_data(y), ym, nPerm, nThreads, seed, raw_data(out));
                outputs[0] = out;
            });
        });
    }
};
