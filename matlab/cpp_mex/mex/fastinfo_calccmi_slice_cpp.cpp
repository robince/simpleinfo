#include "fastinfo_mex_adapter_utils.hpp"

#include "fastinfo_kernels.hpp"
#include "fastinfo_typed_kernels.hpp"

class MexFunction : public fastinfo::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 7 || outputs.size() > 1) {
            fail("fastinfo_calccmi_slice_cpp:usage",
                "fastinfo_calccmi_slice_cpp(X, Xm, Y, Ym, Z, Zm, Nthread) expects seven inputs.");
        }

        const auto xm = scalar_to_size(inputs[1], "Xm");
        const auto ym = scalar_to_size(inputs[3], "Ym");
        const auto zm = scalar_to_size(inputs[5], "Zm");
        const auto nThreads = scalar_to_size(inputs[6], "Nthread");

        dispatch_integer_array(inputs[0], "X", [&](const auto& x) {
            validate_discrete_array_view(x, xm, "X");
            const auto dims = x.getDimensions();
            if (dims.size() != 2) {
                fail("fastinfo_calccmi_slice_cpp:shape", "X must be a 2-D matrix of shape [Ntrl, Nx].");
            }
            const auto nTrials = dims[0];
            const auto nCols = dims[1];
            dispatch_integer_array(inputs[2], "Y", [&](const auto& y) {
                validate_discrete_array_view(y, ym, "Y");
                dispatch_integer_array(inputs[4], "Z", [&](const auto& z) {
                    validate_discrete_array_view(z, zm, "Z");
                    if (y.getNumberOfElements() != nTrials || z.getNumberOfElements() != nTrials) {
                        fail("fastinfo_calccmi_slice_cpp:shape", "Y and Z must contain exactly Ntrl samples.");
                    }
                    auto out = factory_.createArray<double>({nCols, 1});
                    fastinfo::typed::calc_cmi_slice(
                        raw_data(x), nTrials, nCols, xm, raw_data(y), ym, raw_data(z), zm, nThreads, raw_data(out));
                    outputs[0] = out;
                });
            });
        });
    }
};
