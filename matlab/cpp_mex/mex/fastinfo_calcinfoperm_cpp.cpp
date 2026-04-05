#include "fastinfo_mex_adapter_utils.hpp"

#include "fastinfo_kernels.hpp"
#include "fastinfo_typed_kernels.hpp"

class MexFunction : public fastinfo::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if ((inputs.size() != 6 && inputs.size() != 7) || outputs.size() > 1) {
            fail("fastinfo_calcinfoperm_cpp:usage",
                "fastinfo_calcinfoperm_cpp(X, Xm, Y, Ym, Nperm, Nthread, Seed) expects six or seven inputs.");
        }

        const auto xm = scalar_to_size(inputs[1], "Xm");
        const auto ym = scalar_to_size(inputs[3], "Ym");
        const auto nPerm = scalar_to_size(inputs[4], "Nperm");
        const auto nThreads = scalar_to_size(inputs[5], "Nthread");
        const auto seed = inputs.size() == 7 ? scalar_to_uint64(inputs[6], "Seed") : 5489u;
        dispatch_integer_array(inputs[0], "X", [&](const auto& x) {
            validate_discrete_array_view(x, xm, "X");
            dispatch_integer_array(inputs[2], "Y", [&](const auto& y) {
                validate_discrete_array_view(y, ym, "Y");
                if (x.getNumberOfElements() != y.getNumberOfElements()) {
                    fail("fastinfo_calcinfoperm_cpp:shape", "X and Y must contain the same number of samples.");
                }
                auto out = factory_.createArray<double>({nPerm, 1});
                fastinfo::typed::calc_info_perm(
                    raw_data(x), xm, raw_data(y), ym, x.getNumberOfElements(), nPerm, nThreads, seed, raw_data(out));
                outputs[0] = out;
            });
        });
    }
};
