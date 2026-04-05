#include "fastinfo_mex_adapter_utils.hpp"

#include "fastinfo_kernels.hpp"
#include "fastinfo_typed_kernels.hpp"

class MexFunction : public fastinfo::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 6 || outputs.size() > 1) {
            fail("fastinfo_calccmi_cpp:usage",
                "fastinfo_calccmi_cpp(X, Xm, Y, Ym, Z, Zm) expects six inputs.");
        }

        const auto xm = scalar_to_size(inputs[1], "Xm");
        const auto ym = scalar_to_size(inputs[3], "Ym");
        const auto zm = scalar_to_size(inputs[5], "Zm");
        dispatch_integer_array(inputs[0], "X", [&](const auto& x) {
            validate_discrete_array_view(x, xm, "X");
            dispatch_integer_array(inputs[2], "Y", [&](const auto& y) {
                validate_discrete_array_view(y, ym, "Y");
                dispatch_integer_array(inputs[4], "Z", [&](const auto& z) {
                    validate_discrete_array_view(z, zm, "Z");
                    if (x.getNumberOfElements() != y.getNumberOfElements() || x.getNumberOfElements() != z.getNumberOfElements()) {
                        fail("fastinfo_calccmi_cpp:shape", "X, Y, and Z must contain the same number of samples.");
                    }
                    outputs[0] = factory_.createScalar(fastinfo::typed::calc_cmi(
                        raw_data(x), xm, raw_data(y), ym, raw_data(z), zm, x.getNumberOfElements()));
                });
            });
        });
    }
};
