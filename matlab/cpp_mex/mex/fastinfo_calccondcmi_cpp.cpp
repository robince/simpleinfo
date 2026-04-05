#include "fastinfo_mex_adapter_utils.hpp"

#include "fastinfo_kernels.hpp"
#include "fastinfo_typed_kernels.hpp"

class MexFunction : public fastinfo::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 8 || outputs.size() > 2) {
            fail("fastinfo_calccondcmi_cpp:usage",
                "fastinfo_calccondcmi_cpp(X, Xm, Y, Ym, Z, Zm, K, Km) expects eight inputs.");
        }

        const auto xm = scalar_to_size(inputs[1], "Xm");
        const auto ym = scalar_to_size(inputs[3], "Ym");
        const auto zm = scalar_to_size(inputs[5], "Zm");
        const auto km = scalar_to_size(inputs[7], "Km");
        dispatch_integer_array(inputs[0], "X", [&](const auto& x) {
            validate_discrete_array_view(x, xm, "X");
            dispatch_integer_array(inputs[2], "Y", [&](const auto& y) {
                validate_discrete_array_view(y, ym, "Y");
                dispatch_integer_array(inputs[4], "Z", [&](const auto& z) {
                    validate_discrete_array_view(z, zm, "Z");
                    dispatch_integer_array(inputs[6], "K", [&](const auto& k) {
                        validate_discrete_array_view(k, km, "K");
                        if (x.getNumberOfElements() != y.getNumberOfElements()
                            || x.getNumberOfElements() != z.getNumberOfElements()
                            || x.getNumberOfElements() != k.getNumberOfElements()) {
                            fail("fastinfo_calccondcmi_cpp:shape", "X, Y, Z, and K must contain the same number of samples.");
                        }

                        auto total = factory_.createScalar(0.0);
                        auto contributions = factory_.createArray<double>({km, 1});
                        double totalValue = 0.0;
                        fastinfo::typed::calc_cond_cmi(
                            raw_data(x), xm, raw_data(y), ym, raw_data(z), zm, raw_data(k), km,
                            x.getNumberOfElements(), &totalValue, raw_data(contributions));
                        total[0] = totalValue;
                        outputs[0] = total;
                        if (outputs.size() > 1) {
                            outputs[1] = contributions;
                        }
                    });
                });
            });
        });
    }
};
