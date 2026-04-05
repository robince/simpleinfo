#include "fastinfo_mex_adapter_utils.hpp"

#include "fastinfo_kernels.hpp"

class MexFunction : public fastinfo::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 2 || outputs.size() > 1) {
            fail("fastinfo_eqpop_cpp:usage", "fastinfo_eqpop_cpp(X, Nb) expects two inputs.");
        }

        const auto x = require_double_array(inputs[0], "X");
        if (x.getDimensions().size() > 2 || !(x.getDimensions()[0] == 1 || x.getDimensions().size() == 1 || x.getDimensions()[1] == 1)) {
            fail("fastinfo_eqpop_cpp:shape", "X must be a real vector.");
        }
        const auto nBins = scalar_to_size(inputs[1], "Nb");
        if (x.getNumberOfElements() < nBins) {
            fail("fastinfo_eqpop_cpp:shape", "Nb cannot exceed the number of samples.");
        }

        auto out = factory_.createArray<int32_t>(x.getDimensions());
        fastinfo::eqpop(raw_data(x), x.getNumberOfElements(), nBins, raw_data(out));
        outputs[0] = out;
    }
};
