#include "fastinfo_mex_adapter_utils.hpp"

#include "fastinfo_eqpop_kernels.hpp"

class MexFunction : public fastinfo::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() > 1) {
            fail("fastinfo_eqpop_slice_cpp:usage",
                "fastinfo_eqpop_slice_cpp(X, Nb, Nthread) expects two or three inputs.");
        }

        const auto x = require_double_array(inputs[0], "X");
        if (x.getDimensions().size() != 2) {
            fail("fastinfo_eqpop_slice_cpp:shape", "X must be a real matrix of shape [Ntrl, Npage].");
        }
        const auto dims = x.getDimensions();
        const auto nRows = dims[0];
        const auto nCols = dims[1];
        const auto nBins = scalar_to_size(inputs[1], "Nb");
        const auto nThreads = inputs.size() == 3 ? scalar_to_size(inputs[2], "Nthread") : 0;
        if (nRows < nBins) {
            fail("fastinfo_eqpop_slice_cpp:shape", "Nb cannot exceed the number of rows in X.");
        }

        auto out = factory_.createArray<double>(dims);
        fastinfo::eqpop_slice(raw_data(x), nRows, nCols, nBins, nThreads, raw_data(out));
        outputs[0] = out;
    }
};
