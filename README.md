# Simpleinfo: simple calculation of information-theoretic quantities

`simpleinfo` is a small tutorial-style package for computing binned
information-theoretic quantities in both Python and MATLAB.

The repository now contains two layers:

- top-level tutorial/reference functions such as `calcinfo`, `calccmi`,
  `calcinfoperm`, `eqpopbin`, `rebin`, `numbase2dec`, and `numdec2base`
- an optimized `fastinfo` namespace in MATLAB and Python

The MATLAB optimized runtime is backed by C++ MEX with OpenMP and follows the
modern `buildtool`-based structure used in `gcmi`.

## Python

Install from the repository root:

```bash
python -m pip install -e .
```

Run the Python tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

For Python, the batched `simpleinfo.fastinfo` APIs use trial-last, C-contiguous
matrices: `shape == (n_pages, n_trials)`. This differs from MATLAB and from the
top-level tutorial/reference batch functions, which keep the MATLAB-style
trial-first layout. If you need to adapt a trial-first matrix `x`, use
`np.ascontiguousarray(x.T)` before calling `simpleinfo.fastinfo`.

The public optimized Python API includes:

- `simpleinfo.fastinfo.calcinfo`
- `simpleinfo.fastinfo.calccmi`
- `simpleinfo.fastinfo.calcinfoperm`
- `simpleinfo.fastinfo.calcinfo_slice`
- `simpleinfo.fastinfo.calcinfomatched`
- `simpleinfo.fastinfo.calccondcmi`
- `simpleinfo.fastinfo.calccmi_slice`
- `simpleinfo.fastinfo.calcinfoperm_slice`
- `simpleinfo.fastinfo.eqpop`
- `simpleinfo.fastinfo.eqpop_sorted`
- `simpleinfo.fastinfo.eqpop_slice`
- `simpleinfo.fastinfo.eqpop_sorted_slice`
- `simpleinfo.fastinfo.calcpairwiseinfo`
- `simpleinfo.fastinfo.calcpairwiseinfo_slice`
- `simpleinfo.fastinfo.get_threads`
- `simpleinfo.fastinfo.set_threads`

`simpleinfo.calccondcmi` and `simpleinfo.fastinfo.calccondcmi` return
`(total, contributions)`, where `contributions[k]` is the globally weighted
contribution from condition `K == k` and `total == contributions.sum()`. In
other words, the returned total is the decomposition over both `Z` and `K`, not
just `I(X;Y|Z)`.

`simpleinfo.fastinfo.set_threads(n)` returns the previous global thread count
for the active backend. `simpleinfo.fastinfo.get_threads()` returns the current
global thread count, or `None` when the fallback backend is active.

Python `fastinfo` accepts integer label arrays. Numba can specialize kernels to
the input dtype automatically, but some validated Python paths may still
normalize inputs to `int64` internally. So integer arrays are the right
user-facing input type, but the MATLAB fast path is currently the stricter
zero-copy implementation.

Typical Python usage:

```python
import numpy as np
import simpleinfo

xbin = simpleinfo.fastinfo.eqpop(x_continuous, 8)
I = simpleinfo.fastinfo.calcinfo(xbin, 8, y, yb)

X_fast = np.ascontiguousarray(X_trial_first.T)
simpleinfo.fastinfo.set_threads(4)
I_pages = simpleinfo.fastinfo.calcinfo_slice(X_fast, 8, y, yb)
Iperm = simpleinfo.fastinfo.calcinfoperm_slice(X_fast, 8, y, yb, 256, seed=123)
```

If you upgrade the package in place after changes to the cached Numba kernels
and hit unexpected import or compilation errors, clear Python cache artifacts
such as `__pycache__` before retrying.

## MATLAB

Build the native runtime from the repository root inside MATLAB:

```matlab
buildtool compile
buildtool test
```

The public optimized MATLAB API lives under:

- `fastinfo.calcinfo`
- `fastinfo.calccmi`
- `fastinfo.calcinfoperm`
- `fastinfo.calcinfo_slice`
- `fastinfo.calcinfomatched`
- `fastinfo.calccondcmi`
- `fastinfo.calccmi_slice`
- `fastinfo.calcinfoperm_slice`
- `fastinfo.eqpop`
- `fastinfo.eqpop_sorted`
- `fastinfo.eqpop_slice`
- `fastinfo.eqpop_sorted_slice`
- `fastinfo.calcpairwiseinfo`
- `fastinfo.calcpairwiseinfo_slice`

`calccondcmi` and `fastinfo.calccondcmi` return `[I, IK]`, where `IK(ki)` is
the globally weighted contribution for `K == ki - 1` and `I == sum(IK)`.

For the discrete-information kernels, the intended fast MATLAB inputs are
integer label arrays:

- `int16`
- `int32`
- `int64`

These classes use the zero-copy MEX dispatch path. `double` integer-valued
arrays are not the intended fast-path input type; convert them to an integer
class first.

`fastinfo.eqpop` and `fastinfo.eqpop_sorted` return integer labels, so the
intended MATLAB workflow is:

```matlab
xbin = fastinfo.eqpop(x, nb);
I = fastinfo.calcinfo(xbin, nb, y, yb);
```

Since `eqpop` returns integer labels directly, that path stays on the fast
typed route without an extra conversion step.

Typical MATLAB usage:

```matlab
xbin = fastinfo.eqpop(x, 8);
I = fastinfo.calcinfo(xbin, 8, y, yb);

Ipages = fastinfo.calcinfo_slice(X, 8, y, yb, Threads=4);
Iperm = fastinfo.calcinfoperm_slice(X, 8, y, yb, 256, Threads=4, Seed=123);
```

## Benchmarks

MATLAB benchmarking lives under [benchmarks/run_matlab_benchmarks.m](benchmarks/run_matlab_benchmarks.m).
It measures current `fastinfo` performance against the tutorial/reference path,
OpenMP scaling, and optional legacy Fortran MEX comparisons when the old
`info` repo is available.
