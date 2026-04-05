# Simpleinfo: simple calculation of information-theoretic quantities

`simpleinfo` is a small tutorial-style package for computing binned
information-theoretic quantities in both Python and MATLAB.

The repository now contains two layers:

- top-level tutorial/reference functions such as `calcinfo`, `calccmi`,
  `eqpopbin`, `rebin`, `numbase2dec`, and `numdec2base`
- an optimized `fastinfo` namespace in MATLAB and Python

The MATLAB optimized runtime is backed by C++ MEX with OpenMP and follows the
modern `buildtool`-based structure used in `gcmi`.

## Python

Install from the repository root:

```bash
uv pip install -e .
```

Install with optional Numba acceleration:

```bash
uv pip install -e ".[opt]"
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

Python `fastinfo` accepts integer label arrays. Numba can specialize kernels to
the input dtype automatically, but the current validated Python path may still
normalize some inputs to `int64` internally. So integer arrays are the right
user-facing input type, but the MATLAB fast path is currently the stricter
zero-copy implementation.

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

## Benchmarks

MATLAB benchmarking lives under [benchmarks/run_matlab_benchmarks.m](benchmarks/run_matlab_benchmarks.m).
It measures current `fastinfo` performance against the tutorial/reference path,
OpenMP scaling, and optional legacy Fortran MEX comparisons when the old
`info` repo is available.
