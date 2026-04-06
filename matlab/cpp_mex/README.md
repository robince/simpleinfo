# MATLAB C++ MEX Runtime

This directory contains the optimized MATLAB-native C++ MEX runtime used by the
`fastinfo.*` wrapper functions.

## Scope

The first native runtime provides:

- `fastinfo_calcinfo_cpp`
- `fastinfo_calccmi_cpp`
- `fastinfo_calcinfoperm_cpp`
- `fastinfo_calcinfo_slice_cpp`
- `fastinfo_eqpop_cpp`
- `fastinfo_eqpop_sorted_cpp`

OpenMP is enabled by default for the slice and permutation kernels.

## Build

Before building or using the wrappers from a fresh MATLAB session, add the
repository MATLAB path from the repo root:

```matlab
setup_simpleinfo
```

From the repository root inside MATLAB:

```matlab
buildtool compile
buildtool test
buildtool package
```

Compiled binaries are written to:

- `matlab/cpp_mex/bin/<MATLAB release>/<mexext>/`

## Wrapper API

End users should call the MATLAB wrappers under `+fastinfo`, not the raw MEX
entrypoints directly.
