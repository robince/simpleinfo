# Fastinfo Port Overview

## Goal

Extend `simpleinfo` with an optimized native runtime while keeping the current
top-level functions as a tutorial-friendly reference interface.

The optimized layer should:

- mirror the practical scope of the legacy `info` package where that scope is
  still useful
- follow the modern MATLAB build and packaging style used in `gcmi`
- expose backend-neutral public names so MATLAB and Python stay aligned
- use OpenMP internally by default rather than exposing public `_omp`
  entrypoints

## Public API Direction

### Tutorial / reference interface

Keep the existing simple top-level MATLAB and Python functions as the primary
easy-to-read interface:

- `calcinfo`
- `calccmi`
- `calcinfoperm`
- `eqpopbin`
- `rebin`
- `numbase2dec`
- `numdec2base`

These remain the readable, tutorial-oriented functions. They should not depend
on the optimized runtime by default in the first implementation phase.

### Optimized interface

Expose the native runtime under a single `fastinfo` namespace.

MATLAB:

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

Python:

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

Rationale:

- tutorial functions stay short and approachable
- optimized functions are clearly separated without a deep package hierarchy
- the namespace is backend-neutral and can be mirrored in Python cleanly
- user-facing fast-path names should match the tutorial names rather than use a
  separate underscore style

## Interface Rules

### General wrapper style

Public wrapper functions should use:

- positional arguments for the scientific inputs
- name-value options for runtime controls

Examples:

```matlab
I = fastinfo.calcinfo(x, xb, y, yb);
I = fastinfo.calcinfo(x, xb, y, yb, Threads=8);
xbin = fastinfo.eqpop(x, nb);
```

Likely initial wrapper options:

- `Threads`
- `Validate`

Bias correction options can remain in the tutorial path first unless there is a
clear reason to move them into the native wrappers immediately.

### OpenMP policy

OpenMP is an implementation detail, not a public namespace distinction.

That means:

- no public `_omp` MATLAB or Python API
- no separate public `fastinfo_omp`
- wrappers call the OpenMP-enabled native kernels by default when available

Non-OpenMP kernels can still exist internally for:

- fallback builds
- testing
- debugging

### Bin-count arguments

For the optimized information kernels, require explicit bin/state counts:

- `fastinfo.calcinfo(x, xb, y, yb)`
- `fastinfo.calccmi(x, xb, y, yb, z, zb)`

Do not infer `xb`, `yb`, or `zb` from `max(x)+1` in the fast path.

Rationale:

- it matches the native counting kernels directly
- it avoids silently accepting malformed label sets
- it keeps MATLAB and Python contracts unambiguous

### Integer input policy

For the discrete `fastinfo` kernels, integer label arrays are the intended
public input type.

MATLAB fast path:

- zero-copy MEX dispatch currently supports `int16`, `int32`, and `int64`
- users should convert integer-valued `double` arrays before calling the fast
  discrete kernels
- `fastinfo.eqpop` and `fastinfo.eqpop_sorted` return integer labels, so the
  normal path `eqpop -> calcinfo` stays on the typed fast route

Python fast path:

- integer arrays are the intended public input type
- Numba can specialize kernels by dtype automatically
- some validated Python paths may still normalize inputs to `int64`
  internally, so dtype specialization is currently less strict than the MATLAB
  zero-copy path

## Binning Semantics

### `fastinfo.eqpop`

`fastinfo.eqpop` is intended for real-valued data.

New behavior should prioritize mathematical consistency over legacy rank-only
compatibility:

- equal values must not be split across bins
- exact `nb` bins must be produced
- if ties make exact equipopulation binning impossible, the function errors
- the error message should direct users to `rebin` for already discrete or
  heavily quantized data

This intentionally differs from the old native implementation, which could
split ties.

Suggested error shape:

> Cannot form `nb` equal-population bins without splitting tied values. If the
> input is discrete or strongly quantized, use `rebin` instead.

### `fastinfo.eqpop_sorted`

`fastinfo.eqpop_sorted` has the same tie policy as `fastinfo.eqpop`, but
assumes the input is already sorted and should avoid sorting internally.

This remains important for pairwise workflows and other paths where sorted
inputs are already available.

### `rebin`

`rebin` should be ported for both MATLAB and Python, but does not need a MEX
implementation initially.

`rebin` is the intended tool for:

- already discrete labels
- quantized data with many repeated values
- workflows where exact `nb` equipopulation binning is not well-defined

## Utility Functions To Keep Cross-Language

The following should exist in both MATLAB and Python as ordinary vectorized
functions:

- `rebin`
- `numbase2dec`
- `numdec2base`

These do not need native code first. The goal is:

- clear contracts
- efficient vectorized implementations
- identical semantics across both languages

## Native Kernel Scope

### Implemented kernel scope

The current optimized runtime includes:

- scalar MI
- scalar CMI
- permutation / bootstrap MI
- slice MI with OpenMP over columns
- matched-column batched MI
- conditional-per-condition CMI contributions
- CMI slice
- permutation MI slice
- `eqpop`
- `eqpop_sorted`
- `eqpop_slice`
- `eqpop_sorted_slice`
- pairwise helper workflows built on sorted binning

The typed low-level MEX entrypoints exist only as internal implementation
details. The public API remains the `fastinfo.*` wrapper layer.

## Proposed Repository Layout

### MATLAB

```text
matlab/
  calcinfo.m
  calccmi.m
  calcinfoperm.m
  eqpopbin.m
  rebin.m
  numbase2dec.m
  numdec2base.m
  +fastinfo/
    calcinfo.m
    calccmi.m
    calcinfoperm.m
    calcinfo_slice.m
    eqpop.m
    eqpop_sorted.m
  cpp_mex/
    README.md
    include/
    src/
    mex/
    tooling/
    tests/
    bin/<release>/<mexext>/
```

### Python

```text
python/src/simpleinfo/
  __init__.py
  core.py
  fastinfo/
    __init__.py
    _api.py
    _fallback.py
```

Python `fastinfo` is now a NumPy-first implementation with optional Numba
acceleration.

## Python Runtime Plan

The Python optimized layer should follow the same public API shape as MATLAB,
but should not introduce compiled-extension complexity in the first phase.

Recommended staged approach:

1. keep `simpleinfo.fastinfo` API-matched with MATLAB
2. use NumPy as the baseline implementation
3. use optional Numba acceleration for the hot counting and permutation paths
4. defer any Python C++ extension or shared native-core binding until there is a
   demonstrated need for it

Rationale:

- MATLAB already carries the first native build complexity through MEX and
  OpenMP
- NumPy first keeps the Python side easy to install and easy to test
- optional Numba can provide most of the practical speedup without committing to
  pybind11 or a second compiled toolchain immediately

## Build And Release Direction

Follow the `gcmi` pattern for MATLAB native builds:

- root `buildfile.m`
- `matlab/cpp_mex/tooling/*` for config, compile, package, clean, and test
- release-specific binary output under `matlab/cpp_mex/bin/<release>/<mexext>/`
- GitHub Actions for:
  - Python CI
  - MATLAB build and smoke test
  - packaged MEX release artifacts

Key principle:

- build tooling should understand compiler family and OpenMP flags centrally
- native wrappers should be thin and use the modern MATLAB C++ MEX interface

## C++ Design Direction

Follow the algorithm mapping in the legacy spec, with the deliberate tie-policy
change for equipopulation binning.

Core design:

- one shared C++ counting / entropy library
- thin MEX wrappers per public capability
- heap-backed scratch buffers
- OpenMP over columns or permutation replicates
- dense histogram counting for MI and CMI
- zero-copy typed dispatch for MATLAB integer inputs on the main discrete kernels

Likely internal modules:

- histogram utilities
- entropy-from-counts utilities
- MI / CMI kernels
- permutation utilities
- eqpop binning
- MATLAB adapter utilities

## Testing Strategy

### MATLAB

Add regression tests for:

- numerical agreement between tutorial and fast paths on supported inputs
- tie-handling error behavior in `fastinfo.eqpop`
- shape handling for vector and slice inputs
- thread-count override plumbing

### Python

Add tests for:

- tutorial/reference functions
- `rebin`
- `numbase2dec`
- `numdec2base`
- API consistency for the `fastinfo` namespace

### Cross-language

Maintain a small shared fixture set so MATLAB and Python agree on:

- label conventions
- MI / CMI outputs
- base-conversion utilities
- tie-related binning failures

## Phased Plan

1. Lock the public names and function signatures.
2. Set up MATLAB native build tooling modeled on `gcmi`.
3. Implement shared C++ utilities plus `eqpop_sorted` and scalar MI first.
4. Add `eqpop`, `calccmi`, and `calcinfo_slice`.
5. Add permutation kernels and OpenMP tuning.
6. Port `rebin`, `numbase2dec`, and `numdec2base` cleanly for MATLAB and
   Python.
7. Add CI and packaged MATLAB release artifacts.

## Remaining Interface Questions

These are now treated as decided:

1. User-facing optimized names should match the tutorial names:
   - `fastinfo.calcinfo`
   - `fastinfo.calccmi`
   - `fastinfo.calcinfoperm`
   - `fastinfo.calcinfo_slice`

2. `fastinfo.eqpop_sorted` should be public from the start.
