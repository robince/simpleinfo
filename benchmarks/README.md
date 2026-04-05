# Benchmarks

The main MATLAB benchmark entrypoint is:

- `run_matlab_benchmarks`
- `run_matlab_dtype_benchmarks`
- `run_matlab_comparative_benchmarks`
- `run_python_benchmarks.py`
- `run_python_dtype_benchmarks.py`
- `run_python_comparative_benchmarks.py`
- `run_comparative_benchmarks.py`

It is designed to:

- measure `fastinfo` runtime on representative scalar, slice, and binning
  workloads
- compare against the tutorial/reference implementations by default
- optionally compare against the legacy Fortran MEX implementation from the
  `info` repo when available
- report OpenMP scaling for the slice and permutation workloads

Example:

```matlab
addpath(fullfile(pwd, 'benchmarks'));
results = run_matlab_benchmarks('Compile', true, 'LegacyRepo', '/path/to/info');
```

Python:

```bash
./.venv/bin/python benchmarks/run_python_benchmarks.py --mode quick
./.venv/bin/python benchmarks/run_python_dtype_benchmarks.py --mode quick --thread-counts 1 2 4
```

Benchmark artifacts are written under:

- `build/benchmarks/`

Comparative benchmark examples:

```bash
./.venv/bin/python benchmarks/run_python_comparative_benchmarks.py --mode quick
./.venv/bin/python benchmarks/run_comparative_benchmarks.py --mode quick --thread-counts 1 2 4
```

MATLAB:

```matlab
addpath(fullfile(pwd, 'benchmarks'));
results = run_matlab_comparative_benchmarks('Mode', 'quick', 'ThreadCounts', [1 2 4]);
dtypeResults = run_matlab_dtype_benchmarks('Mode', 'quick', 'ThreadCounts', [1 2 4]);
```

The dtype harnesses focus specifically on the discrete fast kernels and compare:

- `int16`
- `int32`
- `int64`

for:

- `calcinfo`
- `calcinfo_slice`
- `calcinfomatched`
- `calccmi`
- `calccmi_slice`
- `calcinfoperm`
- `calcinfoperm_slice`

The comparative harness is split into two passes:

- an all-path equivalence pass over deterministic estimators on shared simulated
  data with and without effects
- a threaded scaling pass for the accelerated implementations, verifying that
  multi-threaded outputs still match the single-thread baseline
