# Benchmarks

The main MATLAB benchmark entrypoint is:

- `run_matlab_benchmarks`
- `run_python_benchmarks.py`

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
```

Benchmark artifacts are written under:

- `build/benchmarks/`
