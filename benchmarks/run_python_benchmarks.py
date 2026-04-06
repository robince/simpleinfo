from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import simpleinfo


def fixed_repeat_time(fn, repeats):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return float(np.median(np.asarray(timings)))


def measure_case(case, repeats):
    result = {"name": case["name"]}
    result["fast"] = fixed_repeat_time(case["fast"], repeats)
    if case.get("reference") is not None:
        result["reference"] = fixed_repeat_time(case["reference"], repeats)
    return result


def build_cases(mode):
    rng = np.random.default_rng(42)
    cases = []

    if mode == "quick":
        ntrl = 1500
        nx = 192
        nperm = 96
    else:
        ntrl = 4000
        nx = 512
        nperm = 256

    x = rng.integers(0, 16, size=ntrl, dtype=np.int64)
    y = rng.integers(0, 8, size=ntrl, dtype=np.int64)
    z = rng.integers(0, 4, size=ntrl, dtype=np.int64)
    X = rng.integers(0, 16, size=(nx, ntrl), dtype=np.int64)
    Ymatched = rng.integers(0, 8, size=(nx, ntrl), dtype=np.int64)

    cases.append({
        "name": "calcinfo_scalar",
        "fast": lambda: simpleinfo.fastinfo.calcinfo(x, 16, y, 8),
        "reference": lambda: simpleinfo.calcinfo(x, 16, y, 8, bias=False),
    })
    cases.append({
        "name": "calcinfo_slice",
        "fast": lambda: simpleinfo.fastinfo.calcinfo_slice(X, 16, y, 8),
        "reference": lambda: np.array([simpleinfo.calcinfo(X[row], 16, y, 8, bias=False) for row in range(X.shape[0])]),
    })
    cases.append({
        "name": "calcinfomatched",
        "fast": lambda: simpleinfo.fastinfo.calcinfomatched(X, 16, Ymatched, 8),
        "reference": lambda: simpleinfo.calcinfomatched(X.T, 16, Ymatched.T, 8, bias=False),
    })
    cases.append({
        "name": "calccmi_slice",
        "fast": lambda: simpleinfo.fastinfo.calccmi_slice(X, 16, y, 8, z, 4),
        "reference": lambda: simpleinfo.calccmi_slice(X.T, 16, y, 8, z, 4, bias=False),
    })
    cases.append({
        "name": "calcinfoperm",
        "fast": lambda: simpleinfo.fastinfo.calcinfoperm(x, 16, y, 8, nperm, seed=123),
        "reference": lambda: simpleinfo.calcinfoperm(x, 16, y, 8, nperm, bias=False),
    })

    Xperm = rng.integers(0, 16, size=(48 if mode == "quick" else 96, ntrl), dtype=np.int64)
    cases.append({
        "name": "calcinfoperm_slice",
        "fast": lambda: simpleinfo.fastinfo.calcinfoperm_slice(Xperm, 16, y, 8, nperm, seed=123),
        "reference": lambda: simpleinfo.calcinfoperm_slice(Xperm.T, 16, y, 8, nperm, bias=False),
    })

    cont = rng.standard_normal(size=40000 if mode == "quick" else 120000)
    cont_sorted = np.sort(cont)
    cont_pages = rng.standard_normal(size=(32, 8000 if mode == "quick" else 16000))
    cont_pages_sorted = np.sort(cont_pages, axis=1)
    cases.append({
        "name": "eqpop_sorted",
        "fast": lambda: simpleinfo.fastinfo.eqpop_sorted(cont_sorted, 8),
        "reference": None,
    })
    cases.append({
        "name": "eqpop_slice",
        "fast": lambda: simpleinfo.fastinfo.eqpop_slice(cont_pages, 8),
        "reference": lambda: np.vstack([simpleinfo.fastinfo.eqpop(cont_pages[row], 8) for row in range(cont_pages.shape[0])]),
    })
    cases.append({
        "name": "eqpop_sorted_slice",
        "fast": lambda: simpleinfo.fastinfo.eqpop_sorted_slice(cont_pages_sorted, 8),
        "reference": lambda: np.vstack(
            [simpleinfo.fastinfo.eqpop_sorted(cont_pages_sorted[row], 8) for row in range(cont_pages_sorted.shape[0])]
        ),
    })
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repeats = args.repeats if args.repeats is not None else (3 if args.mode == "quick" else 7)
    cases = [measure_case(case, repeats) for case in build_cases(args.mode)]
    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": simpleinfo.fastinfo._api.BACKEND,
        "mode": args.mode,
        "repeats": repeats,
        "cases": cases,
    }

    output = args.output
    if output is None:
        output_dir = REPO_ROOT / "build" / "benchmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"python_fastinfo_benchmarks_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"Wrote benchmark results to {output}")


if __name__ == "__main__":
    main()
