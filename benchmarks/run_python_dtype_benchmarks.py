from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import simpleinfo
from benchmarks.comparative_fastinfo import XB, YB, ZB, PERM_SEED, discrete_matrix, discrete_vector


def _cast_discrete(values: np.ndarray, dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    return np.asarray(values, dtype=dtype)


def _py_fast_layout(values: np.ndarray, dtype_name: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype_name)).T)


def _median_runtime(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return float(np.median(np.asarray(timings, dtype=np.float64)))


def _resolve_regimes(mode: str, regimes: list[str] | None) -> list[str]:
    if regimes:
        return [str(regime).lower() for regime in regimes]
    mode = str(mode).lower()
    if mode == "quick":
        return ["medium"]
    if mode == "full":
        return ["large"]
    raise ValueError(f"Unsupported mode: {mode}")


def _regime_config(regime: str) -> dict[str, int]:
    regime = str(regime).lower()
    if regime == "small":
        return {
            "ntrl": 256,
            "nx": 128,
            "nmatched": 128,
            "nperm": 64,
        }
    if regime == "medium":
        return {
            "ntrl": 1024,
            "nx": 512,
            "nmatched": 512,
            "nperm": 128,
        }
    if regime == "large":
        return {
            "ntrl": 4096,
            "nx": 2048,
            "nmatched": 1024,
            "nperm": 256,
        }
    raise ValueError(f"Unsupported regime: {regime}")


def _build_dtype_inputs(regime: str) -> dict[str, Any]:
    cfg = _regime_config(regime)
    shared = discrete_vector(cfg["ntrl"], YB, 9001)
    z = discrete_vector(cfg["ntrl"], ZB, 9002)
    y = (shared + z + discrete_vector(cfg["ntrl"], 2, 9003)) % YB
    return {
        "regime": regime,
        "config": cfg,
        "x_slice": (
            shared[:, None]
            + 2 * z[:, None]
            + YB * discrete_matrix(cfg["ntrl"], cfg["nx"], 2, 9004)
            + discrete_matrix(cfg["ntrl"], cfg["nx"], 2, 9005)
        ) % XB,
        "y": y,
        "z": z,
        "x_matched": (
            discrete_matrix(cfg["ntrl"], cfg["nmatched"], YB, 9010)
            + YB * discrete_matrix(cfg["ntrl"], cfg["nmatched"], 2, 9011)
            + discrete_matrix(cfg["ntrl"], cfg["nmatched"], 2, 9012)
        ) % XB,
        "y_matched": (
            discrete_matrix(cfg["ntrl"], cfg["nmatched"], YB, 9010)
            + discrete_matrix(cfg["ntrl"], cfg["nmatched"], 2, 9013)
        ) % YB,
        "nperm": cfg["nperm"],
    }


def _calibrated_repeats(fn, warmup: int, target_seconds: float, max_repeats: int) -> int:
    sample_time = _median_runtime(fn, 1, warmup)
    if sample_time <= 0.0:
        return max_repeats
    repeats = int(np.ceil(target_seconds / sample_time))
    return max(1, min(max_repeats, repeats))


def _benchmark_cases(inputs: dict[str, np.ndarray], dtype_name: str) -> list[dict[str, Any]]:
    x_scalar = _cast_discrete(inputs["x_slice"][:, 0], dtype_name)
    y = _cast_discrete(inputs["y"], dtype_name)
    z = _cast_discrete(inputs["z"], dtype_name)
    x_slice = _py_fast_layout(inputs["x_slice"], dtype_name)
    x_matched = _py_fast_layout(inputs["x_matched"], dtype_name)
    y_matched = _py_fast_layout(inputs["y_matched"], dtype_name)

    return [
        {
            "operation": "calcinfo",
            "kind": "scalar",
            "fn": lambda: simpleinfo.fastinfo.calcinfo(x_scalar, XB, y, YB),
        },
        {
            "operation": "calcinfo_slice",
            "kind": "threaded",
            "fn": lambda: simpleinfo.fastinfo.calcinfo_slice(x_slice, XB, y, YB),
        },
        {
            "operation": "calcinfomatched",
            "kind": "threaded",
            "fn": lambda: simpleinfo.fastinfo.calcinfomatched(x_matched, XB, y_matched, YB),
        },
        {
            "operation": "calccmi",
            "kind": "scalar",
            "fn": lambda: simpleinfo.fastinfo.calccmi(x_scalar, XB, y, YB, z, ZB),
        },
        {
            "operation": "calccmi_slice",
            "kind": "threaded",
            "fn": lambda: simpleinfo.fastinfo.calccmi_slice(x_slice, XB, y, YB, z, ZB),
        },
        {
            "operation": "calcinfoperm",
            "kind": "threaded",
            "fn": lambda: simpleinfo.fastinfo.calcinfoperm(x_scalar, XB, y, YB, int(inputs["nperm"]), seed=PERM_SEED),
        },
        {
            "operation": "calcinfoperm_slice",
            "kind": "threaded",
            "fn": lambda: simpleinfo.fastinfo.calcinfoperm_slice(x_slice, XB, y, YB, int(inputs["nperm"]), seed=PERM_SEED),
        },
    ]


def run_python_dtype_benchmarks(
    mode: str,
    regimes: list[str] | None,
    thread_counts: list[int],
    repeats: int | None,
    warmup: int,
    dtypes: list[str],
    target_seconds_per_dtype: float,
    max_repeats: int,
) -> dict[str, Any]:
    backend = simpleinfo.fastinfo._api.BACKEND
    if backend != "numba":
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "backend": backend,
            "available": False,
            "reason": "Python dtype scaling benchmarks require the Numba backend.",
            "mode": mode,
            "regimes": _resolve_regimes(mode, regimes),
            "thread_counts": [int(x) for x in thread_counts],
            "repeats": None if repeats is None else int(repeats),
            "warmup": int(warmup),
            "target_seconds_per_dtype": float(target_seconds_per_dtype),
            "max_repeats": int(max_repeats),
            "dtypes": list(dtypes),
            "cases": [],
        }

    regime_names = _resolve_regimes(mode, regimes)
    original_threads = simpleinfo.fastinfo.set_threads(thread_counts[0])
    cases: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    try:
        for regime_name in regime_names:
            inputs = _build_dtype_inputs(regime_name)
            configs.append({"regime": regime_name, **inputs["config"]})
            for dtype_name in dtypes:
                specs = _benchmark_cases(inputs, dtype_name)
                measurements_per_dtype = sum(len(thread_counts) if spec["kind"] == "threaded" else 1 for spec in specs)
                target_seconds_per_measurement = target_seconds_per_dtype / float(measurements_per_dtype)
                for spec in specs:
                    if spec["kind"] == "threaded":
                        baseline = None
                        baseline_time = None
                        for threads in thread_counts:
                            simpleinfo.fastinfo.set_threads(int(threads))
                            current = np.asarray(spec["fn"](), dtype=np.float64)
                            repeats_used = repeats if repeats is not None else _calibrated_repeats(
                                spec["fn"], warmup, target_seconds_per_measurement, max_repeats
                            )
                            current_time = _median_runtime(spec["fn"], repeats_used, warmup)
                            if baseline is None:
                                baseline = current
                                baseline_time = current_time
                            cases.append({
                                "regime": regime_name,
                                "dtype": dtype_name,
                                "operation": spec["operation"],
                                "kind": spec["kind"],
                                "threads": int(threads),
                                "repeats": int(repeats_used),
                                "seconds": current_time,
                                "speedup_vs_1": float(baseline_time / current_time),
                                "max_abs_diff_vs_1": float(np.max(np.abs(current - baseline))),
                            })
                    else:
                        simpleinfo.fastinfo.set_threads(int(thread_counts[0]))
                        repeats_used = repeats if repeats is not None else _calibrated_repeats(
                            spec["fn"], warmup, target_seconds_per_measurement, max_repeats
                        )
                        current_time = _median_runtime(spec["fn"], repeats_used, warmup)
                        cases.append({
                            "regime": regime_name,
                            "dtype": dtype_name,
                            "operation": spec["operation"],
                            "kind": spec["kind"],
                            "threads": int(thread_counts[0]),
                            "repeats": int(repeats_used),
                            "seconds": current_time,
                            "speedup_vs_1": 1.0,
                            "max_abs_diff_vs_1": 0.0,
                        })
    finally:
        simpleinfo.fastinfo.set_threads(int(original_threads))

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": backend,
        "available": True,
        "mode": mode,
        "regimes": regime_names,
        "thread_counts": [int(x) for x in thread_counts],
        "repeats": None if repeats is None else int(repeats),
        "warmup": int(warmup),
        "target_seconds_per_dtype": float(target_seconds_per_dtype),
        "max_repeats": int(max_repeats),
        "dtypes": list(dtypes),
        "configs": configs,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--regimes", nargs="+", default=None)
    parser.add_argument("--thread-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtypes", nargs="+", default=["int16", "int32", "int64"])
    parser.add_argument("--target-seconds-per-dtype", type=float, default=10.0)
    parser.add_argument("--max-repeats", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = run_python_dtype_benchmarks(
        args.mode,
        args.regimes,
        args.thread_counts,
        args.repeats,
        args.warmup,
        args.dtypes,
        args.target_seconds_per_dtype,
        args.max_repeats,
    )

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = REPO_ROOT / "build" / "benchmarks" / f"python_fastinfo_dtypes_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"Wrote dtype benchmark results to {output}")


if __name__ == "__main__":
    main()
