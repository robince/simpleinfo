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
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import simpleinfo

XB = 16
YB = 8
ZB = 4
PERM_SEED = 5489
PRIME = np.int64(2_147_483_647)


def mode_config(mode: str) -> dict[str, int]:
    mode = str(mode).lower()
    if mode == "quick":
        return {
            "equiv_ntrl": 768,
            "equiv_nx": 24,
            "equiv_nmatched": 16,
            "equiv_nperm": 24,
            "scaling_ntrl": 2_048,
            "scaling_nx": 192,
            "scaling_nmatched": 128,
            "scaling_nperm": 96,
        }
    if mode == "full":
        return {
            "equiv_ntrl": 2_048,
            "equiv_nx": 64,
            "equiv_nmatched": 48,
            "equiv_nperm": 64,
            "scaling_ntrl": 6_144,
            "scaling_nx": 512,
            "scaling_nmatched": 320,
            "scaling_nperm": 256,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def _hash_grid(ntrl: int, npage: int, seed: int) -> np.ndarray:
    t = (np.arange(ntrl, dtype=np.int64)[:, None] + 1)
    p = (np.arange(npage, dtype=np.int64)[None, :] + 1)
    state = (
        np.int64(seed)
        + np.int64(104_729) * t
        + np.int64(17_077) * p
        + np.int64(433) * t * p
        + np.int64(811) * t * t
        + np.int64(233) * p * p
    ) % PRIME
    return state.astype(np.int64, copy=False)


def discrete_vector(ntrl: int, nbins: int, seed: int) -> np.ndarray:
    return (_hash_grid(ntrl, 1, seed).reshape(ntrl) % np.int64(nbins)).astype(np.int64, copy=False)


def discrete_matrix(ntrl: int, npage: int, nbins: int, seed: int) -> np.ndarray:
    return (_hash_grid(ntrl, npage, seed) % np.int64(nbins)).astype(np.int64, copy=False)


def build_equivalence_inputs(mode: str) -> dict[str, Any]:
    cfg = mode_config(mode)
    ntrl = cfg["equiv_ntrl"]
    nx = cfg["equiv_nx"]
    nmatched = cfg["equiv_nmatched"]

    mi_none_y = discrete_vector(ntrl, YB, 2001)
    mi_none = {
        "x_scalar": discrete_vector(ntrl, XB, 1001),
        "y": mi_none_y,
        "x_slice": discrete_matrix(ntrl, nx, XB, 3001),
        "x_matched": discrete_matrix(ntrl, nmatched, XB, 4001),
        "y_matched": discrete_matrix(ntrl, nmatched, YB, 5001),
    }

    mi_shared = discrete_vector(ntrl, YB, 6001)
    mi_effect_y = (mi_shared + (discrete_vector(ntrl, 2, 6002) % 2)) % YB
    mi_effect = {
        "x_scalar": (mi_shared + YB * discrete_vector(ntrl, 2, 6003) + discrete_vector(ntrl, 2, 6004)) % XB,
        "y": mi_effect_y,
        "x_slice": (
            mi_shared[:, None]
            + YB * discrete_matrix(ntrl, nx, 2, 6005)
            + discrete_matrix(ntrl, nx, 2, 6006)
        ) % XB,
        "x_matched": None,
        "y_matched": None,
    }
    mi_shared_match = discrete_matrix(ntrl, nmatched, YB, 6101)
    mi_effect["x_matched"] = (
        mi_shared_match
        + YB * discrete_matrix(ntrl, nmatched, 2, 6102)
        + discrete_matrix(ntrl, nmatched, 2, 6103)
    ) % XB
    mi_effect["y_matched"] = (mi_shared_match + discrete_matrix(ntrl, nmatched, 2, 6104)) % YB

    cmi_none_z = discrete_vector(ntrl, ZB, 7001)
    cmi_none = {
        "x_scalar": (3 * cmi_none_z + discrete_vector(ntrl, XB, 7002)) % XB,
        "y": (2 * cmi_none_z + discrete_vector(ntrl, YB, 7003)) % YB,
        "z": cmi_none_z,
        "x_slice": (3 * cmi_none_z[:, None] + discrete_matrix(ntrl, nx, XB, 7004)) % XB,
    }

    cmi_effect_z = discrete_vector(ntrl, ZB, 8001)
    cmi_shared = discrete_vector(ntrl, YB, 8002)
    cmi_effect = {
        "x_scalar": (
            cmi_shared
            + 3 * cmi_effect_z
            + YB * discrete_vector(ntrl, 2, 8003)
            + discrete_vector(ntrl, 2, 8004)
        ) % XB,
        "y": (cmi_shared + 2 * cmi_effect_z + discrete_vector(ntrl, 2, 8005)) % YB,
        "z": cmi_effect_z,
        "x_slice": (
            cmi_shared[:, None]
            + 3 * cmi_effect_z[:, None]
            + YB * discrete_matrix(ntrl, nx, 2, 8006)
            + discrete_matrix(ntrl, nx, 2, 8007)
        ) % XB,
    }

    return {
        "config": cfg,
        "mi_none": mi_none,
        "mi_effect": mi_effect,
        "cmi_none": cmi_none,
        "cmi_effect": cmi_effect,
    }


def build_scaling_inputs(mode: str) -> dict[str, np.ndarray]:
    cfg = mode_config(mode)
    ntrl = cfg["scaling_ntrl"]
    nx = cfg["scaling_nx"]
    nmatched = cfg["scaling_nmatched"]
    nperm = cfg["scaling_nperm"]

    shared = discrete_vector(ntrl, YB, 9001)
    z = discrete_vector(ntrl, ZB, 9002)
    y = (shared + z + discrete_vector(ntrl, 2, 9003)) % YB

    return {
        "x_slice": (
            shared[:, None]
            + 2 * z[:, None]
            + YB * discrete_matrix(ntrl, nx, 2, 9004)
            + discrete_matrix(ntrl, nx, 2, 9005)
        ) % XB,
        "y": y,
        "z": z,
        "x_matched": (
            discrete_matrix(ntrl, nmatched, YB, 9010)
            + YB * discrete_matrix(ntrl, nmatched, 2, 9011)
            + discrete_matrix(ntrl, nmatched, 2, 9012)
        ) % XB,
        "y_matched": (
            discrete_matrix(ntrl, nmatched, YB, 9010)
            + discrete_matrix(ntrl, nmatched, 2, 9013)
        ) % YB,
        "nperm": nperm,
    }


def _py_fast_layout(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64).T)


def _serialize(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "values": array.reshape(-1).tolist(),
    }


def _max_abs_diff(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def reference_calcinfo_slice(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray([simpleinfo.calcinfo(x[:, col], XB, y, YB, bias=False) for col in range(x.shape[1])], dtype=np.float64)


def equivalence_case_specs(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "scenario": "mi_none",
            "operation": "calcinfo",
            "naive": lambda: simpleinfo.calcinfo(inputs["mi_none"]["x_scalar"], XB, inputs["mi_none"]["y"], YB, bias=False),
            "fast": lambda: simpleinfo.fastinfo.calcinfo(inputs["mi_none"]["x_scalar"], XB, inputs["mi_none"]["y"], YB),
        },
        {
            "scenario": "mi_effect",
            "operation": "calcinfo",
            "naive": lambda: simpleinfo.calcinfo(inputs["mi_effect"]["x_scalar"], XB, inputs["mi_effect"]["y"], YB, bias=False),
            "fast": lambda: simpleinfo.fastinfo.calcinfo(inputs["mi_effect"]["x_scalar"], XB, inputs["mi_effect"]["y"], YB),
        },
        {
            "scenario": "mi_none",
            "operation": "calcinfo_slice",
            "naive": lambda: reference_calcinfo_slice(inputs["mi_none"]["x_slice"], inputs["mi_none"]["y"]),
            "fast": lambda: simpleinfo.fastinfo.calcinfo_slice(_py_fast_layout(inputs["mi_none"]["x_slice"]), XB, inputs["mi_none"]["y"], YB),
        },
        {
            "scenario": "mi_effect",
            "operation": "calcinfo_slice",
            "naive": lambda: reference_calcinfo_slice(inputs["mi_effect"]["x_slice"], inputs["mi_effect"]["y"]),
            "fast": lambda: simpleinfo.fastinfo.calcinfo_slice(_py_fast_layout(inputs["mi_effect"]["x_slice"]), XB, inputs["mi_effect"]["y"], YB),
        },
        {
            "scenario": "mi_none",
            "operation": "calcinfomatched",
            "naive": lambda: simpleinfo.calcinfomatched(inputs["mi_none"]["x_matched"], XB, inputs["mi_none"]["y_matched"], YB, bias=False),
            "fast": lambda: simpleinfo.fastinfo.calcinfomatched(
                _py_fast_layout(inputs["mi_none"]["x_matched"]),
                XB,
                _py_fast_layout(inputs["mi_none"]["y_matched"]),
                YB,
            ),
        },
        {
            "scenario": "mi_effect",
            "operation": "calcinfomatched",
            "naive": lambda: simpleinfo.calcinfomatched(inputs["mi_effect"]["x_matched"], XB, inputs["mi_effect"]["y_matched"], YB, bias=False),
            "fast": lambda: simpleinfo.fastinfo.calcinfomatched(
                _py_fast_layout(inputs["mi_effect"]["x_matched"]),
                XB,
                _py_fast_layout(inputs["mi_effect"]["y_matched"]),
                YB,
            ),
        },
        {
            "scenario": "cmi_none",
            "operation": "calccmi",
            "naive": lambda: simpleinfo.calccmi(
                inputs["cmi_none"]["x_scalar"], XB, inputs["cmi_none"]["y"], YB, inputs["cmi_none"]["z"], ZB, bias=False
            ),
            "fast": lambda: simpleinfo.fastinfo.calccmi(
                inputs["cmi_none"]["x_scalar"], XB, inputs["cmi_none"]["y"], YB, inputs["cmi_none"]["z"], ZB
            ),
        },
        {
            "scenario": "cmi_effect",
            "operation": "calccmi",
            "naive": lambda: simpleinfo.calccmi(
                inputs["cmi_effect"]["x_scalar"], XB, inputs["cmi_effect"]["y"], YB, inputs["cmi_effect"]["z"], ZB, bias=False
            ),
            "fast": lambda: simpleinfo.fastinfo.calccmi(
                inputs["cmi_effect"]["x_scalar"], XB, inputs["cmi_effect"]["y"], YB, inputs["cmi_effect"]["z"], ZB
            ),
        },
        {
            "scenario": "cmi_none",
            "operation": "calccmi_slice",
            "naive": lambda: simpleinfo.calccmi_slice(
                inputs["cmi_none"]["x_slice"], XB, inputs["cmi_none"]["y"], YB, inputs["cmi_none"]["z"], ZB, bias=False
            ),
            "fast": lambda: simpleinfo.fastinfo.calccmi_slice(
                _py_fast_layout(inputs["cmi_none"]["x_slice"]), XB, inputs["cmi_none"]["y"], YB, inputs["cmi_none"]["z"], ZB
            ),
        },
        {
            "scenario": "cmi_effect",
            "operation": "calccmi_slice",
            "naive": lambda: simpleinfo.calccmi_slice(
                inputs["cmi_effect"]["x_slice"], XB, inputs["cmi_effect"]["y"], YB, inputs["cmi_effect"]["z"], ZB, bias=False
            ),
            "fast": lambda: simpleinfo.fastinfo.calccmi_slice(
                _py_fast_layout(inputs["cmi_effect"]["x_slice"]), XB, inputs["cmi_effect"]["y"], YB, inputs["cmi_effect"]["z"], ZB
            ),
        },
    ]


def run_python_equivalence(mode: str) -> list[dict[str, Any]]:
    inputs = build_equivalence_inputs(mode)
    cases = []
    for spec in equivalence_case_specs(inputs):
        naive = np.asarray(spec["naive"](), dtype=np.float64)
        fast = np.asarray(spec["fast"](), dtype=np.float64)
        cases.append({
            "scenario": spec["scenario"],
            "operation": spec["operation"],
            "naive": _serialize(naive),
            "fast": _serialize(fast),
            "max_abs_diff": _max_abs_diff(naive, fast),
        })
    return cases


def _median_runtime(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def run_python_scaling(mode: str, thread_counts: list[int], repeats: int, warmup: int) -> dict[str, Any]:
    backend = simpleinfo.fastinfo._api.BACKEND
    if backend != "numba":
        return {
            "available": False,
            "backend": backend,
            "reason": "Python thread scaling requires the Numba backend.",
            "cases": [],
        }

    inputs = build_scaling_inputs(mode)
    cases = [
        {
            "operation": "calcinfo_slice",
            "fn": lambda: simpleinfo.fastinfo.calcinfo_slice(_py_fast_layout(inputs["x_slice"]), XB, inputs["y"], YB),
        },
        {
            "operation": "calcinfomatched",
            "fn": lambda: simpleinfo.fastinfo.calcinfomatched(
                _py_fast_layout(inputs["x_matched"]), XB, _py_fast_layout(inputs["y_matched"]), YB
            ),
        },
        {
            "operation": "calccmi_slice",
            "fn": lambda: simpleinfo.fastinfo.calccmi_slice(_py_fast_layout(inputs["x_slice"]), XB, inputs["y"], YB, inputs["z"], ZB),
        },
        {
            "operation": "calcinfoperm_slice",
            "fn": lambda: simpleinfo.fastinfo.calcinfoperm_slice(
                _py_fast_layout(inputs["x_slice"]), XB, inputs["y"], YB, inputs["nperm"], seed=PERM_SEED
            ),
        },
    ]

    results = []
    for case in cases:
        baseline = None
        baseline_time = None
        for threads in thread_counts:
            simpleinfo.fastinfo.set_threads(threads)
            current = np.asarray(case["fn"](), dtype=np.float64)
            current_time = _median_runtime(case["fn"], repeats, warmup)
            if baseline is None:
                baseline = current
                baseline_time = current_time
            results.append({
                "operation": case["operation"],
                "threads": int(threads),
                "seconds": current_time,
                "speedup_vs_1": float(baseline_time / current_time) if baseline_time is not None else 1.0,
                "max_abs_diff_vs_1": _max_abs_diff(current, baseline),
            })
        simpleinfo.fastinfo.set_threads(thread_counts[0])
    return {
        "available": True,
        "backend": backend,
        "cases": results,
    }


def run_python_comparative(mode: str, thread_counts: list[int], repeats: int, warmup: int) -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode,
        "backend": simpleinfo.fastinfo._api.BACKEND,
        "thread_counts": [int(x) for x in thread_counts],
        "repeats": int(repeats),
        "warmup": int(warmup),
        "config": mode_config(mode),
        "equivalence": run_python_equivalence(mode),
        "scaling": run_python_scaling(mode, thread_counts, repeats, warmup),
    }


def write_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--thread-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repeats = args.repeats if args.repeats is not None else (3 if args.mode == "quick" else 5)
    results = run_python_comparative(args.mode, args.thread_counts, repeats, args.warmup)

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = REPO_ROOT / "build" / "benchmarks" / f"python_fastinfo_comparative_{stamp}.json"
    write_json(results, output)
    print(f"Wrote benchmark results to {output}")


if __name__ == "__main__":
    main()
