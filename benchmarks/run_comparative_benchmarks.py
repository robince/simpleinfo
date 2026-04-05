from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.comparative_fastinfo import run_python_comparative, write_json

DEFAULT_MATLAB_CMD = "/Applications/MATLAB_R2024b.app/bin/matlab"


def _decode_payload(payload: dict[str, Any]) -> np.ndarray:
    shape = payload["shape"]
    values = np.asarray(payload["values"], dtype=np.float64)
    if shape in (None, [], ()):
        return values.reshape(())
    if isinstance(shape, int):
        shape = [shape]
    return values.reshape(tuple(shape))


def _pairwise_max_abs_diff(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    keys = sorted(arrays)
    out: dict[str, float] = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            out[f"{left}__vs__{right}"] = float(np.max(np.abs(arrays[left] - arrays[right])))
    return out


def _index_equivalence(cases: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(case["scenario"], case["operation"]): case for case in cases}


def compare_equivalence(matlab_results: dict[str, Any], python_results: dict[str, Any]) -> list[dict[str, Any]]:
    matlab_cases = _index_equivalence(matlab_results["equivalence"])
    python_cases = _index_equivalence(python_results["equivalence"])
    shared_keys = sorted(set(matlab_cases) & set(python_cases))
    comparisons = []
    for key in shared_keys:
        matlab_case = matlab_cases[key]
        python_case = python_cases[key]
        arrays = {
            "matlab_naive": _decode_payload(matlab_case["naive"]),
            "python_naive": _decode_payload(python_case["naive"]),
            "matlab_fast": _decode_payload(matlab_case["fast"]),
            "python_fast": _decode_payload(python_case["fast"]),
        }
        pairwise = _pairwise_max_abs_diff(arrays)
        comparisons.append({
            "scenario": key[0],
            "operation": key[1],
            "pairwise_max_abs_diff": pairwise,
            "all_paths_max_abs_diff": max(pairwise.values()) if pairwise else 0.0,
            "matlab_naive_vs_fast": float(matlab_case["max_abs_diff"]),
            "python_naive_vs_fast": float(python_case["max_abs_diff"]),
        })
    return comparisons


def compare_scaling(matlab_results: dict[str, Any], python_results: dict[str, Any]) -> list[dict[str, Any]]:
    if not matlab_results.get("scaling", {}).get("available", False):
        return []
    if not python_results.get("scaling", {}).get("available", False):
        return []

    matlab_cases = {(case["operation"], int(case["threads"])): case for case in matlab_results["scaling"]["cases"]}
    python_cases = {(case["operation"], int(case["threads"])): case for case in python_results["scaling"]["cases"]}
    shared_keys = sorted(set(matlab_cases) & set(python_cases))
    merged = []
    for key in shared_keys:
        matlab_case = matlab_cases[key]
        python_case = python_cases[key]
        merged.append({
            "operation": key[0],
            "threads": key[1],
            "matlab_seconds": float(matlab_case["seconds"]),
            "python_seconds": float(python_case["seconds"]),
            "matlab_speedup_vs_1": float(matlab_case["speedup_vs_1"]),
            "python_speedup_vs_1": float(python_case["speedup_vs_1"]),
            "matlab_max_abs_diff_vs_1": float(matlab_case["max_abs_diff_vs_1"]),
            "python_max_abs_diff_vs_1": float(python_case["max_abs_diff_vs_1"]),
            "python_vs_matlab_time_ratio": float(python_case["seconds"] / matlab_case["seconds"]) if matlab_case["seconds"] else np.inf,
        })
    return merged


def run_matlab_comparative(
    *,
    mode: str,
    thread_counts: list[int],
    repeats: int,
    warmup: int,
    compile_mex: bool,
    matlab_cmd: str,
    output: Path,
) -> dict[str, Any]:
    benchmarks_dir = REPO_ROOT / "benchmarks"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    thread_expr = " ".join(str(int(x)) for x in thread_counts)
    batch = (
        f"addpath('{benchmarks_dir.as_posix()}');"
        f"run_matlab_comparative_benchmarks('Mode','{mode}','ThreadCounts',[{thread_expr}],"
        f"'Repeats',{int(repeats)},'Warmup',{int(warmup)},'Compile',{str(bool(compile_mex)).lower()},"
        f"'Verbose',false,'OutputFile','{output.as_posix()}');"
    )
    command = f"{shlex.quote(matlab_cmd)} -batch {shlex.quote(batch)}"
    completed = subprocess.run(
        ["/bin/zsh", "-lc", command],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        if output.exists():
            try:
                return json.loads(output.read_text())
            except json.JSONDecodeError:
                pass
        raise RuntimeError(
            "MATLAB comparative benchmark failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(output.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--thread-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--matlab-output", type=Path, default=None)
    parser.add_argument("--skip-matlab", action="store_true")
    parser.add_argument("--compile-matlab", action="store_true")
    parser.add_argument("--matlab-cmd", default=DEFAULT_MATLAB_CMD)
    args = parser.parse_args()

    repeats = args.repeats if args.repeats is not None else (3 if args.mode == "quick" else 5)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    python_results = run_python_comparative(args.mode, args.thread_counts, repeats, args.warmup)

    matlab_results = None
    if not args.skip_matlab:
        matlab_output = args.matlab_output or (REPO_ROOT / "build" / "benchmarks" / f"matlab_fastinfo_comparative_{stamp}.json")
        matlab_results = run_matlab_comparative(
            mode=args.mode,
            thread_counts=args.thread_counts,
            repeats=repeats,
            warmup=args.warmup,
            compile_mex=args.compile_matlab,
            matlab_cmd=args.matlab_cmd,
            output=matlab_output,
        )

    combined = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": args.mode,
        "thread_counts": [int(x) for x in args.thread_counts],
        "repeats": int(repeats),
        "warmup": int(args.warmup),
        "python": python_results,
        "matlab": matlab_results,
        "cross_path_equivalence": compare_equivalence(matlab_results, python_results) if matlab_results is not None else [],
        "cross_path_scaling": compare_scaling(matlab_results, python_results) if matlab_results is not None else [],
    }

    output = args.output or (REPO_ROOT / "build" / "benchmarks" / f"fastinfo_comparative_{stamp}.json")
    write_json(combined, output)
    print(f"Wrote comparative benchmark results to {output}")


if __name__ == "__main__":
    main()
