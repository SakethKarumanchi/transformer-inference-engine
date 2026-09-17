#!/usr/bin/env python3
"""bench/microbench/run_all.py -- driver for the nine Stage 0 microbenchmarks.

Builds them, runs them in order, leaves each benchmark's own results file in
bench/results/, and emits one consolidated Stage 0 file that the Stage 10
performance model reads.

Re-runnable by later stages: `--compare <prior consolidated file>` runs the
suite afresh and reports every value that moved, so a machine that has drifted
is found before its numbers are trusted.

A microbenchmark that fails is reported as failed. It never leaves a stale
value from a previous run, and never a placeholder, in the consolidated file:
its entry carries status "failed" and no value at all.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
BUILD_DIR = REPO_ROOT / "build"
CONSOLIDATED = RESULTS_DIR / "stage0_consolidated.json"

# The nine, in the order the contract lists them.
BENCHMARKS = [
    (1, "gpu_bandwidth",          "GB/s"),
    (2, "gpu_fp32_peak",          "GFLOP/s"),
    (3, "cpu_cache_ladder",       "GB/s"),
    (4, "cpu_simd_peak",          "GFLOP/s"),
    (5, "host_device_transfer",   "GB/s"),
    (6, "cublas_sgemm_ref",       "GFLOP/s"),
    (7, "kernel_launch_overhead", "us"),
    (8, "shared_mem_bandwidth",   "GB/s"),
    (9, "occupancy_sweep",        "ms"),
]
BENCHMARK_NAMES = [b[1] for b in BENCHMARKS]

DEFAULT_TIMEOUT_S = 1800


def _default_runner(cmd, cwd=None, timeout=DEFAULT_TIMEOUT_S):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def build(runner=_default_runner) -> dict:
    """Builds through scripts/build.ps1, which pins the VS 2022 toolset."""
    script = REPO_ROOT / "scripts" / "build.ps1"
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
               "-File", str(script)]
    else:
        cmd = ["cmake", "--build", str(BUILD_DIR)]
    rc, out, err = runner(cmd, cwd=str(REPO_ROOT))
    return {"ok": rc == 0, "returncode": rc,
            "stdout_tail": out[-4000:], "stderr_tail": err[-4000:]}


def exe_path(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return BUILD_DIR / f"{name}{suffix}"


def run_one(name: str, warmup: int | None = None, samples: int | None = None,
            runner=_default_runner) -> dict:
    """Runs one microbenchmark. Its own results file is written by the binary.

    Exit codes: 0 all configurations valid; 2 ran but at least one
    configuration is INVALID under the 5% rule; anything else is a failure.
    """
    exe = exe_path(name)
    if not exe.exists():
        return {"status": "failed", "reason": f"binary not built: {exe}"}
    cmd = [str(exe)]
    if warmup is not None:
        cmd.append(str(warmup))
        cmd.append(str(samples if samples is not None else ""))
        cmd = [c for c in cmd if c != ""]
    started = time.time()
    try:
        rc, out, err = runner(cmd, cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": f"timed out after {DEFAULT_TIMEOUT_S}s"}
    elapsed = round(time.time() - started, 3)

    if rc == 0:
        status = "ok"
        reason = None
    elif rc == 2:
        status = "invalid"
        reason = ("ran to completion but at least one configuration exceeded the "
                  "5% standard-deviation limit and is reported INVALID")
    else:
        status = "failed"
        reason = f"exit code {rc}"

    return {"status": status, "reason": reason, "returncode": rc,
            "elapsed_s": elapsed, "stdout": out[-8000:], "stderr": err[-4000:]}


def load_benchmark_results(name: str, results_dir: Path = RESULTS_DIR) -> dict | None:
    path = results_dir / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"_parse_error": str(e)}


def headline_values(doc: dict) -> dict:
    """{configuration: {value, units, valid}} for one benchmark's results file."""
    out = {}
    for rec in (doc or {}).get("records", []):
        out[rec.get("configuration", "?")] = {
            "value": rec.get("value"),
            "units": rec.get("units"),
            "valid": (rec.get("statistics") or {}).get("valid"),
            "stddev_pct_of_median": (rec.get("statistics") or {}).get("stddev_pct_of_median"),
        }
    return out


def shared_versus_global_finding(consolidated: dict) -> dict:
    """Post-run check, deliberately not a unit test.

    The unit tests run in the offline gate before any timing exists, so shared
    memory bandwidth cannot be compared against global memory bandwidth there.
    Here both figures exist, so the comparison is made and reported as a
    finding: shared memory that is not substantially faster than global memory
    points at a measurement error in one of the two.
    """
    def pick(name, needle):
        entry = consolidated["benchmarks"].get(name, {})
        for cfg, v in (entry.get("values") or {}).items():
            if needle in cfg and isinstance(v.get("value"), (int, float)):
                return cfg, v["value"]
        return None, None

    g_cfg, g = pick("gpu_bandwidth", "copy")
    s_cfg, s = pick("shared_mem_bandwidth", "conflict-free")
    if g is None or s is None:
        return {"available": False,
                "reason": "one of the two bandwidth figures is missing; "
                          f"global={g_cfg!r} shared={s_cfg!r}"}
    ratio = s / g if g else None
    return {
        "available": True,
        "global_gb_per_s": g,
        "shared_gb_per_s": s,
        "shared_over_global": ratio,
        "finding": ("shared memory bandwidth exceeds global memory bandwidth, as "
                    "expected for on-chip storage"
                    if ratio and ratio > 1.0 else
                    "shared memory bandwidth does NOT exceed global memory "
                    "bandwidth; one of the two measurements is wrong and both "
                    "should be re-examined before the Stage 10 model uses them"),
    }


def consolidate(run_records: dict, results_dir: Path = RESULTS_DIR) -> dict:
    """Builds the single Stage 0 file. A failed benchmark contributes no value."""
    benchmarks = {}
    for order, name, units in BENCHMARKS:
        rec = run_records.get(name, {"status": "not run",
                                     "reason": "the driver did not run it"})
        entry = {"order": order, "units": units, "status": rec.get("status"),
                 "reason": rec.get("reason"), "elapsed_s": rec.get("elapsed_s")}
        if rec.get("status") in ("ok", "invalid"):
            doc = load_benchmark_results(name, results_dir)
            if doc is None:
                entry["status"] = "failed"
                entry["reason"] = (f"{name} reported success but wrote no results "
                                   f"file at {results_dir / (name + '.json')}")
                entry["values"] = None
            else:
                entry["values"] = headline_values(doc)
                entry["git_commit"] = doc.get("git_commit")
                entry["device"] = doc.get("device")
                entry["cxx_flags"] = doc.get("cxx_flags")
                entry["cuda_flags"] = doc.get("cuda_flags")
        else:
            # Explicitly no value. Never a stale figure from an earlier run.
            entry["values"] = None
        benchmarks[name] = entry

    consolidated = {
        "stage": "stage-0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": benchmarks,
        "all_nine_ran": all(benchmarks[n]["status"] in ("ok", "invalid")
                            for n in BENCHMARK_NAMES),
        "failed": [n for n in BENCHMARK_NAMES if benchmarks[n]["status"] == "failed"],
        "invalid": [n for n in BENCHMARK_NAMES if benchmarks[n]["status"] == "invalid"],
    }
    consolidated["findings"] = {
        "shared_versus_global": shared_versus_global_finding(consolidated)
    }
    return consolidated


def compare_drift(current: dict, prior: dict, tolerance_pct: float = 5.0) -> dict:
    """Reports every value that moved by more than tolerance_pct."""
    moved, appeared, disappeared = [], [], []
    cur_b, old_b = current.get("benchmarks", {}), prior.get("benchmarks", {})
    for name in sorted(set(cur_b) | set(old_b)):
        cv = (cur_b.get(name) or {}).get("values") or {}
        ov = (old_b.get(name) or {}).get("values") or {}
        for cfg in sorted(set(cv) | set(ov)):
            a = (ov.get(cfg) or {}).get("value")
            b = (cv.get(cfg) or {}).get("value")
            if a is None and b is not None:
                appeared.append({"benchmark": name, "configuration": cfg, "value": b})
                continue
            if b is None and a is not None:
                disappeared.append({"benchmark": name, "configuration": cfg, "value": a})
                continue
            if a is None or b is None:
                continue
            delta = b - a
            pct = (abs(delta) / abs(a) * 100.0) if a else float("inf")
            if pct > tolerance_pct:
                moved.append({"benchmark": name, "configuration": cfg,
                              "stage0_value": a, "current_value": b,
                              "delta": delta, "delta_pct": pct})
    return {"tolerance_pct": tolerance_pct,
            "drifted": moved, "appeared": appeared, "disappeared": disappeared,
            "machine_unchanged": not moved and not appeared and not disappeared}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--warmup", type=int, default=None,
                    help="warmup iterations; the protocol minimum is enforced regardless")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--only", action="append", default=None,
                    help="run only these benchmarks (repeatable)")
    ap.add_argument("--out", default=str(CONSOLIDATED))
    ap.add_argument("--compare", default=None,
                    help="prior consolidated file to compare this run against")
    ap.add_argument("--tolerance-pct", type=float, default=5.0)
    args = ap.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_build:
        b = build()
        if not b["ok"]:
            print("BUILD FAILED; not running anything.", file=sys.stderr)
            print(b["stderr_tail"] or b["stdout_tail"], file=sys.stderr)
            return 1

    names = args.only or BENCHMARK_NAMES
    unknown = [n for n in names if n not in BENCHMARK_NAMES]
    if unknown:
        print(f"unknown benchmark(s): {unknown}", file=sys.stderr)
        return 2

    records = {}
    for order, name, _units in BENCHMARKS:
        if name not in names:
            continue
        print(f"[{order}/9] {name} ...", flush=True)
        rec = run_one(name, args.warmup, args.samples)
        records[name] = rec
        print(rec.get("stdout", ""), end="")
        print(f"      -> {rec['status']}"
              + (f": {rec['reason']}" if rec.get("reason") else ""), flush=True)

    consolidated = consolidate(records)
    Path(args.out).write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    print(f"\nconsolidated Stage 0 results: {args.out}")
    print(f"  all nine ran : {consolidated['all_nine_ran']}")
    print(f"  failed       : {consolidated['failed'] or 'none'}")
    print(f"  INVALID      : {consolidated['invalid'] or 'none'}")
    svg = consolidated["findings"]["shared_versus_global"]
    print(f"  finding      : {svg.get('finding') or svg.get('reason')}")

    exit_code = 0
    if consolidated["failed"]:
        exit_code = 1
    elif consolidated["invalid"]:
        exit_code = 2

    if args.compare:
        prior = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        drift = compare_drift(consolidated, prior, args.tolerance_pct)
        print("\ndrift against " + args.compare)
        if drift["machine_unchanged"]:
            print("  no value moved by more than "
                  f"{args.tolerance_pct}%; the machine has not drifted")
        else:
            for d in drift["drifted"]:
                print(f"  MOVED {d['benchmark']} / {d['configuration']}: "
                      f"{d['stage0_value']} -> {d['current_value']} "
                      f"({d['delta_pct']:.2f}%)")
            for d in drift["appeared"]:
                print(f"  NEW   {d['benchmark']} / {d['configuration']}")
            for d in drift["disappeared"]:
                print(f"  GONE  {d['benchmark']} / {d['configuration']}")
            exit_code = exit_code or 3
        Path(args.out).with_suffix(".drift.json").write_text(
            json.dumps(drift, indent=2), encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
