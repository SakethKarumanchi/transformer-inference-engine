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


def _default_runner(cmd, cwd=None, timeout=DEFAULT_TIMEOUT_S, env=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, env=env)
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
            runner=_default_runner, env: dict | None = None) -> dict:
    """Runs one microbenchmark. Its own results file is written by the binary.

    `env` overrides the child environment; Stage 0b uses it to point
    BENCH_RESULTS_DIR at bench/results/stage0b/runN and BENCH_STAGE_ID at
    "stage-0b", so a Stage 0b figure is never written into a Stage 0 path and
    never labelled with the stage whose figures it replaces.

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
        if env is None:
            rc, out, err = runner(cmd, cwd=str(REPO_ROOT))
        else:
            rc, out, err = runner(cmd, cwd=str(REPO_ROOT), env=env)
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


# --------------------------------------------------------------- Stage 0b ---
# Stage 0b re-runs ONLY the two CPU microbenchmarks whose figures were INVALID
# in both Stage 0 suite runs. Everything that produced a valid Stage 0 figure is
# left alone and no Stage 0 results file is written, overwritten or deleted:
# bench/results/ and its run1/ and run2/ subdirectories are evidence and are
# immutable. Stage 0b output goes to bench/results/stage0b/run1/ and run2/, and
# _guard_stage0b_path() refuses any other destination rather than trusting the
# caller to have got it right.

STAGE0B_DIR = RESULTS_DIR / "stage0b"
STAGE0B_BENCHMARKS = ["cpu_cache_ladder", "cpu_simd_peak"]
STAGE0B_WARMUP = 25          # matches Stage 0 (BENCHMARK_PROTOCOL.md 4.2)
STAGE0B_SAMPLES = 30         # matches Stage 0
STAGE0B_CONSOLIDATED = STAGE0B_DIR / "stage0b_consolidated.json"


class Stage0ResultsAreImmutable(RuntimeError):
    """Raised rather than writing anywhere a Stage 0 results file could live."""


def _guard_stage0b_path(path: Path) -> Path:
    """Every Stage 0b write goes through here. Nothing else may."""
    p = Path(path).resolve()
    root = STAGE0B_DIR.resolve()
    if p != root and root not in p.parents:
        raise Stage0ResultsAreImmutable(
            f"refusing to write {p}: Stage 0b writes only under {root}. "
            f"Stage 0 results are evidence and are never overwritten.")
    return p


def stage0b_run_dir(run: int) -> Path:
    if run not in (1, 2):
        raise ValueError(f"Stage 0b has runs 1 and 2, not {run}")
    return _guard_stage0b_path(STAGE0B_DIR / f"run{run}")


def run_stage0b(runs=(1, 2), telemetry: bool = True,
                warmup: int = STAGE0B_WARMUP, samples: int = STAGE0B_SAMPLES,
                runner=_default_runner) -> dict:
    """Runs cpu_cache_ladder and cpu_simd_peak twice, under protocol conditions.

    Run 2 is the reference, as Stage 0 treated it.

    Each benchmark process is optionally shadowed by a CPU frequency sampler
    running in THIS process. The sampler cannot be inside the benchmark's timed
    bracket -- it is not even in the same process -- and its own cost and duty
    cycle are recorded so its contribution to background load is visible rather
    than assumed negligible. It is a recorded run CONDITION, not a measurement.
    """
    sampler_factory = None
    if telemetry:
        try:
            sys.path.insert(0, str(REPO_ROOT / "bench"))
            import machine_state                                # noqa: PLC0415
            sampler_factory = machine_state.CpuTelemetrySampler
        except Exception as e:                                  # noqa: BLE001
            print(f"CPU telemetry sampler unavailable, continuing without it: {e}",
                  file=sys.stderr)

    out = {"stage": "stage-0b",
           "benchmarks": STAGE0B_BENCHMARKS,
           "warmup": warmup, "samples": samples,
           "reference_run": 2,
           "telemetry_requested": telemetry,
           "runs": {}}

    for run in runs:
        d = stage0b_run_dir(run)
        d.mkdir(parents=True, exist_ok=True)
        run_rec = {}
        for name in STAGE0B_BENCHMARKS:
            print(f"[stage-0b run{run}] {name} ...", flush=True)
            env = dict(os.environ)
            env["BENCH_RESULTS_DIR"] = str(d)
            env["BENCH_STAGE_ID"] = "stage-0b"

            sampler = None
            if sampler_factory is not None:
                try:
                    sampler = sampler_factory().start()
                    aff = sampler.affinity
                    print(f"  telemetry sampler pinned to logical CPUs "
                          f"{aff['allowed_logical_cpus']} (mask {aff['mask_hex']}), "
                          f"clear of {aff['excluded_logical_cpus']}", flush=True)
                except Exception as e:                          # noqa: BLE001
                    # An unpinnable sampler sharing a physical core with the
                    # measured thread is worse than no trace. Recorded, not
                    # silently dropped.
                    print(f"  CPU frequency sampler NOT started: {e}",
                          file=sys.stderr)
                    out.setdefault("telemetry_not_started", []).append(
                        {"run": run, "benchmark": name, "reason": str(e)})
                    sampler = None

            rec = run_one(name, warmup, samples, runner=runner, env=env)

            if sampler is not None:
                trace = sampler.stop()
                tpath = _guard_stage0b_path(d / f"{name}_cpu_frequency_trace.json")
                tpath.write_text(json.dumps(trace, indent=2), encoding="utf-8")
                rec["cpu_frequency_trace"] = str(tpath.relative_to(REPO_ROOT))
                rec["cpu_frequency_summary"] = trace["summary"]
                rec["sampler_duty_cycle_of_one_core"] = trace["sampler_duty_cycle_of_one_core"]
                rec["sampler_affinity"] = {
                    "mask_hex": trace["affinity"]["mask_hex"],
                    "allowed_logical_cpus": trace["affinity"]["allowed_logical_cpus"],
                    "excluded_logical_cpus": trace["affinity"]["excluded_logical_cpus"],
                    "applied": trace["affinity_applied"],
                }
                rec["sampler_interval_s"] = trace["interval_s"]
                rec["sampler_per_probe_us"] = trace["per_probe_us"]

            doc = load_benchmark_results(name, results_dir=d)
            if doc is None:
                rec["status"] = "failed"
                rec["reason"] = (f"{name} wrote no results file at "
                                 f"{d / (name + '.json')}")
                rec["values"] = None
            else:
                rec["values"] = headline_values(doc)
                rec["git_commit"] = doc.get("git_commit")
                rec["cxx_flags"] = doc.get("cxx_flags")
                rec["stage_recorded"] = doc.get("stage")
            print(rec.get("stdout", ""), end="")
            print(f"      -> {rec['status']}"
                  + (f": {rec['reason']}" if rec.get("reason") else ""), flush=True)
            run_rec[name] = rec
        out["runs"][f"run{run}"] = run_rec

    # Every configuration that is still INVALID, listed as invalid. Never
    # averaged away, never silently retried.
    invalid = []
    for run_key, benches in out["runs"].items():
        for name, rec in benches.items():
            for cfg, v in (rec.get("values") or {}).items():
                if v.get("valid") is False:
                    invalid.append({"run": run_key, "benchmark": name,
                                    "configuration": cfg,
                                    "stddev_pct_of_median": v.get("stddev_pct_of_median")})
    out["invalid_configurations"] = invalid
    out["n_invalid"] = len(invalid)
    out["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


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
    ap.add_argument("--stage0b", action="store_true",
                    help="Stage 0b mode: run ONLY cpu_cache_ladder and cpu_simd_peak, "
                         "twice, writing to bench/results/stage0b/run1 and run2. "
                         "No Stage 0 results path is written, overwritten or deleted.")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="Stage 0b mode: do not shadow the runs with the CPU "
                         "frequency sampler")
    args = ap.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage0b:
        if not args.no_build:
            b = build()
            if not b["ok"]:
                print("BUILD FAILED; not running anything.", file=sys.stderr)
                print(b["stderr_tail"] or b["stdout_tail"], file=sys.stderr)
                return 1
        data = run_stage0b(telemetry=not args.no_telemetry,
                           warmup=args.warmup or STAGE0B_WARMUP,
                           samples=args.samples or STAGE0B_SAMPLES)
        out_path = _guard_stage0b_path(STAGE0B_CONSOLIDATED)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nconsolidated Stage 0b results: {out_path}")
        if data["invalid_configurations"]:
            print(f"  INVALID configurations ({data['n_invalid']}):")
            for c in data["invalid_configurations"]:
                print(f"    {c['run']} {c['benchmark']} / {c['configuration']}: "
                      f"{c['stddev_pct_of_median']:.3f}% of median")
        else:
            print("  INVALID configurations: none")
        return 2 if data["invalid_configurations"] else 0

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
