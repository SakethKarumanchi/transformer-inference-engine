#!/usr/bin/env python3
"""bench/profile.py -- reusable Nsight Compute wrapper for this project.

Runs a CUDA binary under ncu and extracts the counter set the project
contract requires, per kernel:

    achieved occupancy            theoretical occupancy
    DRAM read throughput          DRAM write throughput
    L2 hit rate                   shared memory bank conflicts
    warp stall reason breakdown   instructions executed
    duration

Metric names are RESOLVED against the installed Nsight Compute rather than
hardcoded: the tool is asked for the metrics it has for the attached chip, and
each logical counter above is mapped onto a name that actually exists. The
mapping used is recorded in the output.

Every required counter appears in the output. One that could not be collected
is marked unavailable together with the reason -- never left blank and never
defaulted to zero, because a zero that means "not collected" is
indistinguishable from a zero that means "no bank conflicts".

If the profiler itself is missing or blocked, this fails loudly. It never
returns an empty result set that a caller could mistake for a clean run.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"

NCU_TIMEOUT_S = 300


class ProfilerUnavailable(RuntimeError):
    """The profiler could not be used at all. Never swallowed."""


# --- the required counter set -------------------------------------------
# Each logical counter maps to candidate metric names in preference order.
# The first candidate whose base name the installed tool reports is used.
REQUIRED_COUNTERS: dict[str, list[str]] = {
    "achieved_occupancy": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ],
    # Nsight Compute reports theoretical occupancy directly, and also exposes
    # the four limiters it comes from. Both are collected: the direct figure is
    # what the model uses, the limiters say which resource bound it, and the
    # derivation below cross-checks the two against each other.
    "theoretical_occupancy_direct": [
        "sm__maximum_warps_per_active_cycle_pct",
    ],
    "theoretical_occupancy_max_warps": [
        "sm__maximum_warps_avg_per_active_cycle",
    ],
    "theoretical_occupancy_limit_blocks": ["launch__occupancy_limit_blocks"],
    "theoretical_occupancy_limit_warps": ["launch__occupancy_limit_warps"],
    "theoretical_occupancy_limit_registers": ["launch__occupancy_limit_registers"],
    "theoretical_occupancy_limit_shared_mem": ["launch__occupancy_limit_shared_mem"],
    "block_size": ["launch__block_size"],
    "dram_read_throughput": [
        "dram__bytes_read.sum.per_second",
        "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    ],
    "dram_write_throughput": [
        "dram__bytes_write.sum.per_second",
        "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    ],
    "l2_hit_rate": [
        "lts__t_sector_hit_rate.pct",
        "lts__t_sectors_lookup_hit.sum",
    ],
    "shared_memory_bank_conflicts": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    ],
    "instructions_executed": [
        "smsp__inst_executed.sum",
        "sm__inst_executed.sum",
    ],
    "duration": [
        "gpu__time_duration.sum",
    ],
}

# The warp stall reason breakdown. Every reason the tool offers is collected;
# a subset would not be a breakdown.
STALL_REASONS = [
    "barrier", "branch_resolving", "dispatch_stall", "drain", "imc_miss",
    "lg_throttle", "long_scoreboard", "math_pipe_throttle", "membar",
    "mio_throttle", "misc", "no_instruction", "not_selected", "selected",
    "short_scoreboard", "sleeping", "tex_throttle", "wait",
]
STALL_METRIC_FMT = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"


def _default_runner(cmd: list[str], timeout: int = NCU_TIMEOUT_S):
    """Runs a command, returning (returncode, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as e:
        raise ProfilerUnavailable(f"{cmd[0]} not found on PATH: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise ProfilerUnavailable(f"{cmd[0]} timed out after {timeout}s") from e


def ncu_path() -> str:
    exe = shutil.which("ncu") or shutil.which("ncu.bat")
    if not exe:
        raise ProfilerUnavailable(
            "Nsight Compute (ncu) is not on PATH. Counter collection cannot be "
            "verified and no profile can be produced.")
    return exe


def ncu_version(runner=_default_runner) -> str:
    rc, out, err = runner([ncu_path(), "--version"])
    if rc != 0:
        raise ProfilerUnavailable(f"ncu --version failed (rc={rc}): {err.strip() or out.strip()}")
    for line in out.splitlines():
        if line.strip().startswith("Version"):
            return line.strip()
    return out.strip().splitlines()[0] if out.strip() else "unknown"


# The tool splits its metrics across several collections. The default query
# returns only the hardware profiling counters, which is why the launch
# attributes and the occupancy metrics have to be asked for by name -- querying
# only the default set silently loses them.
METRIC_COLLECTIONS = ["profiling", "launch", "occupancy"]


def _parse_metric_listing(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        line = line.rstrip()
        if not line or line[0].isspace():
            continue
        if line.startswith("-") or line.startswith("Chip ") or line.startswith("Device "):
            continue
        tok = line.split()[0]
        if tok in ("Metric",):
            continue
        names.add(tok)
    return names


def query_available_metrics(chip: str | None = None, runner=_default_runner) -> set[str]:
    """Asks the installed tool which metrics it has. Never a hardcoded list."""
    names: set[str] = set()
    failures: list[str] = []
    for collection in METRIC_COLLECTIONS:
        cmd = [ncu_path(), "--query-metrics", "--query-metrics-collection", collection]
        if chip:
            cmd += ["--chip", chip]
        rc, out, err = runner(cmd)
        if rc != 0 or not out.strip():
            failures.append(f"{collection}: rc={rc} stderr={err.strip()!r}")
            continue
        names |= _parse_metric_listing(out)

    if not names:
        raise ProfilerUnavailable(
            "ncu --query-metrics returned no metric names for any collection "
            f"({'; '.join(failures) or 'no detail'}). Metric names cannot be "
            "resolved against this installation.")
    return names


def resolve_metrics(chip: str | None = None, runner=_default_runner) -> dict:
    """Maps each required logical counter onto a real metric name.

    Returns {"mapping": {logical: metric or None},
             "unavailable": {logical: reason},
             "metrics": [metric names to request]}
    """
    available = query_available_metrics(chip=chip, runner=runner)

    mapping: dict[str, str | None] = {}
    unavailable: dict[str, str] = {}

    def base_of(metric: str) -> str:
        return metric.split(".")[0]

    for logical, candidates in REQUIRED_COUNTERS.items():
        chosen = None
        for cand in candidates:
            if base_of(cand) in available:
                chosen = cand
                break
        mapping[logical] = chosen
        if chosen is None:
            unavailable[logical] = (
                "no candidate metric exists in this Nsight Compute build for this "
                f"chip; tried {candidates}")

    for reason in STALL_REASONS:
        logical = f"warp_stall_{reason}"
        metric = STALL_METRIC_FMT.format(reason)
        if base_of(metric) in available:
            mapping[logical] = metric
        else:
            mapping[logical] = None
            unavailable[logical] = (
                f"stall reason '{reason}' is not offered by this build for this chip")

    metrics = sorted({m for m in mapping.values() if m})
    if not metrics:
        raise ProfilerUnavailable(
            "not one required metric resolved against this Nsight Compute "
            "installation; refusing to produce an empty profile")
    return {"mapping": mapping, "unavailable": unavailable, "metrics": metrics}


def _parse_ncu_csv(text: str) -> dict[str, dict[str, dict]]:
    """Parses ncu --csv output into {kernel: {metric: {value, unit}}}."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith('"ID"') or ln.lstrip().startswith("ID,"):
            start = i
            break
    if start is None:
        return {}
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    out: dict[str, dict[str, dict]] = {}
    for row in reader:
        kernel = (row.get("Kernel Name") or row.get("Kernel") or "").strip()
        metric = (row.get("Metric Name") or "").strip()
        if not kernel or not metric:
            continue
        raw = (row.get("Metric Value") or "").strip().replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            value = None
        out.setdefault(kernel, {})[metric] = {
            "value": value,
            "raw": row.get("Metric Value", ""),
            "unit": (row.get("Metric Unit") or "").strip(),
        }
    return out


def profile_binary(exe: str, exe_args: list[str] | None = None,
                   chip: str | None = None, runner=_default_runner,
                   kernel_regex: str | None = None) -> dict:
    """Profiles exe and returns the full per-kernel counter set.

    Raises ProfilerUnavailable if the profiler is missing, blocked, or returns
    nothing. It never returns an empty result set silently.
    """
    exe_args = exe_args or []
    resolution = resolve_metrics(chip=chip, runner=runner)

    cmd = [ncu_path(), "--csv", "--metrics", ",".join(resolution["metrics"])]
    if kernel_regex:
        cmd += ["--kernel-name", f"regex:{kernel_regex}"]
    cmd += [exe] + exe_args

    rc, out, err = runner(cmd)
    combined = (out or "") + "\n" + (err or "")
    per_kernel = _parse_ncu_csv(out)

    if not per_kernel:
        blocked_markers = (
            "ERR_NVGPUCTRPERM", "insufficient permissions",
            "The user does not have permission",
            "permission to access NVIDIA GPU Performance Counters",
        )
        hit = next((m for m in blocked_markers if m.lower() in combined.lower()), None)
        detail = combined.strip()[-2000:]
        if hit:
            raise ProfilerUnavailable(
                "Nsight Compute counter collection is BLOCKED for this user.\n"
                f"  command: {' '.join(cmd)}\n"
                f"  marker : {hit}\n"
                f"  output : {detail}\n"
                "  unblock: NVIDIA Control Panel -> Desktop -> Enable Developer "
                "settings, then Developer -> Manage GPU Performance Counters -> "
                "allow access to all users, then a full restart.")
        raise ProfilerUnavailable(
            f"ncu produced no parseable counter rows (rc={rc}).\n"
            f"  command: {' '.join(cmd)}\n"
            f"  output : {detail}")

    mapping = resolution["mapping"]
    unavailable = dict(resolution["unavailable"])

    kernels: dict[str, dict] = {}
    for kernel, metrics in per_kernel.items():
        counters: dict[str, dict] = {}
        for logical, metric in mapping.items():
            if metric is None:
                counters[logical] = {
                    "status": "unavailable",
                    "metric": None,
                    "value": None,
                    "unit": None,
                    "reason": unavailable.get(logical, "metric does not exist"),
                }
                continue
            row = metrics.get(metric)
            if row is None:
                counters[logical] = {
                    "status": "unavailable",
                    "metric": metric,
                    "value": None,
                    "unit": None,
                    "reason": "requested but not returned by the profiler for this kernel",
                }
            elif row["value"] is None:
                counters[logical] = {
                    "status": "unavailable",
                    "metric": metric,
                    "value": None,
                    "unit": row["unit"],
                    "reason": f"profiler returned a non-numeric value {row['raw']!r}",
                }
            else:
                counters[logical] = {
                    "status": "populated",
                    "metric": metric,
                    "value": row["value"],
                    "unit": row["unit"],
                    "reason": None,
                }

        counters["theoretical_occupancy_pct"] = _derive_theoretical_occupancy(counters)
        kernels[kernel] = {"counters": counters}

    return {
        "tool": "Nsight Compute",
        "tool_version": ncu_version(runner=runner),
        "command": cmd,
        "metric_mapping": mapping,
        "metrics_requested": resolution["metrics"],
        "unresolved_counters": unavailable,
        "kernels": kernels,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _derive_theoretical_occupancy(counters: dict) -> dict:
    """Theoretical occupancy, with the arithmetic recorded alongside it.

    blocks/SM = min(the four launch__occupancy_limit_* values)
    warps/SM  = blocks/SM * block_size / 32
    occupancy = warps/SM / sm__maximum_warps_avg_per_active_cycle * 100
    """
    limits = []
    for key in ("theoretical_occupancy_limit_blocks",
                "theoretical_occupancy_limit_warps",
                "theoretical_occupancy_limit_registers",
                "theoretical_occupancy_limit_shared_mem"):
        c = counters.get(key, {})
        if c.get("status") == "populated" and c.get("value"):
            limits.append(c["value"])
    blk = counters.get("block_size", {})
    maxw = counters.get("theoretical_occupancy_max_warps", {})
    if not limits or blk.get("status") != "populated" or maxw.get("status") != "populated":
        return {
            "status": "unavailable",
            "metric": None,
            "value": None,
            "unit": "%",
            "reason": "one or more of the occupancy limit metrics, the block size, "
                      "or the max-warps metric was not collected",
        }
    blocks_per_sm = min(limits)
    warps_per_sm = blocks_per_sm * blk["value"] / 32.0
    pct = 100.0 * warps_per_sm / maxw["value"]

    direct = counters.get("theoretical_occupancy_direct", {})
    cross = None
    if direct.get("status") == "populated" and direct.get("value") is not None:
        cross = {"direct_pct": direct["value"],
                 "agrees_within_1pct": abs(direct["value"] - pct) <= 1.0}

    return {
        "status": "derived",
        "metric": None,
        "value": pct,
        "unit": "%",
        "reason": None,
        "arithmetic": (f"min(limits)={blocks_per_sm} blocks/SM * block_size="
                       f"{blk['value']} / 32 = {warps_per_sm} warps/SM; "
                       f"/ max_warps={maxw['value']} * 100 = {pct:.4f}%"),
        "cross_check_against_profiler_direct_metric": cross,
    }


def summarise(profile: dict) -> str:
    lines = [f"profiler: {profile['tool']} {profile['tool_version']}"]
    for kernel, data in profile["kernels"].items():
        lines.append(f"kernel: {kernel}")
        for logical, c in sorted(data["counters"].items()):
            if c["status"] in ("populated", "derived"):
                lines.append(f"  {logical:42s} {c['status']:9s} "
                             f"{c['value']!r} {c['unit'] or ''}".rstrip())
            else:
                lines.append(f"  {logical:42s} UNAVAILABLE  {c['reason']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", help="CUDA binary to profile")
    ap.add_argument("--arg", action="append", default=[], help="argument for the binary")
    ap.add_argument("--kernel-regex", default=None)
    ap.add_argument("--chip", default=None, help="chip name for metric resolution")
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--list-metrics", action="store_true",
                    help="resolve and print the metric mapping, then exit")
    args = ap.parse_args(argv)

    try:
        if args.list_metrics:
            res = resolve_metrics(chip=args.chip)
            print(json.dumps(res, indent=2))
            return 0
        if not args.exe:
            ap.error("--exe is required unless --list-metrics is given")

        profile = profile_binary(args.exe, args.arg, chip=args.chip,
                                 kernel_regex=args.kernel_regex)
    except ProfilerUnavailable as e:
        print(f"PROFILER UNAVAILABLE: {e}", file=sys.stderr)
        return 2

    print(summarise(profile))
    out = Path(args.out) if args.out else \
        RESULTS_DIR / f"profile_{Path(args.exe).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"profile written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
