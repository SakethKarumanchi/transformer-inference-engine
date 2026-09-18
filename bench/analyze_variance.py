#!/usr/bin/env python3
"""bench/analyze_variance.py -- Stage 0b Phase 1: offline variance diagnosis.

Reads the raw per-sample timing arrays Stage 0 retained in bench/results/run1/
and bench/results/run2/ and asks, for every configuration in both suite runs,
what shape its timing distribution has. It costs no machine time, changes no
benchmark, and is not gated by the measurement checkpoint: the inputs already
exist and are read strictly read-only.

What it answers, per configuration:

  a. Is the excess variance carried by a few extreme samples or by the whole
     distribution? Reported as a robust (median-absolute-deviation) outlier
     count, the run-order position of each flagged sample, and the size of the
     largest deviations relative to the median.
  b. How does standard-deviation-as-percent-of-median relate to the median
     per-sample DURATION across every configuration in the corpus? Reported
     once over the whole set, not per benchmark.
  c. Statistics recomputed with the k largest samples removed, k = 1, 2, 3.

  *** (c) IS A DIAGNOSTIC ONLY. ***
  A trimmed figure never enters HARDWARE.md, never appears as a measured value
  anywhere, and NEVER converts an INVALID run into a valid one. Every trimmed
  number in the emitted JSON sits under a key beginning "diagnostic_only__",
  carries its own "diagnostic_only": true marker and a disclaimer string, and
  is absent from the "measurement" block, which is the only block whose
  numbers are measurements. The verdict in "measurement" is computed from the
  untrimmed samples and from nothing else.

  d. The 16384 B cache-ladder point against the 8192 B point. Both sit inside
     this CPU's 32768 B L1d, so whatever separates them is not DRAM contention.
  e. The decode-shaped GEMMs (M=1) against each other: the two that were
     INVALID in both Stage 0 runs and the three that were valid.

Usage:
    .venv/Scripts/python.exe bench/analyze_variance.py
    .venv/Scripts/python.exe bench/analyze_variance.py --out <path>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
DEFAULT_OUT = RESULTS_DIR / "stage0b" / "variance_analysis.json"

# The protocol limit, restated here only to CLASSIFY already-recorded runs.
# bench/microbench/bench_common.cu owns the rule; this file never relaxes it.
MAX_STDDEV_PCT_OF_MEDIAN = 5.0

# Modified z-score cutoff for the robust outlier test. 3.5 is the conventional
# threshold for the MAD-based modified z-score (Iglewicz & Hoaglin).
MAD_Z_THRESHOLD = 3.5
MAD_TO_SIGMA = 0.6745          # 0.75 quantile of the standard normal

TRIM_DISCLAIMER = (
    "DIAGNOSTIC ONLY. Not a measurement. A trimmed statistic never enters "
    "HARDWARE.md, never appears as a measured value, and never converts an "
    "INVALID run into a valid one."
)


class MissingRawSamples(ValueError):
    """A results record carries summary statistics but no raw sample array.

    Raised rather than falling back to the summary block: the whole point of
    this analysis is the distribution shape, which summary statistics cannot
    supply, and silently returning them would answer a different question
    while looking like an answer to this one.
    """


# ------------------------------------------------------------- statistics ---
def stddev_pct_of_median(samples: list[float]) -> float:
    """Sample standard deviation as a percentage of the median.

    Deliberately identical in construction to bench_compute_stats() in
    bench/microbench/bench_common.cu: Bessel-corrected (divisor n-1) standard
    deviation, median of an even-length set as the mean of the two middle
    elements, percentage taken against |median|.
    """
    n = len(samples)
    if n == 0:
        return 0.0
    med = statistics.median(samples)
    sd = statistics.stdev(samples) if n > 1 else 0.0
    return 100.0 * sd / abs(med) if med != 0.0 else 0.0


def basic_stats(samples: list[float]) -> dict:
    n = len(samples)
    med = statistics.median(samples) if n else 0.0
    sd = statistics.stdev(samples) if n > 1 else 0.0
    return {
        "n": n,
        "mean_ms": statistics.fmean(samples) if n else 0.0,
        "median_ms": med,
        "min_ms": min(samples) if n else 0.0,
        "max_ms": max(samples) if n else 0.0,
        "stddev_ms": sd,
        "stddev_pct_of_median": stddev_pct_of_median(samples),
    }


def verdict_from(samples: list[float], n_expected: int | None = None) -> dict:
    """The protocol verdict, computed from the UNTRIMMED samples only."""
    pct = stddev_pct_of_median(samples)
    reasons = []
    if pct > MAX_STDDEV_PCT_OF_MEDIAN:
        reasons.append(f"stddev is {pct:.3f}% of median, above the "
                       f"{MAX_STDDEV_PCT_OF_MEDIAN:.1f}% protocol limit")
    if n_expected is not None and len(samples) < n_expected:
        reasons.append(f"sample count {len(samples)} is below {n_expected}")
    return {
        "verdict": "INVALID" if reasons else "VALID",
        "stddev_pct_of_median": pct,
        "reasons": reasons,
        "computed_from": "untrimmed samples",
    }


def mad_outliers(samples: list[float], z_threshold: float = MAD_Z_THRESHOLD) -> dict:
    """Robust outlier detection on the median absolute deviation.

    A sample is flagged when |modified z| exceeds z_threshold, where the
    modified z-score is 0.6745 * (x - median) / MAD. MAD is used rather than
    the standard deviation because the standard deviation is itself inflated by
    the very samples being looked for.

    When MAD is exactly zero (more than half the samples identical) the scale
    falls back to the mean absolute deviation, and the fallback is reported.
    """
    n = len(samples)
    if n == 0:
        return {"n_flagged": 0, "flagged": [], "mad_ms": 0.0, "scale_used": "none",
                "z_threshold": z_threshold}
    med = statistics.median(samples)
    deviations = [abs(x - med) for x in samples]
    mad = statistics.median(deviations)
    scale_used = "median absolute deviation"
    if mad == 0.0:
        mad = statistics.fmean(deviations)
        scale_used = "mean absolute deviation (MAD was zero)"
    flagged = []
    if mad > 0.0:
        for i, x in enumerate(samples):
            z = MAD_TO_SIGMA * (x - med) / mad
            if abs(z) > z_threshold:
                flagged.append({
                    "run_order_index": i,          # 0-based position in the timed run
                    "run_order_fraction": (i / (n - 1)) if n > 1 else 0.0,
                    "value_ms": x,
                    "modified_z": z,
                    "pct_of_median": 100.0 * (x - med) / abs(med) if med else 0.0,
                    "direction": "slow" if x > med else "fast",
                })
    return {
        "n_flagged": len(flagged),
        "flagged": flagged,
        "mad_ms": mad,
        "scale_used": scale_used,
        "z_threshold": z_threshold,
    }


def dispersion_shape(samples: list[float], outliers: dict) -> dict:
    """Distinguishes 'a few spikes' from 'broadly dispersed'.

    Two quantities, both scale-free:

      spike_share_of_variance  -- the fraction of the total squared deviation
                                  from the median contributed by the flagged
                                  samples. Near 1 means the excess is carried
                                  by those samples alone.
      iqr_pct_of_median        -- the interquartile range as a percentage of
                                  the median. This ignores the tails entirely,
                                  so a distribution that is broadly dispersed
                                  keeps a large value here while one that is
                                  tight-plus-spikes does not.
    """
    n = len(samples)
    if n < 4:
        return {"spike_share_of_variance": None, "iqr_pct_of_median": None,
                "reason": f"only {n} samples"}
    med = statistics.median(samples)
    total_sq = sum((x - med) ** 2 for x in samples)
    idx = {f["run_order_index"] for f in outliers["flagged"]}
    spike_sq = sum((samples[i] - med) ** 2 for i in idx)
    q = statistics.quantiles(samples, n=4, method="inclusive")
    iqr = q[2] - q[0]
    return {
        "spike_share_of_variance": (spike_sq / total_sq) if total_sq > 0 else 0.0,
        "iqr_ms": iqr,
        "iqr_pct_of_median": 100.0 * iqr / abs(med) if med else 0.0,
        "max_deviation_pct_of_median": 100.0 * max(abs(x - med) for x in samples) / abs(med)
                                       if med else 0.0,
        "reason": None,
    }


def trimmed_diagnostics(samples: list[float]) -> dict:
    """Statistics with the k largest samples removed, k = 1, 2, 3.

    DIAGNOSTIC ONLY -- see the module docstring. Every key here is prefixed so
    that a trimmed number cannot be mistaken for a measurement by a reader or
    by a downstream consumer grepping for measurement field names.
    """
    out = {
        "diagnostic_only": True,
        "disclaimer": TRIM_DISCLAIMER,
        "never_changes_verdict": True,
        "by_k": {},
    }
    ordered = sorted(samples)
    for k in (1, 2, 3):
        if len(ordered) - k < 2:
            out["by_k"][str(k)] = {"diagnostic_only": True,
                                   "reason": "too few samples to trim"}
            continue
        kept = ordered[:len(ordered) - k]
        st = basic_stats(kept)
        out["by_k"][str(k)] = {
            "diagnostic_only": True,
            "disclaimer": TRIM_DISCLAIMER,
            "k_largest_removed": k,
            "diagnostic_only__n": st["n"],
            "diagnostic_only__median_ms": st["median_ms"],
            "diagnostic_only__stddev_ms": st["stddev_ms"],
            "diagnostic_only__stddev_pct_of_median": st["stddev_pct_of_median"],
            "diagnostic_only__would_be_under_limit":
                st["stddev_pct_of_median"] <= MAX_STDDEV_PCT_OF_MEDIAN,
        }
    return out


def run_order_trend(samples: list[float]) -> dict:
    """Least-squares slope of sample value against run order.

    A drift that rises (or falls) monotonically across the run separates a
    thermal / frequency-governance story from random interference, which has
    no preferred direction. Reported with the correlation coefficient so a
    slope on noise is not read as a trend.
    """
    n = len(samples)
    if n < 3:
        return {"slope_ms_per_sample": None, "reason": f"only {n} samples"}
    xs = list(range(n))
    mx, my = statistics.fmean(xs), statistics.fmean(samples)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, samples))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in samples)
    slope = sxy / sxx if sxx else 0.0
    r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
    med = statistics.median(samples)
    first_half = statistics.median(samples[: n // 2])
    second_half = statistics.median(samples[n - n // 2:])
    return {
        "slope_ms_per_sample": slope,
        "slope_pct_of_median_per_sample": 100.0 * slope / abs(med) if med else 0.0,
        "pearson_r_vs_run_order": r,
        "first_half_median_ms": first_half,
        "second_half_median_ms": second_half,
        "second_minus_first_pct": 100.0 * (second_half - first_half) / abs(first_half)
                                  if first_half else 0.0,
        "reason": None,
    }


# ------------------------------------------------------------------ input ---
def load_records(path: Path) -> list[dict]:
    """Loads one benchmark results file. Read-only. Raises when a record has no
    raw per-sample array rather than silently returning summary statistics."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    records = doc.get("records")
    if not records:
        raise MissingRawSamples(f"{path}: no records array")
    for rec in records:
        raw = rec.get("raw_samples_ms")
        if not isinstance(raw, list) or not raw:
            raise MissingRawSamples(
                f"{path}: record {rec.get('configuration')!r} carries no "
                f"raw_samples_ms array; summary statistics cannot substitute "
                f"for the per-sample distribution this analysis needs")
    return records


def analyze_record(rec: dict, run: str, source: str) -> dict:
    samples = rec["raw_samples_ms"]
    recorded = rec.get("statistics") or {}
    outliers = mad_outliers(samples)
    return {
        "run": run,
        "source_file": source,
        "benchmark": rec.get("benchmark"),
        "configuration": rec.get("configuration"),
        "units": rec.get("units"),
        "headline_value": rec.get("value"),
        "n_samples": len(samples),
        "warmup_iterations": rec.get("warmup_iterations"),
        # (a) and the verdict -- measurements, untrimmed
        "measurement": {
            **basic_stats(samples),
            **verdict_from(samples, rec.get("samples_requested")),
        },
        "recorded_stddev_pct_of_median": recorded.get("stddev_pct_of_median"),
        "recorded_valid": recorded.get("valid"),
        "outliers": outliers,
        "dispersion": dispersion_shape(samples, outliers),
        "run_order": run_order_trend(samples),
        # (c) diagnostic only
        "diagnostic_only__trimmed": trimmed_diagnostics(samples),
    }


def collect(results_root: Path, runs=("run1", "run2")) -> list[dict]:
    rows = []
    for run in runs:
        d = results_root / run
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            if path.name.endswith("_consolidated.json"):
                continue
            try:
                source = str(path.relative_to(REPO_ROOT))
            except ValueError:          # a results root outside the repository
                source = str(path)
            for rec in load_records(path):
                rows.append(analyze_record(rec, run, source))
    return rows


# ----------------------------------------------- (b) corpus-wide relation ---
def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def duration_versus_variance(rows: list[dict]) -> dict:
    """(b) The relationship across the WHOLE corpus, not per benchmark."""
    pts = [(r["measurement"]["median_ms"], r["measurement"]["stddev_pct_of_median"])
           for r in rows if r["measurement"]["median_ms"] > 0]
    if len(pts) < 3:
        return {"n": len(pts), "reason": "too few configurations"}
    xs = [math.log10(a) for a, _ in pts]
    ys = [b for _, b in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    pearson_log = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0

    # Decade buckets on the median duration.
    buckets: dict[str, list[float]] = {}
    for med, pct in pts:
        e = math.floor(math.log10(med))
        key = f"1e{e} <= median_ms < 1e{e + 1}"
        buckets.setdefault(key, []).append(pct)
    bucket_rows = {}
    for key in sorted(buckets, key=lambda k: float(k.split("e")[1].split(" ")[0])):
        v = buckets[key]
        bucket_rows[key] = {
            "n_configurations": len(v),
            "median_stddev_pct": statistics.median(v),
            "max_stddev_pct": max(v),
            "n_over_5pct": sum(1 for x in v if x > MAX_STDDEV_PCT_OF_MEDIAN),
        }
    invalid = [p for p in pts if p[1] > MAX_STDDEV_PCT_OF_MEDIAN]
    valid = [p for p in pts if p[1] <= MAX_STDDEV_PCT_OF_MEDIAN]
    return {
        "n_configurations": len(pts),
        "pearson_r_log10_median_vs_stddev_pct": pearson_log,
        "spearman_rho_median_vs_stddev_pct": _spearman([a for a, _ in pts], ys),
        "by_duration_decade": bucket_rows,
        "median_duration_ms_of_invalid_configs":
            statistics.median([a for a, _ in invalid]) if invalid else None,
        "median_duration_ms_of_valid_configs":
            statistics.median([a for a, _ in valid]) if valid else None,
        "n_invalid": len(invalid),
        "n_valid": len(valid),
    }


# ------------------------------------------------ (d) and (e) comparisons ---
def _find(rows, benchmark, needle, run=None):
    return [r for r in rows
            if r["benchmark"] == benchmark
            and needle in (r["configuration"] or "")
            and (run is None or r["run"] == run)]


def _digest(r: dict) -> dict:
    return {
        "run": r["run"],
        "configuration": r["configuration"],
        "median_ms": r["measurement"]["median_ms"],
        "min_ms": r["measurement"]["min_ms"],
        "max_ms": r["measurement"]["max_ms"],
        "stddev_pct_of_median": r["measurement"]["stddev_pct_of_median"],
        "verdict": r["measurement"]["verdict"],
        "n_outliers_flagged": r["outliers"]["n_flagged"],
        "outlier_run_order_indices": [f["run_order_index"] for f in r["outliers"]["flagged"]],
        "spike_share_of_variance": r["dispersion"]["spike_share_of_variance"],
        "iqr_pct_of_median": r["dispersion"]["iqr_pct_of_median"],
        "max_deviation_pct_of_median": r["dispersion"]["max_deviation_pct_of_median"],
        "run_order_pearson_r": r["run_order"]["pearson_r_vs_run_order"],
        "second_minus_first_half_pct": r["run_order"]["second_minus_first_pct"],
    }


def l1_anomaly(rows: list[dict]) -> dict:
    """(d) 16384 B against 8192 B. Both are inside the 32768 B L1d."""
    pairs = {}
    for ws in (8192, 16384, 4096, 32768):
        got = _find(rows, "cpu_cache_ladder", f"working_set={ws} B,")
        pairs[str(ws)] = [_digest(r) for r in got]
    return {
        "note": ("8192 B and 16384 B both sit inside this CPU's 32768 B L1d, so no "
                 "DRAM-side contention can reach either. 4096 B and 32768 B are "
                 "included as the neighbouring L1-resident points."),
        "l1d_bytes": 32768,
        "points": pairs,
    }


def decode_shapes(rows: list[dict]) -> dict:
    """(e) M=1 GEMM shapes: the two INVALID in both runs against the three valid."""
    out = {}
    for r in rows:
        if r["benchmark"] != "cublas_sgemm_ref":
            continue
        cfg = r["configuration"] or ""
        if "M=1 " not in cfg and not cfg.endswith("M=1"):
            if " M=1," not in cfg:
                continue
        out.setdefault(cfg, []).append(_digest(r))
    return {
        "note": ("qkv_projection and attn_output_projection at M=1 were INVALID in "
                 "both Stage 0 runs; ffn_up, ffn_down and lm_head at M=1 were not. "
                 "cublas_sgemm_ref is NOT modified or re-run by Stage 0b: it is the "
                 "prefill denominator and PERSISTENT.md D3 records its M sweep as a "
                 "placeholder Stage 3 replaces. This block is analysis only."),
        "by_configuration": out,
    }


# ----------------------------------------------------------------- report ---
def build_report(results_root: Path = RESULTS_DIR) -> dict:
    rows = collect(results_root)
    invalid = [r for r in rows if r["measurement"]["verdict"] == "INVALID"]
    return {
        "analysis": "stage-0b phase 1 variance diagnosis",
        "inputs_read_only": True,
        "source_runs": ["bench/results/run1", "bench/results/run2"],
        "protocol_limit_pct_of_median": MAX_STDDEV_PCT_OF_MEDIAN,
        "trimmed_statistics_policy": TRIM_DISCLAIMER,
        "n_configurations": len(rows),
        "n_invalid": len(invalid),
        "invalid_configurations": [
            {"run": r["run"], "benchmark": r["benchmark"],
             "configuration": r["configuration"],
             "stddev_pct_of_median": r["measurement"]["stddev_pct_of_median"]}
            for r in invalid],
        "b_duration_versus_variance": duration_versus_variance(rows),
        "d_l1_resident_anomaly": l1_anomaly(rows),
        "e_decode_shapes": decode_shapes(rows),
        "configurations": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=str(RESULTS_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    report = build_report(Path(args.results_root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"variance analysis written: {out}")
    print(f"  configurations analysed : {report['n_configurations']}")
    print(f"  INVALID under the 5% rule: {report['n_invalid']}")
    b = report["b_duration_versus_variance"]
    print(f"  (b) spearman rho median-duration vs stddev%%: "
          f"{b.get('spearman_rho_median_vs_stddev_pct'):.3f}")
    print(f"      median duration of INVALID configs: "
          f"{b.get('median_duration_ms_of_invalid_configs')} ms")
    print(f"      median duration of VALID configs  : "
          f"{b.get('median_duration_ms_of_valid_configs')} ms")
    print("  (c) trimmed statistics are diagnostic only and change no verdict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
