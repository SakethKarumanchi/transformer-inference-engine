"""correctness.py -- the D2 correctness gate, and the divergence statistics behind it.

WHAT THIS FILE IS. The single implementation of the correctness check every stage
from 3 onward must pass. It is importable by a later stage and runnable as a
script. It computes the full Stage 2 divergence statistic set over engine logits
and reference logits, and applies the D2 gate exactly as BENCHMARK_PROTOCOL.md
section 5 defines it.

IT REIMPLEMENTS NOTHING. The reference logits come from
reference/reference_impl.py, which is CALLED, never transcribed. The engine
logits come from the committed `gpt2_tool --dump-logits`, which writes a binary
block this file parses. Timing lives in C, in bench/microbench/bench_common.h,
and nothing here times anything.

NO THRESHOLD IS HARDCODED HERE. The tolerance is a required argument. Its
default source is the value written into BENCHMARK_PROTOCOL.md section 5, which
`read_tolerance()` PARSES out of that file rather than duplicating in code, so
the two cannot drift apart. A caller that supplies neither gets an error, not a
guess.

THE GATE, as section 5 fixes it. The gate figure is the maximum ELEMENTWISE
ABSOLUTE difference over the FULL logit vector at EVERY position of a prefill,
computed in float32 on both sides. It is absolute rather than relative because
the upper bound that makes the tolerance meaningful -- the gap between the top-1
and top-2 reference logits -- is a quantity in absolute logit units, and a
relative threshold cannot be compared against it. Relative statistics are still
computed and reported, against a denominator of |reference| + 1e-6 elementwise
with the count of elements below that floor stated, but they do not gate.

PASS requires all three, and any one failing is a FAIL:
  1. maximum absolute difference at or below the threshold;
  2. top-1 agreement at EVERY position;
  3. the greedy-decoded token sequence matches the reference on the fixed
     prompt set.

A FAIL IS RETURNED, NOT RAISED, so the harness can record it alongside timings
that were honestly taken. An exception would discard measurements that are valid
as measurements whatever the verdict.

THE ORACLE IS NOT CHEAP AND MUST NOT OVERLAP A TIMED BRACKET. It holds the full
weight set in numpy and again in torch, over a gigabyte resident, single-threaded
on the CPU. This script runs to completion and exits before any timed run
starts, which is also why its results file carries no timings.

ARRAY SIZES. At the longest D3 length the comparison holds two 128 x 50257
float32 blocks, roughly 25 MB each. That is comfortable in memory and is read
from the engine's BINARY dump rather than from any text form -- a text format
would round the very quantity being compared. The results JSON carries
statistics and the margin table, never the raw logit arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PROTOCOL = os.path.join(REPO, "BENCHMARK_PROTOCOL.md")
FINGERPRINT = os.path.join(REPO, "bench", "results", "machine_fingerprint.json")
BUILD_INFO = os.path.join(REPO, "build", "generated", "build_info.h")
GIT_INFO = os.path.join(REPO, "build", "generated", "git_info.h")
D3_FIXTURE = os.path.join(REPO, "tests", "fixtures", "benchmark_prompts.tsv")
DEFAULT_OUT = os.path.join(REPO, "bench", "results", "stage4", "stage4_correctness.json")

# The relative-difference denominator floor. A fixed property of the statistic,
# not a tolerance: it only keeps the relative figure finite where a reference
# logit is near zero, and the count of elements it affects is always reported.
REL_EPS = 1e-6

# The token the tolerance is published under in BENCHMARK_PROTOCOL.md section 5.
TOLERANCE_TOKEN = "D2_MAX_ABS_LOGIT_DIFF"


# ------------------------------------------------------------- the tolerance --

def read_tolerance(protocol_path: str = PROTOCOL) -> float:
    """Reads the D2 threshold out of BENCHMARK_PROTOCOL.md section 5.

    The value is READ rather than duplicated in code so that the document and
    the gate cannot drift apart. A missing value is an error naming the file --
    never a default, because a silently defaulted tolerance is exactly the
    arbitrary threshold section 5 exists to forbid.
    """
    with open(protocol_path, "r", encoding="utf-8") as f:
        text = f.read()
    found = re.findall(TOLERANCE_TOKEN + r"\s*=\s*([0-9.eE+-]+)", text)
    if not found:
        raise ValueError(
            f"{protocol_path} carries no {TOLERANCE_TOKEN} value. BENCHMARK_PROTOCOL.md "
            "section 5 is the only source for the D2 threshold; pass one explicitly only "
            "when deliberately measuring divergence before the threshold exists.")
    # Stage 4: TWO machine-readable lines is a document that has been amended
    # without superseding its old value, and picking the first would mean the
    # gate silently ran against whichever one happened to come first in the file.
    if len(found) > 1:
        raise ValueError(
            f"{protocol_path} carries {len(found)} {TOLERANCE_TOKEN} values ({found}). "
            "BENCHMARK_PROTOCOL.md section 5 must carry EXACTLY ONE machine-readable "
            "line; an amendment supersedes the old value in prose and leaves one line.")
    # The pattern accepts any number of significant figures: Stage 4's re-derived
    # value is two (2.3e-03) where Stage 3's was one (6e-03).
    return float(found[0])


# ------------------------------------------------- D2 under the corrected rule --

def _values_at_precision(lower: float, upper: float, sig: int) -> list:
    """Every value with exactly `sig` significant figures lying in [lower, upper]."""
    if not (lower > 0 and upper > 0) or upper < lower:
        return []
    out = set()
    e_lo = math.floor(math.log10(lower)) - (sig - 1)
    e_hi = math.floor(math.log10(upper)) - (sig - 1)
    for e in range(e_lo, e_hi + 1):
        step = 10.0 ** e
        m_min = max(int(math.ceil(lower / step - 1e-12)), 10 ** (sig - 1))
        m_max = min(int(math.floor(upper / step + 1e-12)), 10 ** sig - 1)
        for m in range(m_min, m_max + 1):
            out.add(m * step)
    return sorted(out)


def d2_corrected_rule(min_margin: float, max_divergence: float,
                      window_minimum: float = 9.0, max_sig: int = 6) -> dict:
    """THE CORRECTED D2 RULE, as Stage 4 writes it into BENCHMARK_PROTOCOL.md §5.

    Upper bound: the minimum reference top-1/top-2 margin over the WHOLE D3 set,
    all positions of all four rows, divided by 3. A property of the REFERENCE
    alone, which is what makes it a bound the implementation cannot move.

    Lower bound: 3x the maximum observed absolute divergence over the WHOLE set,
    taken over BOTH engine paths measured -- the prefill path and the
    cached-decode path.

    Window check: margin / divergence must be at least `window_minimum`.

    Value: the geometric mean of the two bounds, rounded to the FEWEST
    significant figures at which some value lies inside the window, starting at
    one; among the values at that precision lying inside the window, the one
    nearest the geometric mean in LOG distance. The safety bound decides the
    rounding; the rounding never decides the safety bound.

    WHY THE PREVIOUS RULE'S PREMISE FAILED: Stage 3 evaluated its window check
    at the longest length because its rule said to, and in the same section
    established that the premise of that rule -- that margin degrades with
    length -- is a nested-prefix artifact that does not hold across the four
    INDEPENDENT D3 rows. The set-wide minimum is the honest bound, and under it
    Stage 3's committed 6e-03 leaves only 1.30x rather than the 3x its own
    condition (b) required.

    Returns the full arithmetic. Never raises on a closed window: a closed
    window is a FINDING, reported with the value left unchanged.
    """
    lower = 3.0 * float(max_divergence)
    upper = float(min_margin) / 3.0
    ratio = (float(min_margin) / float(max_divergence)) if max_divergence > 0 else None

    window_open = bool(lower < upper)
    window_wide_enough = bool(ratio is not None and ratio >= window_minimum)

    result = {
        "rule": ("lower = 3 x the set-wide maximum absolute divergence over BOTH engine "
                 "paths; upper = the set-wide minimum reference top-1/top-2 margin / 3; "
                 "value = the geometric mean, rounded to the fewest significant figures at "
                 "which some value lies inside the window, nearest the geometric mean in "
                 "log distance among those"),
        "set_wide_max_abs_divergence": float(max_divergence),
        "set_wide_min_reference_margin": float(min_margin),
        "lower_bound": lower,
        "lower_bound_derivation": f"3 x {max_divergence:.6e} = {lower:.6e}",
        "upper_bound": upper,
        "upper_bound_derivation": f"{min_margin:.6e} / 3 = {upper:.6e}",
        "window_ratio_margin_over_divergence": ratio,
        "window_minimum_required": window_minimum,
        "window_open": window_open,
        "window_wide_enough": window_wide_enough,
        "usable": bool(window_open and window_wide_enough),
    }

    if not result["usable"]:
        result["value"] = None
        result["significant_figures"] = None
        result["candidates"] = []
        result["geometric_mean"] = (math.sqrt(lower * upper) if window_open else None)
        result["finding"] = (
            "the window is CLOSED: the lower bound is at or above the upper bound, so no "
            "value satisfies both conditions" if not window_open else
            f"the window is open but narrower than {window_minimum}x "
            f"(margin / divergence = {ratio:.3f}x)")
        return result

    geo = math.sqrt(lower * upper)
    chosen, chosen_sig, candidates = None, None, []
    for sig in range(1, max_sig + 1):
        candidates = _values_at_precision(lower, upper, sig)
        if candidates:
            chosen_sig = sig
            chosen = min(candidates, key=lambda v: abs(math.log(v / geo)))
            break

    result["geometric_mean"] = geo
    result["significant_figures"] = chosen_sig
    result["candidates"] = candidates
    result["candidate_log_distances"] = {f"{v:.6e}": abs(math.log(v / geo))
                                         for v in candidates}
    result["value"] = chosen
    result["achieved_factor_above_divergence"] = (
        chosen / max_divergence if max_divergence > 0 else None)
    result["achieved_factor_below_margin"] = (
        min_margin / chosen if chosen else None)
    return result


# ------------------------------------------------- bit-for-bit comparison ------

def bitwise_comparison(a, b, label: str = "") -> dict:
    """EXACT float32 equality, elementwise. Reports the COUNT of differing
    elements and the maximum absolute difference -- both zero for a pass.

    Separate from divergence_statistics on purpose: this is not a tolerance
    question. Two code paths that run the same inner loops over the same
    elements in the same order must agree exactly, and reporting that as a small
    difference inside a tolerance would hide the one thing worth knowing."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return {"label": label, "comparable": False,
                "reason": f"shapes differ: {a.shape} against {b.shape}",
                "bit_for_bit": False}
    neq = (a != b)
    n = int(neq.sum())
    if n:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        maxdiff = float(d.max())
        idx = np.argwhere(neq)[:16]
        first = [[int(x) for x in row] for row in idx]
    else:
        maxdiff, first = 0.0, []
    return {
        "label": label,
        "comparable": True,
        "elements": int(a.size),
        "differing_elements": n,
        "max_abs_difference": maxdiff,
        "bit_for_bit": n == 0,
        "first_differing_indices": first,
        "rule": "exact float32 equality; no tolerance is applied here",
    }


# ------------------------------------------------------------- the logit dump --

def read_logit_dump(path: str) -> np.ndarray:
    """Parses the binary block `gpt2_tool --dump-logits` writes: the 8-byte magic
    "TIE2LOGI", int32 positions, int32 vocab_size, then positions*vocab_size
    float32 in row order. Returns a [positions, vocab_size] float32 array."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"TIE2LOGI":
            raise ValueError(f"{path}: bad magic {magic!r}; this is not a logit dump")
        npos, nvocab = struct.unpack("<ii", f.read(8))
        raw = f.read(npos * nvocab * 4)
    if len(raw) != npos * nvocab * 4:
        raise ValueError(f"{path}: short dump, {len(raw)} bytes for {npos}x{nvocab} float32")
    return np.frombuffer(raw, dtype=np.float32).reshape(npos, nvocab).copy()


# ------------------------------------------------------------- the statistics --

def _percentiles(a: np.ndarray) -> dict:
    q = np.percentile(a, [50, 90, 99, 99.9])
    return {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "p99_9": float(q[3]), "max": float(a.max())}


def divergence_statistics(engine_logits, reference_logits, token_ids=None,
                          label: str = "") -> dict:
    """The full Stage 2 statistic set, computed in float32 on both sides over the
    FULL logit vector at EVERY position. NO THRESHOLD IS APPLIED HERE -- this
    function describes the divergence and says nothing about whether it passes.
    Keeping description and judgement apart is what lets the same code run before
    a threshold exists and after one is set.
    """
    c = np.asarray(engine_logits, dtype=np.float32)
    r = np.asarray(reference_logits, dtype=np.float32)
    if c.shape != r.shape:
        raise ValueError(f"shape mismatch: engine {c.shape} vs reference {r.shape}")
    if c.ndim != 2:
        raise ValueError(f"expected [positions, vocab]; got {c.shape}")

    absdiff = np.abs(c - r).astype(np.float64)
    denom = np.abs(r).astype(np.float64) + REL_EPS
    reldiff = absdiff / denom
    below_floor = int(np.count_nonzero(np.abs(r) < REL_EPS))

    flat = int(np.argmax(absdiff))
    pos, tok = divmod(flat, c.shape[1])

    top1 = np.argmax(c, axis=1)
    top1_ref = np.argmax(r, axis=1)
    agree_mask = (top1 == top1_ref)
    top1_agree = int(np.count_nonzero(agree_mask))
    disagreeing = [int(i) for i in np.flatnonzero(~agree_mask)]

    top5_agree = 0
    top5_disagreeing = []
    for t in range(c.shape[0]):
        a = set(np.argpartition(-c[t], 5)[:5].tolist())
        b = set(np.argpartition(-r[t], 5)[:5].tolist())
        if a == b:
            top5_agree += 1
        else:
            top5_disagreeing.append(t)

    # The margin: the gap between the top-1 and the top-2 REFERENCE logits at
    # each position. This is how much divergence greedy decoding can absorb at
    # that position before the selected token flips, and it is a property of the
    # REFERENCE alone -- no change to the engine can move it. That is why it,
    # and not the observed divergence, is what makes the tolerance meaningful.
    srt = np.sort(r, axis=1)
    margins = (srt[:, -1] - srt[:, -2]).astype(np.float64)

    out = {
        "label": label,
        "positions": int(c.shape[0]),
        "vocab_size": int(c.shape[1]),
        "max_abs_diff": float(absdiff.max()),
        "max_abs_diff_position": int(pos),
        "max_abs_diff_token_id": int(tok),
        "max_rel_diff": float(reldiff.max()),
        "rel_diff_denominator": "|reference| + 1e-6, elementwise",
        "elements_below_rel_floor": below_floor,
        "mean_abs_diff": float(absdiff.mean()),
        "median_abs_diff": float(np.median(absdiff)),
        "rms_abs_diff": float(np.sqrt((absdiff ** 2).mean())),
        "abs_diff_percentiles": _percentiles(absdiff),
        "rel_diff_percentiles": _percentiles(reldiff),
        "top1_agreement_positions": top1_agree,
        "top1_agreement_fraction": top1_agree / c.shape[0],
        "top1_disagreeing_positions": disagreeing,
        "top5_set_agreement_positions": top5_agree,
        "top5_set_agreement_fraction": top5_agree / c.shape[0],
        "top5_disagreeing_positions": top5_disagreeing,
        "reference_top1_top2_margin": {
            "min": float(margins.min()),
            "median": float(np.median(margins)),
            "max": float(margins.max()),
            "argmin_position": int(np.argmin(margins)),
            "per_position": [float(x) for x in margins],
        },
        "margin_min_over_max_abs_diff": (
            float(margins.min() / absdiff.max()) if absdiff.max() > 0 else None),
    }
    if token_ids is not None:
        out["token_ids"] = [int(i) for i in token_ids]
    return out


# ---------------------------------------------------------------- the D2 gate --

def apply_d2_gate(per_configuration: list, tolerance: float,
                  greedy_match=None) -> dict:
    """Applies the three PASS conditions of BENCHMARK_PROTOCOL.md section 5 to a
    list of per-configuration statistic blocks. Returns a structured verdict;
    NEVER raises on a FAIL.

    The three conditions are reported separately as well as combined, because
    which one failed is the whole diagnostic value of the gate: an absolute
    overshoot with top-1 intact is a precision story, and a top-1 flip inside
    tolerance is a decision-boundary story.
    """
    if tolerance is None:
        raise ValueError("the D2 tolerance is a required argument and has no default here; "
                         "pass it, or read it with read_tolerance()")
    tolerance = float(tolerance)

    worst_abs = max((c["max_abs_diff"] for c in per_configuration), default=0.0)
    worst_abs_label = next((c["label"] for c in per_configuration
                            if c["max_abs_diff"] == worst_abs), None)
    min_margin = min((c["reference_top1_top2_margin"]["min"]
                      for c in per_configuration), default=None)
    min_margin_label = next((c["label"] for c in per_configuration
                             if c["reference_top1_top2_margin"]["min"] == min_margin), None)

    # Condition 1. "At or below" -- a difference exactly equal to the threshold
    # passes, which is what makes the threshold a stated boundary rather than an
    # open interval whose edge nobody can test.
    within = bool(worst_abs <= tolerance)
    offenders = [{"configuration": c["label"], "max_abs_diff": c["max_abs_diff"]}
                 for c in per_configuration if c["max_abs_diff"] > tolerance]

    # Condition 2.
    top1_all = all(c["top1_agreement_fraction"] == 1.0 for c in per_configuration)
    top1_offenders = [{"configuration": c["label"],
                       "positions": c["positions"],
                       "agreeing": c["top1_agreement_positions"],
                       "disagreeing_positions": c["top1_disagreeing_positions"]}
                      for c in per_configuration if c["top1_agreement_fraction"] != 1.0]

    # Condition 3. An unsupplied greedy comparison is NOT treated as satisfied.
    if greedy_match is None:
        greedy_ok = False
        greedy_note = ("NOT CHECKED in this invocation, and therefore NOT SATISFIED: the "
                       "greedy-sequence condition of section 5 requires an engine and a "
                       "reference generation over the fixed prompt set")
    else:
        greedy_ok = bool(greedy_match)
        greedy_note = "checked" if greedy_ok else "the sequences differ"

    verdict = "PASS" if (within and top1_all and greedy_ok) else "FAIL"
    return {
        "verdict": verdict,
        "tolerance": tolerance,
        "tolerance_form": ("elementwise ABSOLUTE difference over the full logit vector at "
                           "every position of a prefill, float32 on both sides; the gate "
                           "figure is the maximum over all positions and all vocabulary "
                           "entries"),
        "tolerance_source": f"BENCHMARK_PROTOCOL.md section 5, {TOLERANCE_TOKEN}",
        "conditions": {
            "max_abs_diff_within_tolerance": {
                "pass": within,
                "worst_max_abs_diff": worst_abs,
                "worst_configuration": worst_abs_label,
                "rule": "max absolute difference <= tolerance",
                "offending_configurations": offenders,
            },
            "top1_agreement_at_every_position": {
                "pass": top1_all,
                "offending_configurations": top1_offenders,
            },
            "greedy_sequence_matches_reference": {
                "pass": greedy_ok,
                "note": greedy_note,
            },
        },
        "observed": {
            "worst_max_abs_diff": worst_abs,
            "worst_max_abs_diff_configuration": worst_abs_label,
            "minimum_reference_margin": min_margin,
            "minimum_reference_margin_configuration": min_margin_label,
            "tolerance_over_worst_divergence": (
                tolerance / worst_abs if worst_abs > 0 else None),
            "minimum_margin_over_tolerance": (
                min_margin / tolerance if min_margin is not None else None),
        },
    }


def check(engine_logits, reference_logits, tolerance, token_ids=None,
          label: str = "", greedy_match=None) -> dict:
    """One configuration, end to end: statistics then gate. Returns a dict with
    both. Never raises on a FAIL."""
    stats = divergence_statistics(engine_logits, reference_logits, token_ids, label)
    gate = apply_d2_gate([stats], tolerance, greedy_match=greedy_match)
    return {"statistics": stats, "gate": gate, "verdict": gate["verdict"]}


# ------------------------------------------------------------- the provenance --

def _parse_defines(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#define BENCH_"):
                parts = line.split(None, 2)
                if len(parts) == 3:
                    out[parts[1]] = parts[2].strip().strip('"')
    return out


def build_provenance(stage: str) -> dict:
    """Every BENCHMARK_PROTOCOL.md section 9 provenance field, taken from the SAME
    source the C layer writes rather than re-derived by a different route:
    build/generated/build_info.h and build/generated/git_info.h, both generated
    by CMake. The device string is assembled from the stored environment
    fingerprint and its source is named, because this run touches no device and
    has no CUDA context to ask.
    """
    flags = _parse_defines(BUILD_INFO)
    git = _parse_defines(GIT_INFO)
    device = "unavailable"
    device_source = "none"
    if os.path.exists(FINGERPRINT):
        with open(FINGERPRINT, "r", encoding="utf-8") as f:
            fp = json.load(f)
        gpu = fp.get("gpu", {})
        name, cc = gpu.get("name"), gpu.get("compute_cap")
        if name and cc:
            device = f"{name} sm_{cc.replace('.', '')}"
            device_source = ("bench/results/machine_fingerprint.json; this run touches no "
                             "device -- the engine and the oracle are both on the CPU")
    return {
        "stage": stage,
        "git_commit": git.get("BENCH_GIT_COMMIT", "unavailable"),
        "git_worktree": git.get("BENCH_GIT_WORKTREE", "unavailable"),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_timestamp": flags.get("BENCH_BUILD_TIMESTAMP", "unavailable"),
        "device": device,
        "device_source": device_source,
        "cxx_compiler": flags.get("BENCH_CXX_COMPILER", "unavailable"),
        "cxx_flags": flags.get("BENCH_CXX_FLAGS", "unavailable"),
        "cuda_compiler": flags.get("BENCH_CUDA_COMPILER", "unavailable"),
        "cuda_flags": flags.get("BENCH_CUDA_FLAGS", "unavailable"),
        "provenance_source": ("build/generated/build_info.h and "
                              "build/generated/git_info.h, the same files the C layer "
                              "compiles into every results file"),
    }


# ------------------------------------------------------------------ the fixture --

def read_prompt_fixture(path: str = D3_FIXTURE) -> list:
    """Reads the fixed prompt set. Columns: id, target_tokens, verified_tokens,
    text. The text is the LAST tab-separated column, matching the C driver's
    reader, so the two cannot disagree about which column is the prompt."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 3:
                continue
            row = {"id": parts[0], "text": parts[-1]}
            try:
                row["target_tokens"] = int(parts[1])
            except ValueError:
                row["target_tokens"] = None
            try:
                row["verified_tokens"] = int(parts[2])
            except (ValueError, IndexError):
                row["verified_tokens"] = None
            rows.append(row)
    return rows


def prompt_set_identity(path: str) -> dict:
    """The prompt set's identity, carried into every results file. A measurement
    whose prompt set is not recorded cannot be compared with another."""
    status = "UNLABELLED"
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
    if "NOT A PLACEHOLDER" in head or "FIXED PROMPT SET" in head:
        status = "FIXED"
    elif "PLACEHOLDER" in head:
        status = "PLACEHOLDER"
    rows = read_prompt_fixture(path)
    return {
        "status": status,
        "decision": "D3",
        "source": os.path.relpath(path, REPO).replace("\\", "/"),
        "rows": [{"id": r["id"], "target_tokens": r["target_tokens"],
                  "verified_tokens": r["verified_tokens"]} for r in rows],
    }


# --------------------------------------------------------------- script mode --

def _run_engine(engine: str, prompt_path: str, truncate: int, dump: str,
                generate: int = 0, timeout: int = 1200,
                kv_cache: str = "off", via_decode: bool = False):
    """One engine invocation. `kv_cache` and `via_decode` are Stage 4's: the
    default pair ("off", False) is the Stage 3 invocation exactly, so the
    no-cache measurement is taken by the same command line as before."""
    cmd = [engine, "--prompt-file", prompt_path, "--truncate", str(truncate),
           "--cproj", "as-stored", "--dump-logits", dump]
    if via_decode:
        cmd += ["--via-decode"]
    elif kv_cache == "on":
        cmd += ["--kv-cache", "on"]
    if generate:
        cmd += ["--generate", str(generate)]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        sys.stderr.write(out.stdout + "\n" + out.stderr + "\n")
        raise RuntimeError(f"{engine} exited {out.returncode}")
    ids = None
    for line in out.stdout.splitlines():
        if line.startswith("ids:"):
            ids = [int(t) for t in line[4:].split()]
    if ids is None:
        raise RuntimeError("the engine printed no id sequence")
    return ids, out.stdout


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engine", required=True, help="path to the built gpt2_tool")
    p.add_argument("--fixture", default=D3_FIXTURE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--work-dir", default=None,
                   help="where the logit dumps and the oracle arrays are cached")
    p.add_argument("--tolerance", default=None,
                   help="the D2 threshold. Omitted, it is read from "
                        "BENCHMARK_PROTOCOL.md section 5")
    p.add_argument("--measure-only", action="store_true",
                   help="compute and report the divergence and margin statistics and apply NO "
                        "gate. This is the mode that runs BEFORE the threshold exists, so that "
                        "the run which justifies a threshold is never shaped by one")
    p.add_argument("--reuse-dumps", action="store_true",
                   help="reuse cached engine dumps and oracle arrays if present, so a gating "
                        "pass re-reads the measurement pass's arrays instead of re-running "
                        "the engine")
    p.add_argument("--greedy-tokens", type=int, default=3,
                   help="tokens to greedily generate for the sequence-match condition")
    p.add_argument("--paths", choices=("both", "nocache-only"), default="both",
                   help="Stage 4: 'both' additionally measures the prefill path in cache "
                        "mode and the cached-decode path (DECISION E checks 1, 2 and 3). "
                        "'nocache-only' is the Stage 3 behaviour exactly")
    p.add_argument("--stage", default="stage-4")
    a = p.parse_args(argv)

    work = a.work_dir or os.path.join(REPO, "build", "correctness_work")
    os.makedirs(work, exist_ok=True)

    tolerance = None
    if not a.measure_only:
        tolerance = float(a.tolerance) if a.tolerance is not None else read_tolerance()

    rows = read_prompt_fixture(a.fixture)
    if not rows:
        print(f"no data rows in {a.fixture}", file=sys.stderr)
        return 2

    from reference.reference_impl import ReferenceGPT2
    import torch

    print(f"fixture: {a.fixture} ({len(rows)} rows)")
    print(f"mode   : {'MEASURE ONLY, no gate applied' if a.measure_only else 'GATE'}")
    if tolerance is not None:
        print(f"D2 tolerance: {tolerance:g} (absolute, max over all positions and all vocab)")

    # The engine runs first, so the oracle's gigabyte of resident weights is not
    # held while the engine is working.
    engine_runs = {}
    for r in rows:
        L = r["target_tokens"]
        prompt_path = os.path.join(work, f"{r['id']}.txt")
        with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(r["text"])
        dump = os.path.join(work, f"{r['id']}_engine.bin")
        ids_path = os.path.join(work, f"{r['id']}_ids.json")
        if a.reuse_dumps and os.path.exists(dump) and os.path.exists(ids_path):
            with open(ids_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"  engine {r['id']}: reusing the cached dump ({cached['tokens']} tokens)")
            engine_runs[r["id"]] = {"dump": dump, "ids": cached["ids"],
                                    "greedy": cached["greedy"],
                                    "dump_prefill_cache": cached.get("dump_prefill_cache"),
                                    "dump_via_decode": cached.get("dump_via_decode")}
            continue
        gen = a.greedy_tokens if L == min(x["target_tokens"] for x in rows) else 0
        print(f"  engine {r['id']}: prefill L={L}, no-cache path"
              + (f", greedy {gen} tokens" if gen else ""))
        ids, _ = _run_engine(a.engine, prompt_path, L, dump, generate=gen)
        prompt_ids = ids[:L]
        greedy = ids if gen else None
        entry = {"dump": dump, "ids": prompt_ids, "greedy": greedy}

        # ---- Stage 4. Two more runs per row, when the engine offers the flags.
        # DECISION E check 1 (the existing gate on the prefill path in CACHE
        # mode as well as no-cache) and check 2 (the whole logit matrix from the
        # CACHED DECODE PATH alone). Both write the same TIE2LOGI format, so the
        # same comparator reads all three.
        if a.paths != "nocache-only":
            cache_dump = os.path.join(work, f"{r['id']}_engine_cache.bin")
            print(f"  engine {r['id']}: prefill L={L}, CACHE path")
            _run_engine(a.engine, prompt_path, L, cache_dump, kv_cache="on")
            entry["dump_prefill_cache"] = cache_dump

            dec_dump = os.path.join(work, f"{r['id']}_engine_viadecode.bin")
            print(f"  engine {r['id']}: {L - 1} cached decode steps after a 1-token prefill "
                  f"(check 2)")
            _run_engine(a.engine, prompt_path, L, dec_dump, via_decode=True)
            entry["dump_via_decode"] = dec_dump

        engine_runs[r["id"]] = entry
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump({"tokens": len(prompt_ids), "ids": prompt_ids, "greedy": greedy,
                       "dump_prefill_cache": entry.get("dump_prefill_cache"),
                       "dump_via_decode": entry.get("dump_via_decode")}, f)

    # The oracle, once, reused across all four lengths.
    oracle = ReferenceGPT2()
    print(f"oracle: verified {oracle.n_verified} tensors against the inventory; "
          f"torch {torch.__version__}, threads {oracle.threads}, device cpu")

    per_configuration = []          # the GATING set: the prefill path, no-cache
    per_configuration_cache = []    # check 1, the prefill path in CACHE mode
    per_configuration_decode = []   # check 2, the CACHED DECODE path
    path_consistency = []           # check 3, bit-for-bit between the two
    prefill_path_consistency = []   # check 1's two prefill paths against each other
    for r in rows:
        L = r["target_tokens"]
        run = engine_runs[r["id"]]
        engine_logits = read_logit_dump(run["dump"])
        if engine_logits.shape[0] != L:
            print(f"the engine dumped {engine_logits.shape[0]} positions for a configuration "
                  f"of {L} tokens", file=sys.stderr)
            return 2
        ref = oracle.forward(run["ids"]).numpy()
        stats = divergence_statistics(
            engine_logits, ref, run["ids"],
            label=f"prefill, L={L} tokens, D3 row {r['id']}")
        stats["prompt_row_id"] = r["id"]
        stats["target_tokens"] = L
        stats["cproj_reading"] = "as-stored"
        stats["engine_path"] = "prefill, no-cache"
        stats["positions_covered"] = ("EVERY position of the prefill: gpt2_tool --dump-logits "
                                      "writes logits for all positions, so both D2 bounds are "
                                      "measured over the whole prefill")
        per_configuration.append(stats)
        m = stats["reference_top1_top2_margin"]
        print(f"  L={L:4d}  prefill/nocache  max abs {stats['max_abs_diff']:.6g}  rms "
              f"{stats['rms_abs_diff']:.6g}  max rel {stats['max_rel_diff']:.6g}  "
              f"top-1 {stats['top1_agreement_positions']}/{L}  "
              f"margin min {m['min']:.6g} (pos {m['argmin_position']})  "
              f"margin/div {stats['margin_min_over_max_abs_diff']:.2f}x")

        # ---- DECISION E check 1, the same gate on the prefill path in CACHE
        # mode. Predicted identical to the no-cache path, and the prediction is
        # checked rather than assumed: the two are also compared BIT FOR BIT.
        if run.get("dump_prefill_cache"):
            cache_logits = read_logit_dump(run["dump_prefill_cache"])
            cstats = divergence_statistics(
                cache_logits, ref, run["ids"],
                label=f"prefill CACHE path, L={L} tokens, D3 row {r['id']}")
            cstats["prompt_row_id"] = r["id"]
            cstats["target_tokens"] = L
            cstats["cproj_reading"] = "as-stored"
            cstats["engine_path"] = "prefill, cache"
            per_configuration_cache.append(cstats)
            bits = bitwise_comparison(
                cache_logits, engine_logits,
                label=f"prefill cache against prefill no-cache, L={L}, row {r['id']}")
            bits["prompt_row_id"] = r["id"]
            bits["target_tokens"] = L
            prefill_path_consistency.append(bits)
            print(f"  L={L:4d}  prefill/cache    max abs {cstats['max_abs_diff']:.6g}  "
                  f"top-1 {cstats['top1_agreement_positions']}/{L}  "
                  f"against no-cache prefill: "
                  f"{'BIT FOR BIT' if bits['bit_for_bit'] else 'DIFFERS'}"
                  f" ({bits['differing_elements']} elements, max "
                  f"{bits['max_abs_difference']:.6g})")

        # ---- DECISION E check 2, the CACHED DECODE path against the oracle,
        # compared exactly as the gate compares prefill; and check 3, the same
        # matrix against the prefill matrix of the same row, bit for bit.
        if run.get("dump_via_decode"):
            dec_logits = read_logit_dump(run["dump_via_decode"])
            dstats = divergence_statistics(
                dec_logits, ref, run["ids"],
                label=f"cached decode path, every position, L={L} tokens, D3 row {r['id']}")
            dstats["prompt_row_id"] = r["id"]
            dstats["target_tokens"] = L
            dstats["cproj_reading"] = "as-stored"
            dstats["engine_path"] = "cached decode"
            dstats["assembly"] = ("a 1-token prefill followed by one cached decode step per "
                                  "remaining position; every row comes from the cached path")
            per_configuration_decode.append(dstats)
            bits = bitwise_comparison(
                dec_logits, engine_logits,
                label=f"cached decode against prefill, L={L}, row {r['id']}")
            bits["prompt_row_id"] = r["id"]
            bits["target_tokens"] = L
            path_consistency.append(bits)
            dm = dstats["reference_top1_top2_margin"]
            print(f"  L={L:4d}  cached-decode    max abs {dstats['max_abs_diff']:.6g}  rms "
                  f"{dstats['rms_abs_diff']:.6g}  max rel {dstats['max_rel_diff']:.6g}  "
                  f"top-1 {dstats['top1_agreement_positions']}/{L}  "
                  f"margin min {dm['min']:.6g}")
            print(f"           check 3 against the prefill matrix: "
                  f"{'BIT FOR BIT' if bits['bit_for_bit'] else 'DIFFERS'} "
                  f"({bits['differing_elements']} of {bits['elements']} elements, max abs "
                  f"{bits['max_abs_difference']:.6g})")

    # The greedy-sequence condition, on the shortest row, against the same oracle.
    greedy_block = None
    shortest = min(rows, key=lambda r: r["target_tokens"])
    eg = engine_runs[shortest["id"]]["greedy"]
    if eg:
        n_new = len(eg) - shortest["target_tokens"]
        ref_seq = oracle.greedy(engine_runs[shortest["id"]]["ids"], n_new)
        greedy_block = {
            "prompt_row_id": shortest["id"],
            "prompt_tokens": shortest["target_tokens"],
            "tokens_generated": n_new,
            "engine_sequence": [int(x) for x in eg],
            "reference_sequence": [int(x) for x in ref_seq],
            "match": [int(x) for x in eg] == [int(x) for x in ref_seq],
        }
        print(f"  greedy on {shortest['id']}: {n_new} tokens, "
              f"{'MATCH' if greedy_block['match'] else 'DIFFER'}")

    margin_table = [{
        "target_tokens": c["target_tokens"],
        "prompt_row_id": c["prompt_row_id"],
        "positions": c["positions"],
        "max_abs_diff": c["max_abs_diff"],
        "margin_min": c["reference_top1_top2_margin"]["min"],
        "margin_min_position": c["reference_top1_top2_margin"]["argmin_position"],
        "margin_median": c["reference_top1_top2_margin"]["median"],
        "margin_max": c["reference_top1_top2_margin"]["max"],
        "margin_min_over_max_abs_diff": c["margin_min_over_max_abs_diff"],
    } for c in per_configuration]

    doc = build_provenance(a.stage)
    doc["kind"] = "correctness"
    doc["generated_by"] = "bench/correctness.py"
    doc["prompt_set"] = prompt_set_identity(a.fixture)
    doc["reference"] = {
        "implementation": "reference/reference_impl.py, hand-written in torch",
        "call_path": "ReferenceGPT2().forward(ids) and .greedy(ids, n) -- called, never "
                     "reimplemented",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "torch_num_threads": oracle.threads,
        "tensors_verified_against_inventory": oracle.n_verified,
        "ran_before_any_timed_bracket": True,
        "what_agreement_proves": ("the two implementations agree; NOT that either matches "
                                  "GPT-2. Both were written in one session from one reading "
                                  "of the architecture"),
    }
    doc["engine"] = {
        "binary": a.engine,
        "logit_dump_format": ('8-byte magic "TIE2LOGI", int32 positions, int32 vocab_size, '
                              "then positions*vocab_size float32 in row order"),
        "positions_emitted": "every position of the prefill",
    }
    doc["counters"] = {
        "collected": False,
        "reason": ("BENCHMARK_PROTOCOL.md section 6 requires Nsight counters for GPU stages "
                   "from Stage 7 onward. This stage writes no kernel and this run touches no "
                   "device"),
    }
    doc["timings"] = {
        "collected": False,
        "reason": "this is the correctness gate; it times nothing and reports no latency",
    }
    doc["margin_table"] = margin_table
    doc["configurations"] = per_configuration
    doc["greedy"] = greedy_block

    # ---- Stage 4: the two extra paths, the set-wide figures over BOTH of them,
    # and the corrected-rule D2 arithmetic (DECISION A).
    all_paths = per_configuration + per_configuration_cache + per_configuration_decode
    set_wide_max_div = max(c["max_abs_diff"] for c in all_paths)
    set_wide_max_div_label = next(c["label"] for c in all_paths
                                  if c["max_abs_diff"] == set_wide_max_div)
    # The margin is a property of the REFERENCE alone, so it is taken over the
    # four rows once rather than once per engine path.
    set_wide_min_margin = min(c["reference_top1_top2_margin"]["min"]
                              for c in per_configuration)
    set_wide_min_margin_cfg = next(c for c in per_configuration
                                   if c["reference_top1_top2_margin"]["min"]
                                   == set_wide_min_margin)

    doc["set_wide"] = {
        "max_abs_divergence": set_wide_max_div,
        "max_abs_divergence_configuration": set_wide_max_div_label,
        "max_abs_divergence_taken_over": (
            "BOTH engine paths this stage measures -- the prefill path and the cached-decode "
            "path -- over all positions of all four D3 rows"
            if per_configuration_decode else
            "the prefill path only; the cached-decode path was not measured in this run"),
        "min_reference_margin": set_wide_min_margin,
        "min_reference_margin_row": set_wide_min_margin_cfg["prompt_row_id"],
        "min_reference_margin_position":
            set_wide_min_margin_cfg["reference_top1_top2_margin"]["argmin_position"],
        "min_reference_margin_note": ("a property of the REFERENCE alone, so it is taken over "
                                      "the four rows once and not per engine path"),
        "paths_measured": [c["engine_path"] for c in
                           ([per_configuration[0]] if per_configuration else [])
                           + ([per_configuration_cache[0]] if per_configuration_cache else [])
                           + ([per_configuration_decode[0]] if per_configuration_decode else [])],
    }

    doc["d2_corrected_rule"] = d2_corrected_rule(set_wide_min_margin, set_wide_max_div)
    doc["d2_corrected_rule"]["value_in_force_at_run_time"] = tolerance
    doc["d2_corrected_rule"]["decision"] = "DECISION A, Stage 4"

    # The gate is a pure function of the statistics and the threshold, so the
    # verdict under OTHER thresholds is recorded without re-running the engine.
    # Stage 4 is required to record a PASS against the superseded 6e-03 as well
    # as against the re-derived value, and the two are the same measurement.
    if tolerance is not None:
        also = {}
        for name, value in (("superseded_stage3_value", 6e-03),
                            ("stage4_corrected_rule_value",
                             doc["d2_corrected_rule"]["value"])):
            if value is None:
                continue
            also[name] = {
                "tolerance": value,
                "prefill_nocache": apply_d2_gate(
                    per_configuration, value,
                    greedy_match=(greedy_block["match"] if greedy_block else None))["verdict"],
                "prefill_cache": (apply_d2_gate(
                    per_configuration_cache, value,
                    greedy_match=(greedy_block["match"] if greedy_block else None))["verdict"]
                    if per_configuration_cache else None),
                "cached_decode_path": (apply_d2_gate(
                    per_configuration_decode, value,
                    greedy_match=(greedy_block["match"] if greedy_block else None))["verdict"]
                    if per_configuration_decode else None),
            }
        doc["gate_against_both_values"] = {
            "note": ("one measurement, evaluated at two thresholds. The gate is a pure "
                     "function of the statistics and the threshold, so no engine run was "
                     "repeated to produce these verdicts"),
            "values": also,
        }

    doc["stage4_checks"] = {
        "check_1_prefill_both_paths": {
            "description": ("the existing three-condition gate on the prefill path, run with "
                            "the switch set to CACHE and to NO-CACHE; predicted identical"),
            "nocache_configurations": len(per_configuration),
            "cache_configurations": len(per_configuration_cache),
            "cache_gate": (apply_d2_gate(
                per_configuration_cache, tolerance,
                greedy_match=(greedy_block["match"] if greedy_block else None))
                if (per_configuration_cache and tolerance is not None) else None),
            "cache_against_nocache_bit_for_bit": prefill_path_consistency,
            "all_bit_for_bit": (all(b["bit_for_bit"] for b in prefill_path_consistency)
                                if prefill_path_consistency else None),
        },
        "check_2_cached_decode_against_oracle": {
            "description": ("the full positions x vocab logit matrix assembled from the "
                            "CACHED DECODE PATH ALONE, compared to the oracle exactly as the "
                            "gate compares prefill. This divergence enters the D2 lower "
                            "bound (DECISION A)"),
            "configurations": per_configuration_decode,
            "margin_table": [{
                "target_tokens": c["target_tokens"],
                "prompt_row_id": c["prompt_row_id"],
                "positions": c["positions"],
                "max_abs_diff": c["max_abs_diff"],
                "margin_min": c["reference_top1_top2_margin"]["min"],
                "margin_min_position": c["reference_top1_top2_margin"]["argmin_position"],
                "margin_min_over_max_abs_diff": c["margin_min_over_max_abs_diff"],
            } for c in per_configuration_decode],
            "top1_agreement_at_every_position": (
                all(c["top1_agreement_fraction"] == 1.0 for c in per_configuration_decode)
                if per_configuration_decode else None),
            "relative_statistics_note": ("relative statistics are reported and do NOT gate, "
                                         "as in Stage 3"),
        },
        "check_3_cached_decode_against_prefill": {
            "description": ("the cached-path matrix against the prefill matrix of the same "
                            "row, position by position. Bit-for-bit is the expectation; a "
                            "pass only within D2 would be a result to localise, not to hide"),
            "comparisons": path_consistency,
            "all_bit_for_bit": (all(b["bit_for_bit"] for b in path_consistency)
                                if path_consistency else None),
            "total_differing_elements": sum(b["differing_elements"] for b in path_consistency),
        },
    }

    if a.measure_only:
        doc["gate"] = {
            "applied": False,
            "reason": ("MEASURE ONLY. The D2 threshold is derived FROM this run, so applying "
                       "one here would shape the measurement by the value it is meant to "
                       "justify"),
        }
        verdict = None
    else:
        doc["gate"] = apply_d2_gate(
            per_configuration, tolerance,
            greedy_match=(greedy_block["match"] if greedy_block else None))
        verdict = doc["gate"]["verdict"]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    print(f"written: {a.out}")

    if verdict is None:
        print("no gate applied (measure-only)")
        return 0
    print(f"D2 GATE: {verdict}   (prefill path, no-cache)")
    for name, cond in doc["gate"]["conditions"].items():
        print(f"  {'pass' if cond['pass'] else 'FAIL'}  {name}")
    cg = doc["stage4_checks"]["check_1_prefill_both_paths"]["cache_gate"]
    if cg:
        print(f"D2 GATE: {cg['verdict']}   (prefill path, CACHE mode -- DECISION E check 1)")
    s4 = doc["stage4_checks"]
    if s4["check_3_cached_decode_against_prefill"]["comparisons"]:
        print(f"check 2: cached-decode path against the oracle, set-wide max abs "
              f"{max(c['max_abs_diff'] for c in per_configuration_decode):.6e}")
        print(f"check 3: cached decode against prefill -- "
              f"{'BIT FOR BIT on every row' if s4['check_3_cached_decode_against_prefill']['all_bit_for_bit'] else 'DIFFERS'}")
    cr = doc["d2_corrected_rule"]
    print("DECISION A, the corrected rule, from THIS run's measurement:")
    print(f"  set-wide max abs divergence over both paths: {cr['set_wide_max_abs_divergence']:.6e}")
    print(f"  set-wide minimum reference margin          : {cr['set_wide_min_reference_margin']:.6e}")
    print(f"  lower = {cr['lower_bound_derivation']}   upper = {cr['upper_bound_derivation']}")
    print(f"  window margin/divergence = {cr['window_ratio_margin_over_divergence']:.3f}x "
          f"(at least {cr['window_minimum_required']}x required) -> "
          f"{'OPEN' if cr['usable'] else 'NOT USABLE'}")
    if cr["usable"]:
        print(f"  geometric mean {cr['geometric_mean']:.6e}, "
              f"{cr['significant_figures']} significant figures, value {cr['value']:.1e}")
        print(f"  achieved {cr['achieved_factor_above_divergence']:.2f}x above divergence and "
              f"{cr['achieved_factor_below_margin']:.2f}x below margin")
    else:
        print(f"  FINDING: {cr['finding']}")
    # A FAIL is REPORTED, not raised. The exit code lets a caller branch on it;
    # the results file is written either way.
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
