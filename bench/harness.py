"""harness.py -- the benchmark orchestrator.

WHAT IT IS, AND WHAT IT DELIBERATELY IS NOT. This is the top-level orchestrator
for every timed run from Stage 3 onward. It selects configurations, enforces the
run conditions, drives a C timing binary as a subprocess, reads that binary's
JSON, runs the correctness gate, compares against a prior stage for regression,
and writes one results file in the BENCHMARK_PROTOCOL.md section 9 format.

IT CONTAINS NO TIMING CODE. Warmup, sampling, the median and standard deviation,
and the 5%-of-median validity rule live in C, in bench/microbench/bench_common.h,
once, where they are already unit-tested. Reimplementing any of them in Python
would create a second place for the protocol rules to drift, which is the thing
bench_common exists to prevent. TECHNICAL_SPEC.md section 4 calls
bench/stage2_forward_bench.c "temporary; Stage 3's harness replaces it"; the
resolution taken here is that the harness replaces it as the ORCHESTRATOR and it
remains the timing driver.

THE 5%-OF-MEDIAN LIMIT IS NOT A PARAMETER. There is no argument, environment
variable or keyword anywhere in this file that changes it, and there is no retry
path: a configuration cannot be re-run into validity by this code because this
code cannot re-run a configuration. An INVALID verdict is recorded as INVALID and
the numbers behind it are kept.

ROBUST STATISTICS ARE DIAGNOSTIC. Where a trimmed figure appears it is labelled
diagnostic_*, it never replaces a reported value, and it never converts an
INVALID run into a valid one.


================================================================================
W3 -- THE TIMING CONSTRUCTION FOR FAST CONFIGURATIONS
================================================================================
PERSISTENT.md section 8 W3. Stage 0's M = 1 GEMM configurations had medians of
30-90 microseconds with the variance carried by single interruption events, and
were INVALID in at least one of two runs in all five shapes. Stage 2's decode
steps were five orders of magnitude above that band and were unaffected, so W3
was untested by anything measured before Stage 3. Stage 4 adds a KV cache and
decode steps get fast again. The construction below is set here and every later
stage inherits it.

THE ARITHMETIC.

Take n samples of a workload whose uninterrupted time is B, of which k are hit
by one interruption costing 0.5*B. The sample standard deviation as a percentage
of the median is

    sd% = 50 * sqrt( k*(n-k) / (n*(n-1)) )

At n = 30, k = 1 that is 50 * sqrt(29/870) = 9.13%, roughly twice the 5% limit,
and it agrees with W3's own figure of 0.5/sqrt(29) = 9.29% to within the
difference between the two ways of writing it.

Solving for the 5% limit at k = 1 gives sd% = 50/sqrt(n), so n > 100. But that
is the WRONG QUESTION, because k is not fixed at 1: interruptions arrive at a
rate, so k grows with n. Take a rate lambda per unit time and a cost delta per
event. Over a per-sample bracket of duration T the event count has mean
lambda*T and standard deviation sqrt(lambda*T), so

    relative sd = delta*sqrt(lambda*T)/T = delta*sqrt(lambda/T)

which depends on T and NOT on n. MORE SAMPLES DO NOT REDUCE IT. Dispersion falls
as one over the square root of the PER-SAMPLE DURATION.

Checking the form against the observed case -- B = 30-90 us, one event in thirty
samples, each costing 0.5*B -- gives delta = 0.5B, lambda = (1/30)/B, and a
relative sd of 0.5*sqrt(1/30) = 9.13%, reproducing the figure above.

Batching R iterations into one timed bracket multiplies T by R and divides the
relative dispersion by sqrt(R): 0.5*sqrt(p/R) for a per-iteration event
probability p. At p = 1/30, R = 10 gives 2.89% and R = 30 gives 1.67%.

THE CONSTRUCTION, (a) THROUGH (f).

(a) SAMPLE COUNT: 30, the project standard, against a protocol minimum of 20.
    Not reduced, and NOT raised as a remedy for dispersion -- the arithmetic
    above is precisely why raising it is not a remedy.

(b) CONSTRUCTION: adaptive, decided from a MEASURED probe and never from an
    expectation. Before each configuration's timed set the harness runs one
    untimed probe iteration and measures it. If the probe is at or above a
    per-sample floor of 10 ms, the harness uses the repeated-single-iteration
    construction. Below it, the harness uses a BATCHED construction with
    R = ceil(10 ms / probe) iterations inside one timed bracket, reporting the
    per-iteration cost as the bracket divided by R. The 10 ms floor is chosen so
    that an interruption of the absolute size Stage 0 observed -- of order tens
    of microseconds -- is a fraction of a percent of the bracket rather than half
    of it.

(c) THE TRADE-OFF, stated plainly: a batched sample reports an AMORTISED MEAN
    over R iterations, so the per-iteration distribution is no longer observable
    and individual interruption events can no longer be counted or
    characterised. The dispersion statistic still tests dispersion, but of
    BATCHES rather than of iterations. This is a real loss of diagnostic
    resolution, accepted in exchange for a measurement the 5% rule can validate
    honestly.

(d) DECODE UNDER BATCHING, the rule Stage 4 will need: a decode step advances the
    context, so R consecutive decode steps are R DIFFERENT shapes. When a batched
    sample spans contexts, the harness records the context RANGE covered and
    reports the figure as an amortised cost over [c, c+R), never as the cost at a
    single context. Where the implementation permits restoring the cache state
    outside the timed bracket so that R identical steps are measured, that is
    preferred, and which of the two was done is recorded.

(e) THE 5%-OF-MEDIAN LIMIT IS NEVER LOOSENED, by this stage or by this harness,
    under any circumstance. It is not a parameter and this file exposes no way to
    change it.

(f) The harness records, per configuration, which construction was used, the
    probe value it was chosen from, the batch factor R, and the effective
    per-sample bracket duration. A construction that is not recorded is not
    reproducible.

STATE OF THE BATCHED PATH AS OF STAGE 3. Stage 3's own configurations are
seconds-scale and every one of them takes the single-iteration path. The batched
path is therefore BUILT and UNIT-TESTED here but NOT EXERCISED by a real timed
run in this stage. The harness emits `--repeat R` to the timing command when
R > 1; the Stage 2 timing driver does not implement that flag, because adding a
batching path to its timed bracket would have been a change to timing code, which
Stage 3 was not permitted to make. Stage 4, which is the first stage whose
configurations are fast enough for R > 1 to be selected, is where the C side of
the batched bracket is added.

STAGE 4 AMENDMENT, 2026-10-05. Three things changed, and nothing else.

W3 -- THE SILENT-FABRICATION PATH IS CLOSED, AND THE C LOOP IS STILL NOT BUILT.
Stage 4's own configurations are all far above the 10 ms floor, so R = 1 was
again selected everywhere and the batched C bracket was again not needed. What
Stage 4 added instead is a REFUSAL: before this harness divides any bracket by
R, it requires the driver's own record for that configuration to carry
`repeat_applied == R`. If it does not -- because the driver ignored the flag, or
reported nothing -- the whole run HARD-FAILS with a non-zero exit naming the
configuration, R, what the driver reported, and W3. It never divides an
unconfirmed bracket. It also asserts `repeat_applied == 1` on every R = 1
configuration, so the confirmation is not only checked where it would be
convenient. The C-side batching loop is owned by the first stage that actually
selects R > 1.

W14 -- THE CROSS-SESSION SHIFT TRAVELS INSIDE THE ARTIFACT. The same binary
measured in the Stage 2 and Stage 3 sessions differed by roughly 9 to 12 percent
on unchanged code, about 2.3 times the 4.4 percent noise floor, and the
regression check cannot see it: the environment fingerprint does not include the
process set. Every comparison record now carries `process_set_gated: false` and
a pointer to PERSISTENT.md section 8, W14, so a reader of the results file is
told what the comparison does not control. The comparison LOGIC is otherwise
unchanged.

CONFIGURATIONS ARE KEYED BY PATH AS WELL AS BY LENGTH. Stage 4 times each
length on both engine paths, and a cache configuration must never be compared
against a no-cache one as a regression. A no-cache configuration keeps the
Stage 3 key exactly -- `prefill|L=16` -- so it pairs with the Stage 3 baseline
on its own; a cache configuration gets a distinct key with a `|cache` suffix,
which the Stage 3 file does not contain, so the comparison refuses it by the
existing "does not appear in the prior file" rule rather than by a new one. The
cached-against-uncached change is the stage's RESULT and is reported as that,
labelled, in the MEASUREMENTS.md entry -- not as a regression here.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "bench"))
sys.path.insert(0, REPO)

import correctness as C                                    # noqa: E402

# ---- protocol constants. NOT parameters. ------------------------------------
# The validity limit is duplicated nowhere: the C layer owns it and writes its
# verdict into the results file. The copy here exists ONLY so the harness can
# assert the C layer's verdict rather than trust it, and it is never applied as a
# threshold the harness can relax.
PROTOCOL_MAX_STDDEV_PCT = 5.0            # BENCHMARK_PROTOCOL.md section 3
NOISE_FLOOR_PCT = 4.4                    # section 4.1, Stage 0. USED, never re-derived here.
PROTOCOL_WARMUP = 25                     # section 4.2, adjusted for this machine
PROTOCOL_SAMPLES = 30                    # section 3, the project standard
W3_PER_SAMPLE_FLOOR_MS = 10.0            # W3 (b)
EXPECTED_CLOCK_MHZ = 1365                # section 4 / PERSISTENT.md D8
EXPECTED_COMPARED_FIELDS = 25            # W12: 26 minus the provenance-only build timestamp

D3_FIXTURE = os.path.join(REPO, "tests", "fixtures", "benchmark_prompts.tsv")
D3_PREFILL_LENGTHS = (16, 32, 64, 128)
D3_DECODE_CONTEXTS = (32, 64, 128)

DEFAULT_OUT = os.path.join(REPO, "bench", "results", "stage4", "stage4_harness.json")
DEFAULT_CUBLAS_OUT = os.path.join(REPO, "bench", "results", "stage3", "stage3_cublas_d3.json")

# Stage 4. The owning stage of this harness invocation, used as the default
# BENCH_STAGE_ID for the C layer and as the "stage" field of the results file.
STAGE_ID = "stage-4"

# W14. The limitation that travels inside every comparison record, because a
# prose-only caveat in a document is invisible to the Stage 10 consumer that
# reads these files.
PROCESS_SET_NOTE = "see PERSISTENT.md §8 W14"

# Stage 4's paired interleaved configuration list, in the order the machine sees
# it: each cache configuration immediately followed by its no-cache control, so
# any drift within the session falls on both members of a pair. Prefill L = 128
# has NO no-cache control by decision -- it would add about 26 minutes to the
# longest-running configuration class on this machine -- and is labelled in the
# entry as having no baseline of either kind.
STAGE4_CONFIGS = (
    ("prefill", 16, "cache"), ("prefill", 16, "nocache"),
    ("prefill", 32, "cache"), ("prefill", 32, "nocache"),
    ("prefill", 64, "cache"), ("prefill", 64, "nocache"),
    ("prefill", 128, "cache"),
    ("decode", 32, "cache"), ("decode", 32, "nocache"),
    ("decode", 64, "cache"), ("decode", 64, "nocache"),
    ("decode", 128, "cache"), ("decode", 128, "nocache"),
    ("gemm", 768, None), ("gemm", 3072, None),
)


def format_configs(items) -> str:
    """The driver's --configs argument: `kind:tokens[:path]`, comma separated,
    in order. The order IS the measurement design, so it is passed explicitly
    rather than reconstructed from two independent length lists."""
    out = []
    for kind, tokens, path in items:
        out.append(f"{kind}:{tokens}" + (f":{path}" if path else ""))
    return ",".join(out)

# The substantive fingerprint fields a regression comparison requires to be
# identical between the two results files. The build timestamp is deliberately
# NOT among them -- it moves on every clean build and carries no information
# about the machine (W12). The git commit is not among them either: a later stage
# is a different commit by construction.
SUBSTANTIVE_FINGERPRINT_FIELDS = (
    "device", "cxx_compiler", "cxx_flags", "cuda_compiler", "cuda_flags", "cpu_timer",
)


# ======================================================== the W3 construction ==

def select_construction(probe_ms: float, floor_ms: float = W3_PER_SAMPLE_FLOOR_MS) -> dict:
    """Chooses the W3 timing construction from a MEASURED probe.

    At or above the floor: the repeated-single-iteration construction, R = 1.
    Below it: the batched construction, R = ceil(floor / probe), with the
    per-iteration cost reported as the bracket divided by R.

    Returns the full construction record W3 (f) requires: which construction, the
    probe it was chosen from, R, and the effective per-sample bracket duration.
    """
    if not (probe_ms > 0) or not math.isfinite(probe_ms):
        raise ValueError(f"the probe must be a positive finite duration in ms; got {probe_ms!r}")
    if probe_ms >= floor_ms:
        return {
            "construction": "single_iteration",
            "probe_ms": probe_ms,
            "per_sample_floor_ms": floor_ms,
            "repeat_factor_R": 1,
            "effective_bracket_ms": probe_ms,
            "amortised": False,
            "note": ("the probe is at or above the per-sample floor, so one iteration per "
                     "timed bracket and the per-iteration distribution stays observable"),
        }
    R = int(math.ceil(floor_ms / probe_ms))
    return {
        "construction": "batched",
        "probe_ms": probe_ms,
        "per_sample_floor_ms": floor_ms,
        "repeat_factor_R": R,
        "effective_bracket_ms": probe_ms * R,
        "amortised": True,
        "note": ("W3 (c) TRADE-OFF: this sample is an AMORTISED MEAN over R iterations. The "
                 "per-iteration distribution is no longer observable and individual "
                 "interruption events can no longer be counted or characterised; the "
                 "dispersion statistic tests dispersion of BATCHES, not of iterations"),
    }


def predicted_dispersion_pct(n: int, k: int) -> float:
    """W3's closed form: sd% = 50 * sqrt(k*(n-k) / (n*(n-1))) for k of n samples
    hit by one interruption costing half the uninterrupted time. Reproduced here
    so the arithmetic in this file's header is executable rather than asserted."""
    if n < 2 or k < 0 or k > n:
        raise ValueError(f"n must be >= 2 and 0 <= k <= n; got n={n}, k={k}")
    return 50.0 * math.sqrt((k * (n - k)) / (n * (n - 1)))


# ================================================================= preflight ==

def _default_verify_fingerprint():
    import machine_state as ms
    stored_path = os.path.join(REPO, "bench", "results", "machine_fingerprint.json")
    with open(stored_path, "r", encoding="utf-8") as f:
        stored = json.load(f)
    return ms.verify_fingerprint(ms.capture_fingerprint(), stored)


def _default_verify_lock(mhz: int):
    import machine_state as ms
    return ms.verify_clock_lock(mhz)


def preflight(expected_mhz: int = EXPECTED_CLOCK_MHZ,
              verify_fingerprint=None, verify_lock=None,
              expected_compared_fields: int = EXPECTED_COMPARED_FIELDS) -> dict:
    """The two conditions that must hold before the first timed bracket.

    The harness REFUSES TO RUN unless both pass, and the refusal names WHICH of
    the two failed. Proceeding past either would produce numbers that cannot be
    compared with any other stage's, which is worse than producing none.
    """
    verify_fingerprint = verify_fingerprint or _default_verify_fingerprint
    verify_lock = verify_lock or _default_verify_lock

    fp = verify_fingerprint()
    fp_ok = bool(fp.get("match"))
    n_compared = fp.get("fields_compared")
    fp_reasons = []
    if not fp_ok:
        for d in fp.get("differences", []):
            fp_reasons.append(f"{d['field']}: stored {d['stored']!r}, current {d['current']!r}")
    if n_compared != expected_compared_fields:
        fp_ok = False
        fp_reasons.append(
            f"the comparison covered {n_compared} fields, not the expected "
            f"{expected_compared_fields} (W12: 26 fields minus the provenance-only build "
            "timestamp)")

    lk = verify_lock(expected_mhz)
    lk_ok = bool(lk.get("locked"))

    checks = {
        "environment_fingerprint": {
            "pass": fp_ok,
            "fields_compared": n_compared,
            "fields_expected": expected_compared_fields,
            "differences": fp.get("differences", []),
            "fields_excluded_from_comparison": fp.get("fields_excluded_from_comparison", []),
            "provenance_reported_not_compared": fp.get("provenance", []),
            "reasons": fp_reasons,
        },
        "graphics_clock_lock": {
            "pass": lk_ok,
            "expected_mhz": expected_mhz,
            "detail": lk,
        },
    }
    failed = [name for name, c in checks.items() if not c["pass"]]
    return {
        "pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "rule": ("BENCHMARK_PROTOCOL.md section 4: the fingerprint is re-verified before the "
                 "first timed run of every stage, and the graphics clock lock is not optional "
                 "on this machine (section 4.2)"),
    }


# ======================================================== the prompt set ======

def resolve_prompt_set(fixture: str | None, allow_other: bool) -> dict:
    """The harness reads the D3 fixed prompt set and REFUSES any other prompt
    source unless the caller names it explicitly. The identity of the set that
    actually ran is recorded either way: a timing whose prompt set is unrecorded
    cannot be compared with another timing, which is the whole content of
    BENCHMARK_PROTOCOL.md section 4."""
    path = os.path.abspath(fixture or D3_FIXTURE)
    identity = C.prompt_set_identity(path)
    is_d3 = os.path.abspath(path) == os.path.abspath(D3_FIXTURE)
    if not is_d3 and not allow_other:
        raise SystemExit(
            f"REFUSED: {path} is not the D3 fixed prompt set "
            f"({D3_FIXTURE}). Pass --allow-other-prompt-set to run against it deliberately; "
            "the set that ran is recorded in the results file either way.")
    identity["is_d3_fixed_set"] = is_d3
    identity["explicitly_overridden"] = bool(not is_d3)
    return identity


# =================================================== driving the C timing binary

_PROBE_RE = re.compile(
    r"^probe (?P<workload>prefill|decode) (?:L|c)=(?P<tokens>\d+) row=(?P<row>\S+) "
    r"seconds=(?P<seconds>[0-9.eE+-]+)(?: path=(?P<path>cache|nocache))?")


def parse_probe_output(text: str) -> list:
    """Reads the timing driver's `--probe` lines. A probe is a SCHEDULING INPUT,
    not a measurement, and is never recorded as a latency anywhere.

    Stage 4: the driver appends `path=cache|nocache`. The group is OPTIONAL so
    this parser still reads a Stage 3 driver's output, where there was one path
    and it was not named; a line without it is treated as the no-cache path,
    which is what that driver ran."""
    out = []
    for line in text.splitlines():
        m = _PROBE_RE.match(line.strip())
        if m:
            out.append({
                "workload": m.group("workload"),
                "tokens": int(m.group("tokens")),
                "prompt_row_id": m.group("row"),
                "probe_ms": float(m.group("seconds")) * 1000.0,
                "path": m.group("path") or "nocache",
                "tag": "probe, NOT A MEASUREMENT",
            })
    return out


def make_configuration_key(workload: str, tokens: int, path: str | None) -> str:
    """The comparison key. A no-cache configuration keeps the Stage 3 key
    exactly, so it pairs with the Stage 3 baseline without a special case; a
    cache configuration gets a distinct key, so the two paths can never be
    compared with each other as a regression."""
    base = f"prefill|L={tokens}" if workload == "prefill" else f"decode|c={tokens}"
    return base + ("|cache" if path == "cache" else "")


def parse_configs_arg(text: str) -> list:
    """`prefill:16:cache,decode:32:nocache,gemm:768` -> the ordered tuple list."""
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 2 and parts[0] == "gemm":
            out.append(("gemm", int(parts[1]), None))
        elif len(parts) == 3 and parts[0] in ("prefill", "decode") \
                and parts[2] in ("cache", "nocache"):
            out.append((parts[0], int(parts[1]), parts[2]))
        else:
            raise ValueError(f"unparseable configuration item {item!r}; expected "
                             "prefill:L:cache|nocache, decode:c:cache|nocache or gemm:N")
    if not out:
        raise ValueError("the configuration list is empty")
    return out


def run_probe(command: list, fixture: str, lengths, contexts, env=None,
              timeout: int = 7200, configs=None) -> list:
    cmd = list(command) + ["--probe", "--fixture", fixture,
                           "--lengths", ",".join(str(x) for x in lengths),
                           "--contexts", ",".join(str(x) for x in contexts),
                           "--warmup", str(PROTOCOL_WARMUP),
                           "--samples", str(PROTOCOL_SAMPLES)]
    if configs:
        cmd += ["--configs", format_configs(configs)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       timeout=timeout, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        raise RuntimeError(f"the probe pass exited {r.returncode}")
    probes = parse_probe_output(r.stdout)
    if not probes:
        # The driver probes prefill and decode only: the isolated GEMM cases are
        # fixed shapes with no fixture row and no engine path, so a chunk made up
        # only of GEMM configurations legitimately produces no probe lines. That
        # is distinct from a driver that failed to emit them, which is still an
        # error -- so the absence is accepted ONLY when nothing probeable was
        # requested.
        probeable = [c for c in (configs or []) if c[0] in ("prefill", "decode")]
        if configs is not None and not probeable:
            return []
        raise RuntimeError("the probe pass produced no parseable probe lines")
    return probes


def run_timed(command: list, fixture: str, lengths, contexts, out_stem: str,
              results_dir: str, constructions: dict, prompt_set: dict,
              env=None, timeout: int = 36000, configs=None,
              stage: str = STAGE_ID) -> dict:
    """Drives the timing binary once over every configuration and returns the JSON
    it wrote. The harness does not recompute a single statistic from the raw
    samples; it carries them through.

    `--repeat R` is emitted only where the W3 selector chose a batched
    construction. Stage 3 selects R = 1 everywhere, so the flag is not emitted
    in this stage. See the header for why the C side of the batched bracket is
    Stage 4's to add.
    """
    repeats = {k: v["repeat_factor_R"] for k, v in constructions.items()}
    distinct = sorted(set(repeats.values()))
    cmd = list(command) + [
        "--fixture", fixture,
        "--lengths", ",".join(str(x) for x in lengths),
        "--contexts", ",".join(str(x) for x in contexts),
        "--warmup", str(PROTOCOL_WARMUP),
        "--samples", str(PROTOCOL_SAMPLES),
        "--out", out_stem,
        "--prompt-set-status", prompt_set["status"],
        "--prompt-set-decision", prompt_set["decision"],
        "--prompt-set-work-item", "W11",
    ]
    # Stage 4: the ORDER is the measurement design, so it is handed to the driver
    # explicitly. One invocation runs every configuration, so the order the
    # driver is given is the order the machine sees.
    if configs:
        cmd += ["--configs", format_configs(configs)]
    # An EMPTY set of constructions means nothing probeable was requested -- a
    # chunk of isolated GEMM configurations -- so there is no R to carry and no
    # mixed invocation to refuse. An R of 1 is likewise not emitted.
    if distinct and distinct != [1]:
        if len(distinct) != 1:
            raise RuntimeError(
                "a batched construction was selected for some configurations and not others: "
                f"R values {distinct}. One invocation carries one R, so the caller must split "
                "the configurations. No figure is reported from a mixed invocation.")
        cmd += ["--repeat", str(distinct[0])]

    run_env = dict(os.environ if env is None else env)
    run_env["BENCH_RESULTS_DIR"] = results_dir
    run_env.setdefault("BENCH_STAGE_ID", stage)
    os.makedirs(results_dir, exist_ok=True)

    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       timeout=timeout, env=run_env)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr + "\n")
        raise RuntimeError(f"the timing pass exited {r.returncode}")
    path = os.path.join(results_dir, out_stem + ".json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ========================================================= records and validity

def configuration_key(record: dict) -> str:
    """A comparison key that is stable across stages. Built from the workload and
    the shape, never from the free-text configuration string, which later stages
    will reword."""
    ann = record.get("annotations", {})
    cnt = record.get("counters", {})
    workload = ann.get("workload", "unknown")
    if workload == "prefill":
        return make_configuration_key("prefill", int(cnt.get("tokens", -1)),
                                      ann.get("prefill_path"))
    if workload == "decode":
        return make_configuration_key("decode", int(cnt.get("context_tokens", -1)),
                                      ann.get("decode_path"))
    if workload == "isolated_gemm":
        return (f"isolated_gemm|M={int(cnt.get('M', -1))},"
                f"N={int(cnt.get('N', -1))},K={int(cnt.get('K', -1))}")
    return f"{workload}|{record.get('configuration', '')}"


def _diagnostic_robust(samples: list) -> dict:
    """DIAGNOSTIC ONLY. A trimmed view of the distribution, for reading a
    dispersion failure. It never replaces a reported value and never converts an
    INVALID run into a valid one."""
    if len(samples) < 5:
        return {"diagnostic_note": "too few samples for a trimmed view"}
    s = sorted(samples)
    k = max(1, len(s) // 10)
    core = s[k:len(s) - k]
    med = statistics.median(core)
    sd = statistics.stdev(core) if len(core) > 1 else 0.0
    return {
        "diagnostic_trimmed_fraction_each_tail": k / len(s),
        "diagnostic_trimmed_median_ms": med,
        "diagnostic_trimmed_stddev_ms": sd,
        "diagnostic_trimmed_stddev_pct_of_median": (sd / med * 100.0) if med > 0 else None,
        "diagnostic_note": ("DIAGNOSTIC ONLY. Labelled, never reported as a result, never "
                            "used to revise a VALID/INVALID verdict. The reported median and "
                            "standard deviation are the C layer's, over all samples"),
    }


def build_configuration_records(c_doc: dict, constructions: dict) -> list:
    """Turns the C layer's records into the harness's per-configuration records:
    raw samples carried through, the C layer's statistics carried through, the W3
    construction record attached, and the validity verdict restated and
    cross-checked rather than recomputed."""
    out = []
    for rec in c_doc.get("records", []):
        key = configuration_key(rec)
        st = rec["statistics"]
        samples = rec.get("raw_samples_ms", [])
        ann = rec.get("annotations", {})

        # The C layer owns the rule. The harness asserts its verdict rather than
        # trusting it, and a disagreement is reported rather than resolved.
        implied = st["stddev_pct_of_median"] <= PROTOCOL_MAX_STDDEV_PCT
        agreement = (bool(st["valid"]) == implied)

        out.append({
            "configuration_key": key,
            "workload": ann.get("workload"),
            "configuration": rec.get("configuration"),
            "prompt_row_id": ann.get("prompt_row_id"),
            "units": rec.get("units"),
            "value": rec.get("value"),
            "warmup_iterations": rec.get("warmup_iterations"),
            "samples_requested": rec.get("samples_requested"),
            "raw_samples_ms": samples,
            "statistics": st,
            "valid": bool(st["valid"]),
            "verdict": "VALID" if st["valid"] else "INVALID",
            "invalid_reason": st.get("invalid_reason") or None,
            "validity_rule": (f"std dev > {PROTOCOL_MAX_STDDEV_PCT}% of median => INVALID. "
                              "Applied by the C layer; never loosened, never retried, never "
                              "averaged away. This harness has no retry path"),
            "validity_crosscheck_agrees_with_c_layer": agreement,
            "construction": constructions.get(key),
            "counters": rec.get("counters", {}),
            "annotations": ann,
            "diagnostics": _diagnostic_robust(samples),
        })
    return out


# ======================================= W3: the repeat-confirmation refusal ==

class RepeatNotConfirmed(RuntimeError):
    """Raised when the driver did not confirm the repeat factor the harness
    selected. It is a hard failure for the WHOLE run, not a per-configuration
    warning: a results file containing one fabricated per-iteration figure is
    not partially trustworthy."""


def verify_repeat_confirmation(records: list, constructions: dict) -> dict:
    """THE REFUSAL THIS HARNESS WILL NOT DIVIDE PAST (W3).

    Before any bracket is divided by R, the driver's own record for that
    configuration must carry `repeat_applied == R`. The harness does not trust
    the flag it emitted to have been honoured -- a driver that ignored
    `--repeat R` and ran one iteration, while this code divided the bracket by
    R, would report a per-iteration figure that no iteration ever produced.

    Checked for EVERY configuration, not only the batched ones: an R = 1
    configuration must report `repeat_applied == 1`. A confirmation that is
    only checked where it is convenient is not a check.

    Returns a structured record on success. Raises RepeatNotConfirmed, naming
    the configuration, R, what the driver reported and W3, on failure.
    """
    failures, checked = [], []
    for r in records:
        key = r["configuration_key"]
        con = constructions.get(key)
        # A configuration with no construction record (the isolated GEMM cases,
        # which are not probed) is still required to report repeat_applied == 1.
        R = int(con["repeat_factor_R"]) if con else 1
        counters = r.get("counters", {}) or {}
        raw = counters.get("repeat_applied", None)
        applied = None
        if raw is not None:
            try:
                applied = int(raw)
            except (TypeError, ValueError):
                applied = None
        requested = counters.get("repeat_requested", None)

        if applied != R:
            failures.append({
                "configuration_key": key,
                "repeat_factor_R_selected": R,
                "repeat_applied_reported_by_driver": (
                    "none -- the driver's record carries no repeat_applied field"
                    if raw is None else raw),
                "repeat_requested_reported_by_driver": requested,
                "reason": (f"the timing driver did not confirm R={R} for {key}. "
                           "PERSISTENT.md section 8, W3: this harness never divides an "
                           "unconfirmed bracket by R, and never reports a per-iteration "
                           "figure that no iteration produced"),
            })
        else:
            checked.append({"configuration_key": key, "repeat_factor_R": R,
                            "repeat_applied": applied})

    if failures:
        lines = ["W3 REFUSAL: the timing driver did not confirm the selected repeat factor."]
        for f in failures:
            lines.append(f"  {f['configuration_key']}: R={f['repeat_factor_R_selected']} "
                         f"selected, driver reported repeat_applied="
                         f"{f['repeat_applied_reported_by_driver']}")
        lines.append("No bracket was divided by R and no per-iteration figure was written. "
                     "The C-side batched bracket is NOT implemented; it is owned by the "
                     "first stage that selects R > 1 (PERSISTENT.md section 8, W3).")
        raise RepeatNotConfirmed("\n".join(lines))

    return {
        "checked": checked,
        "all_confirmed": True,
        "rule": ("every configuration's repeat_applied, as reported by the timing driver, "
                 "must equal the R this harness selected -- including R = 1. An "
                 "unconfirmed bracket is never divided"),
        "work_item": "PERSISTENT.md section 8, W3",
    }


# ================================================== the regression comparison ==

def compare_regression(current_records: list, current_doc: dict, prior_path: str,
                       noise_floor_pct: float = NOISE_FLOOR_PCT) -> dict:
    """Compares medians against a NAMED prior stage's results file.

    A REGRESSION is a slowdown of the median on a configuration that is VALID in
    BOTH files, at the SAME prompt set and the SAME configuration, exceeding the
    noise floor.

    The comparison REFUSES, and says why, when the prompt sets differ, when
    either side is INVALID, or when a substantive fingerprint field differs
    between the two files. A silent comparison across a changed condition is the
    failure this check exists to prevent -- and it is the reason this function
    reports refusals as prominently as it reports regressions.

    WHAT IT STRUCTURALLY CANNOT CATCH: any change smaller than the noise floor.
    A real 3% regression on a machine with a 4.4% floor is indistinguishable from
    noise here, and the honest report of it is "no measurable change".
    """
    if not prior_path:
        return {"performed": False,
                "reason": "no prior results file was named, so no comparison was attempted"}
    if not os.path.exists(prior_path):
        return {"performed": False,
                "reason": f"the named prior results file does not exist: {prior_path}"}
    with open(prior_path, "r", encoding="utf-8") as f:
        prior = json.load(f)

    rel = os.path.relpath(prior_path, REPO).replace("\\", "/")

    # Condition: the substantive fingerprint fields must match between the files.
    fp_diffs = []
    for field in SUBSTANTIVE_FINGERPRINT_FIELDS:
        a, b = prior.get(field, "<absent>"), current_doc.get(field, "<absent>")
        if a != b:
            fp_diffs.append({"field": field, "prior": a, "current": b})

    prior_by_key = {}
    for rec in prior.get("records", []):
        prior_by_key[configuration_key(rec)] = rec

    comparisons, regressions, refusals = [], [], []
    for cur in current_records:
        key = cur["configuration_key"]
        p = prior_by_key.get(key)
        if p is None:
            refusals.append({"configuration_key": key, "comparable": False,
                             "reason": "the configuration does not appear in the prior file",
                             "process_set_gated": False,
                             "process_set_note": PROCESS_SET_NOTE})
            continue

        cur_ps = cur["annotations"].get("prompt_set_source")
        pri_ps = p.get("annotations", {}).get("prompt_set_source")
        cur_st = cur["annotations"].get("prompt_set_status")
        pri_st = p.get("annotations", {}).get("prompt_set_status")
        cur_row = cur["annotations"].get("prompt_row_id")
        pri_row = p.get("annotations", {}).get("prompt_row_id")

        reasons = []
        if (cur_ps, cur_st) != (pri_ps, pri_st) or cur_row != pri_row:
            reasons.append(
                f"the prompt sets differ: prior {pri_ps!r} ({pri_st}, row {pri_row!r}), "
                f"current {cur_ps!r} ({cur_st}, row {cur_row!r}). "
                "BENCHMARK_PROTOCOL.md section 4 holds the prompt set constant across stages "
                "or the comparison is void")
        if not cur["valid"]:
            reasons.append("the current configuration is INVALID")
        if not p["statistics"]["valid"]:
            reasons.append("the prior configuration is INVALID")
        if fp_diffs:
            reasons.append("substantive fingerprint fields differ between the two files: "
                           + "; ".join(f"{d['field']}" for d in fp_diffs))

        if reasons:
            refusals.append({"configuration_key": key, "comparable": False,
                             "reason": " | ".join(reasons),
                             "process_set_gated": False,
                             "process_set_note": PROCESS_SET_NOTE})
            continue

        a = p["statistics"]["median_ms"]
        b = cur["statistics"]["median_ms"]
        delta = (b - a) / a * 100.0
        if delta > noise_floor_pct:
            outcome = "REGRESSION"
            regressions.append(key)
        elif delta < -noise_floor_pct:
            outcome = "improvement"
        else:
            outcome = "no measurable change"
        comparisons.append({
            "configuration_key": key,
            "comparable": True,
            "prior_median_ms": a,
            "current_median_ms": b,
            "change_pct": delta,
            "noise_floor_pct": noise_floor_pct,
            "outcome": outcome,
            # W14. The process set is NOT part of the environment fingerprint, so
            # this comparison does not control it, and the same binary has moved
            # 9 to 12 percent between sessions on unchanged code.
            "process_set_gated": False,
            "process_set_note": PROCESS_SET_NOTE,
        })

    return {
        "performed": True,
        "prior_results_file": rel,
        "prior_stage": prior.get("stage"),
        "noise_floor_pct": noise_floor_pct,
        "noise_floor_source": ("BENCHMARK_PROTOCOL.md section 4.1, measured by Stage 0 on "
                              "2026-09-17. USED here, not re-derived (W5)"),
        "substantive_fingerprint_differences": fp_diffs,
        "process_set_gated": False,
        "process_set_note": PROCESS_SET_NOTE,
        "process_set_limitation": (
            "W14: the environment fingerprint does not include the process set, so this "
            "comparison does not control it. The same binary measured in the Stage 2 and "
            "Stage 3 sessions differed by roughly 9 to 12 percent on unchanged code, about "
            "2.3 times the 4.4 percent noise floor. Any cross-session comparison below "
            "carries that shift; a comparison against an in-session control does not"),
        "regression_definition": ("a slowdown of the median exceeding the noise floor, on a "
                                 "configuration VALID in BOTH files, at the SAME prompt set "
                                 "and the SAME configuration"),
        "cannot_catch": ("any change smaller than the noise floor. A real regression inside "
                         f"{noise_floor_pct}% is indistinguishable from noise here and is "
                         "reported as no measurable change"),
        "comparisons": comparisons,
        "refusals": refusals,
        "regressions": regressions,
        "any_regression": bool(regressions),
    }


# =============================================================== W9: cuBLAS ====

# The four weight GEMM shapes, as [input, output] -- K then N. Verified against
# src/gpt2_tensor_inventory.json rather than taken from a prompt or a memory:
#   h.*.attn.c_attn.weight  [768, 2304]   K=768  N=2304
#   h.*.attn.c_proj.weight  [768,  768]   K=768  N=768
#   h.*.mlp.c_fc.weight     [768, 3072]   K=768  N=3072
#   h.*.mlp.c_proj.weight   [3072, 768]   K=3072 N=768
WEIGHT_GEMM_SHAPES = {
    "qkv_projection":         {"K": 768,  "N": 2304, "tensor": "h.*.attn.c_attn.weight"},
    "attn_output_projection": {"K": 768,  "N": 768,  "tensor": "h.*.attn.c_proj.weight"},
    "ffn_up":                 {"K": 768,  "N": 3072, "tensor": "h.*.mlp.c_fc.weight"},
    "ffn_down":               {"K": 3072, "N": 768,  "tensor": "h.*.mlp.c_proj.weight"},
}
D3_M_VALUES = (16, 32, 64, 128)
DECODE_M = 1


def verify_shapes_against_inventory(inventory_path: str) -> dict:
    """Checks every K and N against the committed tensor inventory. A shape taken
    from a prompt and not from the artifact is an assumption, not a fact."""
    with open(inventory_path, "r", encoding="utf-8") as f:
        inv = json.load(f)
    by_name = {t["name"]: t for t in inv["tensors"]}
    out = {}
    for shape, spec in WEIGHT_GEMM_SHAPES.items():
        name = spec["tensor"].replace("h.*", "h.0")
        t = by_name.get(name)
        if t is None:
            raise AssertionError(f"{name} is not in {inventory_path}")
        K, N = int(t["shape"][0]), int(t["shape"][1])
        if (K, N) != (spec["K"], spec["N"]):
            raise AssertionError(
                f"{name}: the inventory says [{K}, {N}], this harness expected "
                f"[{spec['K']}, {spec['N']}]")
        out[shape] = {"tensor": name, "inventory_shape": [K, N], "K": K, "N": N}
    return out


_CUBLAS_CFG_RE = re.compile(
    r"^(?P<kind>decode|prefill) (?P<shape>\S+) M=(?P<M>\d+) N=(?P<N>\d+) K=(?P<K>\d+)$")


def extract_cublas_d3(c_doc: dict, shapes: dict) -> dict:
    """Pulls the D3 M values, and the M = 1 decode-shaped rows, out of a full
    cublas_sgemm_ref run. The binary's M sweep is compiled in and already
    contains every M value D3 fixes, so the binary is re-run UNMODIFIED and the
    rows that matter are selected here."""
    prefill, decode, other = [], [], []
    for rec in c_doc.get("records", []):
        m = _CUBLAS_CFG_RE.match(rec.get("configuration", "").strip())
        if not m:
            continue
        shape, M = m.group("shape"), int(m.group("M"))
        if shape not in shapes:
            continue
        st = rec["statistics"]
        row = {
            "shape": shape,
            "tensor": shapes[shape]["tensor"],
            "M": M, "N": int(m.group("N")), "K": int(m.group("K")),
            "gflops": rec.get("value"),
            "units": rec.get("units"),
            "median_ms": st["median_ms"],
            "min_ms": st["min_ms"],
            "max_ms": st["max_ms"],
            "stddev_ms": st["stddev_ms"],
            "stddev_pct_of_median": st["stddev_pct_of_median"],
            "n_samples": st["n"],
            "warmup_iterations": rec.get("warmup_iterations"),
            "valid": bool(st["valid"]),
            "verdict": "VALID" if st["valid"] else "INVALID",
            "invalid_reason": st.get("invalid_reason") or None,
            "raw_samples_ms": rec.get("raw_samples_ms", []),
            "counters": rec.get("counters", {}),
        }
        if M == DECODE_M:
            row["role"] = ("DECODE-SHAPED. Not part of the prefill denominator and used for "
                           "nothing. Fresh evidence on W3's original failure mode at the "
                           "sample count the protocol uses")
            decode.append(row)
        elif M in D3_M_VALUES:
            row["role"] = "prefill denominator at a D3 token count"
            prefill.append(row)
        else:
            row["role"] = ("outside the D3 set; retained because the binary's sweep is "
                           "compiled in and was not modified, and because it is what shows "
                           "the non-monotonicity")
            other.append(row)

    # The non-monotonicity, reported as an observation. The cause is NOT
    # established here and no mechanism is proposed: W9 records it as
    # unestablished and Stage 10 takes the explanation further.
    non_monotonic = []
    for shape in shapes:
        rows = sorted([r for r in prefill + other if r["shape"] == shape],
                      key=lambda r: r["M"])
        for a, b in zip(rows, rows[1:]):
            if b["gflops"] < a["gflops"]:
                non_monotonic.append({
                    "shape": shape,
                    "from_M": a["M"], "to_M": b["M"],
                    "from_gflops": a["gflops"], "to_gflops": b["gflops"],
                    "drop_pct": (a["gflops"] - b["gflops"]) / a["gflops"] * 100.0,
                    "both_valid": a["valid"] and b["valid"],
                })
    return {
        "prefill_denominators_at_d3_m_values": prefill,
        "decode_shaped_rows_M1": decode,
        "rows_outside_the_d3_set": other,
        "non_monotonic_observations": non_monotonic,
        "non_monotonicity_note": ("OBSERVATION ONLY. The cause is NOT established by this "
                                  "stage: no per-kernel profiling was done and no mechanism "
                                  "is proposed. W9 carries the cause as unestablished"),
    }


# ======================================================== results file writer ==

def _counters_block() -> dict:
    return {
        "collected": False,
        "reason": ("BENCHMARK_PROTOCOL.md section 6 requires Nsight Compute counters for GPU "
                   "stages from Stage 7 onward. This stage writes no kernel; the engine it "
                   "times runs on the CPU, and the cuBLAS binary it re-runs is Stage 0's, "
                   "already profiled in its own stage. No profiler run was introduced"),
        "nsight_counters": {
            "collected": False,
            "reason": ("a CPU stage. BENCHMARK_PROTOCOL.md section 6 requires Nsight counters "
                       "from Stage 7 onward, and this stage writes no kernel"),
        },
        "cpu_counters": {
            "collected": False,
            "tool_available": "xperf (Windows Performance Toolkit)",
            "reason": ("PERSISTENT.md section 8, W4: xperf is AVAILABLE on this machine and is "
                       "DELIBERATELY UNUSED, following the Stage 2 and Stage 3 precedent. The "
                       "consequence is stated rather than hidden: without CPU counters, the "
                       "cause of the measured gap is not established at the counter level, and "
                       "an issue-rate limit cannot be distinguished from a cache-miss limit"),
        },
    }


def _headline_block() -> dict:
    return {
        "quoted": False,
        "reason": ("BENCHMARK_PROTOCOL.md section 7 defines the decode headline against "
                   "measured GPU bandwidth and the prefill headline against cuBLAS at the "
                   "engine's shapes. This stage's engine runs on the CPU, so the cuBLAS "
                   "figures recorded here are DENOMINATORS FOR LATER STAGES and not a ratio "
                   "for this one"),
    }


def write_results(path: str, doc: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    return path


def assemble_results(c_doc: dict, records: list, prompt_set: dict, pre: dict,
                     probes: list, constructions: dict, correctness_result,
                     regression: dict, stage: str) -> dict:
    """Builds the results document. Every BENCHMARK_PROTOCOL.md section 9 field is
    present by name. Prefill and decode are separated at EVERY level: the records
    carry a workload field, and the summary indexes them under separate keys.
    Nothing anywhere combines them into one figure."""
    prefill = [r for r in records if r["workload"] == "prefill"]
    decode = [r for r in records if r["workload"] == "decode"]
    other = [r for r in records if r["workload"] not in ("prefill", "decode")]

    def summarise(rs):
        return [{"configuration_key": r["configuration_key"],
                 "prompt_row_id": r["prompt_row_id"],
                 "median_ms": r["statistics"]["median_ms"],
                 "min_ms": r["statistics"]["min_ms"],
                 "max_ms": r["statistics"]["max_ms"],
                 "stddev_pct_of_median": r["statistics"]["stddev_pct_of_median"],
                 "n_samples": r["statistics"]["n"],
                 "construction": (r["construction"] or {}).get("construction"),
                 "repeat_factor_R": (r["construction"] or {}).get("repeat_factor_R"),
                 "verdict": r["verdict"]} for r in rs]

    return {
        # ---- section 9: stage identifier, commit, timestamp, device, flags ----
        "stage": stage,
        "kind": "harness",
        "generated_by": "bench/harness.py",
        "git_commit": c_doc.get("git_commit"),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "c_layer_run_timestamp_utc": c_doc.get("run_timestamp_utc"),
        "build_timestamp": c_doc.get("build_timestamp"),
        "device": c_doc.get("device"),
        "cpu_timer": c_doc.get("cpu_timer"),
        "gpu_timer": c_doc.get("gpu_timer"),
        "cxx_compiler": c_doc.get("cxx_compiler"),
        "cxx_flags": c_doc.get("cxx_flags"),
        "cuda_compiler": c_doc.get("cuda_compiler"),
        "cuda_flags": c_doc.get("cuda_flags"),
        "provenance_source": ("carried through from the C timing layer's own results file, "
                             "which compiles build/generated/build_info.h and git_info.h in. "
                             "The Python layer does not re-derive these by a different route"),
        "protocol": c_doc.get("protocol"),
        "protocol_conditions": {
            "warmup_iterations": PROTOCOL_WARMUP,
            "samples": PROTOCOL_SAMPLES,
            "max_stddev_pct_of_median": PROTOCOL_MAX_STDDEV_PCT,
            "limit_is_a_parameter": False,
            "retry_path_exists": False,
            "note": ("the 5%-of-median limit is applied by the C layer and is not adjustable "
                     "from this harness. There is no retry path, so no configuration can be "
                     "re-run into validity by this code"),
        },
        # ---- section 9: prompt set ----
        "prompt_set": prompt_set,
        "preflight": pre,
        # ---- W3 ----
        "w3_construction": {
            "per_sample_floor_ms": W3_PER_SAMPLE_FLOOR_MS,
            "probes": probes,
            "selected": constructions,
            "arithmetic": {
                "sd_pct_formula": "50 * sqrt(k*(n-k)/(n*(n-1)))",
                "sd_pct_at_n30_k1": predicted_dispersion_pct(30, 1),
                "w3_original_form": "0.5/sqrt(29) = 9.29%",
                "rate_model": ("relative sd = delta*sqrt(lambda/T): depends on the PER-SAMPLE "
                               "DURATION T and NOT on the sample count n. More samples do not "
                               "reduce it; batching divides it by sqrt(R)"),
            },
            "batched_path_exercised_by_a_timed_run": any(
                v["repeat_factor_R"] > 1 for v in constructions.values()),
            "note": ("every Stage 3 configuration is seconds-scale and took the "
                     "single-iteration path. The batched path is built and unit-tested and is "
                     "NOT exercised by a timed run in this stage"),
        },
        # ---- section 9: raw timings, statistics; prefill and decode SEPARATE ----
        "records": records,
        "summary": {
            "prefill": summarise(prefill),
            "decode": summarise(decode),
            "other_workloads": summarise(other),
            "separation_note": ("prefill and decode are separate workloads and are never "
                               "combined into one figure, at any level of this file. Prefill "
                               "is compute-bound; decode is memory-bandwidth-bound"),
        },
        "invalid_configurations": [r["configuration_key"] for r in records if not r["valid"]],
        # ---- section 9: correctness result ----
        "correctness": correctness_result,
        # ---- regression ----
        "regression": regression,
        # ---- section 9: counters, headline ratios ----
        "counters": _counters_block(),
        "headline_ratios": _headline_block(),
    }


# ======================================================================= CLI ===

def _load_correctness_result(path):
    if not path:
        return {"available": False,
                "reason": "no correctness results file was named for this run"}
    if not os.path.exists(path):
        return {"available": False,
                "reason": f"the named correctness results file does not exist: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    gate = doc.get("gate", {})
    return {
        "available": True,
        "source": os.path.relpath(path, REPO).replace("\\", "/"),
        "verdict": gate.get("verdict"),
        "gate_applied": gate.get("applied", True) is not False,
        "tolerance": gate.get("tolerance"),
        "conditions": gate.get("conditions"),
        "prompt_set": doc.get("prompt_set"),
        "margin_table": doc.get("margin_table"),
        "note": ("a FAIL is recorded as a FAIL and does not stop timings already taken from "
                 "being recorded as taken"),
    }


def cmd_run(a) -> int:
    prompt_set = resolve_prompt_set(a.fixture, a.allow_other_prompt_set)
    fixture = os.path.abspath(a.fixture or D3_FIXTURE)
    lengths = [int(x) for x in a.lengths.split(",")] if a.lengths else list(D3_PREFILL_LENGTHS)
    contexts = ([int(x) for x in a.contexts.split(",")] if a.contexts
                else list(D3_DECODE_CONTEXTS))

    pre = preflight(expected_mhz=a.expect_mhz)
    print(json.dumps({"preflight_pass": pre["pass"],
                      "failed_checks": pre["failed_checks"]}, indent=1))
    if not pre["pass"]:
        for name in pre["failed_checks"]:
            print(f"REFUSED: the {name} check did not pass.", file=sys.stderr)
            for r in pre["checks"][name].get("reasons", []):
                print(f"  {r}", file=sys.stderr)
            if name == "graphics_clock_lock":
                print(f"  {json.dumps(pre['checks'][name]['detail'])}", file=sys.stderr)
        print("No timed run was started. BENCHMARK_PROTOCOL.md section 4.", file=sys.stderr)
        return 3
    if a.preflight_only:
        print("preflight only: both checks pass; no timed run started")
        return 0

    command = a.command.split() if isinstance(a.command, str) else list(a.command)

    configs = list(STAGE4_CONFIGS) if a.configs is None else parse_configs_arg(a.configs)
    print("\n-- the configuration list, in the order the machine will see it --")
    for kind, tokens, path in configs:
        print(f"  {kind:8s} {tokens:4d} {path or '-'}")

    print("\n-- W3 probe pass: one untimed iteration per configuration --")
    probes = run_probe(command, fixture, lengths, contexts, configs=configs)
    constructions = {}
    for p in probes:
        key = make_configuration_key(p["workload"], p["tokens"], p.get("path"))
        constructions[key] = select_construction(p["probe_ms"])
        c = constructions[key]
        print(f"  {key:22s} probe {p['probe_ms'] / 1000.0:9.3f} s -> "
              f"{c['construction']}, R={c['repeat_factor_R']}, bracket "
              f"{c['effective_bracket_ms'] / 1000.0:9.3f} s")
    # W3, DECISION C: a selected R > 1 is a STOP, not something to build around.
    batched = {k: v["repeat_factor_R"] for k, v in constructions.items()
               if v["repeat_factor_R"] > 1}
    if batched:
        print("\nW3 STOP: the probe pass selected a BATCHED construction for "
              f"{sorted(batched)}.", file=sys.stderr)
        for k, R in sorted(batched.items()):
            print(f"  {k}: probe {constructions[k]['probe_ms']:.3f} ms, R={R}", file=sys.stderr)
        print("The C-side batched bracket is NOT implemented and is not built "
              "speculatively. No timed run was started and no figure was written.",
              file=sys.stderr)
        return 5
    budget = sum(p["probe_ms"] for p in probes) * (PROTOCOL_WARMUP + PROTOCOL_SAMPLES) / 1000.0
    print(f"  projected budget {budget:.0f} s = {budget / 60.0:.1f} min "
          f"(a PROJECTION from a measured probe, NOT a measurement, and it appears nowhere "
          f"as a latency)")
    if a.probe_only:
        return 0

    print("\n-- timed pass --")
    results_dir = a.c_results_dir or os.path.join(REPO, "build", "harness_c_results")
    c_doc = run_timed(command, fixture, lengths, contexts, a.c_out_stem, results_dir,
                      constructions, prompt_set, configs=configs, stage=a.stage)
    records = build_configuration_records(c_doc, constructions)

    # W3. BEFORE anything reads a median: the driver must have confirmed the
    # repeat factor. A failure here is a hard failure for the whole run and
    # nothing is written.
    try:
        repeat_confirmation = verify_repeat_confirmation(records, constructions)
    except RepeatNotConfirmed as exc:
        print(str(exc), file=sys.stderr)
        return 4
    print(f"W3: repeat confirmed by the driver for all "
          f"{len(repeat_confirmation['checked'])} configurations")

    correctness_result = _load_correctness_result(a.correctness)
    regression = compare_regression(records, c_doc, a.compare)

    doc = assemble_results(c_doc, records, prompt_set, pre, probes, constructions,
                           correctness_result, regression, a.stage)
    doc["w3_construction"]["repeat_confirmation"] = repeat_confirmation
    doc["configuration_order"] = {
        "order": [f"{k}:{t}" + (f":{p}" if p else "") for k, t, p in configs],
        "one_invocation_runs_every_configuration": True,
        "note": ("the driver runs every configuration in ONE invocation, so this list is the "
                 "order the machine saw. Each cache configuration is immediately followed by "
                 "its no-cache control, so any drift within the session falls on both "
                 "members of a pair (PERSISTENT.md section 8, W14)"),
    }
    out = write_results(a.out, doc)

    print(f"\nwritten: {out}")
    for group in ("prefill", "decode", "other_workloads"):
        for r in doc["summary"][group]:
            print(f"  {group:16s} {r['configuration_key']:34s} median "
                  f"{r['median_ms']:12.3f} ms  stddev {r['stddev_pct_of_median']:6.3f}%  "
                  f"{r['verdict']}")
    if doc["invalid_configurations"]:
        print(f"INVALID and reported as INVALID: {doc['invalid_configurations']}")
    print(f"correctness: {correctness_result.get('verdict')}")
    if regression.get("performed"):
        print(f"regression: {'YES ' + str(regression['regressions']) if regression['any_regression'] else 'none'}")
        for r in regression["refusals"]:
            print(f"  REFUSED {r['configuration_key']}: {r['reason']}")
    else:
        print(f"regression: not performed -- {regression['reason']}")
    return 0


def merge_parts(parts: list, compare: str | None, stage: str,
                correctness: str | None) -> dict:
    """Combines several `run` invocations into ONE results document.

    WHY THIS EXISTS, stated because it is a deviation from one-invocation-runs-
    everything. Stage 4's full set is about two hours on this machine, and the
    session's first attempt at it was killed partway through by the host's
    memory-pressure reaper with NOTHING written -- the timing driver emits its
    results only after the whole set finishes, so an interrupted invocation
    leaves no record of the configurations that did complete. Splitting the set
    into chunks bounds that loss to one chunk.

    WHAT THE SPLIT PRESERVES, which is the part that matters for the
    measurement: every cache configuration and its no-cache control sit in the
    SAME chunk, adjacent in time, which is what DECISION B's paired interleaved
    ordering is for -- session drift still falls on both members of a pair. The
    order within each workload class is unchanged. What moves is only the
    process boundary, and each chunk re-verifies the environment fingerprint and
    the clock lock before it starts, which is stricter than verifying once.

    WHAT IT DOES NOT DO: it recomputes no statistic. Every record, every raw
    sample array and every VALID/INVALID verdict is carried through from the
    chunk that measured it, byte for byte. The regression comparison and the W3
    repeat confirmation are re-run over the UNION of the records, because both
    are properties of the whole set rather than of a chunk.
    """
    if not parts:
        raise ValueError("no part files were given to merge")

    records, probes, constructions, order = [], [], {}, []
    seen, part_provenance = set(), []
    for path, doc in parts:
        rel = os.path.relpath(path, REPO).replace("\\", "/")
        for r in doc.get("records", []):
            key = r["configuration_key"]
            if key in seen:
                raise ValueError(
                    f"{key} appears in more than one part file; the chunks overlap and the "
                    f"merged set would double-count it. Second occurrence in {rel}")
            seen.add(key)
            records.append(r)
        w3 = doc.get("w3_construction", {})
        probes += w3.get("probes", [])
        constructions.update(w3.get("selected", {}))
        order += doc.get("configuration_order", {}).get("order", [])
        part_provenance.append({
            "part_file": rel,
            "configurations": [r["configuration_key"] for r in doc.get("records", [])],
            "c_layer_run_timestamp_utc": doc.get("c_layer_run_timestamp_utc"),
            "run_timestamp_utc": doc.get("run_timestamp_utc"),
            "git_commit": doc.get("git_commit"),
            "build_timestamp": doc.get("build_timestamp"),
            "preflight_pass": doc.get("preflight", {}).get("pass"),
        })

    # Every chunk must have been measured from the same build and the same
    # commit, or the merged set is not one measurement.
    for field in ("git_commit", "build_timestamp", "device", "cxx_compiler", "cxx_flags",
                  "cuda_compiler", "cuda_flags", "cpu_timer"):
        values = {doc.get(field) for _, doc in parts}
        if len(values) > 1:
            raise ValueError(
                f"the part files disagree on {field}: {sorted(str(v) for v in values)}. "
                "They are not one measurement and are not merged")

    base = parts[0][1]
    repeat_confirmation = verify_repeat_confirmation(records, constructions)
    regression = compare_regression(records, base, compare)
    correctness_result = _load_correctness_result(correctness)

    doc = assemble_results(base, records, base.get("prompt_set", {}),
                           base.get("preflight", {}), probes, constructions,
                           correctness_result, regression, stage)
    doc["w3_construction"]["repeat_confirmation"] = repeat_confirmation
    doc["configuration_order"] = {
        "order": order,
        "one_invocation_runs_every_configuration": False,
        "note": ("MEASURED IN CHUNKS, one harness invocation per chunk, in the order listed. "
                 "Each cache configuration and its no-cache control are in the SAME chunk and "
                 "adjacent in time, so DECISION B's pairing is intact; the order within each "
                 "workload class is unchanged. The split bounds the loss from an interrupted "
                 "run, after the host's memory-pressure reaper killed a single-invocation "
                 "attempt at the full set with nothing written. No statistic is recomputed by "
                 "the merge"),
    }
    doc["merged_from_parts"] = {
        "n_parts": len(parts),
        "parts": part_provenance,
        "merge_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "statistics_recomputed_by_the_merge": False,
        "raw_samples_carried_through": True,
        "regression_and_repeat_confirmation_rerun_over_the_union": True,
    }
    return doc


def cmd_merge(a) -> int:
    parts = []
    for path in a.parts:
        p = os.path.abspath(path)
        if not os.path.exists(p):
            print(f"part file does not exist: {p}", file=sys.stderr)
            return 2
        with open(p, "r", encoding="utf-8") as f:
            parts.append((p, json.load(f)))
    try:
        doc = merge_parts(parts, a.compare, a.stage, a.correctness)
    except (RepeatNotConfirmed, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 4
    out = write_results(a.out, doc)
    print(f"merged {len(parts)} part files -> {out}")
    for group in ("prefill", "decode", "other_workloads"):
        for r in doc["summary"][group]:
            print(f"  {group:16s} {r['configuration_key']:34s} median "
                  f"{r['median_ms']:12.3f} ms  stddev {r['stddev_pct_of_median']:6.3f}%  "
                  f"{r['verdict']}")
    if doc["invalid_configurations"]:
        print(f"INVALID and reported as INVALID: {doc['invalid_configurations']}")
    reg = doc["regression"]
    if reg.get("performed"):
        print(f"regression: {'YES ' + str(reg['regressions']) if reg['any_regression'] else 'none'}")
        for r in reg["refusals"]:
            print(f"  REFUSED {r['configuration_key']}: {r['reason']}")
    return 0


def cmd_cublas(a) -> int:
    shapes = verify_shapes_against_inventory(
        os.path.join(REPO, "src", "gpt2_tensor_inventory.json"))
    print("weight GEMM shapes verified against src/gpt2_tensor_inventory.json:")
    for name, s in shapes.items():
        print(f"  {name:24s} {s['tensor']:26s} inventory {s['inventory_shape']} "
              f"-> K={s['K']} N={s['N']}")

    results_dir = a.c_results_dir or os.path.join(REPO, "build", "harness_c_results")
    os.makedirs(results_dir, exist_ok=True)
    env = dict(os.environ)
    env["BENCH_RESULTS_DIR"] = results_dir
    env.setdefault("BENCH_STAGE_ID", "stage-3")

    print(f"\nre-running {a.binary} UNMODIFIED at warmup {PROTOCOL_WARMUP}, "
          f"samples {PROTOCOL_SAMPLES}")
    r = subprocess.run([a.binary, str(PROTOCOL_WARMUP), str(PROTOCOL_SAMPLES)],
                       cwd=REPO, capture_output=True, text=True, timeout=7200, env=env)
    sys.stdout.write(r.stdout)
    if r.returncode not in (0, 2):     # 2 = the binary's "some point is INVALID"
        sys.stderr.write(r.stderr + "\n")
        raise RuntimeError(f"cublas_sgemm_ref exited {r.returncode}")

    path = os.path.join(results_dir, "cublas_sgemm_ref.json")
    with open(path, "r", encoding="utf-8") as f:
        c_doc = json.load(f)

    extracted = extract_cublas_d3(c_doc, shapes)
    doc = {
        "stage": a.stage,
        "kind": "cublas_prefill_denominators",
        "generated_by": "bench/harness.py cublas",
        "git_commit": c_doc.get("git_commit"),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "c_layer_run_timestamp_utc": c_doc.get("run_timestamp_utc"),
        "build_timestamp": c_doc.get("build_timestamp"),
        "device": c_doc.get("device"),
        "cpu_timer": c_doc.get("cpu_timer"),
        "gpu_timer": c_doc.get("gpu_timer"),
        "cxx_compiler": c_doc.get("cxx_compiler"),
        "cxx_flags": c_doc.get("cxx_flags"),
        "cuda_compiler": c_doc.get("cuda_compiler"),
        "cuda_flags": c_doc.get("cuda_flags"),
        "protocol": c_doc.get("protocol"),
        "protocol_conditions": {"warmup_iterations": PROTOCOL_WARMUP,
                                "samples": PROTOCOL_SAMPLES,
                                "max_stddev_pct_of_median": PROTOCOL_MAX_STDDEV_PCT},
        "binary": {
            "path": a.binary,
            "source": "bench/microbench/cublas_sgemm_ref.cu",
            "modified_by_this_stage": False,
            "m_sweep": "compiled in: 1, 8, 16, 32, 64, 128, 256, 512, 1024",
            "note": ("the compiled sweep already contains every D3 M value and M=1, so the "
                     "binary was re-run unmodified and the rows that matter were selected "
                     "afterwards. Its command line is [warmup] [samples] only"),
        },
        "prompt_set": {
            "decision": "D3",
            "m_values_are_the_d3_prefill_token_counts": list(D3_M_VALUES),
            "source": "tests/fixtures/benchmark_prompts.tsv",
        },
        "weight_gemm_shapes": shapes,
        "counters": _counters_block(),
        "headline_ratios": _headline_block(),
        "role": ("these are the per-shape PREFILL DENOMINATORS later stages quote. W9 requires "
                 "them per shape and never as one number"),
    }
    doc.update(extracted)
    out = write_results(a.out, doc)
    print(f"\nwritten: {out}")
    print("\nper-shape prefill denominators at the D3 M values:")
    for row in extracted["prefill_denominators_at_d3_m_values"]:
        print(f"  {row['shape']:24s} M={row['M']:<4d} N={row['N']:<5d} K={row['K']:<5d} "
              f"{row['gflops']:9.2f} GFLOP/s  stddev {row['stddev_pct_of_median']:6.3f}%  "
              f"{row['verdict']}")
    print("\nM=1 decode-shaped rows (used for nothing; W3 corroboration):")
    for row in extracted["decode_shaped_rows_M1"]:
        print(f"  {row['shape']:24s} M=1 {row['gflops']:9.2f} GFLOP/s  stddev "
              f"{row['stddev_pct_of_median']:6.3f}%  {row['verdict']}")
    nm = extracted["non_monotonic_observations"]
    print(f"\nnon-monotonic steps observed: {len(nm)} (observation only; cause unestablished)")
    for o in nm:
        print(f"  {o['shape']:24s} M {o['from_M']} -> {o['to_M']}: "
              f"{o['from_gflops']:.2f} -> {o['to_gflops']:.2f} GFLOP/s "
              f"({o['drop_pct']:.1f}% drop, both VALID: {o['both_valid']})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="the engine timing run at the fixed prompt set")
    r.add_argument("--command", required=True,
                   help="the timing binary to drive as a subprocess")
    r.add_argument("--fixture", default=None)
    r.add_argument("--allow-other-prompt-set", action="store_true",
                   help="deliberately run against a prompt source other than the D3 fixed set")
    r.add_argument("--lengths", default=None)
    r.add_argument("--contexts", default=None)
    r.add_argument("--out", default=DEFAULT_OUT)
    r.add_argument("--c-results-dir", default=None,
                   help="where the C layer's intermediate JSON is written")
    r.add_argument("--c-out-stem", default="stage4_forward_c")
    r.add_argument("--configs", default=None,
                   help="the ORDERED configuration list, e.g. "
                        "prefill:16:cache,prefill:16:nocache,gemm:768. Defaults to the "
                        "Stage 4 paired interleaved set")
    r.add_argument("--correctness", default=None,
                   help="a correctness results file whose verdict is carried into the summary")
    r.add_argument("--compare", default=None,
                   help="a prior stage's results file to compare against for regression")
    r.add_argument("--expect-mhz", type=int, default=EXPECTED_CLOCK_MHZ)
    r.add_argument("--stage", default=STAGE_ID)
    r.add_argument("--probe-only", action="store_true")
    r.add_argument("--preflight-only", action="store_true")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("merge",
                       help="combine several `run` invocations into one results document")
    m.add_argument("parts", nargs="+", help="the per-chunk results files, in measured order")
    m.add_argument("--out", default=DEFAULT_OUT)
    m.add_argument("--compare", default=None)
    m.add_argument("--correctness", default=None)
    m.add_argument("--stage", default=STAGE_ID)
    m.set_defaults(func=cmd_merge)

    c = sub.add_parser("cublas", help="W9: the cuBLAS prefill denominators at the D3 M values")
    c.add_argument("--binary", required=True)
    c.add_argument("--out", default=DEFAULT_CUBLAS_OUT)
    c.add_argument("--c-results-dir", default=None)
    c.add_argument("--stage", default="stage-3")
    c.set_defaults(func=cmd_cublas)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
