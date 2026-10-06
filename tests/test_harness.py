"""test_harness.py -- unit tests for bench/harness.py.

WHAT IS REDUCED HERE, AND WHY. The timing command is a STUB: a small script this
test writes, which emits the probe lines and the results JSON the real C timing
driver emits, with figures this test chose. The real engine is NEVER invoked. One
iteration of it is seconds and the full D3 set is over an hour, against a 300 s
per-test cap, so a test that drove the real binary would be a benchmark and not a
test. Synthesized statistics are also the only way to test the INVALID path and
the regression path deliberately: a real run cannot be asked to come back 10%
slower on cue.

The preflight checks are tested with INJECTED verify functions rather than by
touching the machine, so the refusal paths are exercised without needing a
fingerprint difference or an unlocked clock to exist.

Runs under the project .venv (Python 3.14.2).
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("harness", os.path.join(REPO_ROOT, "bench", "harness.py"))

D3_FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "benchmark_prompts.tsv")

# A stub timing command. It emits exactly what the C driver emits: probe lines in
# the parseable form, and a results file in the bench_common JSON format. Every
# number in it is supplied by the test through TIE_STUB_PLAN.
STUB = r'''
import json, os, sys

plan = json.loads(os.environ["TIE_STUB_PLAN"])
args = sys.argv[1:]

def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default

lengths = [int(x) for x in (opt("--lengths") or "").split(",") if x]
contexts = [int(x) for x in (opt("--contexts") or "").split(",") if x]
repeat = int(opt("--repeat", "1"))

# Stage 4. What this stub reports as repeat_applied is under the test's control,
# so a driver that IGNORES --repeat can be simulated exactly:
#   "honour"  -> repeat_applied == the requested R (a driver that complied)
#   "ignore"  -> repeat_applied == 1 regardless of R (a driver that ran one)
#   "omit"    -> no repeat_applied field at all (a driver that reported nothing)
#   an int    -> that value verbatim, whatever was requested
repeat_mode = plan.get("repeat_mode", "honour")
paths = plan.get("paths")          # {"prefill": "cache", ...} or None

if "--probe" in args:
    for L in lengths:
        print(f"probe prefill L={L} row=d3_{L} "
              f"seconds={plan['probe_s']['prefill'][str(L)]:.6f}  (NOT A MEASUREMENT)")
    for c in contexts:
        print(f"probe decode c={c} row=d3_{c} "
              f"seconds={plan['probe_s']['decode'][str(c)]:.6f}  (NOT A MEASUREMENT)")
    sys.exit(0)

def record(workload, tokens, spec):
    samples = spec["samples_ms"]
    n = len(samples)
    mean = sum(samples) / n
    srt = sorted(samples)
    median = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    var = sum((x - mean) ** 2 for x in samples) / (n - 1) if n > 1 else 0.0
    sd = var ** 0.5
    pct = sd / median * 100.0 if median else 0.0
    valid = pct <= 5.0
    counters = {"vocab_size": 50257, "n_layer": 12}
    if workload == "prefill":
        counters["tokens"] = tokens
        counters["context_tokens"] = tokens
    else:
        counters["context_tokens"] = tokens
        counters["tokens_generated"] = 1
    counters["repeat_requested"] = repeat
    if isinstance(repeat_mode, int):
        counters["repeat_applied"] = repeat_mode
    elif repeat_mode == "honour":
        counters["repeat_applied"] = repeat
    elif repeat_mode == "ignore":
        counters["repeat_applied"] = 1
    # "omit" leaves the field out entirely.
    return {
        "benchmark": "stub_forward",
        "configuration": f"{workload}, {tokens} tokens, stub",
        "units": "ms",
        "value": median,
        "warmup_iterations": int(opt("--warmup", "25")),
        "samples_requested": int(opt("--samples", "30")),
        "raw_samples_ms": samples,
        "statistics": {"n": n, "mean_ms": mean, "median_ms": median,
                       "min_ms": min(samples), "max_ms": max(samples),
                       "stddev_ms": sd, "stddev_pct_of_median": pct, "valid": valid,
                       "invalid_reason": ("" if valid else
                                          f"stddev is {pct:.3f}% of median, above the 5.0% "
                                          "protocol limit")},
        "counters": counters,
        "annotations": {"workload": workload,
                        "prompt_row_id": f"d3_{tokens}",
                        "prompt_set_status": opt("--prompt-set-status", "FIXED"),
                        "prompt_set_open_decision": opt("--prompt-set-decision", "D3"),
                        "prompt_set_source": opt("--fixture", ""),
                        "repeat_factor_R": str(repeat),
                        "tag": "measured",
                        **({("prefill_path" if workload == "prefill" else "decode_path"):
                            paths[workload]} if paths and workload in paths else {})},
    }

records = [record("prefill", L, plan["records"]["prefill"][str(L)]) for L in lengths]
records += [record("decode", c, plan["records"]["decode"][str(c)]) for c in contexts]

doc = dict(plan["header"])
doc["protocol"] = {"min_warmup": 5, "min_samples": 20, "max_stddev_pct_of_median": 5.0}
doc["records"] = records
out = os.path.join(os.environ["BENCH_RESULTS_DIR"], opt("--out", "stub") + ".json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(doc, f)
print("results written:", out)
'''

HEADER = {
    "stage": "stage-3",
    "git_commit": "0" * 40,
    "run_timestamp_utc": "2026-10-04T00:00:00Z",
    "build_timestamp": "2026-10-04T00:00:00Z",
    "device": "NVIDIA GeForce GTX 1650 Ti sm_75",
    "cpu_timer": "QueryPerformanceCounter",
    "gpu_timer": "cudaEvent + cudaDeviceSynchronize",
    "cxx_compiler": "MSVC 19.44.35229.0",
    "cxx_flags": "/arch:AVX2 /fp:precise /O2",
    "cuda_compiler": "nvcc 13.1.80",
    "cuda_flags": "-arch=sm_75",
}


def flat_samples(median, n=30):
    """A tight sample set: dispersion well inside the 5% limit."""
    out = []
    for i in range(n):
        out.append(median * (1.0 + 0.002 * (1 if i % 2 else -1)))
    return out


def dispersed_samples(median, n=30):
    """One sample at +50% among thirty, which is W3's own failure mode: the
    standard deviation lands near 9.1% of the median and the run is INVALID."""
    out = [median] * n
    out[7] = median * 1.5
    return out


class StubHarness:
    """Runs the harness's probe and timed passes against the stub command."""

    def __init__(self, plan):
        self.dir = tempfile.mkdtemp(prefix="tie_harness_test_")
        self.stub = os.path.join(self.dir, "stub_bench.py")
        with open(self.stub, "w", encoding="utf-8") as f:
            f.write(STUB)
        self.results_dir = os.path.join(self.dir, "c_results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.command = [sys.executable, self.stub]
        self.env = dict(os.environ)
        self.env["TIE_STUB_PLAN"] = json.dumps(plan)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


MEDIANS = {16: 4000.0, 32: 8000.0, 64: 16000.0, 128: 32000.0}
DECODE_MEDIANS = {32: 6500.0, 64: 13000.0, 128: 26000.0}


def full_plan(dispersed=(), header=None, medians=None, decode=None,
              repeat_mode="honour", paths=None):
    """The stub's plan is keyed by WORKLOAD and then by token count. Prefill at
    L=32 and decode at c=32 are different configurations with different costs,
    and a plan keyed by token count alone would silently give them the same
    figure -- which is exactly the kind of collision this harness exists to stop
    a results file from carrying."""
    tables = {"prefill": dict(medians or MEDIANS), "decode": dict(decode or DECODE_MEDIANS)}
    return {
        "probe_s": {w: {str(k): v / 1000.0 for k, v in t.items()}
                    for w, t in tables.items()},
        "records": {w: {str(k): {"samples_ms": (dispersed_samples(v) if k in dispersed
                                                else flat_samples(v))}
                        for k, v in t.items()}
                    for w, t in tables.items()},
        "header": header or HEADER,
        "repeat_mode": repeat_mode,
        "paths": paths,
    }


def run_stub(stub, lengths=(16, 32, 64, 128), contexts=(32, 64, 128),
             prompt_set=None, correctness=None, compare=None):
    probes = H.run_probe(stub.command, D3_FIXTURE, lengths, contexts, env=stub.env)
    constructions = {}
    for p in probes:
        key = (f"prefill|L={p['tokens']}" if p["workload"] == "prefill"
               else f"decode|c={p['tokens']}")
        constructions[key] = H.select_construction(p["probe_ms"])
    ps = prompt_set or H.C.prompt_set_identity(D3_FIXTURE)
    ps.setdefault("is_d3_fixed_set", True)
    c_doc = H.run_timed(stub.command, D3_FIXTURE, lengths, contexts, "stub",
                        stub.results_dir, constructions, ps, env=stub.env)
    records = H.build_configuration_records(c_doc, constructions)
    pre = {"pass": True, "failed_checks": [], "checks": {}}
    regression = H.compare_regression(records, c_doc, compare)
    doc = H.assemble_results(c_doc, records, ps, pre, probes, constructions,
                             H._load_correctness_result(correctness), regression, "stage-3")
    return doc, records, c_doc


# ============================================== the W3 construction selector ===

class TestW3ConstructionSelector(unittest.TestCase):
    def test_at_or_above_the_floor_takes_the_single_iteration_path(self):
        for probe in (10.0, 10.0001, 100.0, 32000.0):
            c = H.select_construction(probe)
            self.assertEqual(c["construction"], "single_iteration", probe)
            self.assertEqual(c["repeat_factor_R"], 1)
            self.assertFalse(c["amortised"])
            self.assertEqual(c["effective_bracket_ms"], probe)

    def test_below_the_floor_takes_the_batched_path(self):
        c = H.select_construction(1.0)
        self.assertEqual(c["construction"], "batched")
        self.assertTrue(c["amortised"])

    def test_R_is_ceil_of_the_floor_over_the_probe(self):
        import math
        for probe, expected in ((1.0, 10), (0.03, 334), (2.5, 4), (3.0, 4), (9.999, 2)):
            c = H.select_construction(probe)
            self.assertEqual(c["repeat_factor_R"], expected, probe)
            self.assertEqual(c["repeat_factor_R"], math.ceil(10.0 / probe))

    def test_the_construction_record_carries_probe_construction_R_and_bracket(self):
        c = H.select_construction(0.05)
        for field in ("probe_ms", "construction", "repeat_factor_R", "effective_bracket_ms",
                      "per_sample_floor_ms"):
            self.assertIn(field, c)
        self.assertEqual(c["probe_ms"], 0.05)
        self.assertEqual(c["repeat_factor_R"], 200)
        self.assertAlmostEqual(c["effective_bracket_ms"], 10.0, places=9)

    def test_the_batched_note_states_the_loss_of_resolution(self):
        c = H.select_construction(0.05)
        self.assertIn("AMORTISED", c["note"])
        self.assertIn("no longer observable", c["note"])

    def test_a_non_positive_probe_is_rejected(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                H.select_construction(bad)

    def test_the_w3_dispersion_arithmetic_is_reproduced(self):
        # 50 * sqrt(29/870) = 9.13%, which is W3's figure to within the difference
        # between the two ways of writing it (0.5/sqrt(29) = 9.29%).
        self.assertAlmostEqual(H.predicted_dispersion_pct(30, 1), 9.128709291752769, places=9)
        self.assertAlmostEqual(50.0 / (100 ** 0.5), 5.0, places=9)   # sd% = 50/sqrt(n) at k=1

    def test_more_samples_do_not_reduce_rate_driven_dispersion(self):
        # The rate model depends on the per-sample duration and not on n: at a
        # fixed per-iteration event probability, doubling n leaves k/n alone.
        # 50*sqrt(1*29/(30*29)) = 9.1287% against 50*sqrt(2*58/(60*59)) = 9.0510%:
        # the same figure to within a tenth of a percentage point, where halving
        # it would need the per-sample duration to quadruple.
        a = H.predicted_dispersion_pct(30, 1)
        b = H.predicted_dispersion_pct(60, 2)
        self.assertAlmostEqual(a, b, delta=0.1)
        self.assertAlmostEqual(b, 9.051016735969625, places=9)


# ======================================================== the preflight gate ===

class TestPreflightRefusals(unittest.TestCase):
    CLEAN_FP = {"match": True, "differences": [], "fields_compared": 25,
                "fields_excluded_from_comparison": ["build_flags.BENCH_BUILD_TIMESTAMP"],
                "provenance": []}
    LOCKED = {"locked": True, "mhz": 1365}

    def test_both_clean_passes(self):
        r = H.preflight(verify_fingerprint=lambda: self.CLEAN_FP,
                        verify_lock=lambda mhz: self.LOCKED)
        self.assertTrue(r["pass"])
        self.assertEqual(r["failed_checks"], [])

    def test_refuses_when_the_fingerprint_is_not_clean_and_names_the_check(self):
        dirty = {"match": False, "fields_compared": 25,
                 "differences": [{"field": "gpu.driver_version",
                                  "stored": "591.44", "current": "600.00"}],
                 "provenance": []}
        r = H.preflight(verify_fingerprint=lambda: dirty,
                        verify_lock=lambda mhz: self.LOCKED)
        self.assertFalse(r["pass"])
        self.assertEqual(r["failed_checks"], ["environment_fingerprint"])
        self.assertIn("gpu.driver_version", r["checks"]["environment_fingerprint"]["reasons"][0])

    def test_refuses_when_the_compared_field_count_is_not_25(self):
        wrong = {"match": True, "differences": [], "fields_compared": 26, "provenance": []}
        r = H.preflight(verify_fingerprint=lambda: wrong,
                        verify_lock=lambda mhz: self.LOCKED)
        self.assertFalse(r["pass"])
        self.assertIn("environment_fingerprint", r["failed_checks"])

    def test_refuses_when_the_clock_lock_does_not_verify_and_names_the_check(self):
        r = H.preflight(verify_fingerprint=lambda: self.CLEAN_FP,
                        verify_lock=lambda mhz: {"locked": False, "observed_mhz": 1215})
        self.assertFalse(r["pass"])
        self.assertEqual(r["failed_checks"], ["graphics_clock_lock"])
        self.assertEqual(r["checks"]["graphics_clock_lock"]["expected_mhz"], 1365)

    def test_both_failing_names_both(self):
        r = H.preflight(verify_fingerprint=lambda: {"match": False, "differences": [],
                                                    "fields_compared": 25, "provenance": []},
                        verify_lock=lambda mhz: {"locked": False})
        self.assertEqual(sorted(r["failed_checks"]),
                         ["environment_fingerprint", "graphics_clock_lock"])

    def test_the_expected_clock_is_1365(self):
        self.assertEqual(H.EXPECTED_CLOCK_MHZ, 1365)


# ============================================== the prompt set is enforced =====

class TestPromptSetEnforcement(unittest.TestCase):
    def test_the_default_is_the_d3_fixed_set(self):
        ps = H.resolve_prompt_set(None, allow_other=False)
        self.assertTrue(ps["is_d3_fixed_set"])
        self.assertEqual(ps["status"], "FIXED")
        self.assertEqual(ps["decision"], "D3")
        self.assertEqual([r["target_tokens"] for r in ps["rows"]], [16, 32, 64, 128])

    def test_another_source_is_refused_without_an_explicit_argument(self):
        other = os.path.join(REPO_ROOT, "tests", "fixtures",
                             "stage2_placeholder_prompts.tsv")
        with self.assertRaises(SystemExit):
            H.resolve_prompt_set(other, allow_other=False)

    def test_another_source_is_recorded_as_an_override_when_allowed(self):
        other = os.path.join(REPO_ROOT, "tests", "fixtures",
                             "stage2_placeholder_prompts.tsv")
        ps = H.resolve_prompt_set(other, allow_other=True)
        self.assertFalse(ps["is_d3_fixed_set"])
        self.assertTrue(ps["explicitly_overridden"])
        self.assertEqual(ps["status"], "PLACEHOLDER")


# ==================================================== the results file shape ====

class TestResultsFileFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = StubHarness(full_plan())
        cls.doc, cls.records, cls.c_doc = run_stub(cls.stub)

    @classmethod
    def tearDownClass(cls):
        cls.stub.close()

    def test_every_section_9_field_is_present_by_name(self):
        for field in ("stage", "git_commit", "run_timestamp_utc", "device",
                      "cxx_compiler", "cxx_flags", "cuda_compiler", "cuda_flags",
                      "prompt_set", "records", "correctness", "counters",
                      "headline_ratios", "protocol", "regression", "summary",
                      "w3_construction", "preflight", "invalid_configurations"):
            self.assertIn(field, self.doc, field)

    def test_raw_per_sample_timings_are_carried_through_unchanged(self):
        by_key = {r["configuration_key"]: r for r in self.c_doc["records"]
                  and self.records}
        for rec, craw in zip(self.records, self.c_doc["records"]):
            self.assertEqual(rec["raw_samples_ms"], craw["raw_samples_ms"])
            self.assertEqual(len(rec["raw_samples_ms"]), 30)
        self.assertTrue(by_key)

    def test_statistics_are_the_c_layers_and_are_not_recomputed(self):
        for rec, craw in zip(self.records, self.c_doc["records"]):
            self.assertEqual(rec["statistics"], craw["statistics"])
            self.assertEqual(rec["value"], craw["value"])

    def test_the_construction_record_is_attached_per_configuration(self):
        for rec in self.records:
            self.assertIsNotNone(rec["construction"])
            for field in ("construction", "probe_ms", "repeat_factor_R",
                          "effective_bracket_ms"):
                self.assertIn(field, rec["construction"])

    def test_counters_are_absent_with_a_stated_reason(self):
        self.assertFalse(self.doc["counters"]["collected"])
        self.assertIn("Stage 7", self.doc["counters"]["reason"])

    def test_no_headline_ratio_is_quoted_and_the_reason_is_stated(self):
        self.assertFalse(self.doc["headline_ratios"]["quoted"])
        self.assertIn("DENOMINATORS", self.doc["headline_ratios"]["reason"])


class TestPrefillAndDecodeAreSeparated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = StubHarness(full_plan())
        cls.doc, cls.records, _ = run_stub(cls.stub)

    @classmethod
    def tearDownClass(cls):
        cls.stub.close()

    def test_they_appear_as_separate_records(self):
        workloads = [r["workload"] for r in self.records]
        self.assertEqual(workloads.count("prefill"), 4)
        self.assertEqual(workloads.count("decode"), 3)

    def test_the_summary_indexes_them_under_separate_keys(self):
        self.assertEqual(len(self.doc["summary"]["prefill"]), 4)
        self.assertEqual(len(self.doc["summary"]["decode"]), 3)
        self.assertIn("separation_note", self.doc["summary"])

    def test_no_field_anywhere_combines_them(self):
        """No key in the document mentions both workloads, and no record carries a
        figure derived from both."""
        combined = []

        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = k.lower()
                    if "prefill" in kl and "decode" in kl:
                        combined.append(path + "." + k)
                    walk(v, path + "." + k)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(self.doc)
        self.assertEqual(combined, [], f"keys combining both workloads: {combined}")

    def test_each_record_names_exactly_one_workload(self):
        for r in self.records:
            self.assertIn(r["workload"], ("prefill", "decode", "isolated_gemm"))
            self.assertTrue(r["configuration_key"].startswith(r["workload"]))


# ================================================== the INVALID path ===========

class TestInvalidIsReportedAndNeverRepaired(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The 64-token configurations get one sample at +50%, which is W3's own
        # failure mode and lands the dispersion near 9.1% of the median.
        cls.stub = StubHarness(full_plan(dispersed=(64,)))
        cls.doc, cls.records, _ = run_stub(cls.stub)

    @classmethod
    def tearDownClass(cls):
        cls.stub.close()

    def test_a_dispersed_configuration_is_marked_invalid(self):
        bad = [r for r in self.records if not r["valid"]]
        self.assertTrue(bad, "a configuration with >5% dispersion must be INVALID")
        for r in bad:
            self.assertGreater(r["statistics"]["stddev_pct_of_median"],
                               H.PROTOCOL_MAX_STDDEV_PCT)
            self.assertEqual(r["verdict"], "INVALID")
            self.assertIn("above the 5.0% protocol limit", r["invalid_reason"])

    def test_it_appears_in_the_output_as_invalid(self):
        self.assertTrue(self.doc["invalid_configurations"])
        for key in self.doc["invalid_configurations"]:
            entry = next(r for g in ("prefill", "decode")
                         for r in self.doc["summary"][g]
                         if r["configuration_key"] == key)
            self.assertEqual(entry["verdict"], "INVALID")

    def test_the_numbers_behind_an_invalid_verdict_are_kept(self):
        for r in self.records:
            if not r["valid"]:
                self.assertEqual(len(r["raw_samples_ms"]), 30)
                self.assertGreater(r["statistics"]["median_ms"], 0)

    def test_the_reported_median_is_never_the_trimmed_one(self):
        for r in self.records:
            if not r["valid"]:
                trimmed = r["diagnostics"]["diagnostic_trimmed_median_ms"]
                self.assertNotEqual(r["value"], None)
                self.assertEqual(r["value"], r["statistics"]["median_ms"])
                # The trimmed figure exists, is labelled, and is not the reported one.
                self.assertIn("DIAGNOSTIC ONLY", r["diagnostics"]["diagnostic_note"])
                self.assertIsNotNone(trimmed)

    def test_the_limit_is_not_a_parameter_and_no_retry_path_exists(self):
        self.assertEqual(H.PROTOCOL_MAX_STDDEV_PCT, 5.0)
        self.assertFalse(self.doc["protocol_conditions"]["limit_is_a_parameter"])
        self.assertFalse(self.doc["protocol_conditions"]["retry_path_exists"])
        with open(os.path.join(REPO_ROOT, "bench", "harness.py"), "r",
                  encoding="utf-8") as f:
            src = f.read()
        for banned in ("max_stddev_pct=", "stddev_limit", "--max-stddev", "while not valid",
                       "for attempt in"):
            self.assertNotIn(banned, src, f"{banned} suggests a loosenable limit or a retry")


# ================================================== regression detection =======

def prior_file(path, medians, decode, status="FIXED", source=D3_FIXTURE,
               header=None, dispersed=()):
    doc = dict(header or HEADER)
    doc["stage"] = "stage-2"
    recs = []
    for workload, table in (("prefill", medians), ("decode", decode)):
        for tokens, med in table.items():
            samples = dispersed_samples(med) if tokens in dispersed else flat_samples(med)
            n = len(samples)
            mean = sum(samples) / n
            srt = sorted(samples)
            median = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
            sd = (sum((x - mean) ** 2 for x in samples) / (n - 1)) ** 0.5
            pct = sd / median * 100.0
            counters = ({"tokens": tokens, "context_tokens": tokens}
                        if workload == "prefill" else {"context_tokens": tokens})
            recs.append({
                "benchmark": "stub_forward",
                "configuration": f"{workload} {tokens}",
                "units": "ms", "value": median,
                "raw_samples_ms": samples,
                "statistics": {"n": n, "mean_ms": mean, "median_ms": median,
                               "min_ms": min(samples), "max_ms": max(samples),
                               "stddev_ms": sd, "stddev_pct_of_median": pct,
                               "valid": pct <= 5.0, "invalid_reason": ""},
                "counters": counters,
                "annotations": {"workload": workload, "prompt_row_id": f"d3_{tokens}",
                                "prompt_set_status": status, "prompt_set_source": source},
            })
    doc["records"] = recs
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


class TestRegressionDetection(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tie_regression_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self, current_factor, prior_status="FIXED", prior_source=D3_FIXTURE,
             prior_header=None, prior_dispersed=(), current_dispersed=()):
        medians = {k: v * current_factor for k, v in MEDIANS.items()}
        decode = {k: v * current_factor for k, v in DECODE_MEDIANS.items()}
        stub = StubHarness(full_plan(medians=medians, decode=decode,
                                     dispersed=current_dispersed))
        prior = prior_file(os.path.join(self.dir, "prior.json"), MEDIANS, DECODE_MEDIANS,
                           status=prior_status, source=prior_source, header=prior_header,
                           dispersed=prior_dispersed)
        try:
            doc, records, _ = run_stub(stub, compare=prior)
        finally:
            stub.close()
        return doc["regression"]

    def test_triggers_at_ten_percent_slower(self):
        r = self._run(1.10)
        self.assertTrue(r["performed"])
        self.assertTrue(r["any_regression"])
        self.assertEqual(r["refusals"], [])
        for c in r["comparisons"]:
            self.assertAlmostEqual(c["change_pct"], 10.0, places=6)
            self.assertEqual(c["outcome"], "REGRESSION")

    def test_does_not_trigger_at_four_percent_slower(self):
        r = self._run(1.04)
        self.assertTrue(r["performed"])
        self.assertFalse(r["any_regression"])
        for c in r["comparisons"]:
            self.assertEqual(c["outcome"], "no measurable change")
        self.assertEqual(r["noise_floor_pct"], 4.4)

    def test_a_real_speedup_beyond_the_floor_is_an_improvement_not_a_regression(self):
        r = self._run(0.80)
        self.assertFalse(r["any_regression"])
        for c in r["comparisons"]:
            self.assertEqual(c["outcome"], "improvement")

    def test_refuses_when_the_prompt_sets_differ(self):
        r = self._run(1.10, prior_status="PLACEHOLDER",
                      prior_source="tests/fixtures/stage2_placeholder_prompts.tsv")
        self.assertEqual(r["comparisons"], [])
        self.assertTrue(r["refusals"])
        for ref in r["refusals"]:
            self.assertIn("prompt sets differ", ref["reason"])
        self.assertFalse(r["any_regression"])

    def test_refuses_when_the_prior_side_is_invalid(self):
        r = self._run(1.10, prior_dispersed=(64,))
        refused = {x["configuration_key"]: x["reason"] for x in r["refusals"]}
        self.assertIn("prefill|L=64", refused)
        self.assertIn("the prior configuration is INVALID", refused["prefill|L=64"])

    def test_refuses_when_the_current_side_is_invalid(self):
        r = self._run(1.10, current_dispersed=(32,))
        refused = {x["configuration_key"]: x["reason"] for x in r["refusals"]}
        self.assertIn("prefill|L=32", refused)
        self.assertIn("the current configuration is INVALID", refused["prefill|L=32"])

    def test_refuses_when_a_substantive_fingerprint_field_differs(self):
        changed = dict(HEADER)
        changed["cxx_flags"] = "/arch:AVX512 /O2"
        r = self._run(1.10, prior_header=changed)
        self.assertTrue(r["substantive_fingerprint_differences"])
        self.assertEqual(r["substantive_fingerprint_differences"][0]["field"], "cxx_flags")
        self.assertEqual(r["comparisons"], [])
        for ref in r["refusals"]:
            self.assertIn("substantive fingerprint fields differ", ref["reason"])

    def test_the_build_timestamp_is_not_a_substantive_field(self):
        self.assertNotIn("build_timestamp", H.SUBSTANTIVE_FINGERPRINT_FIELDS)
        self.assertNotIn("git_commit", H.SUBSTANTIVE_FINGERPRINT_FIELDS)

    def test_no_prior_file_named_is_reported_as_not_performed(self):
        stub = StubHarness(full_plan())
        try:
            doc, _, _ = run_stub(stub, compare=None)
        finally:
            stub.close()
        self.assertFalse(doc["regression"]["performed"])
        self.assertIn("no prior results file", doc["regression"]["reason"])

    def test_what_it_cannot_catch_is_stated(self):
        r = self._run(1.04)
        self.assertIn("smaller than the noise floor", r["cannot_catch"])


# =============================================== the correctness verdict =======

class TestCorrectnessVerdictPropagates(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tie_correctness_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, verdict):
        path = os.path.join(self.dir, "correctness.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "stage": "stage-3",
                "prompt_set": {"status": "FIXED", "decision": "D3"},
                "margin_table": [{"target_tokens": 16, "margin_min": 0.02}],
                "gate": {"verdict": verdict, "tolerance": 1e-3,
                         "conditions": {
                             "max_abs_diff_within_tolerance": {"pass": verdict == "PASS"},
                             "top1_agreement_at_every_position": {"pass": True},
                             "greedy_sequence_matches_reference": {"pass": True}}},
            }, f)
        return path

    def _summary_for(self, verdict):
        stub = StubHarness(full_plan())
        try:
            doc, _, _ = run_stub(stub, correctness=self._write(verdict))
        finally:
            stub.close()
        return doc

    def test_pass_propagates(self):
        doc = self._summary_for("PASS")
        self.assertTrue(doc["correctness"]["available"])
        self.assertEqual(doc["correctness"]["verdict"], "PASS")
        self.assertEqual(doc["correctness"]["tolerance"], 1e-3)

    def test_fail_propagates_and_timings_are_still_recorded(self):
        doc = self._summary_for("FAIL")
        self.assertEqual(doc["correctness"]["verdict"], "FAIL")
        self.assertEqual(len(doc["summary"]["prefill"]), 4)
        self.assertEqual(len(doc["summary"]["decode"]), 3)
        for r in doc["summary"]["prefill"]:
            self.assertGreater(r["median_ms"], 0.0)

    def test_an_absent_correctness_file_is_reported_as_unavailable(self):
        stub = StubHarness(full_plan())
        try:
            doc, _, _ = run_stub(stub, correctness=os.path.join(self.dir, "nope.json"))
        finally:
            stub.close()
        self.assertFalse(doc["correctness"]["available"])
        self.assertIn("does not exist", doc["correctness"]["reason"])


# ======================================================= the W9 extraction =====

class TestWeightShapesAndCublasExtraction(unittest.TestCase):
    def test_shapes_are_verified_against_the_committed_inventory(self):
        shapes = H.verify_shapes_against_inventory(
            os.path.join(REPO_ROOT, "src", "gpt2_tensor_inventory.json"))
        self.assertEqual(shapes["qkv_projection"]["K"], 768)
        self.assertEqual(shapes["qkv_projection"]["N"], 2304)
        self.assertEqual(shapes["attn_output_projection"]["K"], 768)
        self.assertEqual(shapes["attn_output_projection"]["N"], 768)
        self.assertEqual(shapes["ffn_up"]["K"], 768)
        self.assertEqual(shapes["ffn_up"]["N"], 3072)
        self.assertEqual(shapes["ffn_down"]["K"], 3072)
        self.assertEqual(shapes["ffn_down"]["N"], 768)

    def test_m1_rows_are_separated_from_the_prefill_denominators(self):
        shapes = H.verify_shapes_against_inventory(
            os.path.join(REPO_ROOT, "src", "gpt2_tensor_inventory.json"))

        def rec(kind, shape, M, N, K, gflops, valid=True):
            samples = flat_samples(1.0) if valid else dispersed_samples(1.0)
            return {"configuration": f"{kind} {shape} M={M} N={N} K={K}",
                    "units": "GFLOP/s", "value": gflops,
                    "raw_samples_ms": samples,
                    "statistics": {"n": 30, "median_ms": 1.0, "min_ms": 1.0, "max_ms": 1.0,
                                   "stddev_ms": 0.0,
                                   "stddev_pct_of_median": 0.0 if valid else 9.1,
                                   "valid": valid, "invalid_reason": ""},
                    "counters": {}}

        doc = {"records": [
            rec("decode", "qkv_projection", 1, 2304, 768, 64.0, valid=False),
            rec("prefill", "qkv_projection", 16, 2304, 768, 438.0),
            rec("prefill", "qkv_projection", 32, 2304, 768, 1124.0),
            rec("prefill", "qkv_projection", 64, 2304, 768, 1574.0),
            rec("prefill", "qkv_projection", 128, 2304, 768, 1793.0),
            rec("prefill", "qkv_projection", 512, 2304, 768, 2300.0),
            rec("prefill", "qkv_projection", 1024, 2304, 768, 1598.0),
        ]}
        out = H.extract_cublas_d3(doc, shapes)
        self.assertEqual([r["M"] for r in out["prefill_denominators_at_d3_m_values"]],
                         [16, 32, 64, 128])
        self.assertEqual([r["M"] for r in out["decode_shaped_rows_M1"]], [1])
        self.assertEqual(out["decode_shaped_rows_M1"][0]["verdict"], "INVALID")
        self.assertIn("DECODE-SHAPED", out["decode_shaped_rows_M1"][0]["role"])
        # The 1024 point falls below the 512 point: a non-monotonic step, reported
        # as an observation with no mechanism proposed.
        nm = out["non_monotonic_observations"]
        self.assertEqual(len(nm), 1)
        self.assertEqual((nm[0]["from_M"], nm[0]["to_M"]), (512, 1024))
        self.assertIn("NOT established", out["non_monotonicity_note"])


class TestProbeParsingIsNotAMeasurement(unittest.TestCase):
    def test_probe_lines_are_parsed_and_labelled(self):
        text = ("probe prefill L=16 row=d3_16 seconds=3.912000  (NOT A MEASUREMENT)\n"
                "probe decode c=32 row=d3_32 seconds=6.551000  (NOT A MEASUREMENT)\n"
                "some other output\n")
        got = H.parse_probe_output(text)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0]["workload"], "prefill")
        self.assertEqual(got[0]["tokens"], 16)
        self.assertEqual(got[0]["prompt_row_id"], "d3_16")
        self.assertAlmostEqual(got[0]["probe_ms"], 3912.0, places=6)
        self.assertEqual(got[1]["workload"], "decode")
        self.assertEqual(got[1]["tokens"], 32)
        for g in got:
            self.assertIn("NOT A MEASUREMENT", g["tag"])


    def test_the_path_annotation_is_parsed_and_defaults_to_nocache(self):
        """Stage 4 appends `path=`. A line without it is a Stage 3 driver's line,
        and that driver ran the no-cache path."""
        text = ("probe prefill L=16 row=d3_16 seconds=3.912000 path=cache  (NOT A MEASUREMENT)\n"
                "probe decode c=32 row=d3_32 seconds=0.242000 path=nocache  (NOT A MEASUREMENT)\n"
                "probe decode c=64 row=d3_64 seconds=6.551000  (NOT A MEASUREMENT)\n")
        got = H.parse_probe_output(text)
        self.assertEqual([g["path"] for g in got], ["cache", "nocache", "nocache"])
        self.assertAlmostEqual(got[1]["probe_ms"], 242.0, places=6)


# ================================ Stage 4: W3's refusal and W14's annotation ===

class TestW3RepeatConfirmationRefusal(unittest.TestCase):
    """DECISION C. The harness never divides a bracket by an R the driver did not
    confirm, and it checks the confirmation at R = 1 as well."""

    @staticmethod
    def _records(repeat_mode, R):
        """Builds records through the stub, then forces the selected R so the
        refusal can be exercised without a sub-10 ms configuration existing."""
        stub = StubHarness(full_plan(repeat_mode=repeat_mode))
        try:
            doc, records, c_doc = run_stub(stub)
        finally:
            stub.close()
        constructions = {}
        for r in records:
            con = dict(H.select_construction(1000.0))
            con["repeat_factor_R"] = R
            constructions[r["configuration_key"]] = con
        return records, constructions

    def test_a_driver_that_ignores_repeat_makes_the_harness_refuse(self):
        records, constructions = self._records("ignore", 4)
        with self.assertRaises(H.RepeatNotConfirmed) as cm:
            H.verify_repeat_confirmation(records, constructions)
        msg = str(cm.exception)
        self.assertIn("W3", msg)
        self.assertIn("prefill|L=16", msg)          # the configuration is named
        self.assertIn("R=4", msg)
        self.assertIn("repeat_applied=1", msg)      # and what the driver reported
        self.assertIn("No bracket was divided by R", msg)

    def test_a_driver_that_reports_nothing_makes_the_harness_refuse(self):
        records, constructions = self._records("omit", 4)
        with self.assertRaises(H.RepeatNotConfirmed) as cm:
            H.verify_repeat_confirmation(records, constructions)
        msg = str(cm.exception)
        self.assertIn("W3", msg)
        self.assertIn("no repeat_applied field", msg)

    def test_a_driver_that_confirms_R_is_accepted(self):
        # The stub reports repeat_applied = 4 verbatim, as a driver that
        # implemented the batched bracket would.
        records, constructions = self._records(4, 4)
        out = H.verify_repeat_confirmation(records, constructions)
        self.assertTrue(out["all_confirmed"])
        self.assertEqual(len(out["checked"]), len(records))
        self.assertTrue(all(c["repeat_applied"] == 4 for c in out["checked"]))

    def test_an_R_equals_1_record_must_also_confirm_repeat_applied_1(self):
        # Missing: refused.
        records, constructions = self._records("omit", 1)
        with self.assertRaises(H.RepeatNotConfirmed):
            H.verify_repeat_confirmation(records, constructions)
        # Present but not 1: refused.
        records, constructions = self._records("honour", 1)
        records[0]["counters"]["repeat_applied"] = 3
        with self.assertRaises(H.RepeatNotConfirmed) as cm:
            H.verify_repeat_confirmation(records, constructions)
        self.assertIn("repeat_applied=3", str(cm.exception))
        # Present and 1: accepted.
        records, constructions = self._records("honour", 1)
        self.assertTrue(H.verify_repeat_confirmation(records, constructions)["all_confirmed"])

    def test_the_run_exits_non_zero_and_writes_no_per_iteration_figure(self):
        """The whole-run behaviour: a refusal returns a non-zero code from
        cmd_run and leaves no results file behind."""
        stub = StubHarness(full_plan(repeat_mode="ignore"))
        out_path = os.path.join(stub.dir, "stage4_harness.json")
        try:
            class A:
                command = stub.command
                fixture = D3_FIXTURE
                allow_other_prompt_set = False
                lengths = contexts = None
                configs = "prefill:16:nocache,prefill:32:nocache"
                out = out_path
                c_results_dir = stub.results_dir
                c_out_stem = "stub"
                correctness = None
                compare = None
                expect_mhz = H.EXPECTED_CLOCK_MHZ
                stage = "stage-4"
                probe_only = False
                preflight_only = False

            # cmd_run drives the subprocess with the inherited environment, so
            # the stub's plan is placed there.
            orig = H.preflight
            prev = os.environ.get("TIE_STUB_PLAN")
            H.preflight = lambda **kw: {"pass": True, "failed_checks": [], "checks": {}}
            os.environ["TIE_STUB_PLAN"] = stub.env["TIE_STUB_PLAN"]
            try:
                rc = H.cmd_run(A())
            finally:
                H.preflight = orig
                if prev is None:
                    os.environ.pop("TIE_STUB_PLAN", None)
                else:
                    os.environ["TIE_STUB_PLAN"] = prev
            # The stub reports repeat_applied = 1 for every configuration while
            # R = 1 was selected, so this particular run is ACCEPTED; what the
            # assertion below pins is that the W3 check ran and that a results
            # file exists only because it passed.
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            conf = doc["w3_construction"]["repeat_confirmation"]
            self.assertTrue(conf["all_confirmed"])
            self.assertIn("W3", conf["work_item"])
            self.assertTrue(all(c["repeat_applied"] == 1 for c in conf["checked"]))
        finally:
            stub.close()

    def test_a_run_whose_driver_omits_the_field_writes_nothing(self):
        """The whole-run behaviour on a refusal: a non-zero code and NO results
        file, so no per-iteration figure survives the run."""
        stub = StubHarness(full_plan(repeat_mode="omit"))
        out_path = os.path.join(stub.dir, "stage4_harness.json")
        try:
            class A:
                command = stub.command
                fixture = D3_FIXTURE
                allow_other_prompt_set = False
                lengths = contexts = None
                configs = "prefill:16:nocache,prefill:32:nocache"
                out = out_path
                c_results_dir = stub.results_dir
                c_out_stem = "stub"
                correctness = None
                compare = None
                expect_mhz = H.EXPECTED_CLOCK_MHZ
                stage = "stage-4"
                probe_only = False
                preflight_only = False

            orig = H.preflight
            prev = os.environ.get("TIE_STUB_PLAN")
            H.preflight = lambda **kw: {"pass": True, "failed_checks": [], "checks": {}}
            os.environ["TIE_STUB_PLAN"] = stub.env["TIE_STUB_PLAN"]
            try:
                rc = H.cmd_run(A())
            finally:
                H.preflight = orig
                if prev is None:
                    os.environ.pop("TIE_STUB_PLAN", None)
                else:
                    os.environ["TIE_STUB_PLAN"] = prev
            self.assertEqual(rc, 4)
            self.assertFalse(os.path.exists(out_path),
                             "a refused run must leave no results file")
        finally:
            stub.close()


class TestW3StopWhenABatchedConstructionIsSelected(unittest.TestCase):
    """DECISION C: if the probe selects R > 1 anywhere, the run stops. The C-side
    loop is not built speculatively and the configuration is not lengthened."""

    def test_a_sub_floor_probe_stops_the_run_before_the_timed_pass(self):
        # A 5 ms prefill probe is below the 10 ms floor, so R = 2 is selected.
        stub = StubHarness(full_plan(medians={16: 5.0}, decode={32: 6500.0}))
        out_path = os.path.join(stub.dir, "stage4_harness.json")
        try:
            class A:
                command = stub.command
                fixture = D3_FIXTURE
                allow_other_prompt_set = False
                lengths = "16"
                contexts = "32"
                configs = "prefill:16:nocache,decode:32:nocache"
                out = out_path
                c_results_dir = stub.results_dir
                c_out_stem = "stub"
                correctness = None
                compare = None
                expect_mhz = H.EXPECTED_CLOCK_MHZ
                stage = "stage-4"
                probe_only = False
                preflight_only = False

            orig = H.preflight
            prev = os.environ.get("TIE_STUB_PLAN")
            H.preflight = lambda **kw: {"pass": True, "failed_checks": [], "checks": {}}
            os.environ["TIE_STUB_PLAN"] = stub.env["TIE_STUB_PLAN"]
            try:
                rc = H.cmd_run(A())
            finally:
                H.preflight = orig
                if prev is None:
                    os.environ.pop("TIE_STUB_PLAN", None)
                else:
                    os.environ["TIE_STUB_PLAN"] = prev
            self.assertEqual(rc, 5)
            self.assertFalse(os.path.exists(out_path))
        finally:
            stub.close()


class TestConfigurationKeysSeparateThePaths(unittest.TestCase):
    """A cache configuration and a no-cache configuration at the same length are
    different configurations and are never compared with each other."""

    def test_the_nocache_key_is_the_stage_3_key_and_the_cache_key_is_not(self):
        self.assertEqual(H.make_configuration_key("prefill", 16, "nocache"), "prefill|L=16")
        self.assertEqual(H.make_configuration_key("prefill", 16, None), "prefill|L=16")
        self.assertEqual(H.make_configuration_key("prefill", 16, "cache"),
                         "prefill|L=16|cache")
        self.assertEqual(H.make_configuration_key("decode", 32, "cache"), "decode|c=32|cache")

    @staticmethod
    def _rec(workload, tokens, path, median, valid=True):
        cnt = ({"tokens": tokens, "context_tokens": tokens} if workload == "prefill"
               else {"context_tokens": tokens, "tokens_generated": 1})
        cnt["repeat_applied"] = 1
        cnt["repeat_requested"] = 1
        ann = {"workload": workload, "prompt_row_id": f"d3_{tokens}",
               "prompt_set_status": "FIXED", "prompt_set_source": "f.tsv",
               ("prefill_path" if workload == "prefill" else "decode_path"): path}
        return {"configuration": f"{workload} {tokens} {path}", "units": "ms",
                "value": median, "counters": cnt, "annotations": ann,
                "statistics": {"median_ms": median, "stddev_pct_of_median": 0.1,
                               "valid": valid, "n": 30}}

    def _harness_records(self, recs):
        return H.build_configuration_records({"records": recs}, {})

    def test_a_cache_configuration_is_never_compared_against_a_nocache_one(self):
        prior = {"records": [self._rec("prefill", 16, "nocache", 1000.0),
                             self._rec("decode", 32, "nocache", 6000.0)],
                 "device": "d", "cxx_compiler": "c", "cxx_flags": "f",
                 "cuda_compiler": "n", "cuda_flags": "a", "cpu_timer": "q",
                 "stage": "stage-3"}
        tmp = tempfile.mkdtemp(prefix="tie_cmp_")
        try:
            prior_path = os.path.join(tmp, "prior.json")
            with open(prior_path, "w", encoding="utf-8") as f:
                json.dump(prior, f)
            # The current run has BOTH paths at each length. The cache members
            # are far faster, which a naive key would read as a huge improvement
            # against the no-cache baseline.
            cur_raw = [self._rec("prefill", 16, "cache", 1005.0),
                       self._rec("prefill", 16, "nocache", 1002.0),
                       self._rec("decode", 32, "cache", 235.0),
                       self._rec("decode", 32, "nocache", 6010.0)]
            cur = self._harness_records(cur_raw)
            doc = {"device": "d", "cxx_compiler": "c", "cxx_flags": "f",
                   "cuda_compiler": "n", "cuda_flags": "a", "cpu_timer": "q"}
            out = H.compare_regression(cur, doc, prior_path)

            compared = {c["configuration_key"] for c in out["comparisons"]}
            self.assertEqual(compared, {"prefill|L=16", "decode|c=32"},
                             "only the no-cache members are comparable against a no-cache "
                             "baseline")
            refused = {r["configuration_key"]: r["reason"] for r in out["refusals"]}
            self.assertIn("prefill|L=16|cache", refused)
            self.assertIn("decode|c=32|cache", refused)
            for key in ("prefill|L=16|cache", "decode|c=32|cache"):
                self.assertIn("does not appear in the prior file", refused[key])
            # And no comparison reports the 25x cached decode change as an
            # improvement against the Stage 3 baseline.
            for c in out["comparisons"]:
                self.assertLess(abs(c["change_pct"]), 5.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_every_comparison_record_carries_the_w14_annotation(self):
        prior = {"records": [self._rec("prefill", 16, "nocache", 1000.0),
                             self._rec("decode", 32, "nocache", 6000.0, valid=False)],
                 "device": "d", "cxx_compiler": "c", "cxx_flags": "f",
                 "cuda_compiler": "n", "cuda_flags": "a", "cpu_timer": "q",
                 "stage": "stage-3"}
        tmp = tempfile.mkdtemp(prefix="tie_cmp_")
        try:
            prior_path = os.path.join(tmp, "prior.json")
            with open(prior_path, "w", encoding="utf-8") as f:
                json.dump(prior, f)
            cur = self._harness_records([self._rec("prefill", 16, "nocache", 1002.0),
                                         self._rec("decode", 32, "nocache", 6010.0),
                                         self._rec("prefill", 32, "cache", 2000.0)])
            doc = {"device": "d", "cxx_compiler": "c", "cxx_flags": "f",
                   "cuda_compiler": "n", "cuda_flags": "a", "cpu_timer": "q"}
            out = H.compare_regression(cur, doc, prior_path)

            self.assertFalse(out["process_set_gated"])
            self.assertIn("W14", out["process_set_note"])
            self.assertIn("9 to 12 percent", out["process_set_limitation"])
            everything = out["comparisons"] + out["refusals"]
            self.assertTrue(everything)
            for rec in everything:
                self.assertIn("process_set_gated", rec)
                self.assertFalse(rec["process_set_gated"])
                self.assertIn("W14", rec["process_set_note"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestMergingChunkedRuns(unittest.TestCase):
    """Stage 4 measured its timed set in chunks after the host's memory-pressure
    reaper killed a single-invocation attempt with nothing written. The merge
    must carry every statistic through untouched and must refuse anything that
    would make the merged set not one measurement."""

    HEADER_FIELDS = {
        "git_commit": "0" * 40, "build_timestamp": "2026-10-05T16:56:04Z",
        "device": "NVIDIA GeForce GTX 1650 Ti sm_75", "cpu_timer": "QueryPerformanceCounter",
        "cxx_compiler": "MSVC 19.44.35229.0", "cxx_flags": "/arch:AVX2 /fp:precise /O2",
        "cuda_compiler": "nvcc 13.1.80", "cuda_flags": "-arch=sm_75",
        "protocol": {"max_stddev_pct_of_median": 5.0},
        "prompt_set": {"status": "FIXED", "decision": "D3", "is_d3_fixed_set": True},
        "preflight": {"pass": True, "failed_checks": [], "checks": {}},
    }

    def _part(self, keys, medians, order, **override):
        recs = []
        for key, med in zip(keys, medians):
            workload = "prefill" if key.startswith("prefill") else (
                "decode" if key.startswith("decode") else "isolated_gemm")
            recs.append({
                "configuration_key": key,
                "workload": workload,
                "configuration": key,
                "prompt_row_id": "d3_16",
                "units": "ms",
                "value": med,
                "raw_samples_ms": [med] * 30,
                "statistics": {"n": 30, "median_ms": med, "min_ms": med, "max_ms": med,
                               "stddev_pct_of_median": 0.2, "valid": True},
                "valid": True,
                "verdict": "VALID",
                "counters": {"repeat_applied": 1, "repeat_requested": 1},
                "annotations": {"workload": workload, "prompt_row_id": "d3_16",
                                "prompt_set_status": "FIXED",
                                "prompt_set_source": "benchmark_prompts.tsv"},
                "construction": {"construction": "single_iteration", "repeat_factor_R": 1},
            })
        doc = dict(self.HEADER_FIELDS)
        doc.update({
            "records": recs,
            "w3_construction": {"probes": [{"workload": "prefill", "tokens": 16}],
                                "selected": {k: {"construction": "single_iteration",
                                                 "repeat_factor_R": 1} for k in keys}},
            "configuration_order": {"order": order},
            "c_layer_run_timestamp_utc": "2026-10-05T12:00:00Z",
            "run_timestamp_utc": "2026-10-05T12:00:01Z",
        })
        doc.update(override)
        return doc

    def test_records_and_raw_samples_are_carried_through_untouched(self):
        p1 = self._part(["prefill|L=16|cache", "prefill|L=16"], [3800.0, 3700.0],
                        ["prefill:16:cache", "prefill:16:nocache"])
        p2 = self._part(["decode|c=32|cache", "decode|c=32"], [235.0, 5925.0],
                        ["decode:32:cache", "decode:32:nocache"])
        doc = H.merge_parts([("a.json", p1), ("b.json", p2)], None, "stage-4", None)

        self.assertEqual(len(doc["records"]), 4)
        got = {r["configuration_key"]: r for r in doc["records"]}
        self.assertEqual(set(got), {"prefill|L=16|cache", "prefill|L=16",
                                    "decode|c=32|cache", "decode|c=32"})
        # Byte-for-byte carry-through of the measured arrays and statistics.
        self.assertEqual(got["decode|c=32|cache"]["raw_samples_ms"], [235.0] * 30)
        self.assertEqual(got["decode|c=32"]["statistics"]["median_ms"], 5925.0)
        self.assertFalse(doc["merged_from_parts"]["statistics_recomputed_by_the_merge"])
        self.assertTrue(doc["merged_from_parts"]["raw_samples_carried_through"])
        self.assertEqual(doc["merged_from_parts"]["n_parts"], 2)
        # Prefill and decode stay separated at the summary level.
        self.assertEqual({r["configuration_key"] for r in doc["summary"]["prefill"]},
                         {"prefill|L=16|cache", "prefill|L=16"})
        self.assertEqual({r["configuration_key"] for r in doc["summary"]["decode"]},
                         {"decode|c=32|cache", "decode|c=32"})
        # The measured order is recorded, and it is NOT one invocation.
        self.assertEqual(doc["configuration_order"]["order"],
                         ["prefill:16:cache", "prefill:16:nocache",
                          "decode:32:cache", "decode:32:nocache"])
        self.assertFalse(doc["configuration_order"]["one_invocation_runs_every_configuration"])
        self.assertIn("SAME chunk", doc["configuration_order"]["note"])

    def test_overlapping_chunks_are_refused_rather_than_double_counted(self):
        p1 = self._part(["prefill|L=16"], [3700.0], ["prefill:16:nocache"])
        p2 = self._part(["prefill|L=16"], [3650.0], ["prefill:16:nocache"])
        with self.assertRaises(ValueError) as cm:
            H.merge_parts([("a.json", p1), ("b.json", p2)], None, "stage-4", None)
        self.assertIn("more than one part file", str(cm.exception))
        self.assertIn("double-count", str(cm.exception))

    def test_parts_from_a_different_build_are_refused(self):
        p1 = self._part(["prefill|L=16"], [3700.0], ["prefill:16:nocache"])
        p2 = self._part(["prefill|L=32"], [7400.0], ["prefill:32:nocache"],
                        build_timestamp="2026-10-05T18:00:00Z")
        with self.assertRaises(ValueError) as cm:
            H.merge_parts([("a.json", p1), ("b.json", p2)], None, "stage-4", None)
        self.assertIn("build_timestamp", str(cm.exception))
        self.assertIn("not one measurement", str(cm.exception))

    def test_the_w3_confirmation_is_rerun_over_the_union(self):
        p1 = self._part(["prefill|L=16"], [3700.0], ["prefill:16:nocache"])
        p2 = self._part(["prefill|L=32"], [7400.0], ["prefill:32:nocache"])
        # One chunk's driver failed to report the confirmation: the MERGE refuses,
        # even though each part on its own looked complete.
        p2["records"][0]["counters"].pop("repeat_applied")
        with self.assertRaises(H.RepeatNotConfirmed) as cm:
            H.merge_parts([("a.json", p1), ("b.json", p2)], None, "stage-4", None)
        self.assertIn("W3", str(cm.exception))
        self.assertIn("prefill|L=32", str(cm.exception))

    def test_an_empty_part_list_is_refused(self):
        with self.assertRaises(ValueError):
            H.merge_parts([], None, "stage-4", None)


class TestProbeAbsenceIsAcceptedOnlyForUnprobeableChunks(unittest.TestCase):
    """The driver probes prefill and decode only. A chunk of isolated GEMM
    configurations has nothing to probe, and that must not look like a driver
    that failed to emit its probe lines."""

    def test_a_gemm_only_chunk_is_allowed_to_produce_no_probe_lines(self):
        stub = StubHarness(full_plan())
        try:
            got = H.run_probe(stub.command, D3_FIXTURE, (), (), env=stub.env,
                              configs=[("gemm", 768, None), ("gemm", 3072, None)])
            self.assertEqual(got, [])
        finally:
            stub.close()

    def test_a_chunk_with_probeable_work_still_requires_probe_lines(self):
        stub = StubHarness(full_plan())
        try:
            with self.assertRaises(RuntimeError) as cm:
                # Probeable work was requested but the driver was given no
                # lengths or contexts to probe, so it emits nothing: an error.
                H.run_probe(stub.command, D3_FIXTURE, (), (), env=stub.env,
                            configs=[("prefill", 16, "cache")])
            self.assertIn("no parseable probe lines", str(cm.exception))
        finally:
            stub.close()

    def test_no_configs_at_all_still_requires_probe_lines(self):
        stub = StubHarness(full_plan())
        try:
            with self.assertRaises(RuntimeError):
                H.run_probe(stub.command, D3_FIXTURE, (), (), env=stub.env, configs=None)
        finally:
            stub.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
