#!/usr/bin/env python3
"""Unit test for bench/analyze_variance.py.

Structural, and offline in the strongest sense: it builds its own synthetic
sample arrays and its own synthetic results files, so it asserts properties of
the analysis rather than properties of the machine. It runs in the offline gate
before any Stage 0b measurement exists.

The assertion this file exists for is the third one down: a trimmed statistic is
a diagnostic and can never become a measurement or turn an INVALID run valid.
"""

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


av = _load("analyze_variance", REPO_ROOT / "bench" / "analyze_variance.py")


def _results_doc(records):
    return {"stage": "stage-0", "git_commit": "0" * 40,
            "run_timestamp_utc": "2026-09-17T00:00:00Z",
            "device": "test", "cxx_flags": "/arch:AVX2",
            "records": records}


def _record(cfg, samples, benchmark="synthetic", with_raw=True):
    rec = {"benchmark": benchmark, "configuration": cfg, "units": "GB/s",
           "value": 1.0, "warmup_iterations": 25, "samples_requested": len(samples),
           "statistics": {"n": len(samples), "median_ms": 1.0,
                          "stddev_pct_of_median": 0.0, "valid": True}}
    if with_raw:
        rec["raw_samples_ms"] = samples
    return rec


class TestRobustOutlierDetection(unittest.TestCase):
    """(a) -- is the excess carried by a few extreme samples?"""

    CLEAN = [2.100, 2.102, 2.099, 2.101, 2.100, 2.103, 2.098, 2.101,
             2.100, 2.102, 2.099, 2.100, 2.101, 2.100, 2.102, 2.099,
             2.101, 2.100, 2.103, 2.098]

    def test_a_clean_array_has_no_flagged_outlier(self):
        r = av.mad_outliers(self.CLEAN)
        self.assertEqual(r["n_flagged"], 0,
                         f"flagged {r['flagged']} in a clean array")

    def test_an_injected_outlier_is_flagged_at_its_run_order_position(self):
        samples = list(self.CLEAN)
        samples[13] = 3.400                 # +62% of the median, one sample
        r = av.mad_outliers(samples)
        self.assertGreaterEqual(r["n_flagged"], 1)
        positions = [f["run_order_index"] for f in r["flagged"]]
        self.assertIn(13, positions)
        flagged = next(f for f in r["flagged"] if f["run_order_index"] == 13)
        self.assertEqual(flagged["direction"], "slow")
        self.assertGreater(flagged["pct_of_median"], 50.0)

    def test_the_injected_outlier_carries_the_variance(self):
        samples = list(self.CLEAN)
        samples[13] = 3.400
        shape = av.dispersion_shape(samples, av.mad_outliers(samples))
        self.assertGreater(shape["spike_share_of_variance"], 0.9)

    def test_a_broadly_dispersed_array_is_not_reported_as_spike_carried(self):
        # deliberately wide, with no single dominant sample
        samples = [2.0 + 0.4 * math.sin(i) for i in range(30)]
        shape = av.dispersion_shape(samples, av.mad_outliers(samples))
        self.assertLess(shape["spike_share_of_variance"], 0.5)
        self.assertGreater(shape["iqr_pct_of_median"], 5.0)


class TestStddevPercentComputation(unittest.TestCase):
    """(b) -- the figure this whole analysis classifies against."""

    def test_hand_checked_value_on_a_synthetic_array(self):
        # [1,2,3,4,5]: mean 3, median 3, sum of squared deviations 10,
        # sample variance 10/4 = 2.5, sample stddev sqrt(2.5), and
        # 100 * sqrt(2.5) / 3 = 52.70462766947299 %
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected = 100.0 * math.sqrt(2.5) / 3.0
        self.assertAlmostEqual(av.stddev_pct_of_median(samples), expected, places=12)
        self.assertAlmostEqual(expected, 52.70462766947299, places=10)

    def test_an_identical_array_has_zero_spread(self):
        self.assertEqual(av.stddev_pct_of_median([4.0] * 20), 0.0)

    def test_it_matches_the_c_harness_construction(self):
        """Bessel-corrected divisor n-1, median of an even set as the mean of the
        two middle elements -- the same construction as bench_compute_stats()."""
        samples = [1.0, 2.0, 3.0, 4.0]
        # median = (2+3)/2 = 2.5; mean 2.5; sum sq dev = 2.25+0.25+0.25+2.25 = 5;
        # variance 5/3; stddev sqrt(5/3); pct = 100*sqrt(5/3)/2.5
        self.assertAlmostEqual(av.stddev_pct_of_median(samples),
                               100.0 * math.sqrt(5.0 / 3.0) / 2.5, places=12)


class TestTrimmedStatisticsAreDiagnosticOnly(unittest.TestCase):
    """(c) -- the rule this stage must not break."""

    # median 2.0; one sample at 2.6 pushes the spread over 5% of median
    SAMPLES = [2.0] * 29 + [2.6]

    def test_the_untrimmed_configuration_is_invalid(self):
        v = av.verdict_from(self.SAMPLES)
        self.assertEqual(v["verdict"], "INVALID")
        self.assertGreater(v["stddev_pct_of_median"], 5.0)
        self.assertEqual(v["computed_from"], "untrimmed samples")

    def test_trimming_would_pass_but_the_verdict_stays_invalid(self):
        trimmed = av.trimmed_diagnostics(self.SAMPLES)
        k1 = trimmed["by_k"]["1"]
        # trimming the single large sample leaves an array with zero spread
        self.assertTrue(k1["diagnostic_only__would_be_under_limit"])
        self.assertLess(k1["diagnostic_only__stddev_pct_of_median"], 5.0)
        # and it changes nothing about the verdict
        self.assertEqual(av.verdict_from(self.SAMPLES)["verdict"], "INVALID")

    def test_every_trimmed_number_sits_under_a_diagnostic_key(self):
        trimmed = av.trimmed_diagnostics(self.SAMPLES)
        self.assertTrue(trimmed["diagnostic_only"])
        self.assertIn("DIAGNOSTIC ONLY", trimmed["disclaimer"])
        for k, block in trimmed["by_k"].items():
            self.assertTrue(block["diagnostic_only"], f"k={k}")
            numeric = [key for key, val in block.items()
                       if isinstance(val, (int, float)) and not isinstance(val, bool)]
            for key in numeric:
                self.assertTrue(
                    key.startswith("diagnostic_only__") or key == "k_largest_removed",
                    f"k={k}: numeric key {key!r} is not marked diagnostic")

    def test_no_trimmed_number_appears_in_the_measurement_block(self):
        rec = _record("synthetic cfg", self.SAMPLES)
        row = av.analyze_record(rec, "run1", "synthetic.json")
        measurement = row["measurement"]
        self.assertEqual(measurement["verdict"], "INVALID")
        for key in measurement:
            self.assertNotIn("trim", key.lower())
            self.assertNotIn("diagnostic", key.lower())
        # the measurement statistics are those of the full, untrimmed array
        self.assertEqual(measurement["n"], len(self.SAMPLES))
        self.assertEqual(measurement["max_ms"], max(self.SAMPLES))
        self.assertAlmostEqual(measurement["stddev_pct_of_median"],
                               av.stddev_pct_of_median(self.SAMPLES), places=12)
        # and the trimmed block is present, separately, marked
        self.assertTrue(row["diagnostic_only__trimmed"]["diagnostic_only"])

    def test_an_invalid_configuration_is_listed_as_invalid_in_the_report(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "run1").mkdir()
            (root / "run1" / "synthetic.json").write_text(
                json.dumps(_results_doc([_record("spiky", self.SAMPLES),
                                         _record("clean", [2.0] * 30)])),
                encoding="utf-8")
            report = av.build_report(root)
        self.assertEqual(report["n_invalid"], 1)
        self.assertEqual(report["invalid_configurations"][0]["configuration"], "spiky")
        self.assertIn("DIAGNOSTIC ONLY", report["trimmed_statistics_policy"])


class TestMissingRawSamplesRaises(unittest.TestCase):
    """The analysis is about distribution shape. Summary statistics cannot
    substitute for it, so a file without raw arrays is an error, not a
    degraded answer."""

    def test_a_record_without_raw_samples_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "no_raw.json"
            path.write_text(json.dumps(_results_doc(
                [_record("summary only", [1.0] * 30, with_raw=False)])),
                encoding="utf-8")
            with self.assertRaises(av.MissingRawSamples):
                av.load_records(path)

    def test_an_empty_raw_array_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty_raw.json"
            path.write_text(json.dumps(_results_doc([_record("empty", [])])),
                            encoding="utf-8")
            with self.assertRaises(av.MissingRawSamples):
                av.load_records(path)

    def test_a_file_with_raw_arrays_loads(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ok.json"
            path.write_text(json.dumps(_results_doc([_record("fine", [1.0] * 30)])),
                            encoding="utf-8")
            recs = av.load_records(path)
            self.assertEqual(len(recs), 1)
            self.assertEqual(len(recs[0]["raw_samples_ms"]), 30)


class TestRunOrderTrend(unittest.TestCase):
    """Separates a drift across the run from direction-free interference."""

    def test_a_monotone_ramp_is_detected(self):
        samples = [2.0 + 0.01 * i for i in range(30)]
        t = av.run_order_trend(samples)
        self.assertGreater(t["pearson_r_vs_run_order"], 0.99)
        self.assertGreater(t["second_minus_first_pct"], 0.0)

    def test_a_flat_series_has_no_trend(self):
        t = av.run_order_trend([2.0] * 30)
        self.assertEqual(t["slope_ms_per_sample"], 0.0)
        self.assertEqual(t["pearson_r_vs_run_order"], 0.0)


class TestStage0ResultsAreReadOnly(unittest.TestCase):
    def test_the_report_declares_its_inputs_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "run2").mkdir()
            (root / "run2" / "s.json").write_text(
                json.dumps(_results_doc([_record("c", [1.0] * 30)])), encoding="utf-8")
            before = (root / "run2" / "s.json").read_bytes()
            report = av.build_report(root)
            after = (root / "run2" / "s.json").read_bytes()
        self.assertTrue(report["inputs_read_only"])
        self.assertEqual(before, after, "the analysis modified its input file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
