#!/usr/bin/env python3
"""Unit test for bench/machine_state.py. Structural; touches no device."""

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ms = _load("machine_state", REPO_ROOT / "bench" / "machine_state.py")


class TestFingerprintVerify(unittest.TestCase):
    STORED = {
        "os": "Windows 11 10.0.26200",
        "python": "3.14.2",
        "gpu": {"name": "NVIDIA GeForce GTX 1650 Ti", "driver_version": "591.44"},
        "build_flags": {"BENCH_CXX_FLAGS": "/O2 /arch:AVX2"},
        "captured_utc": "2026-09-16T00:00:00Z",
    }

    def test_changed_field_is_detected(self):
        current = {**self.STORED,
                   "gpu": {**self.STORED["gpu"], "driver_version": "600.00"},
                   "captured_utc": "2026-10-01T00:00:00Z"}
        r = ms.verify_fingerprint(current, self.STORED)
        self.assertFalse(r["match"])
        fields = [d["field"] for d in r["differences"]]
        self.assertIn("gpu.driver_version", fields)
        d = next(d for d in r["differences"] if d["field"] == "gpu.driver_version")
        self.assertEqual(d["stored"], "591.44")
        self.assertEqual(d["current"], "600.00")

    def test_identical_fingerprint_matches(self):
        current = {**self.STORED, "captured_utc": "2026-12-25T00:00:00Z"}
        r = ms.verify_fingerprint(current, self.STORED)
        self.assertTrue(r["match"], f"unexpected differences: {r['differences']}")

    def test_changed_compiler_flags_are_detected(self):
        """The flags are frozen for the project; a change voids comparisons."""
        current = {**self.STORED,
                   "build_flags": {"BENCH_CXX_FLAGS": "/O2 /arch:AVX512"}}
        r = ms.verify_fingerprint(current, self.STORED)
        self.assertFalse(r["match"])
        self.assertIn("build_flags.BENCH_CXX_FLAGS",
                      [d["field"] for d in r["differences"]])

    def test_missing_and_added_fields_are_reported(self):
        current = {k: v for k, v in self.STORED.items() if k != "python"}
        current["new_field"] = 1
        r = ms.verify_fingerprint(current, self.STORED)
        fields = {d["field"]: d for d in r["differences"]}
        self.assertEqual(fields["python"]["current"], "<absent>")
        self.assertEqual(fields["new_field"]["stored"], "<absent>")


class TestVramIntegrityDetector(unittest.TestCase):
    def test_clean_buffer_passes(self):
        buf = bytes([0xAA]) * 4096
        r = ms.compare_patterns(buf, buf)
        self.assertTrue(r["ok"])
        self.assertEqual(r["bad_bytes"], 0)
        self.assertIsNone(r["first_bad_offset"])

    def test_deliberately_corrupted_buffer_is_detected(self):
        written = bytearray([0xAA]) * 4096
        read_back = bytearray(written)
        read_back[1234] ^= 0x01          # one flipped bit
        r = ms.compare_patterns(bytes(written), bytes(read_back))
        self.assertFalse(r["ok"])
        self.assertEqual(r["first_bad_offset"], 1234)
        self.assertEqual(r["bad_bytes"], 1)
        self.assertIn("differ on readback", r["reason"])

    def test_truncated_readback_is_detected(self):
        written = bytes([0x55]) * 4096
        r = ms.compare_patterns(written, written[:4000])
        self.assertFalse(r["ok"])
        self.assertIn("length mismatch", r["reason"])


class TestStabilizationTime(unittest.TestCase):
    @staticmethod
    def _series(mhz_values, step=2.0):
        return [{"t_s": i * step, "clocks.sm": f"{m} MHz"}
                for i, m in enumerate(mhz_values)]

    def test_value_is_derived_from_the_series(self):
        # falls 1800 -> 1500 then holds; with window 5 the first sample whose
        # next five are within 30 MHz is the one at t = 8.0 s
        series = self._series([1800, 1740, 1680, 1600, 1500, 1495, 1490,
                               1492, 1488, 1491, 1489])
        r = ms.compute_stabilization_time(series, window=5, band_mhz=30)
        self.assertIsNotNone(r["seconds"])
        self.assertEqual(r["seconds"], 8.0)
        self.assertEqual(r["settled_mhz"], 1500)
        self.assertIsNone(r["reason"])

    def test_an_unsettled_series_returns_none_with_a_reason(self):
        series = self._series([1800, 1400, 1800, 1400, 1800, 1400,
                               1800, 1400, 1800, 1400, 1800])
        r = ms.compute_stabilization_time(series, window=5, band_mhz=30)
        self.assertIsNone(r["seconds"])
        self.assertIn("never held", r["reason"])

    def test_too_few_samples_returns_none_with_a_reason_not_a_default(self):
        r = ms.compute_stabilization_time(self._series([1500, 1500]),
                                          window=5, band_mhz=30)
        self.assertIsNone(r["seconds"])
        self.assertIn("usable clock samples", r["reason"])

    def test_a_series_that_is_stable_from_the_start_returns_zero(self):
        r = ms.compute_stabilization_time(self._series([1500] * 8),
                                          window=5, band_mhz=30)
        self.assertEqual(r["seconds"], 0.0)


class TestSpreadCalculation(unittest.TestCase):
    def test_correct_percentage_over_two_synthetic_runs(self):
        a = {"gpu_bandwidth": 100.0, "gpu_fp32_peak": 1000.0}
        b = {"gpu_bandwidth": 110.0, "gpu_fp32_peak": 1000.0}
        r = ms.spread_percent(a, b)
        # |110-100| / 105 * 100 = 9.523809523809524
        self.assertAlmostEqual(r["per_key_pct"]["gpu_bandwidth"],
                               10.0 / 105.0 * 100.0, places=10)
        self.assertAlmostEqual(r["per_key_pct"]["gpu_fp32_peak"], 0.0, places=12)
        self.assertAlmostEqual(r["max_pct"], 10.0 / 105.0 * 100.0, places=10)
        self.assertAlmostEqual(r["mean_pct"], (10.0 / 105.0 * 100.0) / 2, places=10)

    def test_identical_runs_have_zero_spread(self):
        a = {"x": 42.0, "y": -7.5}
        r = ms.spread_percent(a, dict(a))
        self.assertEqual(r["max_pct"], 0.0)

    def test_keys_present_in_only_one_run_are_reported(self):
        r = ms.spread_percent({"x": 1.0, "only_a": 2.0}, {"x": 1.0, "only_b": 3.0})
        self.assertEqual(sorted(r["keys_missing_from_one_run"]), ["only_a", "only_b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
