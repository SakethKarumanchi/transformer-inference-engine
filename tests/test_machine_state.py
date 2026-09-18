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


class TestTelemetrySourceClassification(unittest.TestCase):
    """Stage 0b Phase 2. HARDWARE.md 5.3 records that Stage 0's CPU telemetry was
    static and unusable -- a constant 2496 MHz and a constant package temperature
    across every sample of both runs while CPU load varied between 4% and 36%.
    The classifier exists so that failure cannot recur silently: a source is
    reported live only when its own readings move."""

    @staticmethod
    def _flat(value, n=15):
        return [value] * n

    def test_a_source_whose_readings_never_change_is_static(self):
        r = ms.classify_telemetry_source(
            "fake_nominal_mhz",
            self._flat(2496), self._flat(2496), self._flat(2496),
            per_probe_us=6.0, unit="MHz")
        self.assertFalse(r["live"])
        self.assertEqual(r["verdict"], "static")
        self.assertEqual(r["constant_value"], 2496)
        self.assertIn("static nominal read", r["reason"])

    def test_a_source_that_moves_under_load_and_returns_is_live(self):
        idle = [172.1, 171.2, 173.0, 172.7, 169.9]
        load = [143.7, 141.4, 142.7, 140.0, 141.4]
        recovery = [167.3, 170.3, 170.9, 168.5, 170.0]
        r = ms.classify_telemetry_source("fake_live", idle, load, recovery,
                                         per_probe_us=12.5, unit="percent")
        self.assertTrue(r["live"])
        self.assertEqual(r["verdict"], "live")
        self.assertIsNone(r["reason"])
        self.assertGreater(r["load_shift_pct_of_idle_median"], 2.0)
        self.assertEqual(r["phases"]["load"]["min"], 140.0)
        self.assertEqual(r["phases"]["idle"]["max"], 173.0)

    def test_a_source_that_barely_moves_is_not_called_live(self):
        idle = [100.0, 100.1, 100.0, 99.9, 100.0]
        load = [100.2, 100.1, 100.0, 100.1, 100.0]
        recovery = [100.0, 100.0, 100.1, 99.9, 100.0]
        r = ms.classify_telemetry_source("fake_barely", idle, load, recovery,
                                         per_probe_us=6.0, unit="percent")
        self.assertFalse(r["live"])
        self.assertEqual(r["verdict"], "static")
        self.assertIn("below the", r["reason"])

    def test_the_per_probe_cost_is_recorded_on_every_verdict(self):
        for idle, load, rec, cost in (
                (self._flat(1), self._flat(1), self._flat(1), 6.0),
                ([1, 2, 3], [9, 9, 9], [1, 2, 3], 12.5),
                ([], [], [], 1360246.0)):
            r = ms.classify_telemetry_source("s", idle, load, rec, per_probe_us=cost)
            self.assertIn("per_probe_us", r)
            self.assertEqual(r["per_probe_us"], cost)

    def test_a_phase_with_no_readings_is_unavailable_not_invented(self):
        r = ms.classify_telemetry_source("s", [1.0, 2.0], [], [1.0],
                                         per_probe_us=6.0)
        self.assertEqual(r["verdict"], "unavailable")
        self.assertFalse(r["live"])
        self.assertIn("load", r["reason"])
        self.assertNotIn("phases", r)

    def test_the_probe_cost_helper_returns_a_real_median(self):
        calls = {"n": 0}

        def fake_probe():
            calls["n"] += 1

        cost = ms.measure_probe_cost_us(fake_probe, calls=25)
        self.assertEqual(calls["n"], 25)
        self.assertIsInstance(cost, float)
        self.assertGreaterEqual(cost, 0.0)


class TestNoTelemetryInsideATimedBracket(unittest.TestCase):
    """BENCHMARK_PROTOCOL.md: timing brackets computation only. Telemetry is
    sampled outside the bracket under all circumstances.

    Asserted by reading the two CPU benchmark sources and inspecting what lies
    between each bracket's t0 and t1, rather than by trusting a comment. The
    same scan also proves allocation and initialisation are outside the bracket.
    """

    SOURCES = ["cpu_cache_ladder.c", "cpu_simd_peak.c"]

    # Anything that talks to the OS, allocates, or reads a sensor. A timed
    # bracket may contain the computation and the monotonic counter, nothing more.
    FORBIDDEN = [
        "PdhCollectQueryData", "PdhGetFormattedCounterValue", "CallNtPowerInformation",
        "Get-Counter", "Get-CimInstance", "nvidia-smi", "telemetry",
        "GetSystemTime", "GetTickCount", "QueryPerformanceCounter",
        "malloc(", "calloc(", "realloc(", "free(", "_aligned_malloc", "_aligned_free",
        "fopen", "fprintf", "printf", "snprintf", "memset(", "getenv",
        "SetThreadAffinityMask", "SetThreadPriority", "SetPriorityClass",
        "bench_pin_current_thread", "bench_restore_current_thread",
        "bench_write_results", "Sleep(",
    ]

    @staticmethod
    def _brackets(text):
        """Every region between a t0 assignment and the next t1 assignment."""
        out = []
        start = 0
        while True:
            i = text.find("double t0 = bench_cpu_time_seconds();", start)
            if i < 0:
                break
            j = text.find("double t1 = bench_cpu_time_seconds();", i)
            assert j > i, "a timed bracket opened and was never closed"
            out.append(text[i + len("double t0 = bench_cpu_time_seconds();"):j])
            start = j + 1
        return out

    def test_every_timed_bracket_is_free_of_telemetry_and_allocation(self):
        for name in self.SOURCES:
            path = REPO_ROOT / "bench" / "microbench" / name
            text = path.read_text(encoding="utf-8")
            brackets = self._brackets(text)
            self.assertGreaterEqual(len(brackets), 1,
                                    f"{name}: no timed bracket found")
            for k, body in enumerate(brackets):
                for token in self.FORBIDDEN:
                    self.assertNotIn(
                        token, body,
                        f"{name} timed bracket {k} contains {token!r}; timing "
                        f"brackets computation only")

    def test_thread_placement_is_applied_outside_every_bracket(self):
        """It must be present in the file -- Stage 0b added it -- and outside
        every timed region, which the previous test covers."""
        for name in self.SOURCES:
            text = (REPO_ROOT / "bench" / "microbench" / name).read_text(encoding="utf-8")
            self.assertIn("bench_pin_current_thread(", text,
                          f"{name}: Stage 0b thread placement is missing")
            self.assertIn("bench_restore_current_thread()", text,
                          f"{name}: thread placement is never restored")

    def test_the_sampler_declares_it_is_outside_the_bracket(self):
        self.assertIn("sampled_inside_any_timed_bracket",
                      (REPO_ROOT / "bench" / "machine_state.py").read_text(encoding="utf-8"))


class TestCoreMappingAndSamplerAffinity(unittest.TestCase):
    """Operator decision 2: the telemetry sampler must be pinned clear of the
    measured thread's logical CPU AND of that CPU's SMT sibling, which shares the
    physical core's execution ports and L1d. The mapping is QUERIED, not assumed:
    the conventional interleaving is a convention, not a guarantee."""

    SYNTHETIC = {
        "available": True,
        "n_physical_cores": 4,
        "n_logical_cpus": 8,
        "logical_to_physical": {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
        "smt_siblings": {0: [1], 1: [0], 2: [3], 3: [2],
                         4: [5], 5: [4], 6: [7], 7: [6]},
    }

    def test_the_sibling_of_an_excluded_cpu_is_also_excluded(self):
        r = ms.sampler_affinity_mask((0, 2), self.SYNTHETIC)
        self.assertTrue(r["available"])
        self.assertEqual(r["excluded_logical_cpus"], [0, 1, 2, 3])
        self.assertEqual(r["allowed_logical_cpus"], [4, 5, 6, 7])
        self.assertEqual(r["mask"], 0b11110000)
        self.assertEqual(r["mask_hex"], "0xf0")

    def test_the_mask_names_only_allowed_cpus(self):
        r = ms.sampler_affinity_mask((0, 2), self.SYNTHETIC)
        for cpu in r["excluded_logical_cpus"]:
            self.assertEqual((r["mask"] >> cpu) & 1, 0, f"cpu {cpu} is in the mask")
        for cpu in r["allowed_logical_cpus"]:
            self.assertEqual((r["mask"] >> cpu) & 1, 1, f"cpu {cpu} is not in the mask")

    def test_a_non_conventional_mapping_is_honoured_not_assumed(self):
        """If this CPU paired 0 with 4 instead of 0 with 1, the mask must follow
        the mapping rather than the convention."""
        odd = {"available": True, "n_physical_cores": 4, "n_logical_cpus": 8,
               "logical_to_physical": {0: 0, 4: 0, 1: 1, 5: 1,
                                       2: 2, 6: 2, 3: 3, 7: 3},
               "smt_siblings": {0: [4], 4: [0], 1: [5], 5: [1],
                                2: [6], 6: [2], 3: [7], 7: [3]}}
        r = ms.sampler_affinity_mask((0, 2), odd)
        self.assertEqual(r["excluded_logical_cpus"], [0, 2, 4, 6])
        self.assertEqual(r["allowed_logical_cpus"], [1, 3, 5, 7])

    def test_excluding_everything_is_refused_rather_than_returning_zero(self):
        r = ms.sampler_affinity_mask((0, 2, 4, 6), self.SYNTHETIC)
        self.assertFalse(r["available"])
        self.assertIn("nowhere to run", r["reason"])
        self.assertNotIn("mask", r)

    def test_an_unavailable_mapping_propagates_its_reason(self):
        r = ms.sampler_affinity_mask((0, 2),
                                     {"available": False, "reason": "no such API"})
        self.assertFalse(r["available"])
        self.assertEqual(r["reason"], "no such API")

    def test_the_live_mapping_is_self_consistent(self):
        m = ms.logical_core_mapping()
        if not m.get("available"):
            self.skipTest(f"mapping unavailable: {m.get('reason')}")
        self.assertGreaterEqual(m["n_physical_cores"], 1)
        self.assertEqual(m["n_logical_cpus"],
                         sum(len(c["logical_cpus"]) for c in m["physical_cores"]))
        for cpu, sibs in m["smt_siblings"].items():
            for sib in sibs:
                self.assertEqual(m["logical_to_physical"][sib],
                                 m["logical_to_physical"][cpu],
                                 "a sibling must sit on the same physical core")


class TestPowerSource(unittest.TestCase):
    """BENCHMARK_PROTOCOL.md 3: a run taken on battery is INVALID outright, so
    this is a gate rather than a note."""

    def test_the_power_source_is_reported_with_the_raw_status(self):
        p = ms.power_source()
        self.assertIn("on_ac", p)
        self.assertIn("protocol", p)
        self.assertIn("INVALID", p["protocol"])
        if p.get("ac_line_status") is not None:
            self.assertIn(p["ac_line_status_text"],
                          ("on AC", "on battery", "unknown", "unrecognised"))

    def test_an_unknown_status_is_not_assumed_to_be_ac(self):
        """ACLineStatus 255 means unknown. The contract is that on_ac is None
        there, never True -- asserted on the parsing rule, since the live machine
        cannot be made to report 255 on demand."""
        p = ms.power_source()
        if p.get("ac_line_status") == 255:
            self.assertIsNone(p["on_ac"])
        else:
            self.assertEqual(p["on_ac"], p.get("ac_line_status") == 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
