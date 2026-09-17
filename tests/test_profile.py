#!/usr/bin/env python3
"""Unit test for bench/profile.py.

Structural: the profiler is never actually invoked. A fake runner supplies the
tool's replies, so the blocked-profiler and missing-metric paths can be tested
without breaking the machine's counter permissions.
"""

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


prof = _load("profile_wrapper", REPO_ROOT / "bench" / "profile.py")

ALL_LOGICAL = list(prof.REQUIRED_COUNTERS) + \
    [f"warp_stall_{r}" for r in prof.STALL_REASONS]


def _metric_listing(metrics):
    lines = ["Device 0", "---------"]
    lines += [f"{m}   Counter   unit   description" for m in metrics]
    return "\n".join(lines)


def _all_base_names():
    bases = set()
    for cands in prof.REQUIRED_COUNTERS.values():
        bases.add(cands[0].split(".")[0])
    for r in prof.STALL_REASONS:
        bases.add(prof.STALL_METRIC_FMT.format(r).split(".")[0])
    return sorted(bases)


def _csv_rows(kernel, pairs):
    head = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
    rows = [f'"0","{kernel}","{m}","{u}","{v}"' for m, u, v in pairs]
    return "\n".join(["==PROF== Connected", head] + rows)


class FakeRunner:
    """Replies to --version, --query-metrics and the profiling run."""

    def __init__(self, available=None, csv=None, run_rc=0, run_stderr=""):
        self.available = available if available is not None else _all_base_names()
        self.csv = csv
        self.run_rc = run_rc
        self.run_stderr = run_stderr
        self.commands = []

    def __call__(self, cmd, timeout=None):
        self.commands.append(cmd)
        if "--version" in cmd:
            return 0, "NVIDIA (R) Nsight Compute\nVersion 2025.4.0.0\n", ""
        if "--query-metrics" in cmd:
            return 0, _metric_listing(self.available), ""
        return self.run_rc, (self.csv or ""), self.run_stderr


def _full_csv(kernel="fake_kernel(float*)"):
    pairs = []
    for logical, cands in prof.REQUIRED_COUNTERS.items():
        pairs.append((cands[0], "unit", "1.5"))
    for r in prof.STALL_REASONS:
        pairs.append((prof.STALL_METRIC_FMT.format(r), "inst", "0.25"))
    return _csv_rows(kernel, pairs)


class TestMetricResolution(unittest.TestCase):
    def test_every_required_counter_appears_in_the_mapping(self):
        res = prof.resolve_metrics(runner=FakeRunner())
        for logical in ALL_LOGICAL:
            self.assertIn(logical, res["mapping"],
                          f"{logical} is missing from the resolved mapping")

    def test_the_contract_counter_set_is_covered(self):
        """Each item the contract names must be reachable from the mapping."""
        mapping = prof.resolve_metrics(runner=FakeRunner())["mapping"]
        self.assertIsNotNone(mapping["achieved_occupancy"])
        self.assertIsNotNone(mapping["theoretical_occupancy_max_warps"])
        self.assertIsNotNone(mapping["dram_read_throughput"])
        self.assertIsNotNone(mapping["dram_write_throughput"])
        self.assertIsNotNone(mapping["l2_hit_rate"])
        self.assertIsNotNone(mapping["shared_memory_bank_conflicts"])
        self.assertIsNotNone(mapping["instructions_executed"])
        self.assertIsNotNone(mapping["duration"])
        self.assertTrue(any(k.startswith("warp_stall_") and v
                            for k, v in mapping.items()),
                        "the warp stall reason breakdown resolved to nothing")

    def test_names_are_resolved_against_the_tool_not_hardcoded(self):
        runner = FakeRunner()
        prof.resolve_metrics(runner=runner)
        self.assertTrue(any("--query-metrics" in c for c in runner.commands),
                        "the wrapper never asked the tool which metrics it has")

    def test_a_metric_absent_from_the_installation_is_marked_unavailable(self):
        available = [b for b in _all_base_names() if b != "lts__t_sector_hit_rate"]
        available = [b for b in available if b != "lts__t_sectors_lookup_hit"]
        res = prof.resolve_metrics(runner=FakeRunner(available=available))
        self.assertIsNone(res["mapping"]["l2_hit_rate"])
        self.assertIn("l2_hit_rate", res["unavailable"])
        self.assertIn("no candidate metric exists", res["unavailable"]["l2_hit_rate"])


class TestProfileOutput(unittest.TestCase):
    def test_all_counters_present_and_populated(self):
        p = prof.profile_binary("fake.exe", runner=FakeRunner(csv=_full_csv()))
        self.assertEqual(len(p["kernels"]), 1)
        counters = next(iter(p["kernels"].values()))["counters"]
        for logical in ALL_LOGICAL:
            self.assertIn(logical, counters, f"{logical} missing from the output")
            self.assertIn(counters[logical]["status"], ("populated", "derived"))
        self.assertEqual(counters["duration"]["value"], 1.5)

    def test_uncollected_counter_is_unavailable_with_its_reason_not_defaulted(self):
        """A blank or a zero would be indistinguishable from a real measurement."""
        pairs = [(c[0], "unit", "1.0") for c in prof.REQUIRED_COUNTERS.values()
                 if c[0] != "lts__t_sector_hit_rate.pct"]
        pairs += [(prof.STALL_METRIC_FMT.format(r), "inst", "0.1")
                  for r in prof.STALL_REASONS]
        p = prof.profile_binary("fake.exe",
                                runner=FakeRunner(csv=_csv_rows("k", pairs)))
        c = next(iter(p["kernels"].values()))["counters"]["l2_hit_rate"]
        self.assertEqual(c["status"], "unavailable")
        self.assertIsNone(c["value"])
        self.assertIsNotNone(c["reason"])
        self.assertIn("not returned by the profiler", c["reason"])

    def test_non_numeric_value_is_unavailable_with_the_raw_text(self):
        pairs = [(c[0], "unit", "1.0") for c in prof.REQUIRED_COUNTERS.values()]
        pairs = [(m, u, "n/a" if m == "gpu__time_duration.sum" else v)
                 for m, u, v in pairs]
        pairs += [(prof.STALL_METRIC_FMT.format(r), "inst", "0.1")
                  for r in prof.STALL_REASONS]
        p = prof.profile_binary("fake.exe",
                                runner=FakeRunner(csv=_csv_rows("k", pairs)))
        c = next(iter(p["kernels"].values()))["counters"]["duration"]
        self.assertEqual(c["status"], "unavailable")
        self.assertIn("non-numeric", c["reason"])
        self.assertIn("n/a", c["reason"])

    def test_metric_missing_from_the_installation_survives_into_the_output(self):
        available = [b for b in _all_base_names()
                     if b not in ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared",)]
        pairs = [(c[0], "unit", "1.0") for c in prof.REQUIRED_COUNTERS.values()]
        pairs += [(prof.STALL_METRIC_FMT.format(r), "inst", "0.1")
                  for r in prof.STALL_REASONS]
        p = prof.profile_binary("fake.exe",
                                runner=FakeRunner(available=available,
                                                  csv=_csv_rows("k", pairs)))
        c = next(iter(p["kernels"].values()))["counters"]["shared_memory_bank_conflicts"]
        self.assertEqual(c["status"], "unavailable")
        self.assertIsNone(c["value"])
        self.assertIn("no candidate metric exists", c["reason"])

    def test_theoretical_occupancy_is_derived_with_its_arithmetic(self):
        pairs = [(c[0], "unit", "1.0") for c in prof.REQUIRED_COUNTERS.values()]
        override = {"launch__occupancy_limit_blocks": "16",
                    "launch__occupancy_limit_warps": "4",
                    "launch__occupancy_limit_registers": "16",
                    "launch__occupancy_limit_shared_mem": "16",
                    "launch__block_size": "256",
                    "sm__maximum_warps_avg_per_active_cycle": "32"}
        pairs = [(m, u, override.get(m, v)) for m, u, v in pairs]
        pairs += [(prof.STALL_METRIC_FMT.format(r), "inst", "0.1")
                  for r in prof.STALL_REASONS]
        p = prof.profile_binary("fake.exe",
                                runner=FakeRunner(csv=_csv_rows("k", pairs)))
        t = next(iter(p["kernels"].values()))["counters"]["theoretical_occupancy_pct"]
        self.assertEqual(t["status"], "derived")
        # min(16,4,16,16)=4 blocks * 256 / 32 = 32 warps; 32/32 * 100 = 100%
        self.assertAlmostEqual(t["value"], 100.0, places=9)
        self.assertIn("min(limits)=4", t["arithmetic"])


class TestFailsLoudly(unittest.TestCase):
    def test_blocked_profiler_raises_with_the_error_text(self):
        runner = FakeRunner(csv="", run_rc=1, run_stderr=(
            "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to "
            "access NVIDIA GPU Performance Counters on the target device."))
        with self.assertRaises(prof.ProfilerUnavailable) as cm:
            prof.profile_binary("fake.exe", runner=runner)
        msg = str(cm.exception)
        self.assertIn("BLOCKED", msg)
        self.assertIn("ERR_NVGPUCTRPERM", msg)
        self.assertIn("Manage GPU Performance Counters", msg)

    def test_empty_output_raises_rather_than_returning_an_empty_result_set(self):
        with self.assertRaises(prof.ProfilerUnavailable) as cm:
            prof.profile_binary("fake.exe", runner=FakeRunner(csv="", run_rc=0))
        self.assertIn("no parseable counter rows", str(cm.exception))

    def test_query_metrics_failure_raises(self):
        class R(FakeRunner):
            def __call__(self, cmd, timeout=None):
                if "--query-metrics" in cmd:
                    return 1, "", "boom"
                return super().__call__(cmd, timeout)
        with self.assertRaises(prof.ProfilerUnavailable) as cm:
            prof.resolve_metrics(runner=R())
        self.assertIn("no metric names", str(cm.exception))

    def test_no_metric_resolving_at_all_raises(self):
        with self.assertRaises(prof.ProfilerUnavailable) as cm:
            prof.resolve_metrics(runner=FakeRunner(available=["something__else"]))
        self.assertIn("not one required metric resolved", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
