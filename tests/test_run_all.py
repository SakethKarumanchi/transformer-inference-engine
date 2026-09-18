#!/usr/bin/env python3
"""Unit test for bench/microbench/run_all.py.

Structural, like the C tests: it never launches a real benchmark. The runner is
injected, so a failure can be simulated without breaking a machine.
"""

import importlib.util
import json
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


run_all = _load("run_all", REPO_ROOT / "bench" / "microbench" / "run_all.py")


def _fake_results_doc(name, configs):
    return {
        "stage": "stage-0",
        "git_commit": "deadbeef",
        "device": "fake sm_75",
        "cxx_flags": "/O2",
        "cuda_flags": "-O3",
        "records": [
            {"benchmark": name, "configuration": cfg, "units": "GB/s",
             "value": val, "raw_samples_ms": [1.0] * 20,
             "statistics": {"valid": True, "stddev_pct_of_median": 0.5}}
            for cfg, val in configs.items()
        ],
    }


class TestRegistration(unittest.TestCase):
    def test_all_nine_are_registered(self):
        self.assertEqual(len(run_all.BENCHMARKS), 9)
        names = run_all.BENCHMARK_NAMES
        for expected in ["gpu_bandwidth", "gpu_fp32_peak", "cpu_cache_ladder",
                         "cpu_simd_peak", "host_device_transfer",
                         "cublas_sgemm_ref", "kernel_launch_overhead",
                         "shared_mem_bandwidth", "occupancy_sweep"]:
            self.assertIn(expected, names, f"{expected} is not registered")
        self.assertEqual([b[0] for b in run_all.BENCHMARKS], list(range(1, 10)),
                         "the nine are registered in contract order 1..9")

    def test_every_registered_benchmark_is_run(self):
        calls = []

        def runner(cmd, cwd=None, timeout=None):
            calls.append(Path(cmd[0]).stem)
            return 0, "ok\n", ""

        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            records = {}
            for _o, name, _u in run_all.BENCHMARKS:
                (results / f"{name}.json").write_text(
                    json.dumps(_fake_results_doc(name, {"copy": 100.0})),
                    encoding="utf-8")
                # exe_path points at build/, which need not exist for this test,
                # so call the runner directly the way run_one does.
                rc, out, err = runner([str(results / name)], cwd=None)
                records[name] = {"status": "ok" if rc == 0 else "failed",
                                 "reason": None, "returncode": rc,
                                 "elapsed_s": 0.0, "stdout": out, "stderr": err}
            self.assertEqual(sorted(calls), sorted(run_all.BENCHMARK_NAMES))

            cons = run_all.consolidate(records, results_dir=results)
            self.assertTrue(cons["all_nine_ran"])
            self.assertEqual(cons["failed"], [])
            self.assertEqual(len(cons["benchmarks"]), 9)


class TestFailureHandling(unittest.TestCase):
    def test_failed_benchmark_leaves_no_value(self):
        """A failure must not leave a stale or placeholder value behind."""
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            # A results file from a PREVIOUS run is present on disk...
            (results / "gpu_bandwidth.json").write_text(
                json.dumps(_fake_results_doc("gpu_bandwidth", {"copy": 999.0})),
                encoding="utf-8")
            # ...but this run's attempt failed.
            records = {name: {"status": "ok", "reason": None, "elapsed_s": 1.0}
                       for name in run_all.BENCHMARK_NAMES}
            records["gpu_bandwidth"] = {"status": "failed",
                                        "reason": "exit code 9", "elapsed_s": 0.1}
            for name in run_all.BENCHMARK_NAMES:
                if name != "gpu_bandwidth":
                    (results / f"{name}.json").write_text(
                        json.dumps(_fake_results_doc(name, {"cfg": 1.0})),
                        encoding="utf-8")

            cons = run_all.consolidate(records, results_dir=results)
            entry = cons["benchmarks"]["gpu_bandwidth"]
            self.assertEqual(entry["status"], "failed")
            self.assertIsNone(entry["values"],
                              "the stale 999.0 from the previous run must not appear")
            self.assertIn("exit code 9", entry["reason"])
            self.assertEqual(cons["failed"], ["gpu_bandwidth"])
            self.assertFalse(cons["all_nine_ran"])
            blob = json.dumps(cons)
            self.assertNotIn("999.0", blob,
                             "no placeholder or stale figure reaches the consolidated file")

    def test_success_without_a_results_file_is_reported_as_failed(self):
        with tempfile.TemporaryDirectory() as td:
            records = {"gpu_bandwidth": {"status": "ok", "reason": None, "elapsed_s": 1.0}}
            cons = run_all.consolidate(records, results_dir=Path(td))
            entry = cons["benchmarks"]["gpu_bandwidth"]
            self.assertEqual(entry["status"], "failed")
            self.assertIn("wrote no results file", entry["reason"])
            self.assertIsNone(entry["values"])

    def test_not_run_benchmarks_are_marked_not_run(self):
        with tempfile.TemporaryDirectory() as td:
            cons = run_all.consolidate({}, results_dir=Path(td))
            for name in run_all.BENCHMARK_NAMES:
                self.assertEqual(cons["benchmarks"][name]["status"], "not run")
                self.assertIsNone(cons["benchmarks"][name]["values"])

    def test_invalid_exit_code_is_distinct_from_failure(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "gpu_bandwidth.json").write_text(
                json.dumps(_fake_results_doc("gpu_bandwidth", {"copy": 50.0})),
                encoding="utf-8")
            records = {"gpu_bandwidth": {"status": "invalid",
                                         "reason": "5% rule", "elapsed_s": 1.0}}
            cons = run_all.consolidate(records, results_dir=results)
            self.assertEqual(cons["invalid"], ["gpu_bandwidth"])
            self.assertEqual(cons["failed"], [])
            self.assertIsNotNone(cons["benchmarks"]["gpu_bandwidth"]["values"])


class TestDriftComparison(unittest.TestCase):
    def _consolidated(self, value):
        return {"stage": "stage-0",
                "benchmarks": {"gpu_bandwidth":
                               {"order": 1, "units": "GB/s", "status": "ok",
                                "values": {"copy": {"value": value, "units": "GB/s",
                                                    "valid": True}}}}}

    def test_changed_value_is_flagged(self):
        prior = self._consolidated(100.0)          # the synthetic Stage 0 figures
        current = self._consolidated(90.0)         # 10% lower
        drift = run_all.compare_drift(current, prior, tolerance_pct=5.0)
        self.assertFalse(drift["machine_unchanged"])
        self.assertEqual(len(drift["drifted"]), 1)
        d = drift["drifted"][0]
        self.assertEqual(d["benchmark"], "gpu_bandwidth")
        self.assertEqual(d["stage0_value"], 100.0)
        self.assertEqual(d["current_value"], 90.0)
        self.assertAlmostEqual(d["delta_pct"], 10.0, places=9)

    def test_unchanged_value_is_not_flagged(self):
        drift = run_all.compare_drift(self._consolidated(100.0),
                                      self._consolidated(101.0),
                                      tolerance_pct=5.0)
        self.assertTrue(drift["machine_unchanged"])
        self.assertEqual(drift["drifted"], [])

    def test_appearing_and_disappearing_values_are_reported(self):
        prior = self._consolidated(100.0)
        current = {"benchmarks": {"gpu_bandwidth": {"values": {"other": {"value": 5.0}}}}}
        drift = run_all.compare_drift(current, prior, tolerance_pct=5.0)
        self.assertEqual([d["configuration"] for d in drift["appeared"]], ["other"])
        self.assertEqual([d["configuration"] for d in drift["disappeared"]], ["copy"])
        self.assertFalse(drift["machine_unchanged"])


class TestSharedVersusGlobalFinding(unittest.TestCase):
    def _cons(self, global_v, shared_v):
        b = {}
        if global_v is not None:
            b["gpu_bandwidth"] = {"values": {"float4 grid-stride copy, x":
                                             {"value": global_v}}}
        if shared_v is not None:
            b["shared_mem_bandwidth"] = {"values": {"conflict-free": {"value": shared_v}}}
        return {"benchmarks": b}

    def test_shared_faster_than_global_is_the_expected_finding(self):
        f = run_all.shared_versus_global_finding(self._cons(100.0, 2000.0))
        self.assertTrue(f["available"])
        self.assertAlmostEqual(f["shared_over_global"], 20.0)
        self.assertIn("as expected", f["finding"])

    def test_shared_not_faster_is_reported_as_a_problem_not_a_test_failure(self):
        f = run_all.shared_versus_global_finding(self._cons(2000.0, 100.0))
        self.assertTrue(f["available"])
        self.assertIn("does NOT exceed", f["finding"])

    def test_missing_figure_is_reported_rather_than_assumed(self):
        f = run_all.shared_versus_global_finding(self._cons(100.0, None))
        self.assertFalse(f["available"])
        self.assertIn("missing", f["reason"])


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"")
    return path


class TestStage0bPathGuard(unittest.TestCase):
    """Stage 0b re-runs two benchmarks and must never touch a Stage 0 results
    file. bench/results/ and its run1/ and run2/ subdirectories are evidence and
    are immutable, so every Stage 0b write goes through _guard_stage0b_path()
    and that guard is asserted here rather than left to review."""

    def test_the_two_stage0b_benchmarks_are_the_two_that_failed(self):
        self.assertEqual(run_all.STAGE0B_BENCHMARKS,
                         ["cpu_cache_ladder", "cpu_simd_peak"])
        for name in run_all.STAGE0B_BENCHMARKS:
            self.assertIn(name, run_all.BENCHMARK_NAMES)

    def test_the_protocol_counts_match_stage_0(self):
        self.assertEqual(run_all.STAGE0B_WARMUP, 25)
        self.assertEqual(run_all.STAGE0B_SAMPLES, 30)

    def test_run_directories_are_under_the_stage0b_root(self):
        for run in (1, 2):
            d = run_all.stage0b_run_dir(run)
            self.assertEqual(d.name, f"run{run}")
            self.assertEqual(d.parent, run_all.STAGE0B_DIR.resolve())

    def test_a_stage_0_results_path_is_refused(self):
        for bad in (run_all.RESULTS_DIR,
                    run_all.RESULTS_DIR / "run1",
                    run_all.RESULTS_DIR / "run2" / "cpu_cache_ladder.json",
                    run_all.RESULTS_DIR / "cpu_simd_peak.json",
                    run_all.CONSOLIDATED):
            with self.assertRaises(run_all.Stage0ResultsAreImmutable,
                                   msg=f"{bad} was not refused"):
                run_all._guard_stage0b_path(bad)

    def test_an_out_of_range_run_number_is_refused(self):
        with self.assertRaises(ValueError):
            run_all.stage0b_run_dir(3)

    def test_the_consolidated_file_is_not_the_stage_0_one(self):
        self.assertNotEqual(run_all.STAGE0B_CONSOLIDATED, run_all.CONSOLIDATED)
        self.assertEqual(run_all._guard_stage0b_path(run_all.STAGE0B_CONSOLIDATED),
                         run_all.STAGE0B_CONSOLIDATED.resolve())


class TestStage0bRoutingAndInvalidReporting(unittest.TestCase):
    """The driver must point the child at a Stage 0b directory, label it
    stage-0b, and list every configuration that is still INVALID as invalid --
    never average one away and never silently retry it."""

    INVALID_DOC = {
        "stage": "stage-0b",
        "records": [
            {"benchmark": "cpu_simd_peak",
             "configuration": "vectorised FMA loop, AVX2",
             "units": "GFLOP/s", "value": 43.0,
             "raw_samples_ms": [1.0] * 30,
             "statistics": {"valid": False, "stddev_pct_of_median": 5.6}},
        ],
    }

    def setUp(self):
        self.calls = []
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._saved_dir = run_all.STAGE0B_DIR
        self._saved_exe = run_all.exe_path
        run_all.STAGE0B_DIR = self.root
        run_all.exe_path = lambda n: _touch(self.root / f"{n}.exe")

    def tearDown(self):
        run_all.STAGE0B_DIR = self._saved_dir
        run_all.exe_path = self._saved_exe
        self._tmp.cleanup()

    def _runner(self, cmd, cwd=None, timeout=None, env=None):
        self.calls.append({"cmd": list(cmd), "env": env})
        return 2, "ran\n", ""          # exit 2: ran, at least one INVALID

    def test_the_child_environment_routes_output_and_labels_the_stage(self):
        run_all.run_stage0b(runs=(1,), telemetry=False, runner=self._runner)
        self.assertEqual(len(self.calls), 2)
        for call in self.calls:
            env = call["env"]
            self.assertEqual(env["BENCH_STAGE_ID"], "stage-0b")
            self.assertEqual(Path(env["BENCH_RESULTS_DIR"]).name, "run1")
            self.assertEqual(Path(env["BENCH_RESULTS_DIR"]).parent,
                             self.root.resolve())
            self.assertEqual(call["cmd"][1:], ["25", "30"])

    def test_no_stage_0_results_path_appears_in_any_child_environment(self):
        run_all.run_stage0b(runs=(1, 2), telemetry=False, runner=self._runner)
        stage0 = run_all.RESULTS_DIR.resolve()
        for call in self.calls:
            d = Path(call["env"]["BENCH_RESULTS_DIR"]).resolve()
            self.assertNotEqual(d, stage0)
            self.assertNotIn(d, (stage0 / "run1", stage0 / "run2"))

    def test_invalid_configurations_are_listed_as_invalid(self):
        run1 = self.root / "run1"
        run1.mkdir(parents=True, exist_ok=True)
        for name in run_all.STAGE0B_BENCHMARKS:
            (run1 / f"{name}.json").write_text(json.dumps(self.INVALID_DOC),
                                               encoding="utf-8")
        out = run_all.run_stage0b(runs=(1,), telemetry=False, runner=self._runner)
        self.assertEqual(out["stage"], "stage-0b")
        self.assertEqual(out["reference_run"], 2)
        self.assertEqual(out["n_invalid"], 2)
        for c in out["invalid_configurations"]:
            self.assertAlmostEqual(c["stddev_pct_of_median"], 5.6)
        # reported as "invalid", which is distinct from a failure
        self.assertEqual(out["runs"]["run1"]["cpu_simd_peak"]["status"], "invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
