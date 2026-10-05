"""test_correctness.py -- unit tests for bench/correctness.py.

WHAT IS REDUCED HERE, AND WHY. Every array in this file is SYNTHETIC and every
token count is reduced: 2 positions against a vocabulary of 8, where the real
gate runs 128 positions against a vocabulary of 50257. The real engine is NEVER
invoked, and neither is the PyTorch oracle. One prefill at the longest D3 length
is tens of seconds of engine time and the oracle holds over a gigabyte resident,
so driving either from a unit test would blow the 300 s per-test cap and would
test the engine rather than the statistics. What is tested here is the
ARITHMETIC: every figure asserted below is hand-computed in the test itself, so
a wrong statistic fails against a number a reader can check by hand rather than
against this module's own output.

The one place a real artifact is read is the Stage 2 correctness results file.
The value compared against is READ FROM THAT FILE, never hardcoded here.

Runs under the project .venv (Python 3.14.2, numpy 2.5.3).
"""

import importlib.util
import json
import math
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE2_CORRECTNESS = os.path.join(REPO_ROOT, "bench", "results", "stage2",
                                  "stage2_correctness.json")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cx = _load("correctness", os.path.join(REPO_ROOT, "bench", "correctness.py"))


# ---------------------------------------------------------------- fixtures ----
# 2 positions x 8 vocabulary entries. Two elements differ and the rest are
# identical, so every statistic below is computable by hand.
#
#   reference row 0 : 1 2 3 4 5 6 7 8          argmax = index 7
#   reference row 1 : 8 7 6 5 4 3 2 1          argmax = index 0
#   engine    row 0 : index 1 is 2.5, not 2.0  -> absolute difference 0.50
#   engine    row 1 : index 7 is 1.25, not 1.0 -> absolute difference 0.25
#
# The 16 absolute differences are therefore fourteen zeros, one 0.25 and one
# 0.50. Neither perturbation touches an argmax, so top-1 and top-5 agreement are
# both complete and the only thing moving is the magnitude.
REF = np.array([[1., 2., 3., 4., 5., 6., 7., 8.],
                [8., 7., 6., 5., 4., 3., 2., 1.]], dtype=np.float32)


def engine_pair():
    eng = REF.copy()
    eng[0, 1] = 2.5
    eng[1, 7] = 1.25
    return eng


class TestStatisticsAgainstHandArithmetic(unittest.TestCase):
    """Every expected value in this class is arithmetic written out, not a figure
    recorded from a previous run of the code under test."""

    def setUp(self):
        self.stats = cx.divergence_statistics(engine_pair(), REF, token_ids=[11, 22],
                                              label="hand-computed case")

    def test_maximum_and_its_location(self):
        # The largest difference is 0.50, at position 0, token id 1.
        self.assertEqual(self.stats["max_abs_diff"], 0.5)
        self.assertEqual(self.stats["max_abs_diff_position"], 0)
        self.assertEqual(self.stats["max_abs_diff_token_id"], 1)

    def test_mean_median_rms(self):
        # mean  = (0.50 + 0.25) / 16 = 0.75 / 16 = 0.046875
        self.assertAlmostEqual(self.stats["mean_abs_diff"], 0.75 / 16, places=12)
        # median of fourteen zeros, one 0.25 and one 0.50, n = 16: the two middle
        # elements of the sorted set are both 0, so the median is 0.
        self.assertEqual(self.stats["median_abs_diff"], 0.0)
        # rms = sqrt((0.50^2 + 0.25^2) / 16) = sqrt(0.3125 / 16) = sqrt(0.01953125)
        self.assertAlmostEqual(self.stats["rms_abs_diff"], math.sqrt(0.3125 / 16), places=12)
        self.assertAlmostEqual(self.stats["rms_abs_diff"], 0.1397542485937369, places=12)

    def test_each_percentile(self):
        # numpy's linear interpolation on the sorted 16-element set
        #   a = [0]*14 + [0.25, 0.50],  index = q * (n - 1) = q * 15
        # p50   : 7.500  -> a[7]  = 0, a[8] = 0                  -> 0
        # p90   : 13.500 -> a[13] = 0,    a[14] = 0.25  -> 0 + 0.500*0.25 = 0.125
        # p99   : 14.850 -> a[14] = 0.25, a[15] = 0.50  -> 0.25 + 0.850*0.25 = 0.4625
        # p99.9 : 14.985 -> a[14] = 0.25, a[15] = 0.50  -> 0.25 + 0.985*0.25 = 0.49625
        p = self.stats["abs_diff_percentiles"]
        self.assertEqual(p["p50"], 0.0)
        self.assertAlmostEqual(p["p90"], 0.125, places=12)
        self.assertAlmostEqual(p["p99"], 0.4625, places=12)
        self.assertAlmostEqual(p["p99_9"], 0.49625, places=12)
        self.assertEqual(p["max"], 0.5)

    def test_shape_and_passthrough_fields(self):
        self.assertEqual(self.stats["positions"], 2)
        self.assertEqual(self.stats["vocab_size"], 8)
        self.assertEqual(self.stats["token_ids"], [11, 22])
        self.assertEqual(self.stats["rel_diff_denominator"], "|reference| + 1e-6, elementwise")


class TestMarginComputation(unittest.TestCase):
    """The top-1 / top-2 margin of the REFERENCE, hand-computed."""

    def test_margin_matches_hand_computed_example(self):
        # row 0: the two largest reference logits are 5.0 and 4.0 -> margin 1.00
        # row 1: the two largest reference logits are 10.0 and 9.75 -> margin 0.25
        ref = np.array([[0.0, 1.0, 5.0, 2.0, 3.0, 4.0, -1.0, -2.0],
                        [10.0, 9.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
        stats = cx.divergence_statistics(ref, ref, label="margins")
        m = stats["reference_top1_top2_margin"]
        self.assertAlmostEqual(m["per_position"][0], 1.0, places=6)
        self.assertAlmostEqual(m["per_position"][1], 0.25, places=6)
        self.assertAlmostEqual(m["min"], 0.25, places=6)
        self.assertAlmostEqual(m["median"], 0.625, places=6)   # (0.25 + 1.00) / 2
        self.assertAlmostEqual(m["max"], 1.0, places=6)
        self.assertEqual(m["argmin_position"], 1)

    def test_margin_over_divergence_ratio(self):
        # Both reference rows of REF have top-1 8 and top-2 7, so every margin is
        # 1.0 and the worst divergence is 0.50: the ratio is 1.0 / 0.50 = 2.0.
        stats = cx.divergence_statistics(engine_pair(), REF)
        self.assertAlmostEqual(stats["reference_top1_top2_margin"]["min"], 1.0, places=6)
        self.assertAlmostEqual(stats["margin_min_over_max_abs_diff"], 2.0, places=6)


class TestRelativeFloor(unittest.TestCase):
    """The relative denominator is |reference| + 1e-6 elementwise, and the count
    of elements whose |reference| falls below that floor is reported."""

    def test_count_below_floor_is_reported_and_correct(self):
        # Row 0 carries three reference values with |r| < 1e-6: 0.0, 5e-7, -3e-7.
        # Row 1 carries none. The expected count is therefore exactly 3.
        ref = np.array([[0.0, 5e-7, -3e-7, 1.0, 2.0, 3.0, 4.0, 5.0],
                        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
        eng = ref.copy()
        eng[0, 0] = 1e-6
        stats = cx.divergence_statistics(eng, ref)
        self.assertEqual(stats["elements_below_rel_floor"], 3)
        # The only differing element is position 0 token 0: |1e-6 - 0| over a
        # denominator of |0| + 1e-6, which is 1.0 up to the float32 rounding of
        # 1e-6 itself (float32(1e-6) = 9.9999999747e-07).
        self.assertAlmostEqual(stats["max_rel_diff"], 1.0, places=6)

    def test_no_elements_below_floor_on_ordinary_logits(self):
        stats = cx.divergence_statistics(engine_pair(), REF)
        self.assertEqual(stats["elements_below_rel_floor"], 0)


class TestTopKAgreement(unittest.TestCase):
    def test_complete_agreement_is_reported_per_position(self):
        stats = cx.divergence_statistics(engine_pair(), REF)
        self.assertEqual(stats["top1_agreement_positions"], 2)
        self.assertEqual(stats["top1_agreement_fraction"], 1.0)
        self.assertEqual(stats["top5_set_agreement_positions"], 2)
        self.assertEqual(stats["top5_set_agreement_fraction"], 1.0)
        self.assertEqual(stats["top1_disagreeing_positions"], [])

    def test_single_position_top1_disagreement_is_caught(self):
        # Position 1 only: the engine puts 9.0 where the reference's argmax 8.0
        # sits at index 0, at index 3 instead. Position 0 is left alone, so a
        # per-position computation must report exactly one disagreement.
        eng = REF.copy()
        eng[1, 3] = 9.0
        stats = cx.divergence_statistics(eng, REF)
        self.assertEqual(stats["top1_agreement_positions"], 1)
        self.assertEqual(stats["top1_agreement_fraction"], 0.5)
        self.assertEqual(stats["top1_disagreeing_positions"], [1])

    def test_single_position_top5_set_disagreement_is_caught(self):
        # The reference top-5 set at row 1 is {0, 1, 2, 3, 4} (logits 8..4). The
        # engine promotes index 7 past index 4, so the SET changes while the
        # argmax does not: top-1 still agrees everywhere and top-5 does not.
        eng = REF.copy()
        eng[1, 7] = 4.5
        stats = cx.divergence_statistics(eng, REF)
        self.assertEqual(stats["top1_agreement_positions"], 2)
        self.assertEqual(stats["top5_set_agreement_positions"], 1)
        self.assertEqual(stats["top5_disagreeing_positions"], [1])


class TestD2Gate(unittest.TestCase):
    def test_identical_pair_passes_with_all_three_conditions_true(self):
        res = cx.check(REF, REF, tolerance=1e-3, label="identical", greedy_match=True)
        self.assertEqual(res["verdict"], "PASS")
        conds = res["gate"]["conditions"]
        self.assertTrue(conds["max_abs_diff_within_tolerance"]["pass"])
        self.assertTrue(conds["top1_agreement_at_every_position"]["pass"])
        self.assertTrue(conds["greedy_sequence_matches_reference"]["pass"])
        self.assertEqual(res["statistics"]["max_abs_diff"], 0.0)

    def test_exactly_at_the_boundary_passes(self):
        # 0.5 is exact in binary, so the difference is exactly the tolerance.
        eng = REF.copy()
        eng[0, 1] = np.float32(2.5)
        self.assertEqual(float(np.abs(eng[0, 1] - REF[0, 1])), 0.5)
        res = cx.check(eng, REF, tolerance=0.5, greedy_match=True)
        self.assertEqual(res["statistics"]["max_abs_diff"], 0.5)
        self.assertEqual(res["verdict"], "PASS")

    def test_one_representable_step_above_the_boundary_fails(self):
        eng = REF.copy()
        eng[0, 1] = np.nextafter(np.float32(2.5), np.float32(np.inf))
        diff = float(np.abs(np.float64(eng[0, 1]) - np.float64(REF[0, 1])))
        self.assertGreater(diff, 0.5)
        res = cx.check(eng, REF, tolerance=0.5, greedy_match=True)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertFalse(res["gate"]["conditions"]["max_abs_diff_within_tolerance"]["pass"])

    def test_fail_is_returned_not_raised(self):
        eng = REF.copy()
        eng[0, 1] = 99.0
        try:
            res = cx.check(eng, REF, tolerance=1e-3, greedy_match=True)
        except Exception as exc:                                  # noqa: BLE001
            self.fail(f"a FAIL must be returned, not raised; got {type(exc).__name__}: {exc}")
        self.assertEqual(res["verdict"], "FAIL")
        self.assertIn("offending_configurations",
                      res["gate"]["conditions"]["max_abs_diff_within_tolerance"])

    def test_top1_failure_alone_is_a_fail_inside_tolerance(self):
        # A top-1 flip can happen while the magnitudes stay small, and the gate
        # must fail on it: condition 2 is independent of condition 1.
        # Row 1's reference argmax is index 0 at 9.00005, with index 1 just below
        # it at 9.0. The engine lifts index 1 to 9.0001, which is above 9.00005,
        # so the argmax MOVES while the largest absolute difference is only
        # 1.0e-04 -- comfortably inside a 1e-03 tolerance. Both values are
        # distinctly representable in float32, whose spacing near 9 is 7.6e-07.
        ref = np.array([[1.0, 1.0001, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        [9.00005, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
        eng = ref.copy()
        eng[1, 1] = 9.0001                        # the argmax moves from index 0 to index 1
        res = cx.check(eng, ref, tolerance=1e-3, greedy_match=True)
        self.assertLess(res["statistics"]["max_abs_diff"], 1e-3)
        self.assertEqual(res["statistics"]["top1_disagreeing_positions"], [1])
        self.assertEqual(res["verdict"], "FAIL")
        self.assertTrue(res["gate"]["conditions"]["max_abs_diff_within_tolerance"]["pass"])
        self.assertFalse(res["gate"]["conditions"]["top1_agreement_at_every_position"]["pass"])

    def test_unsupplied_greedy_condition_is_not_treated_as_satisfied(self):
        res = cx.check(REF, REF, tolerance=1e-3)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertFalse(res["gate"]["conditions"]["greedy_sequence_matches_reference"]["pass"])

    def test_tolerance_is_required(self):
        with self.assertRaises(ValueError):
            cx.apply_d2_gate([cx.divergence_statistics(REF, REF)], None)


class TestToleranceIsReadNotHardcoded(unittest.TestCase):
    def test_tolerance_comes_from_the_protocol_document(self):
        value = cx.read_tolerance()
        self.assertIsInstance(value, float)
        self.assertGreater(value, 0.0)
        # The value must actually appear in the document, which is the whole point
        # of reading it rather than duplicating it in code.
        with open(os.path.join(REPO_ROOT, "BENCHMARK_PROTOCOL.md"), "r",
                  encoding="utf-8") as f:
            self.assertIn(cx.TOLERANCE_TOKEN, f.read())

    def test_a_document_without_the_value_is_an_error_not_a_default(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                         encoding="utf-8") as f:
            f.write("## 5. Correctness gate\n\nno threshold here\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                cx.read_tolerance(path)
        finally:
            os.unlink(path)


class TestAgainstTheStage2Artifact(unittest.TestCase):
    """The Stage 2 correctness file is a real artifact, and the figure compared
    against is READ FROM IT rather than copied into this test."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(STAGE2_CORRECTNESS):
            raise unittest.SkipTest(f"{STAGE2_CORRECTNESS} is not present")
        with open(STAGE2_CORRECTNESS, "r", encoding="utf-8") as f:
            cls.doc = json.load(f)
        # The as-stored configurations are the ones that measured the engine; the
        # TRANSPOSED one is a deliberate negative control and is excluded.
        cls.configs = [c for c in cls.doc["configurations"]
                       if c.get("cproj_reading") == "as-stored"]

    def test_the_file_carries_the_statistics_this_module_produces(self):
        produced = set(cx.divergence_statistics(engine_pair(), REF, token_ids=[1, 2]))
        for c in self.configs:
            missing = set(c) - produced - {"cproj_reading"}
            self.assertEqual(
                missing, set(),
                f"{c['label']} carries fields this module does not produce: {sorted(missing)}")

    def test_reported_maximum_divergence_equals_the_stored_value(self):
        for c in self.configs:
            stored = c["max_abs_diff"]                 # read, never hardcoded
            # A synthetic pair constructed to diverge by exactly the stored
            # amount must be reported as diverging by exactly that amount.
            ref = np.zeros((2, 8), dtype=np.float32)
            eng = ref.copy()
            eng[0, 1] = np.float32(stored)
            stats = cx.divergence_statistics(eng, ref, label=c["label"])
            self.assertEqual(stats["max_abs_diff"], float(np.float32(stored)))
            self.assertEqual(stats["max_abs_diff_position"], 0)
            self.assertEqual(stats["max_abs_diff_token_id"], 1)

    def test_the_stored_margin_ratio_is_reproduced_from_the_stored_parts(self):
        for c in self.configs:
            expected = c["margin_min_over_max_abs_diff"]
            computed = c["reference_top1_top2_margin"]["min"] / c["max_abs_diff"]
            self.assertAlmostEqual(computed, expected, places=9)

    def test_stage2_adopted_no_tolerance(self):
        self.assertFalse(self.doc["tolerance"]["adopted"])
        self.assertEqual(self.doc["tolerance"]["owning_stage"], 3)


class TestLogitDumpParser(unittest.TestCase):
    def test_round_trip_through_the_binary_format(self):
        import struct
        import tempfile
        arr = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"TIE2LOGI")
            f.write(struct.pack("<ii", 2, 8))
            f.write(arr.tobytes())
            path = f.name
        try:
            back = cx.read_logit_dump(path)
            self.assertEqual(back.shape, (2, 8))
            self.assertEqual(back.dtype, np.float32)
            np.testing.assert_array_equal(back, arr)
        finally:
            os.unlink(path)

    def test_bad_magic_is_rejected(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"NOTALOGI" + b"\0" * 8)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                cx.read_logit_dump(path)
        finally:
            os.unlink(path)


class TestPromptSetIdentity(unittest.TestCase):
    def test_the_d3_fixture_reports_itself_as_fixed_with_exact_counts(self):
        rows = cx.read_prompt_fixture(cx.D3_FIXTURE)
        self.assertEqual([r["target_tokens"] for r in rows], [16, 32, 64, 128])
        for r in rows:
            self.assertEqual(r["target_tokens"], r["verified_tokens"])
        identity = cx.prompt_set_identity(cx.D3_FIXTURE)
        self.assertEqual(identity["status"], "FIXED")
        self.assertEqual(identity["decision"], "D3")

    def test_the_stage2_placeholder_fixture_still_reports_itself_as_placeholder(self):
        path = os.path.join(REPO_ROOT, "tests", "fixtures",
                            "stage2_placeholder_prompts.tsv")
        self.assertEqual(cx.prompt_set_identity(path)["status"], "PLACEHOLDER")


if __name__ == "__main__":
    unittest.main(verbosity=2)
