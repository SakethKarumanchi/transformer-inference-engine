#!/usr/bin/env python3
"""Unit test for reference/stage1_oracle.py -- the Stage 1 reference side.

Registered with CTest under the SEPARATE oracle interpreter,
.venv-oracle/Scripts/python.exe, not the project .venv: the oracle libraries
(tokenizers, safetensors) live only in that environment, and the project .venv
is a frozen fingerprint value that nothing is installed into.

Nothing here is skipped. If the oracle environment or the shipped artifacts are
missing, this test FAILS and says which, because a silently skipped reference
check is indistinguishable from a passing one.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = REPO_ROOT / "reference" / "stage1_oracle.py"
WEIGHTS_DIR = REPO_ROOT / "models" / "gpt2"
EXPECTED_IDS = REPO_ROOT / "tests" / "fixtures" / "tokenizer_expected_ids.tsv"
INVENTORY = REPO_ROOT / "src" / "gpt2_tensor_inventory.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


oracle = _load("stage1_oracle", ORACLE_PATH)


class TestOracleEnvironment(unittest.TestCase):
    """(1) the oracle interpreter exists and both reference libraries import."""

    def test_interpreter_is_the_oracle_environment(self):
        exe = Path(sys.executable).resolve()
        self.assertTrue(
            ".venv-oracle" in str(exe),
            "this test must run under .venv-oracle, not %s" % exe)

    def test_reference_libraries_import(self):
        import tokenizers
        import safetensors
        self.assertTrue(tokenizers.__version__)
        self.assertTrue(safetensors.__version__)
        # Recorded, not asserted against a hard-coded version: the session
        # condition is what these report, and MEASUREMENTS.md carries it.
        print("tokenizers %s, safetensors %s" % (tokenizers.__version__,
                                                 safetensors.__version__))


class TestExpectedIdsFixture(unittest.TestCase):
    """(2) regenerating the fixture reproduces the committed one byte for byte."""

    def test_regeneration_is_byte_identical(self):
        committed = EXPECTED_IDS.read_bytes()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "regenerated.tsv"
            oracle.emit(str(out))
            self.assertEqual(out.read_bytes(), committed,
                             "regenerating the expected-ids fixture did not "
                             "reproduce the committed file byte for byte")


class TestInventoryAgainstReference(unittest.TestCase):
    """(3) and (4): names, dtypes, shapes and every byte, exactly."""

    @classmethod
    def setUpClass(cls):
        cls.inv = json.loads(INVENTORY.read_text(encoding="utf-8"))

    def test_names_dtypes_and_shapes_match(self):
        from safetensors import safe_open
        with safe_open(str(WEIGHTS_DIR / "model.safetensors"), framework="np") as ref:
            ref_names = set(ref.keys())
            inv = {t["name"]: t for t in self.inv["tensors"]}
            self.assertEqual(sorted(ref_names), sorted(inv),
                             "the inventory and the reference disagree on the tensor set")
            for name in sorted(ref_names):
                arr = ref.get_tensor(name)
                self.assertEqual(list(arr.shape), list(inv[name]["shape"]), name)
                self.assertEqual(str(arr.dtype), "float32", name)
                self.assertEqual(inv[name]["dtype"], "F32", name)

    def test_bytes_at_the_recorded_offsets_match_exactly(self):
        ok, message = oracle.verify_tensors()
        self.assertTrue(ok, message)
        print(message)


class TestVocabularySize(unittest.TestCase):
    """(5) the reference's vocabulary size against the shipped config."""

    def test_vocab_size_matches_shipped_config(self):
        ref_vocab, cfg_vocab = oracle.verify_vocab_size()
        self.assertEqual(ref_vocab, cfg_vocab)


class TestLocalArtifactsOnly(unittest.TestCase):
    """(6) the oracle resolves its artifacts locally and can never fetch."""

    def test_opened_paths_are_the_shipped_files(self):
        oracle.OPENED_PATHS.clear()
        oracle.verify_vocab_size()
        oracle.read_corpus()
        self.assertTrue(oracle.OPENED_PATHS, "the oracle opened nothing")
        weights_dir = str(WEIGHTS_DIR.resolve())
        fixtures_dir = str((REPO_ROOT / "tests" / "fixtures").resolve())
        for p in oracle.OPENED_PATHS:
            self.assertTrue(p.startswith(weights_dir) or p.startswith(fixtures_dir),
                            "the oracle opened %s, which is neither a shipped "
                            "artifact nor a committed fixture" % p)

    def test_no_hub_identifier_in_the_code_path(self):
        source = ORACLE_PATH.read_text(encoding="utf-8")
        # A hub identifier or a downloading constructor anywhere in this file
        # would mean the oracle could compare against a different artifact than
        # the one on disk, and prove nothing.
        for forbidden in ("from_pretrained", "hf_hub_download", "snapshot_download",
                          "huggingface_hub", "openai-community", "https://"):
            self.assertNotIn(forbidden, source,
                             "reference/stage1_oracle.py contains %r" % forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
