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
DEFAULT_OUT = os.path.join(REPO, "bench", "results", "stage3", "stage3_correctness.json")

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
    m = re.search(TOLERANCE_TOKEN + r"\s*=\s*([0-9.eE+-]+)", text)
    if not m:
        raise ValueError(
            f"{protocol_path} carries no {TOLERANCE_TOKEN} value. BENCHMARK_PROTOCOL.md "
            "section 5 is the only source for the D2 threshold; pass one explicitly only "
            "when deliberately measuring divergence before the threshold exists.")
    return float(m.group(1))


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
                generate: int = 0, timeout: int = 1200):
    cmd = [engine, "--prompt-file", prompt_path, "--truncate", str(truncate),
           "--cproj", "as-stored", "--dump-logits", dump]
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
    p.add_argument("--stage", default="stage-3")
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
                                    "greedy": cached["greedy"]}
            continue
        gen = a.greedy_tokens if L == min(x["target_tokens"] for x in rows) else 0
        print(f"  engine {r['id']}: prefill L={L}"
              + (f", greedy {gen} tokens" if gen else ""))
        ids, _ = _run_engine(a.engine, prompt_path, L, dump, generate=gen)
        prompt_ids = ids[:L]
        greedy = ids if gen else None
        engine_runs[r["id"]] = {"dump": dump, "ids": prompt_ids, "greedy": greedy}
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump({"tokens": len(prompt_ids), "ids": prompt_ids, "greedy": greedy}, f)

    # The oracle, once, reused across all four lengths.
    oracle = ReferenceGPT2()
    print(f"oracle: verified {oracle.n_verified} tensors against the inventory; "
          f"torch {torch.__version__}, threads {oracle.threads}, device cpu")

    per_configuration = []
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
        stats["positions_covered"] = ("EVERY position of the prefill: gpt2_tool --dump-logits "
                                      "writes logits for all positions, so both D2 bounds are "
                                      "measured over the whole prefill")
        per_configuration.append(stats)
        m = stats["reference_top1_top2_margin"]
        print(f"  L={L:4d}  max abs {stats['max_abs_diff']:.6g}  rms "
              f"{stats['rms_abs_diff']:.6g}  max rel {stats['max_rel_diff']:.6g}  "
              f"top-1 {stats['top1_agreement_positions']}/{L}  "
              f"margin min {m['min']:.6g} (pos {m['argmin_position']})  "
              f"margin/div {stats['margin_min_over_max_abs_diff']:.2f}x")

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
    print(f"D2 GATE: {verdict}")
    for name, cond in doc["gate"]["conditions"].items():
        print(f"  {'pass' if cond['pass'] else 'FAIL'}  {name}")
    # A FAIL is REPORTED, not raised. The exit code lets a caller branch on it;
    # the results file is written either way.
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
