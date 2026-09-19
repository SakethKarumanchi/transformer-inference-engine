"""Unit test for reference/reference_impl.py, and the Stage 2 correctness record.

Runs under the project .venv (torch 2.14.0+cu130, numpy 2.5.3). It asserts:

  * the weight reader agrees with src/gpt2_tensor_inventory.json on EVERY tensor,
    on name, dtype, shape, absolute file offset and byte length;
  * the reference's greedy token sequence matches the C engine's on the
    placeholder prompt;
  * the divergence statistics are computed and written, at two sequence lengths.

WHAT IS NOT ASSERTED, DELIBERATELY: any numerical tolerance. BENCHMARK_PROTOCOL.md
section 5 leaves the tolerance to Stage 3 and PERSISTENT.md section 1 D2 is open
and owned by Stage 3. A threshold invented here and then passed by its own author
would be circular and would prove nothing. The divergence is therefore MEASURED
AND REPORTED so Stage 3 can set D2 from an observed distribution, and the checks
that gate this stage are the tolerance-free ones: exact greedy-sequence equality,
and the structural properties asserted in tests/test_model.c.

REDUCED TOKEN COUNTS, STATED: the C engine is the deliberately slow Stage 2
baseline, so this test runs at 8 and 16 tokens and generates 3 tokens. The timed
configurations (prefill 32 and 64, decode at contexts 32, 64 and 128) are NOT
reduced.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from reference.reference_impl import (  # noqa: E402
    DEFAULT_CONFIG, DEFAULT_INVENTORY, DEFAULT_WEIGHTS,
    ReferenceGPT2, parse_tensors, read_header, verify_against_inventory,
)

import torch  # noqa: E402

FIXTURE = os.path.join(REPO, "tests", "fixtures", "stage2_placeholder_prompts.tsv")
REL_EPS = 1e-6

checks = 0
failures = 0


def check(cond, msg):
    global checks, failures
    checks += 1
    if cond:
        print(f"  ok   {msg}")
    else:
        failures += 1
        print(f"  FAIL {msg}")
    return bool(cond)


def read_fixture():
    rows = []
    with open(FIXTURE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) >= 3:
                rows.append({"id": parts[0], "use": parts[1], "text": parts[2]})
    return rows


def read_logit_dump(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"TIE2LOGI", f"bad magic {magic!r}"
        npos, nvocab = struct.unpack("<ii", f.read(8))
        raw = f.read(npos * nvocab * 4)
    assert len(raw) == npos * nvocab * 4, "short logit dump"
    return np.frombuffer(raw, dtype=np.float32).reshape(npos, nvocab).copy()


def run_engine(engine, prompt_path, truncate, dump=None, generate=0, cproj="as-stored"):
    cmd = [engine, "--prompt-file", prompt_path, "--truncate", str(truncate),
           "--cproj", cproj]
    if dump:
        cmd += ["--dump-logits", dump]
    if generate:
        cmd += ["--generate", str(generate)]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=280)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr, file=sys.stderr)
        raise RuntimeError(f"the engine exited {out.returncode}")
    ids = None
    for line in out.stdout.splitlines():
        if line.startswith("ids:"):
            ids = [int(t) for t in line[4:].split()]
    assert ids is not None, "the engine printed no id sequence"
    return ids, out.stdout


def percentiles(a):
    q = np.percentile(a, [50, 90, 99, 99.9])
    return {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "p99_9": float(q[3]), "max": float(a.max())}


def divergence(engine_logits, ref_logits, ids, label):
    """Every statistic computed in float32 on both sides, over the FULL logit
    vector at every position. No threshold is applied to any of them."""
    c = np.asarray(engine_logits, dtype=np.float32)
    r = np.asarray(ref_logits, dtype=np.float32)
    assert c.shape == r.shape, f"{c.shape} vs {r.shape}"
    absdiff = np.abs(c - r).astype(np.float64)
    denom = np.abs(r).astype(np.float64) + REL_EPS
    reldiff = absdiff / denom
    below_floor = int(np.count_nonzero(np.abs(r) < REL_EPS))

    flat = int(np.argmax(absdiff))
    pos, tok = divmod(flat, c.shape[1])

    top1 = [int(x) for x in np.argmax(c, axis=1)]
    top1_ref = [int(x) for x in np.argmax(r, axis=1)]
    top1_agree = sum(1 for a, b in zip(top1, top1_ref) if a == b)
    top5_agree = 0
    for t in range(c.shape[0]):
        a = set(np.argpartition(-c[t], 5)[:5].tolist())
        b = set(np.argpartition(-r[t], 5)[:5].tolist())
        top5_agree += int(a == b)

    # The margin: the gap between the top-1 and top-2 REFERENCE logits at each
    # position. This is what says how much divergence greedy decoding can absorb
    # before a token flips, and it is the most useful thing this stage hands to
    # Stage 3.
    srt = np.sort(r, axis=1)
    margins = (srt[:, -1] - srt[:, -2]).astype(np.float64)

    return {
        "label": label,
        "positions": int(c.shape[0]),
        "vocab_size": int(c.shape[1]),
        "token_ids": [int(i) for i in ids],
        "max_abs_diff": float(absdiff.max()),
        "max_abs_diff_position": int(pos),
        "max_abs_diff_token_id": int(tok),
        "max_rel_diff": float(reldiff.max()),
        "rel_diff_denominator": "|reference| + 1e-6, elementwise",
        "elements_below_rel_floor": below_floor,
        "mean_abs_diff": float(absdiff.mean()),
        "median_abs_diff": float(np.median(absdiff)),
        "rms_abs_diff": float(np.sqrt((absdiff ** 2).mean())),
        "abs_diff_percentiles": percentiles(absdiff),
        "rel_diff_percentiles": percentiles(reldiff),
        "top1_agreement_positions": top1_agree,
        "top1_agreement_fraction": top1_agree / c.shape[0],
        "top5_set_agreement_positions": top5_agree,
        "top5_set_agreement_fraction": top5_agree / c.shape[0],
        "reference_top1_top2_margin": {
            "min": float(margins.min()),
            "median": float(np.median(margins)),
            "max": float(margins.max()),
            "per_position": [float(x) for x in margins],
        },
        "margin_min_over_max_abs_diff": (
            float(margins.min() / absdiff.max()) if absdiff.max() > 0 else None),
    }


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to the built gpt2_tool")
    ap.add_argument("--out-dir", default=os.getcwd(),
                    help="where the correctness results file is written")
    ap.add_argument("--lengths", default="8,16")
    ap.add_argument("--generate", type=int, default=3)
    a = ap.parse_args()

    print("test_reference_impl")
    print("  reduced token counts, stated: " + a.lengths + " for the divergence statistics, "
          f"{a.generate} generated tokens for the greedy comparison; the timed "
          "configurations are NOT reduced")

    try:
        import safetensors  # noqa: F401
        importable = True
    except ImportError:
        importable = False
    print(f"  weight-reader branch: safetensors importable in this environment = {importable}; "
          f"{'library' if importable else 'stdlib + numpy, cross-checked against the inventory'}")

    header, data_offset = read_header(DEFAULT_WEIGHTS)
    parsed = parse_tensors(header, data_offset)
    n = verify_against_inventory(parsed, DEFAULT_INVENTORY)
    check(n == 160, f"the weight reader agrees with the committed inventory on all {n} "
                    "tensors: name, dtype, shape, absolute file offset and byte length")

    model = ReferenceGPT2(cproj_transposed=False)
    check(model.threads == 1, f"torch.set_num_threads(1) took: {model.threads} thread")
    check(str(next(iter([model.device]))) == "cpu", "the oracle runs on the CPU")
    check(model.head.data_ptr() == model.wte.data_ptr(),
          "the oracle's head is TIED to the token embedding (same storage)")

    rows = read_fixture()
    long_text = rows[0]["text"]
    short_text = rows[1]["text"]
    check(len(rows) >= 2 and rows[0]["id"] == "p1",
          f"the placeholder fixture loads ({len(rows)} rows, labelled PLACEHOLDER in its header)")

    tmpdir = tempfile.mkdtemp(prefix="stage2_")
    long_path = os.path.join(tmpdir, "p1.txt")
    short_path = os.path.join(tmpdir, "p2.txt")
    for p, t in ((long_path, long_text), (short_path, short_text)):
        with open(p, "w", encoding="utf-8") as f:
            f.write(t)

    configs = []
    for L in [int(s) for s in a.lengths.split(",")]:
        dump = os.path.join(tmpdir, f"logits_{L}.bin")
        ids, _ = run_engine(a.engine, long_path, L, dump=dump)
        eng = read_logit_dump(dump)
        check(eng.shape[0] == L, f"the engine dumped logits at all {L} positions")
        ref = model.forward(ids).numpy()
        d = divergence(eng, ref, ids, f"prefill, L={L} tokens, placeholder prompt p1")
        d["cproj_reading"] = "as-stored"
        configs.append(d)
        print(f"  L={L:3d}: max abs {d['max_abs_diff']:.6g}, rms {d['rms_abs_diff']:.6g}, "
              f"max rel {d['max_rel_diff']:.6g}, top-1 agreement "
              f"{d['top1_agreement_positions']}/{L}, reference margin min "
              f"{d['reference_top1_top2_margin']['min']:.4f}")
        check(d["top1_agreement_fraction"] == 1.0,
              f"top-1 agreement at every position at L={L} "
              f"({d['top1_agreement_positions']}/{L})")

    # The transposed reading of the twelve square attn.c_proj tensors, run
    # against the same oracle. This is the measurement that makes the choice an
    # observation rather than an assumption.
    L = int(a.lengths.split(",")[0])
    dump_t = os.path.join(tmpdir, "logits_transposed.bin")
    ids_t, _ = run_engine(a.engine, long_path, L, dump=dump_t, cproj="transposed")
    eng_t = read_logit_dump(dump_t)
    ref_t = model.forward(ids_t).numpy()
    d_t = divergence(eng_t, ref_t, ids_t,
                     f"prefill, L={L} tokens, TRANSPOSED reading of h.*.attn.c_proj.weight")
    d_t["cproj_reading"] = "transposed"
    configs.append(d_t)
    check(d_t["max_abs_diff"] > configs[0]["max_abs_diff"],
          "the transposed reading of h.*.attn.c_proj.weight diverges from the oracle far "
          f"more than the as-stored reading ({d_t['max_abs_diff']:.6g} against "
          f"{configs[0]['max_abs_diff']:.6g}), so the reading was SELECTED BY MEASUREMENT")

    # Greedy sequence equality -- one of the two requirements that actually gate
    # this stage, and it carries no tolerance.
    ids_engine, stdout = run_engine(a.engine, short_path, 64, generate=a.generate)
    prompt_ids = ids_engine[:len(ids_engine) - a.generate]
    ids_ref = model.greedy(prompt_ids, a.generate)
    check(ids_engine == ids_ref,
          f"the greedy token sequence matches the reference exactly: engine {ids_engine} "
          f"reference {ids_ref}")

    out = {
        "stage": "stage-2",
        "kind": "correctness",
        "generated_by": "tests/test_reference_impl.py",
        "git_commit": git_commit(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": {
            "implementation": "reference/reference_impl.py, hand-written in torch",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "torch_num_threads": model.threads,
            "weight_reader": "library" if importable else "stdlib + numpy",
            "safetensors_importable": importable,
            "tensors_verified_against_inventory": n,
            "what_agreement_proves": (
                "the two implementations agree; NOT that either matches GPT-2. Both were "
                "written in one session from one reading of the architecture. The external "
                "discriminator is readable English from published weights, which catches a "
                "transposed weight, an untied head, a broken causal mask and a wrong "
                "positional scheme, and does not catch a wrong layernorm epsilon, a plain "
                "gelu substituted for gelu_new, or a mildly wrong attention scale."),
        },
        "prompt_set": {
            "status": "PLACEHOLDER",
            "open_decision": "D3",
            "owning_stage": 3,
            "work_item": "W11",
            "source": "tests/fixtures/stage2_placeholder_prompts.tsv",
        },
        "tolerance": {
            "adopted": False,
            "owning_stage": 3,
            "open_decision": "D2",
            "reason": (
                "BENCHMARK_PROTOCOL.md section 5 sets the numerical tolerance in Stage 3 and "
                "D2 is open. Stage 2 measures and reports the divergence distribution and "
                "adopts no threshold; a threshold invented and passed by its own author "
                "would be circular."),
        },
        "greedy": {
            "prompt_ids": prompt_ids,
            "engine_sequence": ids_engine,
            "reference_sequence": ids_ref,
            "match": ids_engine == ids_ref,
            "tokens_generated": a.generate,
        },
        "configurations": configs,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "stage2_correctness.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"  correctness record written: {path}")

    print(f"test_reference_impl: {checks} checks, {failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
