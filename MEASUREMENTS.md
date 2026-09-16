# MEASUREMENTS.md

*The measurement log. Together with the Stage 10 performance model, this is the deliverable. Everything else in the repository is scaffolding that makes these entries trustworthy.*

**Rule: a stage's prediction block is written and committed BEFORE that stage's code exists. A prediction entered after seeing the result is not a prediction.**

---

## How to read an entry

1. **Prediction** — expected result and hardware reasoning, written first, never edited.
2. **Measurement** — what happened, under `BENCHMARK_PROTOCOL.md` conditions.
3. **Gap** — prediction minus measurement, with the explanation and the Nsight counter that supports it.
4. **What this taught** — the transferable fact.

Wrong predictions are kept verbatim. A log of correct predictions looks fabricated; a log of wrong predictions with counter-backed explanations looks like engineering.

---

## Entry template

```markdown
### Stage N — <name>

**Committed:** <date> · **Commit:** <hash> · **Device:** <HARDWARE.md §1>

#### Prediction  (written <date>, before implementation)

Expected prefill: <Nx or latency>
Expected decode:  <Nx or latency>

Reasoning:
- <hardware parameter from HARDWARE.md, arithmetic shown>
- <why prefill and decode should differ>

Falsified if: <what result would prove the reasoning wrong>

#### Measurement

| Metric | Before | After | Change |
|---|---|---|---|
| Prefill latency (median) | | | |
| Decode latency / token (median) | | | |
| Std dev as % of median | | | |
| Correctness | | | pass / fail |
| % of achievable bandwidth (decode) | | | |
| % of cuBLAS (prefill) | | | |

Nsight counters: achieved occupancy __ / theoretical __ · DRAM throughput __ ·
L2 hit rate __ · bank conflicts __ · dominant warp stall reason __

Conditions: <deviations from BENCHMARK_PROTOCOL.md §4>
Raw samples: `bench/results/stage_N.json`

#### Gap

Predicted <X>, measured <Y>. Difference: <Z>.

Explanation: <the mechanism>
Counter evidence: <which counter supports this, with its value>
Alternatives not ruled out: <what the counters do NOT distinguish, stated plainly>

#### What this taught

<One or two sentences of transferable fact. The sentence spoken aloud in an interview.>
```

---

## Stage 0 — Instrument the machine
*No prediction: this stage measures the machine rather than changing it. Output is `HARDWARE.md`.*
**Status:** not started
Record: any figure where measured reality diverged notably from the spec sheet, with a hypothesis. Whether Nsight Compute can collect counters here. Confirmed compute capability and tensor-core presence.

## Stage 1 — Weights and tokenizer
**Status:** not started

## Stage 2 — Naive C forward pass (baseline)
*Predict the baseline itself from operation counts and measured machine throughput, before running it.*
**Status:** not started

## Stage 3 — Benchmark harness and correctness gate
**Status:** not started
Record: the chosen numerical tolerance and the justification for that specific value.

## Stage 4 — KV cache
**Status:** not started

## Stage 5 — CPU GEMM: cache blocking
**Status:** not started
Expected to reveal the project's central asymmetry: large prefill gain, minimal decode gain. If decode improves substantially, the prediction was wrong and the reason must be found.

## Stage 6 — CPU GEMM: SIMD
**Status:** not started

## Stage 7 — CUDA port
**Status:** not started
First stage requiring Nsight counters.

## Stage 8 — Tiled shared-memory matmul
**Status:** not started
Produces the prefill headline ratio.

## Stage 9 — Flash attention
**Status:** not started
Record: memory versus sequence length across at least five lengths, the crossover point where it becomes necessary on this VRAM, numerical error at the longest tested length, and latency against the Stage 8 attention path.

## Stage 10 — Analytical performance model
*No single speedup. The result is an error distribution.*
**Status:** not started

Record:
- Model structure and every term, with the `HARDWARE.md` figure each consumes
- Retroactive validation across all Stage 5–9 kernel configurations: median error, 90th percentile, worst case
- Breakdown by kernel class — compute-bound versus memory-bound, small versus large
- Position relative to the ~34% published roofline baseline
- Which kernel classes the model predicts badly, and the missing term responsible
- Any term added after seeing data, stated explicitly

## Stage 11 — Systolic dataflow variant
*The performance model's prospective test. Predict with the Stage 10 model BEFORE implementing.*
**Status:** not started
A slower result than Stage 8 is acceptable and expected. Explain it; do not tune until it wins.

## Stage 12 — Dashboard
**Status:** not started

---

## Optional stages

## Stage 13 (optional) — Autotuner
**Status:** not started
Record: search time with and without model-based pruning, and whether the pruned search found the same optimum.

## Stage 14 (optional) — Speculative decoding
**Status:** not started
Record: tokens per second, acceptance rate, and the relationship between them.

## Stage 15 (optional) — INT8 quantization
**Status:** not started
Record: speedup, measured output divergence, and the new tolerance. Divergence is a result, not a caveat.

## Stage 16 (stretch) — Upstream contribution
**Status:** not started

---

## Summary table

*Populated as stages complete. Feeds the dashboard and the resume bullet.*

| Stage | Prefill | Decode/token | Predicted | Measured | Counter-backed |
|---|---|---|---|---|---|
| | | | | | |

**Headline figures:**
- Performance model error: median ____%, p90 ____%, worst ____% (published roofline baseline ~34%)
- Decode: ____% of measured achievable bandwidth
- Prefill: ____% of cuBLAS
- Flash attention: peak memory ____ → ____ at sequence length ____
