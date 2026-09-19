# TECHNICAL_SPEC.md

*What gets built, in what order, in what language, producing what. Claude Code prompts are written against this file.*

---

## 1. Model choice

**GPT-2 small.** 124M parameters, 12 layers, 12 heads, d_model 768, vocab 50257, context 1024. Weights published by OpenAI, mirrored on Hugging Face. *(Confirm every figure against the config file shipped with the weights before hardcoding. Do not trust this table.)*

At FP32, weights occupy roughly 500MB — comfortable in 4GB VRAM alongside activations and KV cache. Architecture is the same shape as every modern decoder-only LLM, so the knowledge transfers upward.

## 2. Language per component

| Component | Language | Reason |
|---|---|---|
| Forward pass, Stages 2–6 | **C** | Every target posting lists C. Nothing here needs C++. |
| CUDA kernels, Stages 7–11 | **CUDA C++** | Required by toolchain. |
| Tokenizer, weight loader | C | Part of the no-libraries claim. |
| Benchmark harness, correctness gate | **Python** | Target postings ask for Python test automation. |
| Performance model | **Python** | Needs plotting, fitting, and fast iteration. |
| Reference implementation | Python + PyTorch | Correctness oracle only. |
| Dashboard | React | Presentation layer, last required stage. |

## 3. Stage sequence

One Claude Code session per stage. Each produces one `MEASUREMENTS.md` entry.

**Stage 0 — Instrument the machine.**
Repo skeleton, build system, microbenchmarks from `HARDWARE.md` §3, `HARDWARE.md` populated, Nsight Compute availability confirmed. No inference code.

**Stage 0b — Diagnose and remeasure the INVALID CPU microbenchmarks.**
Scoped diagnostic-and-remeasurement session, added after Stage 0 completed. Two required
`HARDWARE.md` §2 fields — measured DRAM bandwidth and measured peak FP32 SIMD — are empty
because every DRAM-tier working set and the vectorised AVX2 configuration exceeded the 5%
standard-deviation limit in both Stage 0 suite runs. `PROJECT.md` §7 item 4 requires the
Stage 10 performance model validated across all kernels, and a model with no CPU
memory-bandwidth term and no CPU SIMD ceiling cannot be. Stage 6's framing — predicted ceiling
is vector width, measured will be lower, and the gap is the interesting part — requires a
measured ceiling. Diagnose first, from the raw per-sample timings Stage 0 retained; then
remeasure under reduced background load. No inference code, no new microbenchmarks, no change
to the environment fingerprint. An empty field with a better-evidenced reason is a valid
outcome.

**Stage 1 — Weights and tokenizer.**
safetensors parsing, tensor layout mapping, BPE tokenizer, encode/decode round-trip against the reference.

**Stage 2 — Naive C forward pass.**
Embeddings, multi-head attention, feed-forward, layernorm, sampling. Triple-nested-loop matmul, no optimization. **Deliberately slow.** Output: coherent text, correctness match, baseline prefill and decode latency.

**Stage 3 — Benchmark harness and correctness gate.**
Python harness implementing `BENCHMARK_PROTOCOL.md`. Prefill and decode timed separately. Numerical parity check. Regression detection. Structured output for the dashboard and the performance model.

**Stage 4 — KV cache.**
Decode cost per token drops from growing with sequence length to roughly constant. Correctness-preserving algorithmic change, measured separately.

**Stage 5 — CPU GEMM: cache blocking.**
Tile against measured cache sizes. Expect large prefill gain and minimal decode gain. That asymmetry is the project's central lesson and must be measured, not asserted.

**Stage 6 — CPU GEMM: SIMD.**
Hand-vectorize at the widest available width. Predicted ceiling is vector width; measured will be lower, and the gap is the interesting part.

**Stage 7 — CUDA port.**
Whole forward pass on GPU, weights resident in VRAM. Naive kernels acceptable for non-matmul ops. **Do not port only the matmul** — that would make the measurement a test of PCIe bandwidth.

**Stage 8 — Tiled shared-memory matmul.**
Stage tiles through shared memory. Measured against cuBLAS at real shapes. Produces the prefill headline ratio.

**Stage 9 — Flash attention.**
Tiled, IO-aware attention with online softmax. The N×N score matrix is never materialized in global memory; scores are computed tile by tile in shared memory using a running maximum and a running denominator, rescaling previous partial outputs whenever a tile contains a higher max. Memory goes from quadratic to linear in sequence length.

Forward pass only — no backward pass is needed for inference. **Plain shared-memory staging only.** `cp.async` and `ldmatrix` require sm_80; the primary device is sm_75. Measure peak memory against sequence length to demonstrate the scaling change, and latency against the Stage 8 attention path.

**Stage 10 — Analytical performance model. The differentiator.**
A program taking kernel parameters (tile dimensions, block size, bytes moved, FLOPs, occupancy) plus `HARDWARE.md` measured characteristics, producing a predicted runtime.

- Validated **retroactively** against every kernel from Stages 5–9, giving many data points immediately.
- Validated **prospectively** on Stage 11, which is built after the model exists.
- Reports an **error distribution**, not a single figure: median, worst case, and which kernel classes it predicts well versus badly.
- Compares against the published roofline baseline (~34% average error on DL kernels).
- Where the model is badly wrong, the explanation is a finding, not a failure.

**Stage 11 — Systolic dataflow variant.**
Weight-stationary dataflow in software: each weight fetched once, amortized across a tile. Predicted by the Stage 10 model *before* implementation — this is the model's prospective test.

**May well be slower than Stage 8.** Simulating cell-to-cell propagation adds synchronization a well-tiled kernel does not need. A slower result that is explained is better than a faster one that is not. Frame as a TPU-style weight-stationary dataflow. **Do not claim this is how NVIDIA tensor cores work** — they are documented as units performing small matrix multiply-accumulates. The primary device has no tensor cores at all (Turing TU117); confirm in Stage 0.

**Stage 12 — Dashboard.**
React. Latency waterfall across stages, prefill and decode separated, predicted-versus-measured overlay, performance model error distribution, flash attention memory scaling curve.

### Optional stages

Build only after Stage 12. Each is independently valuable; none is required for done. Do not start one before the required stages are complete.

**Stage 13 (optional) — Autotuner.** Search the kernel configuration space automatically, using the Stage 10 model to prune candidates before running them. Report search time with and without pruning. This makes the performance model *useful* rather than merely accurate, which is a stronger story.

**Stage 14 (optional) — Speculative decoding.** A small draft model proposes several tokens; the full model verifies them in one pass and accepts the matching prefix. Directly attacks the memory-bound decode problem. Headline numbers: tokens per second, and the acceptance rate that explains it.

**Stage 15 (optional) — INT8 quantization.** Weight-only quantization with a fused dequantize-matmul kernel. Adds a second axis to the project: speed versus accuracy. Report both the speedup and the measured output divergence — the divergence is a result, not a caveat.

**Stage 16 (stretch) — Upstream contribution.** Find a gap in `llama.cpp` / `ggml`, implement, verify against the CPU reference across a real test matrix, submit.

## 4. Repository layout

```
/
  PROJECT.md  HARDWARE.md  TECHNICAL_SPEC.md  BENCHMARK_PROTOCOL.md
  MEASUREMENTS.md  LEARNING.md  PERSISTENT.md  README.md
  .gitattributes      # Stage 2: fixtures compared byte for byte are eol=lf
  src/
    main.c  model.c/h  tokenizer.c/h  safetensors.c/h
    gpt2_tensor_inventory.json   # Stage 1, committed; Stage 2 consumes it
    gemm/    gemm.h  gemm_naive.c  gemm_blocked.c  gemm_simd.c
    cuda/    forward.cu  gemm_naive.cu  gemm_tiled.cu
             flash_attention.cu  gemm_systolic.cu
  bench/
    microbench/          # Stage 0, populates HARDWARE.md
    harness.py           # Stage 3
    correctness.py
    profile.py           # Nsight Compute wrapper
    stage2_forward_bench.c   # Stage 2, temporary; Stage 3's harness replaces it
    results/
  model/
    perf_model.py        # Stage 10
    validate.py
    results/
  reference/
    reference_impl.py
  tests/              # one test per source file, registered with CTest
    fixtures/         # committed test inputs (Stage 1 corpora, Stage 2 placeholder prompts)
  scripts/            # build.ps1 and the CMake helpers
  models/             # downloaded weights and tokenizer artifacts; gitignored, never committed
  dashboard/
```

## 5. Invariants

1. No ML library in the inference path. PyTorch is an oracle only.
2. Prefill and decode always measured and reported separately.
3. No stage accepted without passing the correctness gate.
4. No speedup reported without a prediction that preceded it.
5. Every gap explanation backed by Nsight counters or explicitly marked unestablished.
6. The primary measurement device never changes.
7. No sm_80+ intrinsics anywhere.
8. Each stage is one commit-complete Claude Code session.
