CC_PROMPT_FORMAT — Canonical Claude Code prompt format for this project

Read this before writing ANY Claude Code prompt. The kickoff template specifies the deliverable
wrapper and the construction RULES; THIS file specifies the body format of the prompt itself.

== HOUSE FORMAT (mandatory) ==

Header, then a 2–3 sentence scope statement naming what NOT to do:
  Task: Stage [N] — [title] ([one-line scope]).
Then: build framing, recon-then-build statement, explicit exclusions.

Section delimiters are == CAPS == — never markdown headers, never rules.

Fixed section order:

  == PREDICTION (COMMIT THIS FIRST, BEFORE ANY CODE) == — the operator's prediction verbatim.
    First instruction of the session is to write it into MEASUREMENTS.md and commit. Never
    edited, improved, or authored by the agent. Omitted only for stages exempt under RULE 3.

  == AUTHORITATIVE CONTRACT (treat as fixed; verify, do not re-derive) == — verbatim hardware
    figures from HARDWARE.md, verbatim measurement rules from BENCHMARK_PROTOCOL.md, verbatim
    stage definition from TECHNICAL_SPEC.md §3, transcribed interfaces and shapes.

  == [STAGE-SPECIFIC GAP / HEADLINE] (resolve via live-repo determination) == — one section per
    open question, framed if X then Y, else Z, and state which.

  == DETERMINE FROM THE LIVE REPO (file-truth — read the files, do not assume) == — numbered.

  == OUTPUTS == — NEW FILE N / EDIT FILE N — path + description + inline unit-test asserts.

  == CONSTRAINTS RECAP ==

  == AFTER YOU BUILD == — self-run offline gate, HARD CHECKPOINT, then on "proceed" benchmarks +
    profiling + MEASUREMENTS.md + PERSISTENT.md + LEARNING.md + commit / push / PR.

Voice: plain declarative build instructions. No source tags, no terse mode, no objection-first.

Recon-then-build is the DEFAULT. Plan mode only when explicitly opted in (exception: Stage 10).

Open decisions resolve as build-time determinations: "DETERMINE the convention, pick accordingly,
build it, and STATE which you chose and why in your final report."

== PROJECT-SPECIFIC INVARIANTS EVERY PROMPT CARRIES ==

  - Prefill and decode measured and reported separately, always.
  - No ML library in the inference path. PyTorch is a correctness oracle only.
  - No stage accepted without passing the correctness gate.
  - No performance number estimated. Every figure comes from a run that happened.
  - GPU stages collect Nsight counters; gap explanations name the supporting counter.
  - The primary measurement device never changes.
  - No sm_80+ intrinsics: no cp.async, no ldmatrix, no mma. Device is Turing sm_75.
  - A run with std dev above 5% of median is INVALID — report it, do not average it away.
  - The agent never writes or revises a prediction.

== WORKED EXAMPLE — Stage 9 (recon-then-build, no plan gate) ==

Reference shape. Mirror its delimiters, OUTPUTS layout, DETERMINE numbering, and gate.

Task: Stage 9 — Flash attention (tiled IO-aware attention with online softmax).

You are replacing the attention path with a fused, tiled kernel that never materializes the N×N
score matrix in global memory, and measuring both its latency and its memory scaling against the
Stage 8 path. This is a kernel task, not an architecture task — the model, the KV cache, and the
GEMM kernels are unchanged. Do NOT implement a backward pass; inference needs forward only. Do
NOT use cp.async, ldmatrix, or mma — the device is sm_75 and those require sm_80. Do NOT modify
the benchmark harness beyond registration. Recon the live repo first, then build directly.

== PREDICTION (COMMIT THIS FIRST, BEFORE ANY CODE) ==

The operator's prediction for Stage 9, verbatim:

  Prefill attention speedup: [operator's number]
  Peak memory at sequence length [L]: [operator's number]
  Reasoning: [operator's reasoning, verbatim, including errors]
  Falsified if: [operator's falsification condition]

First action: write this block into the Stage 9 entry of MEASUREMENTS.md under "#### Prediction",
commit it alone with the message "stage 9: prediction committed", then begin recon. Do not edit,
improve, correct, or comment on the prediction. If the reasoning contains a mistake, leave it —
the gap analysis addresses it.

== AUTHORITATIVE CONTRACT (treat as fixed; verify, do not re-derive) ==

Device characteristics (HARDWARE.md §1, Stage 0 measured):
  Compute capability:            [verbatim]
  Shared memory per block:       [verbatim]
  Shared memory per SM:          [verbatim]
  Measured shared memory bandwidth: [verbatim]
  Measured achievable bandwidth: [verbatim]
  VRAM total:                    [verbatim]
  Max threads per SM:            [verbatim]

Model shapes (TECHNICAL_SPEC.md §1, confirmed against the shipped config):
  [verbatim: heads, head_dim, d_model, max context]

The flash attention algorithm (implement exactly this; do not substitute an approximation):
  Each thread block owns a tile of Q and iterates over tiles of K and V along the sequence.
  For each KV tile: compute S = Q_tile @ K_tile^T, scaled by 1/sqrt(head_dim).
  Maintain per-row a running maximum m and a running denominator l.
  On each new tile: compute the tile's row max, update m to the larger of old and new,
  rescale the accumulated output and the running denominator by exp(m_old - m_new),
  then accumulate the new tile's contribution.
  The N×N score matrix is never written to global memory.
  Causal masking applies — this is an autoregressive decoder.

Correctness (BENCHMARK_PROTOCOL.md §5):
  Logits compared elementwise against the PyTorch reference, tolerance [verbatim from Stage 3]
  Numerical stability verified specifically at the LONGEST tested sequence length
  Greedy-decoded token sequence matches the reference on the fixed prompt set

Benchmark conditions (BENCHMARK_PROTOCOL.md §3, §4, §6):
  5 warmup discarded, 20 timed samples, median with min/max/std dev
  INVALID if std dev exceeds 5% of median
  Nsight counters: achieved occupancy, DRAM throughput, L2 hit rate, shared memory bank
  conflicts, warp stall reasons, instructions executed

== TILE SHAPE AND SHARED MEMORY BUDGET (resolve via live-repo determination) ==

Q tile rows and KV tile length are not specified. DETERMINE from shared memory per block in
AUTHORITATIVE CONTRACT: the resident working set is the Q tile, the K tile, the V tile, and the
per-row running statistics. Size so this fits with headroom, then sweep at least four candidate
shapes and pick the best measured. STATE the computed starting point, the shapes swept, the
winner, and whether the winner matched the computation. If it did not, name the likely cause and
the counter that supports it — bank conflicts and occupancy are the usual suspects, and both are
measurable, so do not speculate where you can measure.

Head dimension is small enough that the Q tile may be held in registers rather than shared
memory. DETERMINE which is faster on this device and STATE the choice with its measurement.

== DETERMINE FROM THE LIVE REPO (file-truth — read the actual files, do not assume) ==

  1. src/cuda/forward.cu — the current attention path, its call signature, memory layout, and
     how the KV cache is indexed. The flash kernel must be a drop-in replacement. STATE the
     signature matched.
  2. The KV cache layout from Stage 4 — head-major or sequence-major, contiguity. The tiling
     strategy depends on it. STATE the layout and how the kernel strides through it.
  3. Whether decode (single-query attention) should use the flash kernel at all. With one query
     row there is no N×N matrix to avoid, so the fused kernel may be slower than a simple
     bandwidth-bound kernel. DETERMINE by measuring both, and STATE which path decode uses and
     why. Do not assume flash attention helps everywhere.
  4. bench/harness.py — how a stage registers and writes to bench/results/. Follow the existing
     pattern; do not add a second results path.
  5. bench/profile.py — the existing Nsight wrapper and its counter set. Extend if this stage
     needs a counter not already collected; STATE what you added.
  6. Compiler flags currently in the build. The comparison is void if they change between
     stages — STATE them and confirm unchanged.

== OUTPUTS ==

NEW FILE 1 — src/cuda/flash_attention.cu
  Fused tiled attention with online softmax and causal masking, forward only, plain shared-memory
  staging (no sm_80 intrinsics). Matches the existing attention signature (DETERMINE #1). Unit
  test (test_flash_attention.cu) asserting: output matches the naive attention path elementwise
  within tolerance at three sequence lengths including the longest supported; causal mask
  correctness (no attention to future positions); numerical stability at the longest length,
  comparing against a high-precision reference; correct handling of sequence lengths not evenly
  divisible by the tile length.

EDIT FILE 2 — src/cuda/forward.cu
  Route prefill attention through the flash kernel. Route decode per DETERMINE #3. Keep the
  Stage 8 attention path in the tree — it is the baseline this stage is measured against.

EDIT FILE 3 — bench/harness.py
  Register Stage 9. Add the memory sweep: peak memory versus sequence length across at least five
  lengths spanning short to the longest that fits in VRAM. Writes bench/results/stage_9.json with
  raw per-sample timings, the tile-shape sweep, the memory sweep, and the counter collection.

== CONSTRAINTS RECAP ==
Forward pass only. No cp.async, no ldmatrix, no mma — device is sm_75. No changes to the model,
KV cache layout, or GEMM kernels. Stage 8 attention path stays in the tree. Compiler flags
unchanged or the comparison is void. Prefill and decode separated in every timing. Correctness
gate must pass before any number is reported, and numerical stability must be verified at the
longest tested sequence length specifically — a kernel correct at length 128 and wrong at 1024 is
the characteristic failure of a naive online softmax. No performance figure estimated. Every gap
explanation names its supporting counter.

== AFTER YOU BUILD ==
Commit the prediction first. Then author the kernel, its unit tests, the forward-pass routing,
and the harness registration. Then run the offline gate YOURSELF: clean build from the repo root,
full unit test suite covering every file you touched, and the correctness gate against the
PyTorch reference. 5-minute per-test cap; on hang or failure stop, inspect, fix, rerun —
fix-driven reruns only.

When the offline gate is green, STOP. Do NOT run benchmarks, do NOT commit, do NOT push, do NOT
open a PR. Hand me the benchmark-conditions checklist (machine idle, no other GPU or CPU
workload, not thermally throttling, not on battery) plus confirmation that Nsight Compute counter
collection is permitted, as a HARD CHECKPOINT, and WAIT for my "proceed".

On "proceed": run the tile-shape sweep, the full benchmark suite, and the memory sweep under
BENCHMARK_PROTOCOL.md conditions. Collect the Nsight counters. If std dev exceeds 5% of median on
any configuration, declare that run INVALID, report it, and stop — do not average it away, do not
silently retry. Fill the Measurement section of the Stage 9 entry in MEASUREMENTS.md from actual
results. Draft the Gap section: predicted versus measured, the mechanism, and the specific counter
value that evidences it. Where the counters do not distinguish between candidate causes, write
that plainly rather than selecting one.

Then: append future-relevant flags to PERSISTENT.md in the existing entry shape and bump "Last
updated:" -> append the concepts this stage exercised to LEARNING.md under the Stage 9 heading,
marked `unread`, including any concept encountered that is not already listed -> git add -A ->
git ls-files > repo-files.txt -> git add repo-files.txt -> commit descriptively -> push the
branch -> open a PR via the GitHub MCP tool. Report the PR URL. Do not author a resolution doc.

Your final report should state: the signature matched, the KV cache layout and how the kernel
strides it, the tile shapes swept and the winner, whether Q lived in registers or shared memory
and the measurement behind that choice, which path decode uses and why, the measured prefill
attention change, the memory-versus-sequence-length curve and the crossover point, the numerical
error at the longest tested length, the counters collected, and the flags appended.
