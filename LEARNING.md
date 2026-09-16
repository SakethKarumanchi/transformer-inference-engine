# LEARNING.md

*The learning surface. This file is written by Claude Code during the build — each session appends the concepts that stage actually exercised — and consumed by you afterward, cold, as the input to whatever teaching tool you use.*

**You are not expected to understand any of this while the project is being built.** The build produces the artifact; this file produces the syllabus. Work through it after Stage 12.

**Test for each item: explain it aloud, unaided, with no notes and no AI, in under two minutes, including one follow-up "why."** Anything failing that test does not go on the resume.

Mark each: `unread` → `read` → `can explain` → `can defend a follow-up`.

---

## Cross-cutting

- Memory hierarchy: registers, L1, L2, L3, DRAM — latency and bandwidth at each level, and why the ratios matter more than the absolute numbers
- Arithmetic intensity: FLOPs per byte moved, computed for a given kernel
- The roofline model: reading one, placing a kernel on one, knowing which ceiling binds
- Compute-bound versus memory-bound, and how to tell which you are looking at
- Why theoretical peak bandwidth is never reached, and what the shortfall comes from
- Amdahl's law across a staged optimization sequence

## Stage 1 — Weights and tokenizer
- FP32 layout; what precision means in bits
- Why floating-point addition is not associative, and why bit-exact comparison across implementations is therefore impossible
- Byte-pair encoding: how the merge table is built and applied
- Tensor memory layout: row-major versus column-major, strides, why layout changes performance

## Stage 2 — Transformer forward pass
- Self-attention mechanically: Q, K, V projections, scaled dot product, softmax, output projection
- Why attention is quadratic in sequence length
- Multi-head attention: what the split buys, how heads are laid out in memory
- Layernorm: what it normalizes over, why it sits where it does
- Feed-forward block: shape, expansion ratio, why it holds most of the parameters
- Where parameters live versus where *time* goes — and why those distributions differ

## Stage 3 — Measurement
- Warmup effects: page faults, clock ramp, instruction cache
- Why median and not mean
- CUDA event timing and why device synchronization is required before reading a timer
- Choosing a numerical tolerance defensibly

## Stage 4 — KV cache
- Why decode without a cache is quadratic in generated tokens
- What is cached, and its footprint as a function of sequence length, layers, and heads
- Why the KV cache makes decode increasingly bandwidth-bound as context grows
- The memory-versus-recompute tradeoff, and when recompute would win

## Stage 5 — Cache blocking
- Why a naive triple loop misses cache, in terms of access pattern and line reuse
- Working set: computing it for a tile, sizing tiles against measured cache
- How blocking changes arithmetic intensity, quantitatively
- Cache associativity and conflict misses; why some tile sizes underperform their neighbours
- **Why blocking helps prefill and barely helps decode** — the central lesson

## Stage 6 — SIMD
- Vector registers and lane width on this specific CPU
- Why the theoretical ceiling is the vector width, and why real code falls short
- Load/store port pressure
- Alignment requirements and the cost of misalignment
- Why compiler auto-vectorization sometimes matches hand intrinsics and sometimes does not

## Stage 7 — CUDA port
- GPU execution model: thread, warp, block, grid, SM
- Warp divergence and its cost
- GPU memory hierarchy: global, shared, L2, registers, constant
- Coalesced access, and what uncoalesced access costs
- Occupancy: definition, what limits it, why maximum occupancy is not always optimal
- PCIe transfer cost and why weight residency changes the measurement entirely
- Kernel launch overhead and when it dominates
- Reading an Nsight Compute report: which counters answer which question

## Stage 8 — Tiled shared-memory kernel
- Shared memory as a software-managed cache
- Bank conflicts: cause, and how padding avoids them
- The tile-size three-way tradeoff: shared memory per block, registers per thread, occupancy
- `__syncthreads()` semantics and cost
- What cuBLAS does that a hand-written kernel does not, and why the gap remains

## Stage 9 — Flash attention
- Why naive attention materializes an N×N matrix in global memory, and what that costs
- **Online softmax** — computing exact softmax incrementally without seeing the full row: the running maximum, the running denominator, and the rescaling correction applied to previous partial outputs when a new tile contains a higher max
- Why naive tiling of softmax is incorrect, and precisely what the correction fixes
- Numerical stability: why the running max exists at all, and what breaks without it
- Kernel fusion: why the GEMMs, mask, softmax, and output GEMM must be one kernel, and why cuBLAS calls cannot be fused
- IO-awareness: counting HBM traffic rather than FLOPs as the thing to minimize
- Why memory goes from quadratic to linear in sequence length
- Why the recomputation trick exists in the backward pass, and why inference does not need it
- What `cp.async` and `ldmatrix` do, why they require sm_80, and what a sm_75 implementation gives up

## Stage 10 — Performance model
- What an analytical performance model is, and how it differs from profiling
- The roofline model as a predictive tool rather than a descriptive one
- Which terms a useful model needs: bytes moved, FLOPs, occupancy, launch overhead, achieved versus theoretical bandwidth
- Why launch overhead must be a separate additive term, and what happens to predictions without it
- Why models break on out-of-distribution kernels
- Fitting versus validating: why a prospective test is worth more than any amount of retroactive agreement
- Reading an error distribution: why median, p90, and worst case say different things
- Where published roofline models sit (~34% average error on DL kernels) and why that is the honest bar

## Stage 11 — Systolic dataflow
- What a systolic array is and why it was proposed
- Weight-stationary versus output-stationary versus input-stationary, and each tradeoff
- Processing-element utilization during fill and drain
- Why each weight being fetched once is the property that matters
- Why a software simulation may be slower than a tiled kernel, in synchronization terms
- **What tensor cores actually are**, stated accurately, and why the systolic framing is a TPU analogy

## Stage 12 — Presentation
- Nothing technical. Test: does the chart make the prefill/decode asymmetry visible to someone who hasn't read the code?

## Optional stages
- **Autotuner:** search space size, pruning, why an approximate model beats exhaustive search
- **Speculative decoding:** why verifying k tokens costs roughly one forward pass, acceptance rate, why it attacks the memory-bound decode problem specifically
- **INT8 quantization:** symmetric versus asymmetric, per-tensor versus per-channel scales, where accuracy is actually lost, why dequantize-fused matmul beats a separate dequantize pass

---

## The six questions to survive

Answerable cold. Any that fails means that stage is not defensible and should not appear on the resume.

1. Why is decode memory-bound and prefill compute-bound, and what follows from that?
2. How does online softmax stay exact without seeing the whole row?
3. Which optimization gave the biggest gain, and why *that* one?
4. Where was your performance model most wrong, and what term was missing?
5. What percent of the machine's ceiling did you reach, and what accounts for the gap?
6. What would you do next with more time, and why that rather than something else?
