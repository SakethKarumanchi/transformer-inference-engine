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

## Stage 0 — Instrumenting the machine
- Why a spec sheet is not a measurement, and what the gap is made of on a power- and thermally-limited part `unread`
- Boost clock versus sustained clock: why a card that advertises 1485 MHz settles at 1365 MHz under load `unread`
- Telling thermal governance from power limiting by watching power draw and clock move together or apart `unread`
- Clock locking as a measurement instrument rather than a performance setting — `nvidia-smi -lgc`, why pinning the minimum as well as the maximum removes the warmup problem entirely `unread`
- Why a locked clock at a *lower* frequency produces better science than an unlocked clock at a higher one `unread`
- GPU throttle reason bitmasks: `sw_power_cap`, `sw_thermal_slowdown`, `hw_slowdown`, `gpu_idle`, and what each one tells you `unread`
- Median versus mean, and standard deviation as a fraction of median as a validity criterion rather than a summary statistic `unread`
- Why a single interrupted sample invalidates a run, and why averaging it away is falsification `unread`
- The noise floor: measuring it from two full suite runs, and why it sets the minimum claimable speedup `unread`
- Choosing a percentile rather than a mean or a max when adopting a noise floor, and what each choice licenses `unread`
- Warmup: what it is actually warming — clock ramp, caches, TLB, the driver's launch path `unread`
- Monotonic counters versus wall clock; `QueryPerformanceCounter` on Windows as the equivalent of `clock_gettime(CLOCK_MONOTONIC)` `unread`
- CUDA event timing and why an explicit device synchronize must precede reading the timer `unread`
- Theoretical memory bandwidth from first principles: transfers per clock x memory clock x bus width / 8 `unread`
- Why a copy benchmark reaching 89% of theoretical is a good result, and what the missing 11% is `unread`
- Corroborating an event-timed bandwidth figure with DRAM throughput counters, and why agreement between two independent methods is the actual evidence `unread`
- Theoretical FP32 peak from core count x clock x 2, and why the core count is a spec-table lookup rather than a queryable property `unread`
- Why a register-resident FMA loop can reach 99.7% of theoretical while nothing else can `unread`
- The cache ladder: reading plateau edges off a bandwidth-versus-working-set curve to infer effective cache sizes `unread`
- Why measured cache edges can disagree with what the OS reports, and which one later work should use `unread`
- Single-channel versus dual-channel memory, and why one DIMM halves the CPU memory ceiling `unread`
- Pinned versus pageable host memory, and why pinned is roughly twice as fast in both directions `unread`
- Kernel launch overhead as a fixed additive term, and why back-to-back launches and synchronize-per-launch are different numbers that answer different questions `unread`
- Shared memory bank conflicts: why a 32-way conflict costs roughly 30x, measured rather than assumed `unread`
- Occupancy: theoretical from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` versus achieved from the profiler, and why they must never be conflated `unread`
- Register pressure as an occupancy limiter, and computing which limiter binds from queried device properties `unread`
- Why GEMM throughput is non-monotonic in M on a small-SM device, and what a partially filled final wave costs `unread`
- Why the prefill denominator must be quoted per shape rather than as a single number `unread`
- Prefill versus decode as different shapes of the same GEMM, and why M=1 must never be averaged in with M=512 `unread`
- Hardware counters: what `sm__warps_active.avg.pct_of_peak_sustained_active`, `lts__t_sector_hit_rate.pct`, `dram__bytes_read.sum.per_second` and the warp-stall ratios actually measure `unread`
- Warp stall reasons as a diagnostic vocabulary, and why `long_scoreboard` dominating means waiting on global memory `unread`
- Resolving profiler metric names against the installed tool version instead of hardcoding them, and why hardcoded names rot `unread`
- Why an uncollected counter must be reported as unavailable rather than as zero `unread`
- Determining a hardware capability empirically when no API exposes it — the WMMA/HMMA probe `unread`
- The limits of counter evidence: why `sm__pipe_tensor_cycles_active` proves a pipe exists but establishes no throughput `unread`
- Distinguishing a permission limit from a device-capability limit by exit code and message text `unread`
- VRAM integrity testing with known bit patterns, and why consumer GDDR6 without ECC fails silently rather than loudly `unread`
- Compute determinism as a hardware check rather than a code check `unread`
- The environment fingerprint: what must be frozen for two measurements taken weeks apart to be comparable `unread`
- Why background processes show up as invalid runs rather than as uniformly slower runs `unread`
- Provenance tagging — `[queried]`, `[measured]`, `[spec]`, `[derived]` — as a discipline that makes a document auditable `unread`
- Why a field left empty with a stated reason is more valuable than a plausible number `unread`
- Toolchain pinning: why CUDA rejects newer MSVC toolsets, and why `-allow-unsupported-compiler` is not a fix `unread`
- Why the profiler, not the compiler, decided native Windows over WSL2 for this project `unread`

## Stage 0b — Diagnosing an invalid measurement
- **Deviation direction as a diagnostic.** Upward excursions from a clean floor can only be contention; downward excursions from a ceiling cannot be contention at all, because interference never makes a sample faster. Reading direction before reading magnitude separates a machine problem from a benchmark problem for free `unread`
- Robust statistics on timing data: median absolute deviation, the modified z-score, and why the standard deviation cannot be used to find the samples that inflated it `unread`
- Distinguishing a spike-carried distribution from a broadly dispersed one: share of total squared deviation held by flagged samples, versus the interquartile range, which ignores the tails entirely `unread`
- Why trimmed statistics are a diagnostic and never a result, and how to structure an output file so a trimmed number cannot be mistaken for a measurement `unread`
- SMT (hyper-threading) and per-core cache: why L1d and L2 are per-core while L3 is shared, and what a thread migration costs in re-warm `unread`
- Thread affinity and scheduling priority as measurement conditions, not tuning knobs — and why a single-threaded benchmark that does not say where its thread ran is sampling more than one machine `unread`
- `GetLogicalProcessorInformationEx` and logical-to-physical core mapping: why the conventional interleaving is a convention rather than a guarantee `unread`
- Windows performance counters through PDH: rate counters as differentials, why `% Processor Performance` is live and APERF/MPERF-derived while `Processor Frequency` is a static nominal read `unread`
- Proving a telemetry source is live before trusting it: apply a load, require the reading to move and to recover, and measure the per-probe cost against the sample duration it will be used alongside `unread`
- Why a probe costing milliseconds cannot instrument a millisecond measurement, and why telemetry belongs outside the timed bracket under all circumstances `unread`
- CPU hardware performance counters on Windows: the ETW PMU path through the Windows Performance Toolkit (`xperf -pmcsources`), what a Comet Lake PMU exposes, and the limit of 7 simultaneously selectable sources `unread`
- Why an observability tool is itself a load, and when the trace is worth the condition change it causes `unread`
- Shared-L3 contention as a measurement condition: why a working set sized at exactly the L3 capacity is the worst case, and why its bandwidth can exceed the DRAM ceiling `unread`
- Single-channel versus dual-channel memory, and reading a measured figure against the channel-count-derived ceiling `unread`
- Sampling arithmetic: why one sample at +50% among 30 produces a standard deviation near 9.3% of median, and why sub-100-microsecond configurations therefore cannot pass a 5% rule reliably `unread`
- Code provenance versus condition provenance: why `BENCHMARK_PROTOCOL.md` §4 voids a comparison across either, and why a partial re-run conceals a condition change rather than avoiding it `unread`
- Verifying a fix by its predicted signature rather than by its headline number — the new median landing on the old *minimum* is evidence about mechanism that a smaller standard deviation alone is not `unread`
- Writing up a diagnosis that the outcome falsified, and stating which alternatives the evidence still does not separate `unread`

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
