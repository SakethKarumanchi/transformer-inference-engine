# BENCHMARK_PROTOCOL.md

*This file decides what counts as a real number. In an interview the question behind every figure is "how do you know?" — this document is the answer.*

---

## 1. Prefill and decode are different workloads

Never combined into one figure.

**Prefill** processes the whole prompt at once. Matmuls are matrix-by-matrix, each weight reused across many tokens, arithmetic intensity high, work **compute-bound**. Cache blocking, tiling, and SIMD help here.

**Decode** generates one token at a time. Matmuls are matrix-by-vector: every weight read once, used for roughly one multiply-add. Arithmetic intensity near its floor, work **memory-bandwidth-bound**. Blocking and tiling help far less; what helps is moving fewer bytes.

A single "tokens per second" figure hides this and is the most common way these projects mislead. Report both, always.

## 2. Timing method

- **CPU:** `clock_gettime(CLOCK_MONOTONIC)`. Never wall clock.
- **GPU:** CUDA events around the region, explicit device synchronize before reading. Kernel launches are asynchronous; timing without synchronization measures launch overhead, not execution.
- Timing brackets computation only. Weight loading, tokenization, and startup excluded and reported separately.

## 3. Run structure

- **Warmup:** minimum 5 discarded iterations.
- **Samples:** minimum 20 timed iterations per configuration.
- **Reported:** median, with min, max, std dev recorded alongside.
- **INVALID if:** std dev exceeds 5% of median, another workload was running, thermal throttling occurred, or the machine was on battery. An invalid run is reported as invalid — never averaged away, never silently retried.

## 4. Fixed conditions

Held constant across every stage or the comparison is void: same primary device, same prompt set, same token counts, same seed, same compiler and flags (recorded in the results file), same model and precision, machine otherwise idle.

**Also held constant — the environment fingerprint from `HARDWARE.md` §5.5:** driver version, CUDA toolkit version, compiler version, GPU clock offsets, power limit, and Windows power plan. Every stage re-verifies the fingerprint before its first timed run. If any value has changed since Stage 0, the comparison against prior stages is void until the change is either reverted or the affected prior stages are re-measured. A clock offset that silently changed between Stage 5 and Stage 9 means those two stages measured different machines.

Any deviation is recorded in that stage's `MEASUREMENTS.md` entry. An unrecorded deviation is a fabricated number.

### 4.1 Noise floor

Stage 0 §5.4 establishes the run-to-run spread of the microbenchmark suite. That spread is the noise floor. **No stage may claim a speedup smaller than the noise floor.** A measured 3% improvement on a machine with 5% run-to-run spread is not an improvement; it is noise, and reporting it as a result is the kind of claim that collapses under a single follow-up question. Where a stage's measured change falls inside the noise floor, report it as "no measurable change" and say what the floor is.

### 4.2 Warmup adjustment

The default of 5 warmup iterations assumes clocks settle quickly. Stage 0 §5.3 measures how long clocks actually take to stabilize on this machine. If stabilization takes longer than 5 iterations of the workload being timed, warmup is increased to cover it and the adjusted value is recorded here and used everywhere.

Adjusted warmup for this machine: ______ *(set in Stage 0)*

## 5. Correctness gate

An optimization that changes the output is not an optimization.

- **Oracle:** the PyTorch reference.
- **Check:** logits compared elementwise after a fixed prompt.
- **Tolerance:** relative error threshold set in Stage 3 and recorded here once chosen. Floating-point reassociation makes bit-exactness unachievable across implementations; the tolerance must be justified, not picked arbitrarily.
- **Also required:** greedy-decoded token sequence matches the reference on a fixed prompt set.
- **Flash attention (Stage 9) additionally:** online softmax must be numerically stable. Verify against the reference at long sequence lengths specifically, where a naive running-max implementation loses precision. Report the observed error at the longest tested length.
- **Quantization stages:** tolerance necessarily loosens. The new tolerance and the observed divergence are both results, reported plainly.

No stage is accepted without passing. A faster wrong answer is a regression.

## 6. Profiling is mandatory, not optional

Every GPU stage from Stage 7 onward collects Nsight Compute counters for its kernels. A gap explanation without counter evidence is a story, and stories do not survive follow-up questions.

**Minimum collected per kernel:** achieved occupancy (and theoretical), DRAM read and write throughput, L2 hit rate, shared memory bank conflicts, warp stall reason breakdown, instructions executed, and duration.

**Rule for gap explanations:** name the counter that supports the explanation. Where the counters do not distinguish between candidate causes, write "the counters do not establish which of X or Y dominates" rather than picking one. An honest unestablished cause is correct output.

CPU stages use whatever equivalent is available (`perf`, VTune, or cache-miss counters) and state which.

## 7. The headline numbers

**Decode — percent of measured achievable bandwidth.**
```
bytes_per_token   = weight bytes read + KV cache bytes read + activation traffic
achieved_bw       = bytes_per_token / decode_time_per_token
headline_decode   = achieved_bw / measured_achievable_bandwidth   [HARDWARE.md §1]
```
Denominator is the *measured* figure, never the spec-sheet peak.

**Prefill — percent of cuBLAS.**
```
headline_prefill = cublas_sgemm_time / custom_kernel_time
```
At the model's actual matrix shapes, not square matrices.

**Flash attention — memory scaling.**
Peak memory versus sequence length, swept across at least five lengths, demonstrating the quadratic-to-linear change. Report the crossover point where flash attention becomes necessary on this device's VRAM.

**Never report** a speedup measured only against the project's own naive baseline as a headline figure.

## 8. Performance model validation (Stage 10)

The model is validated like a model, not demonstrated like a demo.

- **Retroactive set:** every kernel configuration measured in Stages 5–9, including the tile-size sweeps. This should yield dozens of points, not a handful.
- **Prospective set:** Stage 11, predicted before implementation. A model tuned on the retroactive set and then tested prospectively is doing real work; one only ever fitted to data it has seen is not.
- **Reported as a distribution:** median absolute percentage error, 90th percentile, worst case, and a breakdown by kernel class (compute-bound versus memory-bound, small versus large).
- **Reference point:** published roofline analysis averages roughly 34% error on deep learning kernels *(NeuSight, arXiv 2407.13853 — verify before citing)*. State where this model sits relative to that.
- **Failure modes are findings.** If the model is badly wrong on one kernel class, identify the class and the missing term. "The model overpredicts small kernels because launch overhead dominates below N microseconds" is a better result than uniform mediocre accuracy.
- **No fitting to the test set.** If a term is added because Stage 11 was mispredicted, that is stated explicitly and Stage 11 stops counting as a prospective test.

## 9. Results file format

Every run writes structured data to `bench/results/`: stage identifier, git commit hash, timestamp, device, compiler flags, prompt set, per-sample raw timings, computed statistics, correctness result, collected Nsight counters, and headline ratios where applicable.

Raw per-sample timings are retained. Summary statistics without the underlying samples cannot be re-examined, and re-examination is the difference between a measurement and an assertion. The Stage 10 model reads these files directly, so the format is a contract.

## 10. Secondary-device rule

Figures from any device other than the primary are labelled with the device name at every appearance and never enter the main waterfall. A comparison run on borrowed or rented hardware is a separate, explicitly-scoped experiment with its own `MEASUREMENTS.md` entry.
