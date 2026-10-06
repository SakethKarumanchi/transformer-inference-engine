# BENCHMARK_PROTOCOL.md

*This file decides what counts as a real number. In an interview the question behind every figure is "how do you know?" — this document is the answer.*

---

## 1. Prefill and decode are different workloads

Never combined into one figure.

**Prefill** processes the whole prompt at once. Matmuls are matrix-by-matrix, each weight reused across many tokens, arithmetic intensity high, work **compute-bound**. Cache blocking, tiling, and SIMD help here.

**Decode** generates one token at a time. Matmuls are matrix-by-vector: every weight read once, used for roughly one multiply-add. Arithmetic intensity near its floor, work **memory-bandwidth-bound**. Blocking and tiling help far less; what helps is moving fewer bytes.

A single "tokens per second" figure hides this and is the most common way these projects mislead. Report both, always.

## 2. Timing method

- **CPU:** the platform's monotonic counter. On this machine that is `QueryPerformanceCounter`: `clock_gettime(CLOCK_MONOTONIC)` is the POSIX equivalent and is not available under MSVC, and every results file from Stage 0 onward records `QueryPerformanceCounter`, as does `bench/microbench/bench_common.h`. Never wall clock.
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

**Measured noise floor for this machine, set by Stage 0 on 2026-09-17: 4.4%.**

Measurement behind it: the full microbenchmark suite was run twice in one session under the locked clock regime (`bench/results/run1/` and `bench/results/run2/`), and the per-configuration spread computed as `|b - a| / mean(a, b) * 100` by `machine_state.py spread_percent`. Over the 49 configurations that were VALID in both runs: **median 0.317%, mean 1.407%, 90th percentile 4.401%, maximum 19.913%.** Over all 96 configurations regardless of validity: mean 2.059%, maximum 19.913%. Full output in `bench/results/machine_state_5_4_timing_spread.json`.

The adopted floor is the 90th percentile, 4.401%, rounded to **4.4%**, rather than the median. The median of 0.317% describes the best-behaved configurations and would license claims this machine cannot actually support; the maximum of 19.913% comes from a single small-M GEMM and would suppress real results everywhere else.

Two qualifications travel with the figure and must be repeated wherever it is used:

1. **Small-M GEMM shapes are materially noisier than everything else.** The worst spreads all came from `cublas_sgemm_ref` at M = 16 to 32, where launch and setup latency dominate the arithmetic. A claim about **decode**-shaped work (M = 1) needs a wider margin than 4.4%; a claim about **prefill**-shaped work at large M is comfortably inside it.
2. **This floor was measured on a machine with irreducible background load** — a kernel-mode anti-cheat driver, an audio driver that hooks process creation, and Windows Defender real-time scanning, none of which could be closed. It is a floor for *this* machine as it actually runs, which is the right floor for this project, but it is not a floor for an idle headless machine.

### 4.2 Warmup adjustment

The default of 5 warmup iterations assumes clocks settle quickly. Stage 0 §5.3 measures how long clocks actually take to stabilize on this machine. If stabilization takes longer than 5 iterations of the workload being timed, warmup is increased to cover it and the adjusted value is recorded here and used everywhere.

Adjusted warmup for this machine: **25 iterations** *(set in Stage 0, 2026-09-17)*

Arithmetic behind it. Stage 0 §5.3 measured clock stabilization under the locked regime at **0.075 s**. The reference timed workload is `gpu_bandwidth`, whose measured median iteration is **3.118 ms**. The protocol default of 5 warmup iterations covers 5 x 3.118 ms = **15.6 ms**, which is *less* than 0.075 s and therefore insufficient by the rule above. 25 warmup iterations cover 25 x 3.118 ms = **77.9 ms**, which exceeds 0.075 s. 25 was used for every configuration in Stage 0 and is used everywhere from here.

A note on why this is conservative rather than strictly required: the project locks the graphics clock with `nvidia-smi -lgc 1365,1365`, which pins the **minimum** as well as the maximum, so the card already sits at its operating clock before the first warmup iteration and there is no boost ramp to warm through — the 5.3 log recorded 1365 MHz on 122 of 122 samples including idle. The 25 iterations still do useful work warming caches, TLBs and the driver's kernel-launch path. Were the lock ever absent, the unlocked card's stabilization figure of 3.502 s would apply instead, and it understates the problem badly: the unlocked card does not stabilize at all, it decays from 1815 MHz to 1365 MHz over five minutes. **No warmup value can rescue a measurement taken on an unlocked clock here; the lock is not optional.**

## 5. Correctness gate

An optimization that changes the output is not an optimization.

- **Oracle:** the PyTorch reference.
- **Check:** logits compared elementwise after a fixed prompt.
- **Tolerance — D2. The value in force is `2.3e-03`, an ABSOLUTE difference in logit units, RE-DERIVED by Stage 4 on 2026-10-05 under the corrected rule below.** D2 was first RESOLVED in Stage 3 on **2026-10-04 local time** — the date of the correctness run the resolution rests on, `bench/results/stage3/stage3_correctness.json`, stamped `2026-10-05T04:43:45Z`, which is 2026-10-04 21:43:45 at this machine's UTC-07:00 — at `6e-03`. **That value is SUPERSEDED**; the Stage 3 derivation below is kept as provenance, with the superseded statements marked. Floating-point reassociation makes bit-exactness unachievable across implementations, so the tolerance must be justified, not picked arbitrarily. Machine-readable form, which `bench/correctness.py` parses so the value is read rather than duplicated in code — and there is **exactly one** such line in this document, which the parser now enforces: `D2_MAX_ABS_LOGIT_DIFF = 2.3e-3`

  **The form of the check.** The gate figure is the maximum **elementwise ABSOLUTE** difference over the **full logit vector at every position** of a prefill, computed in float32 on both sides against the PyTorch reference. It is absolute rather than relative because the upper bound that makes the tolerance meaningful — the gap between the top-1 and top-2 reference logits — is a quantity in absolute logit units, and a relative threshold cannot be compared against it. Relative statistics are still computed and reported — maximum relative difference against a denominator of `|reference| + 1e-6` elementwise, with the count of elements below that floor stated — but **they do not gate**. `gpt2_tool --dump-logits` emits logits for every position, so both bounds below are measured over the whole prefill and not over a subset.

  **The application rule.** PASS requires all three, and any one failing is a FAIL reported as a FAIL: (1) maximum absolute difference **at or below** the value in force — `2.3e-03` since the Stage 4 amendment, ~~`6e-03`~~ before it; (2) top-1 agreement at **every** position; (3) the greedy-decoded token sequence matches the reference on the fixed prompt set (§4). A difference exactly equal to the threshold passes, which is what makes the threshold a stated boundary rather than an open interval whose edge nobody can test. The check is implemented once, in `bench/correctness.py`, and returns a FAIL rather than raising, so timings honestly taken are still recorded as taken.

  **Measured figures the arithmetic rests on** — *these are Stage 3's, retained as provenance; Stage 4 re-measured every one of them and reproduced all four rows exactly, including the set-wide minimum margin, over two additional engine paths* — from `bench/results/stage3/stage3_correctness.json`, the D3 fixed prompt set, the Stage 2 engine, `--cproj as-stored`:

  | D3 prefill length | max absolute divergence | minimum reference top-1/top-2 margin | position of that minimum | margin ÷ divergence | top-1 agreement |
  |---|---|---|---|---|---|
  | L = 16 | 3.967285e-04 | 5.915833e-02 | 13 | 149.12x | 16 of 16 |
  | L = 32 | 4.272461e-04 | **7.812500e-03** | 11 | 18.29x | 32 of 32 |
  | L = 64 | 4.425049e-04 | 4.189301e-02 | 1 | 94.67x | 64 of 64 |
  | L = 128 | **7.019043e-04** | 4.366302e-02 | 118 | 62.21x | 128 of 128 |

  **Condition (a) — the LOWER bound, from observed divergence at the longest D3 length.** *(SUPERSEDED by the Stage 4 amendment, which takes this bound over the WHOLE set and over both engine paths rather than at the longest length. Retained as provenance.)* The threshold must exceed the maximum observed absolute divergence by a factor of at least 3. At L = 128 that divergence is **7.019043e-04**, so the lower bound is `3 x 7.019043e-04` = **2.105713e-03**.

  **Condition (b) — the UPPER bound, from the minimum decision margin at the longest D3 length.** *(SUPERSEDED by the Stage 4 amendment, which takes this bound over the WHOLE set. This is the rule whose premise failed; see the amendment for why. Retained as provenance.)* The threshold must fall below the minimum reference top-1/top-2 margin by a factor of at least 3. At L = 128 that margin is **4.366302e-02** at position 118, so the upper bound is `4.366302e-02 / 3` = **1.455434e-02**.

  **Condition (d) — the window is not too narrow to use.** `4.366302e-02 / 7.019043e-04` = **62.21x**, comfortably above the required factor of 9 end to end.

  **Condition (c) — the value.** *(SUPERSEDED. The value this paragraph selects, `6e-03`, is no longer in force.)* The geometric mean of the two bounds is `sqrt(2.105713e-03 x 1.455434e-02)` = **5.535997e-03**. Rounded to one significant figure that is **6e-03**, and 6e-03 keeps both ratios at or above 3. **Achieved safety factors: 6e-03 / 7.019043e-04 = 8.55x above the observed divergence, and 4.366302e-02 / 6e-03 = 7.28x below the minimum margin.** (5e-03 also clears both at 7.12x and 8.73x; 6e-03 is what the stated rounding rule selects.)

  **Which bound is a property of what, and the circularity that remains.** The lower bound is a property of **this engine's current divergence** and moves if the engine changes — a Stage 5 blocking rewrite reassociates differently and will produce its own figure. The upper bound is a property of the **reference alone**, and no change to the engine can move it. That asymmetry is why the upper bound is the one that makes the check meaningful: a threshold justified only against the thing being measured is justified by its own subject.

  **A qualification that travels with the figure, and must be repeated wherever it is used.** The minimum margin over the **whole** D3 set is **7.812500e-03**, at L = 32 position 11 — smaller than at L = 128. The margin therefore does **not** degrade monotonically with length across this set, because the four D3 rows are **independent prose and not nested prefixes of one string**: each row's margin minimum is a property of its own text. Stage 2's observation that the minimum margin more than halved from L = 8 to L = 16 was a nested-prefix effect, where adding positions can only lower a minimum, and it does not generalise here. Against that set-wide minimum, 6e-03 leaves only **1.30x**, below the factor of 3 condition (b) asks for. Had the upper bound been evaluated set-wide the window would be `[2.105713e-03, 2.604167e-03]`, a span of only 1.24x end to end, and **no one-significant-figure value lies inside it**: 2e-03 fails the lower bound at 2.85x and 3e-03 fails the upper at 2.60x. The set-wide margin-to-divergence factor is 11.13x, which does still clear condition (d). What protects the gate in practice is that condition (2) is checked **directly** rather than inferred from the tolerance: top-1 agreement was complete at all four lengths, 16 of 16, 32 of 32, 64 of 64 and 128 of 128.

  **Corroboration, not input.** Stage 2 measured 3.1281e-04 at L = 8 and at L = 16 on placeholder inputs. The D3 figures of 3.97e-04 to 7.02e-04 are the same order of magnitude; unlike Stage 2's two nested lengths, divergence here does grow slowly with length. The Stage 2 figures were not used to derive either bound.

  **Amended by Stage 4 on 2026-10-05.** The value in force is now **`2.3e-03`**. The window was re-derived because it was open at Stage 4 and may not be after Stage 5 reassociates the matmul.

  **Why the previous rule's premise failed.** Stage 3's conditions (a) and (b) were both evaluated **at the longest D3 length**, on the premise that the decision margin degrades with length — so that the longest row would be the binding one. In the same section Stage 3 established that this premise is false for this prompt set: the four D3 rows are **independent prose, not nested prefixes of one string**, so each row's margin minimum is a property of its own text, and the set-wide minimum falls at **L = 32, position 11** rather than at L = 128. The "margin more than halved from L = 8 to L = 16" observation that motivated the rule was a nested-prefix effect, where adding positions can only lower a minimum. Evaluating the upper bound at the longest length therefore used a margin **5.6 times larger** than the one the gate actually has to respect (4.366302e-02 against 7.812500e-03), and the committed `6e-03` fails condition (b) against the set-wide minimum at **1.30x**, where (b) requires 3x. The rule, not the measurement, was wrong.

  **The corrected rule.**

  - **Upper bound:** the minimum reference top-1/top-2 margin over the **WHOLE** D3 set — all positions of all four rows — divided by 3. A property of the **reference alone**, which no change to the engine can move.
  - **Lower bound:** **3x** the maximum observed absolute divergence over the **WHOLE** D3 set, taken over **BOTH** engine paths the measuring stage exercises — for Stage 4, the prefill path and the cached-decode path — using that stage's **own** measurement, never an earlier stage's.
  - **Window check:** margin ÷ divergence must be at least **9**. (Note, worth stating because it is not obvious from the wording: this is the same condition as `lower < upper`, since `3d < m/3` is `m/d > 9`. A ratio below 9 presents as a closed window.)
  - **Value:** the **geometric mean** of the two bounds, rounded to the **FEWEST significant figures at which some value lies inside the window**, starting at one; among the values at that precision lying inside the window, the one **nearest the geometric mean in log distance**. The safety bound decides the rounding; the rounding never decides the safety bound.
  - **Both achieved factors are reported.**

  **The arithmetic, from Stage 4's own measurement** (`bench/results/stage4/stage4_correctness.json`, the D3 fixed prompt set, `--cproj as-stored`, the Stage 4 engine, over the prefill path **and** the cached-decode path):

  - Set-wide maximum absolute divergence over both paths: **7.019043e-04**, at L = 128. The cached-decode path's divergence is **identical** to the prefill path's at every one of the four rows, because the two paths agree **bit for bit** (0 differing elements of 12,061,680 over the set), so taking the maximum over both paths does not raise it.
  - Set-wide minimum reference top-1/top-2 margin: **7.812500e-03**, at **d3_32, position 11** — reproduced **exactly** against Stage 3's figure, which is what licenses its use here.
  - Lower bound: `3 x 7.019043e-04` = **2.105713e-03**.
  - Upper bound: `7.812500e-03 / 3` = **2.604167e-03**.
  - Window: `7.812500e-03 / 7.019043e-04` = **11.130x**, above the required 9x. The window is **OPEN**.
  - Geometric mean: `sqrt(2.105713e-03 x 2.604167e-03)` = **2.341715e-03**.
  - Rounding: **no one-significant-figure value lies inside** `[2.105713e-03, 2.604167e-03]` — 2e-03 is below the lower bound and 3e-03 is above the upper. At two significant figures the candidates inside the window are **2.2e-03, 2.3e-03, 2.4e-03, 2.5e-03 and 2.6e-03**. Nearest the geometric mean in log distance is **2.3e-03** (log distance 0.0180, against 0.0246 for 2.4e-03).
  - **Achieved factors: `2.3e-03 / 7.019043e-04` = 3.28x above the observed divergence, and `7.812500e-03 / 2.3e-03` = 3.40x below the minimum margin.** Both clear the required 3x, which the superseded `6e-03` did not on the upper side.

  **What the amendment does not change.** The form of the check, the three PASS conditions, the absolute-rather-than-relative choice, and the fact that condition (2) — top-1 agreement at every position — is checked **directly** rather than inferred from the tolerance. Stage 4 recorded a PASS against **both** values: against the superseded `6e-03` and against the re-derived `2.3e-03`, on the prefill path in both engine modes.

  **The circularity that still remains.** Unchanged from Stage 3's statement of it: the lower bound is a property of **this engine's current divergence** and will move when Stage 5's blocking rewrite reassociates differently. The upper bound remains a property of the reference alone. Stage 4 widened the evidence for the lower bound — two engine paths instead of one — but did not remove the asymmetry.
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
