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
**Status:** complete, 2026-09-17
Record: any figure where measured reality diverged notably from the spec sheet, with a hypothesis. Whether Nsight Compute can collect counters here. Confirmed compute capability and tensor-core presence.

**Device:** NVIDIA GeForce GTX 1650 Ti (TU117), compute capability 7.5, 16 SM, 4 GB, 55 W hard power limit — `HARDWARE.md` §1.

#### Conditions

Deviations from `BENCHMARK_PROTOCOL.md` §4 are recorded here rather than left implicit.

- **This session ran ELEVATED**, deliberately and as an exception to the project's usual posture, so that it could apply `nvidia-smi -lgc` itself rather than handing the command to the operator (`IsInRole(Administrator)` = True, user `MSI\saket`). The separate finding that **Nsight Compute counter collection works for a NON-elevated user** was established by the operator before this session and was **not re-tested here**.
- **The session was unattended.** The operator started it and was not available to answer. The hard checkpoint was produced as a written report and the proceed sequence ran without waiting. Every decision taken without the operator is listed at the end of this entry.
- **Graphics clock locked at 1365 MHz** for every figure below, applied at step 1a and verified in effect (not assumed) before any microbenchmark ran. **Memory clock is NOT locked and cannot be** on this device, so the clock regime is partially, not fully, controlled.
- **Warmup 25 iterations, 30 timed samples** per configuration, against protocol minima of 5 and 20.
- **The machine was not idle in the strict sense and could not be made so.** A kernel-mode anti-cheat driver (Riot Vanguard, `vgc` / `vgtray`), an audio driver that hooks process creation (Nahimic), MSI vendor services, and Windows Defender real-time scanning were all resident throughout. The operator had already closed overlay and monitoring software; this is what remained. Nothing was closed by this session.
- **The full suite was run twice**, as `BENCHMARK_PROTOCOL.md` §5.4 check 3 requires, to establish run-to-run spread. Run 2 is the reference run. Raw per-sample timings for both are retained: `bench/results/run1/` and `bench/results/run2/`.
- **One discarded run.** Early in the session `build/gpu_bandwidth.exe --help` was invoked; that binary has no such flag and ran the benchmark, writing a results file before the machine-state sequence and before the clock lock. That file was deleted and its number discarded. Microbenchmark 1 was re-run under protocol conditions.

Raw samples: `bench/results/` — per-benchmark files, `stage0_consolidated.json`, `run1/`, `run2/`, `machine_state_5_3_sustained.json`, `machine_state_5_3_sustained_locked.json`, `machine_state_5_4_stability.json`, `machine_state_5_4_timing_spread.json`, `profile_gpu_bandwidth.json`, `profile_occupancy_sweep.json`, `stage0_drift_run1_vs_run2.json`.

#### The nine measured values

Reference run 2 unless stated. Std dev is as a percentage of median, the protocol's own validity criterion.

| # | Microbenchmark | Value | Std dev | Verdict |
|---|---|---|---|---|
| 1 | GPU bandwidth (device-to-device, float4 grid-stride, 512 MiB working set = 512x L2) | **170.882 GB/s** | 2.54% | VALID (run 1 INVALID at 5.84%) |
| 2 | GPU FP32 peak (32 register-held FMA chains, no global traffic) | **2786.49 GFLOP/s** | 2.20% | VALID (run 1: 2787.41, 1.89%, VALID) |
| 3 | CPU cache ladder — L1-resident (8 KiB) | **127.38 GB/s** | 1.21% | VALID. **SUPERSEDED by Stage 0b 2026-09-18: 143.85 GB/s at 4 KiB, 1.272%, VALID.** The 8 KiB point was INVALID in Stage 0b run 2 (6.365%) |
| 3 | CPU cache ladder — L2-resident (256 KiB) | **99.64 GB/s** | 1.62% | VALID. **SUPERSEDED by Stage 0b 2026-09-18: 113.85 GB/s at 256 KiB, 4.457%, VALID** |
| 3 | CPU cache ladder — L3-resident (4 MiB) | **62.31 GB/s** | 4.77% | VALID. **SUPERSEDED by Stage 0b 2026-09-18: 70.81 GB/s at 4 MiB, 4.621%, VALID** |
| 3 | CPU cache ladder — DRAM-resident | *(no value)* | 5.32%–16.45% | **INVALID in both runs — field left empty.** Stage 0b 2026-09-18 re-measured and it is **still INVALID in both runs** (5.35–21.46%); the field stays empty with a better reason |
| 4 | CPU SIMD peak — vectorised AVX2 | *(no value)* | 5.60% (run 1: 5.86%) | **INVALID in both runs — field left empty.** **RESOLVED by Stage 0b 2026-09-18: 48.411 GFLOP/s, 2.773%, VALID (run 1: 49.506, 1.075%, VALID)** |
| 4 | CPU SIMD peak — scalar reference of the same loop | 8.347 GFLOP/s | 2.51% | VALID. Stage 0b: 8.565 GFLOP/s, 1.847%, VALID |
| 5 | Host-to-device, pinned | **12.589 GB/s** | 1.97% | VALID |
| 5 | Device-to-host, pinned | **10.573 GB/s** | 4.56% | VALID |
| 5 | Host-to-device, pageable | **5.528 GB/s** | 4.37% | VALID |
| 5 | Device-to-host, pageable | **4.952 GB/s** | 4.42% | VALID (run 1 INVALID at 17.93%) |
| 6 | cuBLAS SGEMM — best VALID **prefill** figure observed | **2344.35 GFLOP/s** (ffn_down, M=512, N=768, K=3072) | 0.40% | VALID in run 1; the same configuration was INVALID in run 2 at 8.78%. Best prefill figure VALID in the reference run: **1943.01 GFLOP/s** (attn_output_projection, M=512, N=768, K=768, 4.51%) |
| 6 | cuBLAS SGEMM — **decode** (M=1), reported separately and never merged with prefill | ffn_up **72.64**, ffn_down **61.66**, lm_head **85.13** GFLOP/s | 3.40% / 1.86% / 1.66% | VALID. qkv_projection and attn_output_projection at M=1 were **INVALID in both runs** and have no value <br><br>**CORRECTION, Stage 0b, 2026-09-18 — the three figures above are NOT valid in both runs and the word VALID in this row is wrong.** The error was found by re-reading the retained per-sample arrays in `bench/results/run1/` and `run2/`, and this entry's own "Every INVALID run" table below already contradicted it. File truth, both runs: qkv_projection M=1 **5.21% / 5.34%** (INVALID in both); attn_output_projection M=1 **7.41% / 9.53%** (INVALID in both); ffn_up M=1 **10.42% run 1 INVALID** / 3.40% run 2 valid; ffn_down M=1 **13.82% run 1 INVALID** / 1.86% run 2 valid; lm_head M=1 1.66% run 1 valid / **11.20% run 2 INVALID**. So: **all five M=1 shapes were INVALID in at least one run**; only qkv_projection and attn_output_projection failed in both; and the quoted lm_head figure of 85.13 GFLOP/s at 1.66% is a **run-1 number quoted in an entry that names run 2 as the reference run**, where that configuration was INVALID at 11.20%. **Which M=1 shape fails is not stable between runs**, so this is a stochastic hit rate under the 30-sample rule, not a property of any shape. The original text above is left intact as the record of what was written. The three figures must not be used as valid decode denominators. This error also propagated into the prompt briefing the Stage 0b session, which repeated the claim; the retained data corrected it. `cublas_sgemm_ref.cu` was NOT modified and NOT re-run by Stage 0b — it is the prefill denominator and `PERSISTENT.md` D3 records its M sweep as a placeholder Stage 3 replaces |
| 7 | Kernel launch overhead — synchronize per launch | **13.104 µs** | 3.40% | VALID (run 1 INVALID at 7.37%) |
| 7 | Kernel launch overhead — back-to-back, one sync per batch of 1000 | **8.275 µs** | 3.18% | VALID |
| 8 | Shared memory bandwidth — conflict-free | **3069.52 GB/s** | 0.79% | VALID |
| 8 | Shared memory bandwidth — 32-way bank conflict | **100.00 GB/s** | 2.20% | VALID |
| 9 | Occupancy sweep — achieved vs theoretical, from the profiler | 98.93 / 97.98 / 97.93 % achieved against 100% theoretical at 12 / 36 / 64 regs; **48.82% achieved against 50.00% theoretical at 96 regs (register-limited)** | — | VALID; achieved occupancy read from Nsight Compute, never assumed equal to theoretical |

CPU timing used `QueryPerformanceCounter` (the highest-resolution monotonic counter on this platform; `clock_gettime` is not available under MSVC). GPU timing used CUDA events with an explicit `cudaDeviceSynchronize` before reading.

#### Where measured reality diverged from the spec sheet

**This is the first honest number in the project, and there are six of them.**

**1. Tensor cores are present on a die the project expected to lack them.** `HARDWARE.md` §1 and `PERSISTENT.md` Q6 both expected TU117 to ship without tensor cores. Measured: a WMMA 16x16x16 HMMA kernel compiled for `sm_75`, ran, and returned all 256 elements exactly 32.0; Nsight Compute reported `sm__inst_executed_pipe_tensor_op_hmma.sum` = 4 and `sm__pipe_tensor_cycles_active.sum` = 512. *Hypothesis:* the GTX 16-series is a market-segmentation product built on the same Turing SM; NVIDIA markets no tensor cores and omits RT cores, but the HMMA datapath inside the SM is present and functional. *What the counters do NOT establish:* any throughput. 512 cycles for 4 instructions is 128 cycles per instruction, far off full-rate silicon, and these counters cannot separate full-rate hardware at very low occupancy from a reduced implementation. **No tensor-core throughput figure was measured and none is claimed.** Consequence: Stage 11's justification against cuBLAS SGEMM was made conditional on the absence of tensor cores, and that condition does not hold — Stage 11 is flagged to revisit its framing, and is deliberately not re-framed here. Separately, `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)` returns SUCCESS on this card, so math mode is **not** a valid tensor-core test; recorded so nobody repeats it.

**2. The card never reaches, let alone sustains, its rated boost clock.** Rated clock is 1485 MHz and `nvidia-smi` advertises a 2100 MHz maximum. Measured on the unlocked card under sustained load: it boosted to 1815 MHz, hit the SW power cap at t = 3.5 s and never cleared it, then decayed **monotonically to 1365 MHz over five minutes** — a 24.8% clock loss — while power draw fell from 54.87 W to 36.62 W. *Hypothesis:* the 55 W power limit has literally zero headroom (current = default = maximum = 55 W), and this is a laptop whose CPU and GPU share one thermal solution, so the card is power-capped from the first second and thermally governed thereafter. The falling power draw alongside falling clocks is the decisive evidence that the tail is **thermal**, not power: a power-limited card holds its power budget. **This is the strongest justification for locking the clock in this project, and it was measured rather than assumed.** Locked at 1365 MHz, the same five-minute load produced 1365 MHz on 122 of 122 samples, peak 82 C instead of 87 C, 33 W instead of 55 W, and **no throttle reason of any kind**.

**3. The driver's thermal target is below the slowdown temperature it reports.** `nvidia-smi` reports a 92 C slowdown limit and a 97 C shutdown limit. Measured: SW thermal slowdown (`0x0000000000000020`) was asserted at **86 C**, six degrees below the reported limit, and peak temperature never exceeded 87 C. *Hypothesis:* the reported slowdown temperature is a hardware protection threshold, while the driver enforces a lower software thermal target on this mobile part. Either way, **92 C is not the temperature at which this card starts losing clock — 86 C is**, and any later stage reasoning about thermal headroom must use the measured figure.

**4. System RAM is single-channel, which halves the CPU-side memory ceiling.** Queried: one Samsung M471A1K43DB1-CWE DIMM, 8 GiB, rated 3200 MT/s, **running at 2933 MT/s**, in `ChannelA-DIMM0` with no second module. Theoretical ceiling is therefore 2933 x 8 B x **1** channel = **23.464 GB/s** `[derived]`, against roughly 46.9 GB/s for the same parts dual-channel. *This is a real constraint on every CPU bandwidth figure the project will produce and it is flagged explicitly for Stages 5 and 6*, where cache blocking and SIMD are evaluated against a memory system that is half as wide as the part numbers suggest.

**5. An earlier concern about DRAM bandwidth exceeding theoretical did NOT reproduce.** Gate-run figures taken on a non-idle machine in a previous session had put DRAM-resident peak as high as 24 GB/s, above the 23.464 GB/s theoretical ceiling — which would indicate a byte-counting or timing error rather than a fast machine. Under protocol conditions the DRAM-tier medians came in at **18.26–21.40 GB/s, all below the ceiling**. The discrepancy is resolved and was an artefact of the uncontrolled machine. It must nonetheless be recorded that **every DRAM-tier configuration was INVALID on standard deviation in both runs**, so no DRAM bandwidth value is reported at all — see below.

**6. cuBLAS SGEMM throughput is strongly non-monotonic in M.** Best observed prefill figure is at M=512 (ffn_down, 2344.35 GFLOP/s), and throughput *falls* at M=1024 (1598.62 GFLOP/s) for the same shape. A similar peak-then-fall appears in every shape family. *Hypothesis:* cuBLAS switches kernels and tiling heuristics at shape thresholds, and the selection is not monotone in M on a 16-SM device where a large M can leave a partially filled final wave. **The counters collected in this stage do not establish this** — confirming it would need per-kernel profiling across the M sweep, which Stage 0 did not do. It is recorded as an open observation, not an explanation, and it matters because microbenchmark 6 is the prefill denominator for the entire project: **the denominator is shape-dependent and must be quoted per shape, never as one number.**

A seventh figure is worth recording as a non-divergence: **measured GPU bandwidth reached 88.99% of theoretical peak** (170.882 / 192.032 GB/s), which is high for a copy benchmark, and the Nsight counters corroborate it independently — `dram__bytes_read.sum.per_second` 87.152 GB/s plus `dram__bytes_write.sum.per_second` 87.025 GB/s gives 174.18 GB/s counter-side against 170.882 GB/s event-timed. Likewise **measured FP32 peak reached 99.68% of the locked-clock ceiling** (2786.49 / 2795.52 GFLOP/s), which is the expected result for a register-resident FMA loop and confirms both the clock lock and the FLOP accounting.

#### Machine state findings and the decision taken

- **Clock offsets: NOT OBTAINABLE, and recorded as unverified rather than zero.** `nvidia-smi -q` contains no clock-offset field whatsoever; Applications Clocks and Default Applications Clocks return "Requested functionality has been deprecated". This session additionally retried the vendor WMI path **with elevation**, which had previously failed with Access denied: `root\WMI MSI_VGA` now reads but returns 18 unlabeled integer instances with no documented schema and no offset field, and `MSI_ACPI` returns "Not supported". **Q9 cannot be closed by query on this machine even with administrative rights.** No offset of zero is recorded anywhere.
- **Decision D8, pre-decided by the operator and not reopened:** MSI Dragon Center user scenario "Balanced" for the life of the project, with clocks locked. Fan profile is not applicable — no fan-profile control is exposed for the Balanced scenario. Date fixed for the clock lock: **2026-09-17**. Whether Balanced is truly stock remains **UNVERIFIED**.
- **Clock lock applied and verified.** `nvidia-smi -lgc 1365,1365` returned "GPU clocks set to (gpuClkMin 1365, gpuClkMax 1365)" and exit 0. The value was derived from the 5.3 sustained-load result and not chosen arbitrarily — see below. Verified in effect rather than assumed: 10 of 10 samples at 1365 MHz under load via `verify-lock`, then 122 of 122 samples across a full five-minute load.
- **How 1365 MHz was chosen, including where the specified criterion failed.** The instruction was to lock at the highest clock the card holds for the full five minutes at its 55 W limit **without a throttle reason appearing**. **No such clock exists on this card**: the SW power cap is asserted at t = 3.5 s at 1815 MHz and never clears, so every clock the card runs under load is a throttled clock. The nearest honest realisation of the intent is the highest clock actually *sustained* across the whole window, which is the observed floor of **1365 MHz**, and it is a supported clock step. The choice is vindicated by the result: at 1365 MHz the card showed **no throttle reason at all** for five minutes.
- **Memory clock cannot be locked.** `nvidia-smi -lmc` returns "Setting locked Memory clocks is not supported for GPU 00000000:01:00.0." with **exit code 0** from a non-elevated shell — exit 0 plus that message is a device-capability limit, not a permission limit, so elevation would not change it. It was **not re-attempted** in this stage. **Consequence: the D8 mitigation is PARTIAL.** The graphics clock is pinned; the memory clock is not, and whether Balanced is stock is unverified. The §5.4 VRAM integrity check is therefore the only evidence available on memory stability in this project.
- **Power limit has zero headroom:** current, default and maximum are all 55.00 W. Power-limit tuning is not an available variable.
- **Frozen fingerprint values confirmed unchanged:** Windows power plan Ultimate Performance (GUID `69f3d390-66d3-4992-be72-f285309743af`), hardware-accelerated GPU scheduling enabled (`HwSchMode = 2`), on AC power throughout.
- **CPU telemetry during the sustained load is not trustworthy, and no conclusion is drawn from it in either direction.** CPU frequency read a constant 2496 MHz and package temperature a constant 73.05 C across all 87 samples of the unlocked run and all 122 of the locked run, while CPU load varied between 4% and 36%. Two series that never move while their driver does are static nominal reads, not live measurements. The question of whether the Ultimate Performance plan raises CPU heat enough to pull GPU boost down therefore **cannot be answered from this data**. What *is* independently established is that the GPU asserted a thermal throttle reason and lost 24.8% of its clock on the unlocked card, which satisfies the condition for flagging a Balanced-plan re-test before Stage 5. That flag is in `PERSISTENT.md`. **The power plan was not changed** — it is a frozen fingerprint value.

#### Stability checks (§5.4)

| Check | Result |
|---|---|
| VRAM integrity | **PASS.** 2048 MiB buffer, patterns `0x00` / `0xFF` / `0xAA` / `0x55`, 3 rounds each = 12 write/readback cycles, every one bit-exact with zero bad bytes, run immediately after the five-minute thermal load with the GPU at 82–87 C. This is the only evidence available on memory stability, because the memory clock cannot be pinned |
| Compute determinism | **PASS.** The same GEMM (M=768, N=2304, K=768) run twice produced bit-identical output buffers |
| Timing reproducibility | Full suite run twice. Spread over the 49 configurations VALID in both runs: **median 0.317%, mean 1.407%, p90 4.401%, max 19.913%** |
| Background load | Enumerated and recorded above. Nothing was closed by this session |

#### Noise floor

**Adopted: 4.4%**, the 90th percentile of run-to-run spread across configurations valid in both suite runs. No stage may claim a speedup smaller than this. Written into `BENCHMARK_PROTOCOL.md` §4.1 with the arithmetic. Two qualifications travel with it: small-M GEMM shapes are materially noisier and need a wider margin, and the floor was measured on a machine carrying irreducible background load that could not be removed.

**Adjusted warmup: 25 iterations**, written into `BENCHMARK_PROTOCOL.md` §4.2. Arithmetic: measured stabilization under the lock is 0.075 s; the reference workload `gpu_bandwidth` has a measured median iteration of 3.118 ms; 5 x 3.118 ms = 15.6 ms does not cover 0.075 s, and 25 x 3.118 ms = 77.9 ms does.

#### Every INVALID run

Of **96 configurations x 2 runs**, 49 configurations were valid in both runs, 30 were valid in one run and invalid in the other, and **17 were invalid in both runs**. None of them was retried, none was averaged away, and no value from an invalid configuration appears in `HARDWARE.md`.

The dominant pattern is telling and is recorded as a finding: **the same configuration is frequently valid in one run and invalid in the other with medians that agree to well under one percent.** That is the signature of an occasional multi-millisecond interruption inflating the standard deviation, not of an unstable machine or a badly built benchmark. The most likely cause is the irreducible background load documented above — a kernel-mode anti-cheat driver, an audio driver that hooks process creation, and Windows Defender real-time scanning. **The counters collected in this stage do not establish that cause**, and it is offered as a hypothesis rather than an explanation.

| Benchmark | Configuration | INVALID in | std dev run 1 | std dev run 2 |
|---|---|---|---|---|
| `gpu_bandwidth` | float4 grid-stride copy, 268435456 B/buffer, grid=512 block=256 | run 1 only | 5.84% | 2.54% |
| `cpu_cache_ladder` | working_set=8192 B, 268435456 B read per sample | run 1 only | 8.67% | 1.21% |
| `cpu_cache_ladder` | working_set=16384 B, 268435456 B read per sample | **both runs** | 12.03% | 5.89% |
| `cpu_cache_ladder` | working_set=32768 B, 268435456 B read per sample | run 2 only | 4.02% | 6.84% |
| `cpu_cache_ladder` | working_set=65536 B, 268435456 B read per sample | run 2 only | 4.90% | 5.17% |
| `cpu_cache_ladder` | working_set=131072 B, 268435456 B read per sample | run 1 only | 7.21% | 1.93% |
| `cpu_cache_ladder` | working_set=262144 B, 268435456 B read per sample | run 1 only | 13.33% | 1.62% |
| `cpu_cache_ladder` | working_set=524288 B, 268435456 B read per sample | run 1 only | 5.51% | 1.99% |
| `cpu_cache_ladder` | working_set=1048576 B, 268435456 B read per sample | run 1 only | 9.63% | 3.20% |
| `cpu_cache_ladder` | working_set=4194304 B, 268435456 B read per sample | run 1 only | 7.34% | 4.77% |
| `cpu_cache_ladder` | working_set=8388608 B, 268435456 B read per sample | **both runs** | 10.34% | 16.45% |
| `cpu_cache_ladder` | working_set=16777216 B, 268435456 B read per sample | **both runs** | 9.70% | 12.52% |
| `cpu_cache_ladder` | working_set=33554432 B, 268435456 B read per sample | **both runs** | 10.47% | 9.23% |
| `cpu_cache_ladder` | working_set=67108864 B, 268435456 B read per sample | **both runs** | 8.54% | 9.14% |
| `cpu_cache_ladder` | working_set=134217728 B, 268435456 B read per sample | **both runs** | 7.73% | 5.32% |
| `cpu_simd_peak` | vectorised FMA loop, AVX2 | **both runs** | 5.86% | 5.60% |
| `host_device_transfer` | pageable device-to-host | run 1 only | 17.93% | 4.42% |
| `cublas_sgemm_ref` | decode qkv_projection M=1 N=2304 K=768 | **both runs** | 5.21% | 5.34% |
| `cublas_sgemm_ref` | prefill qkv_projection M=8 N=2304 K=768 | run 1 only | 5.27% | 4.99% |
| `cublas_sgemm_ref` | prefill qkv_projection M=32 N=2304 K=768 | **both runs** | 5.90% | 5.85% |
| `cublas_sgemm_ref` | prefill qkv_projection M=512 N=2304 K=768 | run 2 only | 0.34% | 13.53% |
| `cublas_sgemm_ref` | prefill qkv_projection M=1024 N=2304 K=768 | run 1 only | 6.65% | 3.22% |
| `cublas_sgemm_ref` | decode attn_output_projection M=1 N=768 K=768 | **both runs** | 7.41% | 9.53% |
| `cublas_sgemm_ref` | prefill attn_output_projection M=32 N=768 K=768 | **both runs** | 13.54% | 11.14% |
| `cublas_sgemm_ref` | prefill attn_output_projection M=64 N=768 K=768 | **both runs** | 5.43% | 5.14% |
| `cublas_sgemm_ref` | decode ffn_up M=1 N=3072 K=768 | run 1 only | 10.42% | 3.40% |
| `cublas_sgemm_ref` | prefill ffn_up M=16 N=3072 K=768 | run 2 only | 4.96% | 9.01% |
| `cublas_sgemm_ref` | prefill ffn_up M=32 N=3072 K=768 | **both runs** | 9.19% | 5.01% |
| `cublas_sgemm_ref` | prefill ffn_up M=256 N=3072 K=768 | run 1 only | 12.93% | 0.81% |
| `cublas_sgemm_ref` | prefill ffn_up M=1024 N=3072 K=768 | **both runs** | 6.44% | 7.15% |
| `cublas_sgemm_ref` | decode ffn_down M=1 N=768 K=3072 | run 1 only | 13.82% | 1.86% |
| `cublas_sgemm_ref` | prefill ffn_down M=32 N=768 K=3072 | run 1 only | 5.32% | 4.73% |
| `cublas_sgemm_ref` | prefill ffn_down M=512 N=768 K=3072 | run 2 only | 0.40% | 8.78% |
| `cublas_sgemm_ref` | prefill ffn_down M=1024 N=768 K=3072 | run 1 only | 5.25% | 3.29% |
| `cublas_sgemm_ref` | decode lm_head M=1 N=50257 K=768 | run 2 only | 1.66% | 11.20% |
| `cublas_sgemm_ref` | prefill lm_head M=8 N=50257 K=768 | run 1 only | 11.92% | 4.45% |
| `cublas_sgemm_ref` | prefill lm_head M=32 N=50257 K=768 | run 1 only | 6.45% | 4.72% |
| `cublas_sgemm_ref` | prefill lm_head M=64 N=50257 K=768 | **both runs** | 6.95% | 7.65% |
| `kernel_launch_overhead` | sync per launch (launch to completion) | run 1 only | 7.37% | 3.40% |
| `occupancy_sweep` | acc=16 block=64 regs/thread=36 blocks/SM=16 limiter=blocks | run 1 only | 14.54% | 0.33% |
| `occupancy_sweep` | acc=16 block=128 regs/thread=36 blocks/SM=8 limiter=warps | run 1 only | 9.69% | 0.51% |
| `occupancy_sweep` | acc=40 block=32 regs/thread=64 blocks/SM=16 limiter=blocks | run 2 only | 0.40% | 9.34% |
| `occupancy_sweep` | acc=40 block=128 regs/thread=64 blocks/SM=8 limiter=warps | run 2 only | 4.56% | 8.79% |
| `occupancy_sweep` | acc=40 block=256 regs/thread=64 blocks/SM=4 limiter=warps | run 1 only | 5.75% | 0.45% |
| `occupancy_sweep` | acc=40 block=1024 regs/thread=64 blocks/SM=1 limiter=warps | **both runs** | 5.29% | 6.95% |
| `occupancy_sweep` | acc=72 block=128 regs/thread=96 blocks/SM=5 limiter=registers | run 1 only | 5.44% | 0.10% |
| `occupancy_sweep` | acc=72 block=512 regs/thread=96 blocks/SM=1 limiter=registers | **both runs** | 5.20% | 7.35% |

#### Fields left empty, and why

Two `HARDWARE.md` fields are deliberately empty. Neither carries a placeholder, an estimate, or a rounded guess.

| Field | Reason |
|---|---|
| §2 **Measured DRAM bandwidth** | Every DRAM-tier working set (8, 16, 32, 64, 128 MiB) exceeded the 5% standard-deviation limit in **both** suite runs. No valid figure exists. For orientation only, the invalid medians spanned 18.26–21.40 GB/s against a 23.464 GB/s single-channel theoretical ceiling |
| §2 **Measured peak FP32 (SIMD)** | The vectorised AVX2 configuration exceeded the limit in **both** runs (5.86%, 5.60%). The scalar reference *was* valid at 8.347 GFLOP/s, and the 5.19x vectorised-over-scalar ratio confirms the compiler genuinely vectorised rather than silently dropping it — but a ratio is not a peak figure and none is substituted |

**FOLLOW-UP, 2026-09-18 — Stage 0b.** A scoped diagnostic-and-remeasurement session was run against both of these fields; see the Stage 0b entry below. Outcome: **§2 Measured peak FP32 (SIMD) is now POPULATED at 48.411 GFLOP/s** (2.773%, VALID, valid in both Stage 0b runs). **§2 Measured DRAM bandwidth REMAINS EMPTY** — every DRAM-tier working set was INVALID again in both Stage 0b runs — but with a better-evidenced reason: the dispersion there is broad rather than spike-carried (zero robust outliers at 8, 16 and 32 MiB in reference run 2, against interquartile ranges of 19.88 / 20.75 / 10.70% of median), which is shared-L3 and single-channel-DRAM contention that no change inside the benchmark can control. Stage 0b additionally **replaced all five ladder values** in `HARDWARE.md` §2, because it changed `cpu_cache_ladder.c` and the background load and §4 voids a comparison across either.

`HARDWARE.md` §6 (secondary devices) is also unpopulated, because no secondary device was used.

#### cuBLAS reference shapes and their provenance

Microbenchmark 6 is the prefill denominator for the entire project, so it runs at the shapes GPT-2 small actually performs, never square. Architecture parameters — `n_embd` 768, `n_head` 12, `n_layer` 12, `n_ctx` 1024, `vocab_size` 50257 — were read from **the published `config.json` shipped with the `openai-community/gpt2` weights, fetched 2026-09-16** (HTTP 200). The five distinct GEMM shapes the forward pass performs, as (N, K):

| GEMM | N | K |
|---|---|---|
| QKV projection | 2304 | 768 |
| Attention output projection | 768 | 768 |
| Feed-forward up projection | 3072 | 768 |
| Feed-forward down projection | 768 | 3072 |
| Language-model head | 50257 | 768 |

M is the token count and was swept across **1, 8, 16, 32, 64, 128, 256, 512, 1024**, spanning short prefill to the maximum context the architecture supports. **M = 1 is the decode case and is reported separately from the prefill shapes in the results file; a single combined figure is not a valid output of this benchmark.**

**These shapes are UNCONFIRMED until Stage 1 verifies them against the config file shipped with the weights.** That flag, and the Stage 3 flag to fix the prompt set and token counts, are recorded in `PERSISTENT.md`.

#### Decisions taken without the operator

The session was unattended, so these were resolved rather than asked. Each is listed for review.

1. **Locked the graphics clock at 1365 MHz** although the specified criterion — highest clock held for five minutes with no throttle reason — has no satisfying value on this card, because the SW power cap never clears under load. Took the highest *sustained* clock instead. Vindicated: no throttle reason appeared at 1365 MHz.
2. **Ran the suite twice and treated run 2 as the reference.** A second run is independently required by §5.4 check 3, so this is not a retry for a better number; both runs are retained in full and every invalid configuration from both is listed above.
3. **Deleted one accidental pre-lock `gpu_bandwidth` result** produced by invoking the binary with a `--help` flag it does not support, and re-ran the benchmark under protocol conditions.
4. **Used a throwaway scratchpad load kernel for the 5.3 sustained load** rather than looping a repository benchmark, so that thermal-soak runs could not write into `bench/results/` and be mistaken for protocol measurements. The kernel is not part of the repository.
5. **Used throwaway scratchpad probes for the tensor-core determination and the full device-property dump**, for the same reason — Stage 0's scope does not include adding source files beyond the specified outputs.
6. **Adopted the 90th percentile rather than the median or the maximum as the noise floor**, with the reasoning stated in `BENCHMARK_PROTOCOL.md` §4.1.
7. **Did not attempt `nvidia-smi -lmc`**, on the prior finding that it is a device-capability limit returning exit 0. Not re-tested.
8. **Left two `HARDWARE.md` fields empty** rather than quoting an invalid measurement.

#### What this taught

A spec sheet describes a die; a machine is a die inside a thermal and power envelope, and the two disagree by 25% here. The rated 1485 MHz boost clock is not a number this card can hold for five seconds, let alone five minutes — it power-caps immediately and then decays to 1365 MHz while its power draw *falls*, which is how you tell thermal governance from power limiting. Locking the clock at the measured floor did not cost performance so much as buy comparability: it removed every throttle reason, collapsed clock variance to zero across 122 samples, and turned the warmup question from a real problem into a formality. The corollary is the one worth carrying: **on this machine an unlocked measurement is not a slow measurement, it is a measurement of a different machine at its start than at its end.**

## Stage 0b — Diagnose and remeasure the INVALID CPU microbenchmarks
*No prediction: Stage 0b is a continuation of Stage 0, which is exempt under the project's prediction rule because it measures the machine rather than changing it. No prediction was written, invented or substituted.*
**Status:** complete, 2026-09-18

Two required `HARDWARE.md` §2 fields were empty after Stage 0 because their measurements were INVALID in both suite runs. `PROJECT.md` §7 item 4 requires the Stage 10 performance model validated across all kernels, and a model with no CPU memory-bandwidth term and no CPU SIMD ceiling cannot be. `TECHNICAL_SPEC.md` §3 Stage 6 is framed as predicted-ceiling-is-vector-width versus measured, which requires a measured ceiling to exist.

**Device:** NVIDIA GeForce GTX 1650 Ti (TU117) present but idle — this stage times no GPU kernel. The measured device is the CPU: Intel Core i5-10300H, Comet Lake-H, 4 physical / 8 logical cores — `HARDWARE.md` §2.

#### Conditions

- **This session ran ELEVATED** (`IsUserAnAdmin()` = True and `IsInRole(Administrator)` = True, both checked, neither assumed), so that it could apply the clock lock itself.
- **Graphics clock locked at 1365 MHz** and **verified in effect before the first timed run, not assumed**: `verify-lock --mhz 1365` returned `locked: true` with 10 of 10 samples at exactly 1365 MHz and zero off-target. Re-verified still in effect immediately before the runs. The reason is thermal, not timing: this is a laptop whose CPU and GPU share one thermal solution, `HARDWARE.md` §5.3 records the locked card idling at 1365 MHz / 66 C / 13.89 W against the unlocked card's 300 MHz / 51 C / 3.86 W, and Stage 0's CPU figures were taken with the lock applied. **`nvidia-smi -lmc` was never attempted** — it is a device-capability limit on this card that returns exit 0 with "not supported", and elevation does not change that.
- **Power: on AC** for every timed run, checked programmatically via `GetSystemPowerStatus` (`ACLineStatus` = 1, battery 98%) rather than assumed. `BENCHMARK_PROTOCOL.md` §3 makes a run on battery INVALID outright.
- **Environment fingerprint verified.** Stage 0 never committed `bench/results/machine_fingerprint.json`, so `machine_state.py verify` had nothing to compare against and the every-stage verification required by `BENCHMARK_PROTOCOL.md` §4 and `PERSISTENT.md` §7 was unenforceable. Stage 0b compared the live capture field-by-field against `HARDWARE.md` §5.5 by hand, found every field identical, then **wrote the fingerprint file before the first timed run** so it records the state the measurements were taken under. `verify` then ran clean: `match: true`, `differences: []`, **26 fields compared**, exit 0.
- **Compiler and flags UNCHANGED and confirmed byte-identical** to `HARDWARE.md` §5.5: host `/DWIN32 /D_WINDOWS /EHsc /W3 /arch:AVX2 /fp:precise /MD /O2 /Ob2 /DNDEBUG`, CUDA `-D_WINDOWS -Xcompiler=" /EHsc" -O3 --generate-line-info -Xptxas=-O3 -Xptxas=-v -Xcompiler=/arch:AVX2 -Xcompiler=/fp:precise -arch=sm_75`, MSVC 19.44.35229.0, nvcc 13.1.80, sm_75, driver 591.44. No comparison in this stage is void on flags.
- **Warmup 25, 30 timed samples**, matching Stage 0 exactly.
- **Both benchmarks were run twice. Run 2 is the reference**, as Stage 0 did. Raw per-sample timings for both retained: `bench/results/stage0b/run1/` and `run2/`. **No Stage 0 results file was written, overwritten or deleted** — Stage 0b output is routed through a guard that raises on any path outside `bench/results/stage0b/`, and that guard is unit-tested.
- **Only `cpu_cache_ladder` and `cpu_simd_peak` were re-run.** The full suite was not. No benchmark that produced a valid Stage 0 figure was modified or re-run.
- **Background load: see `HARDWARE.md` §5.4 for the full dated Stage 0b entry.** In brief: Nahimic and Intel DSA were closed and confirmed absent; **Riot Vanguard was not running at all this session**, where it was resident throughout Stage 0; Logitech Options+ was closed by the operator but **respawned before the runs** and is recorded as present rather than reported as closed; `msedgewebview2.exe` x6 could not be closed and **was equally resident during Stage 0**, so it is a shared condition rather than a difference; the MSI service stack was left running by deliberate operator decision because stopping it risks silently changing the clock regime (decision D8, `PERSISTENT.md` Q9) with no way to detect it; Windows Defender was not excluded or modified. **The Stage 0b condition is strictly lighter than Stage 0's.**
- **Instrument added, recorded rather than assumed negligible.** A CPU frequency sampler ran in a separate process alongside every benchmark: 20 ms interval, measured per-probe cost 37.5–39.2 us, **duty cycle 0.355–0.372% of one core**. Pinned to logical CPUs 4, 5, 6, 7 (mask `0xf0`), clear of logical CPU 2 where the measured thread runs, clear of CPU 2's SMT sibling CPU 3, and clear of CPU 0 and its sibling CPU 1. The logical-to-physical mapping was **queried** with `GetLogicalProcessorInformationEx(RelationProcessorCore)`, not assumed from the conventional interleaving.
- **Network — a recorded run condition.** Stage 0b ran over a **mobile hotspot** ("iPhone 2", Intel Wi-Fi 6 AX201 160MHz, link 216→551 Mbps), not the connection present during Stage 0. **Windows did NOT mark it metered** (`NetworkCostType = Unrestricted`) and **Windows Update was NOT paused** (`PauseUpdatesExpiryTime` null; `wuauserv` Stopped is not a pause). Driver 591.44 is a frozen fingerprint value and an update installed mid-session would void every comparison against Stage 0 with no undo. No update occurred; the fingerprint verified clean before the first timed run. **One disconnection and reconnection was observed, during the build-and-setup phase and before any timed run began.** No timed run overlapped it and **no run is invalidated by it**. Had one overlapped, that run would have been declared INVALID with the disconnection named, because a connection drop and the interference this stage diagnoses produce the same signature and the data cannot separate them.

#### Phase 1 — what the retained Stage 0 data established, before any code changed

`bench/analyze_variance.py` (new, unit-tested) read the raw per-sample arrays for **all 96 configurations in each of the two Stage 0 suite runs, 192 record-instances**, strictly read-only. **Raw per-sample arrays were present throughout** — had they not been, this phase was impossible as specified and the session would have stopped. Output: `bench/results/stage0b/variance_analysis.json`.

**The strongest single discriminator was deviation DIRECTION.**

| | `cpu_cache_ladder` | `cpu_simd_peak`, vectorised |
|---|---|---|
| Flagged outliers by direction | **54 slow, 1 fast** | — |
| Below-median excursion | — | **16.2% (run 1), 16.7% (run 2)** |
| Above-median excursion | — | 6.0% (run 1), 3.1% (run 2) |

Interference can only ever make a sample slower. The ladder's deviations are upward from a clean floor; the SIMD vector path's are **downward from a ceiling**. That is not one fault with one cause — the two benchmarks were failing for different reasons, and interference was ruled out for the SIMD path on direction alone.

**(a) Spike-carried versus broadly dispersed — both shapes present, in different places.**

| Group | Spike share of variance | IQR as % of median | Reading |
|---|---|---|---|
| Cache-resident ladder points that failed | 0.86–0.99 | 1.5–3.7% | tight baseline plus isolated slow spikes |
| DRAM tier 8–64 MiB, both runs | 0.00–0.59 | 8.4–16.6% | genuinely broad; run 1 at 8 MiB flagged **zero** outliers yet still measured 10.34% |
| 128 MiB run 2 | 0.964 | 2.28% | the exception — six isolated spikes on a flat floor |
| Decode-shape GEMMs | 0.78–0.99 | 1.1–4.5% | uniformly spike-carried |

**(b) Sample duration does NOT predict variance across the corpus.** Spearman rho between median duration and std-dev-percent over all 192 record-instances = **0.194**; Pearson on log10(median) = 0.150. Median duration of INVALID configurations **2.073 ms**, of VALID configurations **0.789 ms** — the INVALID ones are *longer*. Decade buckets are non-monotone (5.06 / 1.20 / 4.64 / 3.69% median std dev from 1e-2 to 1e1 ms). **The "sample too short, one preemption dominates" hypothesis is refuted at corpus scale**, though it holds inside the sub-100 us decode subset — see (e).

**(c) Trimming barely moved the SIMD figure** — 5.862% at k=0 to 5.797 / 5.752 / **5.729%** at k = 1 / 2 / 3. Not a tail effect. **Every trimmed statistic in this stage is a diagnostic only.** In `variance_analysis.json` they sit under keys prefixed `diagnostic_only__`, carry their own `diagnostic_only: true` marker and disclaimer, and are absent from the `measurement` block. No trimmed value appears as a result anywhere in this entry, in `HARDWARE.md`, or in any report. No trimmed value converted an INVALID run to valid, and the unit tests assert that it cannot.

**(d) The 16384 B anomaly — two different mechanisms in the two runs, neither of them DRAM.** 16384 B is inside this CPU's 32768 B L1d by a factor of two, so no DRAM-side contention can reach it, yet it was INVALID in both Stage 0 runs.
- Run 1: a smooth **10-sample hump** — flat at 2.185–2.30 ms for samples 0 through 10, rising to 3.102 ms at sample 20, decaying back to 2.26–2.30. Roughly 25 ms of sustained slowdown, +42% at peak. The robust outlier test flags **zero** samples because a quarter of them lie inside the hump. IQR 18.02%.
- Run 2: the **first four timed samples** (2.538, 2.332, 2.449, 2.368 ms) are slow, then it settles at 2.05–2.15 ms for the remaining 26 — despite 25 warmup iterations on an already-faulted buffer.
- The neighbouring 8192 B point was 1.21% VALID in run 2 and 8.67% INVALID in run 1, with its own contiguous hump at samples 18–21.

What this established: at a 16 KiB working set the excursions are multi-sample, episodic and upward, consistent with the measured thread being descheduled or migrated off its core — L1d and L2 are per-core, so a migration costs a full re-warm. What it did **not** establish: which of those. The retained data carries no per-sample core identity and no per-sample frequency.

**(e) Decode shapes — and a correction to `MEASUREMENTS.md` and to this session's own briefing.** See the correction appended to the Stage 0 entry's "nine measured values" table. File truth: **all five M=1 shapes were INVALID in at least one run**; only `qkv_projection` (5.21% / 5.34%) and `attn_output_projection` (7.41% / 9.53%) failed in both; `ffn_up` failed run 1 at 10.42%, `ffn_down` failed run 1 at 13.82%, `lm_head` failed run 2 at 11.20%. **Which shape fails is not stable between runs.**

Mechanism, and here duration *is* the variable. Medians are 30–90 us, spike share 0.78–0.99 on an IQR of 1.1–4.5%. One sample at +50% among 30 gives a standard deviation of 0.5 / sqrt(29) = **9.3% of median** — which is the observed magnitude. **A single sub-100 us configuration cannot survive one scheduler or driver event under a 30-sample rule.** `cublas_sgemm_ref.cu` was **not modified and not re-run**: it is the prefill denominator and `PERSISTENT.md` D3 records its M sweep as a placeholder Stage 3 replaces. The decode headline in `BENCHMARK_PROTOCOL.md` §7 is unaffected — its denominator, 170.882 GB/s, is valid and these figures are not in that arithmetic. Flagged for Stages 4, 7, 8 and the Stage 10 small-kernel term in `PERSISTENT.md`; not decided here.

#### Phase 2 — CPU telemetry, proven to move before being used

`HARDWARE.md` §5.3 records Stage 0's CPU telemetry as static and unusable. Stage 0b probed every candidate source and classified each from its own readings.

| Source | Verdict | Per-probe cost | Idle | Under applied load | Recovery |
|---|---|---|---|---|---|
| PDH `\Processor Information(_Total)\% Processor Performance` | **LIVE** | **12.5 us** | 167.8–176.2% | 140.0–150.4% | 170.0–174.0% |
| PDH `...\% Performance Limit` | static, constant 100 | 6.1 us | — | — | — |
| PDH `...\Processor Frequency` | static, constant 2496 | 6.3 us | — | — | — |
| WMI `Win32_Processor.CurrentClockSpeed` | **static**, constant 2496 | **1 360 246 us** | — | — | — |
| WMI `MSAcpi_ThermalZoneTemperature` | **static**, constant 69.05 C | **1 360 246 us** | — | — | — |

- **A live CPU frequency source exists.** It moves about 30 percentage points under a deliberately applied 4-process load and returns toward idle when that load stops. Against 2496 MHz nominal: idle approximately 4.19–4.40 GHz, loaded approximately 3.49–3.75 GHz. At 12.5 us against ladder samples of 1.9–15 ms it costs 0.08–0.7%, safe **outside** the bracket.
- **No live CPU package temperature source exists on this machine.** `MSAcpi_ThermalZoneTemperature` is constant under full 8-process load and costs roughly **1.36 seconds** per probe — 600 times a cache-ladder sample. It is static and unusable, twice over. This reproduces and explains Stage 0's static series rather than merely repeating it.
- **Telemetry is never inside a timed bracket.** The sampler runs in a different process from the benchmark, so no probe can be. `tests/test_machine_state.py` additionally scans both C sources between every `t0` and `t1` and fails on any telemetry call, allocation, I/O or OS call there — which also proves allocation and initialisation are outside the ladder's bracket.

#### Phase 3 — branch taken: the CODE branch, plus reduced background load

The diagnosis identified causes in the benchmark code that the data supports, so the first branch applies: fix them, and remeasure every tier per operator decision 1. It also established a residue the benchmark cannot control, so the second branch's remedy — reduce background load — applies on top.

**Evidence for the code cause:** 54 of 55 flagged ladder outliers are slow; the excursions are multi-sample and episodic; the failures concentrate where per-core state matters. The measured thread was free to migrate across 8 logical / 4 physical cores mid-sample, and L1d and L2 are per-core; and it ran at normal priority, where it outranked nothing.

**Evidence for the uncontrollable residue:** at 8–64 MiB the dispersion is broad and spike-free. An 8 MiB working set is exactly the shared 8 MiB L3.

**The change, in both files:** the measured thread is pinned to one logical CPU (2 by default, avoiding CPU 0's interrupt and DPC affinity) and raised to ABOVE_NORMAL priority class with THREAD_PRIORITY_HIGHEST. Applied once, **outside every timed bracket**, restored at the end, and **recorded in the results file whether or not it took** — never assumed. The timed brackets themselves are unchanged.

**What was deliberately NOT changed:** the SIMD trip count, arithmetic and flop derivation. Lengthening a sample so it averages over excursions would have lowered the reported standard deviation by measuring something other than what Stage 0 measured. That is engineering the validity test, not the measurement.

#### Measurement

Reference run 2. Std dev as a percentage of median, the protocol's own validity criterion.

**`cpu_simd_peak` — both runs VALID, where Stage 0 was INVALID in both.**

| Configuration | Run | Value | Median | Min | Max | Std dev | Verdict |
|---|---|---|---|---|---|---|---|
| vectorised FMA loop, AVX2 | **2 (ref)** | **48.411 GFLOP/s** | 5.2880 ms | 5.1683 | 6.0396 | **2.773%** | **VALID** |
| vectorised FMA loop, AVX2 | 1 | 49.506 GFLOP/s | 5.1710 ms | 5.1471 | 5.3906 | 1.075% | VALID |
| scalar reference of the same loop | **2 (ref)** | **8.565 GFLOP/s** | 3.7361 ms | 3.7315 | 4.0622 | 1.847% | VALID |
| scalar reference of the same loop | 1 | 8.472 GFLOP/s | 3.7769 ms | 3.7335 | 4.2290 | 3.163% | VALID |

Vectorised-over-scalar **5.65x** (run 1: 5.84x), against Stage 0's 5.19x — the evidence that the compiler genuinely vectorised survives the change. The two paths are additionally now **proven** to compute the same arithmetic rather than assumed to: vector checksum 114111040 over 8 lanes = 14263880 per lane against scalar 14263877, a relative difference of 2.1e-7. Compiled ISA equals widest supported ISA (AVX2 / AVX2), still asserted.

**`cpu_cache_ladder` — reference run 2, all sixteen points.**

| Working set | GB/s | Median | Min | Max | Std dev | Verdict |
|---|---|---|---|---|---|---|
| 4096 B | **143.85** | 1.866 ms | 1.850 | 1.974 | **1.272%** | **VALID** — quoted L1-resident |
| 8192 B | 143.91 | 1.865 ms | 1.848 | 2.305 | 6.365% | **INVALID** |
| 16384 B | 143.82 | 1.866 ms | 1.850 | 1.938 | 1.284% | VALID |
| 32768 B | 143.80 | 1.867 ms | 1.851 | 1.922 | 1.070% | VALID |
| 65536 B | 116.85 | 2.297 ms | 2.150 | 2.449 | 3.665% | VALID |
| 131072 B | 114.77 | 2.339 ms | 2.187 | 3.000 | 12.284% | **INVALID** |
| 262144 B | **113.85** | 2.358 ms | 2.336 | 2.757 | **4.457%** | **VALID** — quoted L2-resident |
| 524288 B | 70.82 | 3.791 ms | 3.742 | 4.852 | 8.317% | **INVALID** |
| 1048576 B | 70.74 | 3.794 ms | 3.708 | 4.298 | 2.898% | VALID |
| 2097152 B | 70.70 | 3.797 ms | 3.682 | 4.244 | 3.084% | VALID |
| 4194304 B | **70.81** | 3.791 ms | 3.748 | 4.444 | **4.621%** | **VALID** — quoted L3-resident |
| 8388608 B | 32.42 | 8.280 ms | 6.651 | 12.060 | 16.573% | **INVALID** |
| 16777216 B | 22.05 | 12.171 ms | 10.924 | 15.900 | 11.768% | **INVALID** |
| 33554432 B | 18.75 | 14.315 ms | 12.974 | 17.352 | 8.315% | **INVALID** |
| 67108864 B | 18.54 | 14.477 ms | 13.860 | 18.448 | 8.060% | **INVALID** |
| 134217728 B | 18.23 | 14.725 ms | 14.174 | 18.144 | 7.231% | **INVALID** |

Measured plateau edges, reference run 2: **32 KiB, 256 KiB, 4 MiB, 8 MiB, 16 MiB** — the identical five Stage 0 found, from independently pinned code under different background load.

**Every configuration still INVALID, listed as invalid — 19 across both runs. None averaged away, none silently retried, no run repeated to chase a better number.**

Run 1 (11 INVALID): 4096 B 10.947%, 8192 B 11.411%, 16384 B 5.877%, 65536 B 16.891%, 262144 B 5.489%, 4194304 B 6.832%, 8388608 B 17.363%, 16777216 B 21.460%, 33554432 B 9.536%, 67108864 B 6.159%, 134217728 B 5.354%.
Run 2 (8 INVALID): 8192 B 6.365%, 131072 B 12.284%, 524288 B 8.317%, 8388608 B 16.573%, 16777216 B 11.768%, 33554432 B 8.315%, 67108864 B 8.060%, 134217728 B 7.231%.

Run 1 was markedly worse than run 2 across the whole ladder and ran first. Both runs used identical code and identical protocol counts. **This stage does not know why run 1 was worse**, and says so rather than picking a story; run 2 is the reference by the convention Stage 0 already set, not because it is the better number.

#### Gap — diagnosis versus outcome

The prompt asked for this framed as what Phase 1 established, what the remeasurement showed, and where the two disagree. **They disagree on the SIMD mechanism, and the remeasurement wins.**

**Where the diagnosis was confirmed.**
- The DRAM tier behaved exactly as Phase 1 predicted it would. Broad, spike-free dispersion is not addressable by thread placement, and it was not addressed: 8, 16 and 32 MiB in run 2 flagged **zero** robust outliers with interquartile ranges of **19.88 / 20.75 / 10.70%** of median — wider than Stage 0's. The field stays empty, now for a reason with a distribution shape behind it rather than a standard deviation.
- The 16384 B anomaly resolved. It was INVALID in both Stage 0 runs (12.03%, 5.89%) and is **1.284% VALID** in Stage 0b run 2, on a flat L1 plateau of 143.80–143.85 GB/s across 4, 16 and 32 KiB agreeing to within 0.04%. Consistent with per-core cache re-warm after migration having been the cause, as Phase 1 suggested but could not establish.
- The edge detector reproduced the identical five boundaries across a code change, which is a check on the ladder itself rather than on the machine.

**Where the diagnosis was WRONG.** Phase 1 read the SIMD vector path's downward excursions as core frequency governance, and reasoned that thread placement would therefore **not** fix it — the Stage 0b entry was drafted expecting a second INVALID SIMD result to be a likely and acceptable outcome. That reading was not confirmed.

- The deviation direction **flipped**. Below-median excursion fell from 16.2% / 16.7% to **0.5% / 2.3%**; above-median went from 6.0% / 3.1% to 4.2% / 14.2%. After pinning, the vector path behaves like every other benchmark on this machine: a clean floor with occasional upward spikes.
- The new median sits at the **old minimum**, not the old median. Stage 0 median 5.977 / 5.911 ms with minima of 5.008 / 4.922 ms; Stage 0b median **5.171 / 5.288 ms**. The benchmark now attains consistently what it previously attained only intermittently.
- **Counter evidence, and its limit.** The live frequency counter recorded a range of **10.9–14.3% of median** across the Stage 0b runs while the SIMD timings were stable to 1.075% and 2.773%. Frequency was still moving; the timings stopped moving with it. That is evidence against the frequency reading. **It is not conclusive, and the reason is stated rather than buried:** the counter sampled was `\Processor Information(_Total)\% Processor Performance`, an average across all 8 logical CPUs, **not** the frequency of logical CPU 2 specifically. A per-core counter path exists and was not used. This stage therefore does not claim to have measured the frequency of the core under test.

**Best-supported explanation, stated as such.** The Stage 0 "fast" samples were the ones where the measured thread happened to have a physical core to itself; the slower baseline was SMT sibling contention, core migration, or preemption. Pinning to logical CPU 2 makes that the normal case rather than the lucky one.

**Alternatives not ruled out, plainly.** Affinity and priority were applied **together**, and this stage cannot separate their contributions — a run pinned at normal priority was not performed. The `_Total` frequency counter cannot attribute frequency to the core under test, so a frequency contribution is reduced in plausibility but not excluded. The SMT sibling of CPU 2 was not held idle, so sibling contention was reduced by chance rather than by construction. And the background load changed at the same time as the code, so no figure in this entry separates the two: `BENCHMARK_PROTOCOL.md` §4 voids a comparison across either change, which is precisely why all five ladder values were replaced rather than only the DRAM one.

**A reporting obligation, per operator decision 3.** Every figure in this entry is a figure for a benchmark that **outranks ordinary background work and owns a physical core** — ABOVE_NORMAL priority class, THREAD_PRIORITY_HIGHEST, pinned to logical CPU 2. That is what a ceiling measurement should be, and it is **not the same measurement Stage 0 attempted**, which ran unpinned at normal priority. Part of the L1 rise from 127.38 to 143.85 GB/s is that change and part is the lighter load; this stage does not claim to know the split.

**The noise floor is unchanged at 4.4%** and was not re-derived, because this stage did not re-run the full suite and could not. The reasoning that keeps it safe: the Stage 0b condition is **strictly lighter** than Stage 0's — the close list removed load and Riot Vanguard was absent — and a floor measured under heavier load stays conservative under lighter load. A conservative floor only ever suppresses claims; it never licenses one. Flagged in `PERSISTENT.md` for a later stage to consider, not decided here.

#### `HARDWARE.md` fields changed

| Field | Before | After |
|---|---|---|
| §2 Measured peak FP32 (SIMD) | empty, INVALID in both Stage 0 runs | **48.411 GFLOP/s**, 2.773%, VALID (both Stage 0b runs valid) |
| §2 Measured DRAM bandwidth | empty | **still empty** — INVALID in both Stage 0b runs, with a better-evidenced reason |
| §2 L1-resident | 127.38 GB/s (8 KiB) | **143.85 GB/s** (4 KiB), 1.272% |
| §2 L2-resident | 99.64 GB/s (256 KiB) | **113.85 GB/s** (256 KiB), 4.457% |
| §2 L3-resident | 62.31 GB/s (4 MiB) | **70.81 GB/s** (4 MiB), 4.621% |
| §2 Effective cache edges | 32 KiB, 256 KiB, 4 MiB, 8 MiB, 16 MiB | **unchanged** — the identical five, re-measured and confirmed |
| §5.4 background load | Stage 0 entry | Stage 0 entry retained; **separate dated Stage 0b entry added** |

Stage 0 provenance text was appended to, never deleted.

#### Decisions and corrections

1. **Corrected the M=1 validity error in the Stage 0 entry** (appended, not overwritten). The error had also propagated into the prompt briefing this session; the retained per-sample data corrected both.
2. **Wrote `bench/results/machine_fingerprint.json`**, which Stage 0 never committed, before the first timed run, and confirmed `verify` runs clean against it.
3. **Did not lengthen the SIMD trip count**, on the grounds stated above. In hindsight this was the right call for a second reason: the real cause was controllable by placement, and a longer sample would have masked it rather than found it.
4. **Did not use the CPU hardware counters that do exist here.** `xperf` from the Windows Performance Toolkit exposes the live Comet Lake PMU. An ETW PMU trace is itself a substantial added load that would change the very condition this stage exists to clean up. Named, not used, and flagged as available to a later stage as a separate non-timed diagnostic run.

#### What this taught

A standard deviation tells you a run is invalid; only the distribution's **shape and direction** tell you why, and they are free — the raw samples were already on disk. Two configurations failing the same 5% test had opposite causes: the ladder deviated upward from a clean floor, which only contention can do, and the SIMD loop deviated *downward* from a ceiling, which contention cannot do at all. The direction alone separated a machine problem from a benchmark problem before a line of code changed. The sharper lesson is the one that cost a wrong hypothesis: a benchmark that measures a single thread must **say where that thread runs**. Left unpinned on a 4-core SMT laptop, `cpu_simd_peak` was not noisy — it was sampling two different machines, a core it shared and a core it owned, and reporting the mixture as variance. Pinning did not make the measurement quieter so much as make it a measurement of one thing.

## Stage 1 — Weights and tokenizer
*No prediction: Stage 1 loads data and verifies a round-trip. There is no performance result to predict, and no prediction was written, invented or substituted. The exemption covers predictions only — every other measurement rule applied in full.*
**Status:** complete, 2026-09-18

Two reusable C components and their tests: the safetensors weight loader and the byte-level BPE tokenizer. No forward pass, no harness, **no timed run of any kind**, and no number in this entry is a latency.

#### Conditions

- **This session was NOT elevated and did not need to be.** `nvidia-smi -lgc` and `-lmc` were never issued, no clock was locked, no background process was enumerated, and no benchmark-conditions checkpoint was taken. `BENCHMARK_PROTOCOL.md` §2 excludes weight loading and tokenization from timing, and those two are the whole of this stage, so there was nothing to time and no condition to control. **Nothing here is a measurement of the machine.**
- **Environment fingerprint verified**, verbatim: `{"match": true, "differences": [], "fields_compared": 26}`, **exit 0**, from `.venv/Scripts/python.exe bench/machine_state.py verify`. This is the first stage able to run that check against a stored file; Stage 0b wrote it.
- **Compiler and flags UNCHANGED and confirmed byte-identical** to `HARDWARE.md` §5.5: host `/DWIN32 /D_WINDOWS /EHsc /W3 /arch:AVX2 /fp:precise /MD /O2 /Ob2 /DNDEBUG`, CUDA `-D_WINDOWS -Xcompiler=" /EHsc" -O3 --generate-line-info -Xptxas=-O3 -Xptxas=-v -Xcompiler=/arch:AVX2 -Xcompiler=/fp:precise -arch=sm_75`, MSVC 19.44.35229.0, nvcc 13.1.80, sm_75. `CMakeLists.txt` was appended to only — the frozen flag block, the toolset check and every existing target are untouched.
- **Reference environment, confirmed by import rather than by listing:** `.venv-oracle/Scripts/python.exe`, Python **3.12.10**, **tokenizers 0.23.2**, **safetensors 0.8.0**, **numpy 2.5.3**. Invoked only by that explicit path. **Nothing was installed anywhere** — not into `.venv`, not into `.venv-oracle`. The project `.venv` (3.14.2, torch 2.14.0+cu130) was used only for the fingerprint check.
- **Offline gate green:** clean build through `scripts/build.ps1 -Clean`, then **17 of 17 CTest tests pass** in 55.6 s, longest single test 11.9 s against the 300 s cap. One fix-driven rerun, no blind looping.
- Artifacts present on disk before the session, nothing downloaded: `models/gpt2/model.safetensors` 548,105,171 B, `tokenizer.json` 1,355,256 B, `vocab.json` 1,042,301 B, `merges.txt` 456,318 B, `config.json` 665 B, `generation_config.json` 124 B, `tokenizer_config.json` 26 B. All gitignored, proven with `git check-ignore -v` before staging.

#### Q3 — CLOSED. The shipped config confirms all five Stage 0 values

`PERSISTENT.md` §2 Q3 recorded that Stage 0 built the five cuBLAS GEMM shapes — and therefore the prefill denominator for the entire project — on values read from the **published** `config.json` of `openai-community/gpt2`, fetched over the network 2026-09-16. Those were re-read here from the **config file shipped with the downloaded weights**, a different artifact.

| Value | Stage 0, network config 2026-09-16 | Shipped `models/gpt2/config.json` | Verdict |
|---|---|---|---|
| `n_embd` | 768 | 768 | **match** |
| `n_head` | 12 | 12 | **match** |
| `n_layer` | 12 | 12 | **match** |
| `n_ctx` | 1024 | 1024 | **match** |
| `vocab_size` | 50257 | 50257 | **match** |

**Q3 outcome: CONFIRMED. All five identical.** The five (N, K) shapes — QKV 2304×768, attention output 768×768, FFN up 3072×768, FFN down 768×3072, LM head 50257×768 — stand unchanged, **microbenchmark 6 was NOT re-run**, and no timed run was performed in this session. A confirmation is a result: the denominator the whole project rests on is now sourced from the artifact it claims to describe rather than from a network fetch.

Corroborated independently twice over: the reference tokenizer reports vocabulary **50257**, equal to the shipped `vocab_size` (asserted in `tests/test_stage1_oracle.py`), and the enumerated `wte.weight` is **[50257, 768]**.

#### Config fields recorded for Stage 2, so Stage 2 reads an artifact rather than assuming

| Field | Shipped value | What it settles for Stage 2 |
|---|---|---|
| `layer_norm_epsilon` | **1e-05** | the epsilon inside every layernorm |
| `activation_function` | **"gelu_new"** | the tanh approximation of GELU, not the erf form and not ReLU |
| positional scheme | **learned absolute** | `n_positions` 1024 and a stored `wpe.weight` [1024, 768]; no rotary or sinusoidal tensor exists in the file |
| `n_positions` / `n_ctx` | 1024 / 1024 | identical, so no ambiguity about the context bound |
| `attn_pdrop`, `embd_pdrop`, `resid_pdrop` | 0.1 each | training-only; inference applies no dropout |
| `bos_token_id`, `eos_token_id` | 50256, 50256 | the same id, which is also the one added token |
| `architectures` | ["GPT2LMHeadModel"] | a head exists in the architecture even though no head tensor is stored — see below |
| `model_type`, `initializer_range` | "gpt2", 0.02 | recorded for completeness; initializer range is a training artifact |
| `summary_*` | `cls_index`, proj, dropout 0.1 | classification-head fields, unused by the language-model path |
| `tokenizer_config.json` | `model_max_length` 1024 | agrees with `n_ctx` |
| `generation_config.json` | bos/eos 50256, `transformers_version` 4.26.0.dev0 | sampling defaults, which Stage 2 does not use |

#### The weight file as the file presents itself, read from the bytes

| Fact | Value | How it was established |
|---|---|---|
| header length prefix | **unsigned 64-bit little endian**, bytes 0..7, value **14283** | assembled byte by byte so host endianness never enters the result |
| header | UTF-8 JSON, **pure ASCII**, no inter-token whitespace, **no backslash escape of any kind**, 161 members | read and scanned |
| members | **160 tensors + one `__metadata__`** whose value is `{"format":"pt"}` | metadata excluded from the tensor count, asserted |
| dtype | **F32 for all 160** | the only spelling present |
| `data_offsets` | `[begin, end)`, **relative to the start of the data segment**, not to the file | 8 + 14283 + 548090880 = 548105171 = the file size, exactly |
| data segment start | **14291** | 8 + 14283 |
| alignment | **NONE** | 14291 is not a multiple of 8, 16 or 64. The file carries no padding, so a caller must copy bytes out before reading them as aligned floats; `st_tensor_read` does exactly that |

#### Tensor inventory, and where it was recorded

The complete inventory — every tensor's name, dtype, shape, declared `begin`/`end`, absolute `file_offset`, byte length, determined orientation and the evidence class for that orientation, plus the language-model head finding — is committed at **`src/gpt2_tensor_inventory.json`**, produced by the C loader itself (`src/safetensors.c` under `ST_INVENTORY_MAIN`, built as `safetensors_tool` from the same source the unit test links).

**It is deliberately NOT in `bench/results/`.** That directory is defined by `BENCHMARK_PROTOCOL.md` §9 as the structured output of runs and is read by the Stage 10 model as a contract; an inventory is not a benchmark result, carries no timing, and must not be mistakable for one. **This stage wrote nothing to `bench/results/`.**

Composition of the 160: 4 top-level (`wte.weight`, `wpe.weight`, `ln_f.weight`, `ln_f.bias`) plus 13 per layer × 12 layers.

#### Orientation of every 2-D weight, and the limit of what this stage proves

| Tensor (count) | Shape | Orientation | Evidence class |
|---|---|---|---|
| `h.*.attn.c_attn.weight` (12) | [768, 2304] | axis 0 = input, axis 1 = output | **pinned by bias length** — sibling `c_attn.bias` is [2304]; 2304 = 3 × n_embd(768) |
| `h.*.mlp.c_fc.weight` (12) | [768, 3072] | axis 0 = input, axis 1 = output | **pinned by bias length** — sibling bias [3072] = 4 × n_embd |
| `h.*.mlp.c_proj.weight` (12) | [3072, 768] | axis 0 = input, axis 1 = output | **pinned by bias length** — sibling bias [768] = n_embd, and 3072 = 4 × n_embd is the input |
| `wte.weight` (1) | [50257, 768] | axis 0 = index, axis 1 = feature | **pinned by shape arithmetic against the shipped config** — 50257 = vocab_size, 768 = n_embd |
| `wpe.weight` (1) | [1024, 768] | axis 0 = index, axis 1 = feature | **pinned by shape arithmetic against the shipped config** — 1024 = n_ctx |
| `h.*.attn.c_proj.weight` (12) | [768, 768] | **UNRESOLVED BY SHAPE** | axes equal; the sibling bias [768] matches both and therefore discriminates nothing |

The bias-length argument is arithmetic on the file's own shapes and assumes no convention: **a bias has one entry per output**, so where its length matches exactly one extent, that extent is the output axis. It pins every rectangular weight in the file to **[input, output]** — the storage order a `Conv1D`-style checkpoint uses, which is the **transpose** of the `[output, input]` convention a `nn.Linear`-style implementation expects. Where an implementation's convention and the checkpoint's storage convention disagree, the loader is where the reconciliation happens, explicitly; the inventory record carries that statement per tensor.

For the twelve square `attn.c_proj.weight` tensors, shape cannot settle it and this entry does not pretend otherwise. The alternative evidence, **stated as inference and not as established fact**: every rectangular 2-D weight in this same file is pinned by its own bias to [input, output], and these tensors are named and grouped alongside them within the same layer, so the loader reads them the same way.

**THE LIMIT, STATED PLAINLY.** Orientation is a **semantic** mapping and this stage does not verify it. The exact byte comparison below does **not** test it either, because both sides read the same bytes from the same file and both report the shape as stored — no orientation error is detectable that way. The loader is **parsed and shape-checked, not verified**. The proof lands at **Stage 2**, when a forward pass produces logits that either match the reference or do not.

#### The language-model head is TIED, not stored

Settled by inspecting the enumeration, not assumed: **zero** of the 160 tensor names contain `lm_head`, and **exactly one** matrix in the file carries the vocabulary size against the embedding width — `wte.weight` [50257, 768]. The head is therefore not stored, and **Stage 2 must tie it to the token embedding**. A loader that does not tie it fails at Stage 2 in a way that looks like a bug in the head. The Stage 1 loader stores no head tensor because the file declares none; tying is a forward-pass decision and is not Stage 1's to make.

#### Parameter and buffer byte split — recomputed, after a first computation that was WRONG

The twelve `h.*.attn.bias` tensors, shape [1, 1, 1024, 1024], are **registered causal-mask buffers, not learnable parameters**: their contents were checked and every element is 0 or 1, with the [1024, 1024] plane exactly lower-triangular ones.

**The first computation of the split was wrong and is recorded rather than quietly replaced.** It selected the buffer bucket with a suffix test for `attn.bias`, which also matches `c_attn.bias` — so the twelve **learnable QKV bias tensors** were counted as mask buffers. The error was 12 × 2304 × 4 = **110,592 B** moved into the wrong bucket, and it inflated the buffer figure to 50,442,240 B while deflating the parameter count by 27,648 parameters.

Recomputed from the enumerated inventory, bucket by bucket `[derived]`:

| Bucket | Bytes | Arithmetic |
|---|---|---|
| causal-mask buffers, 12 × `h.*.attn.bias` | **50,331,648** | 12 × 1024 × 1024 × 4 = 50,331,648 |
| everything else — the parameters | **497,759,232** | 548,090,880 − 50,331,648 |
| data segment, for reconciliation | **548,090,880** | 50,331,648 + 497,759,232, equal to 548,105,171 − 14,291 |

**Parameter count = 497,759,232 B ÷ 4 B per F32 = 124,439,808 parameters, remainder 0.**

`TECHNICAL_SPEC.md` §1 states "124M parameters" for this model. **They agree** at the precision §1 states: 124,439,808 rounds to 124M. Note that the count includes `wte.weight`'s 50257 × 768 = 38,597,376 parameters once; because the head is tied rather than stored, it adds nothing further.

Why the correction mattered enough to record: a parameter count is an input to Stage 2's operation counts and from there to the Stage 10 model, where a silently wrong figure would be looked for last.

**Recommendation to Stage 2, offered as a recommendation and not as a decision Stage 1 is entitled to take: regenerate the causal mask rather than load it.** It is 50,331,648 B of the file — 9.2% of the data segment — to express a triangular predicate that costs two integer comparisons per element.

#### The exact comparison against the reference

**160 of 160 tensors, exact, zero tolerance, zero differences.** A tolerance would hide precisely the errors the comparison exists to catch: both sides read the same bytes from the same file, so any difference at all is a defect rather than noise.

Covered, per tensor: the **name set** in both directions (nothing present on one side and absent on the other), the **dtype** (all F32 / numpy float32), the **shape**, the **byte length**, and **every byte** read from the file at the absolute `file_offset` the C loader recorded, compared against `safetensors.safe_open`. Differences are reported as the first differing byte with both values, not as a count. No tensor was excluded and no subset was taken.

#### Tokenizer — which artifacts, and what the reference actually does

The three artifacts **agree exactly**: the vocabulary object inside `tokenizer.json` equals `vocab.json` entry for entry, and its merge list equals `merges.txt` line for line after the `#version: 0.2` header. **No disagreement to report.**

**The C tokenizer consumes `tokenizer.json`**, for three reasons. It is the only artifact that **declares the added tokens**; it is the only one that declares the **pre-tokenizer configuration** (ByteLevel, `add_prefix_space` false); and it is the **easier parse**, using only `\"` (311) and `\\` (121) escapes where `vocab.json` uses `\uXXXX` **35,908 times** and would drag UTF-16 surrogate reassembly into the inference path for no gain. `merges.txt` is still read — by the unit test, as an independent cross-check that every sampled merge pair carries the rank the artifact gives it and that the merge count matches.

Two reference behaviours were found by probing and matched rather than assumed:

1. **`<|endoftext|>` in ordinary text is not literal.** The reference splits it out as the added token, id **50256**, before the regex ever sees it, and only `skip_special_tokens=False` decoding round-trips. The C tokenizer reads the added-token table from the artifact and does the same, so the round trip stays byte-exact. Had the tokenizer been built from `vocab.json` + `merges.txt`, this behaviour would have had to be assumed.
2. **The pre-tokenizer's whitespace rule is not the obvious one.** `\s+(?!\S)` makes a whitespace run that is followed by a non-space character **give up its last character to the next piece**, while a run reaching end of input stays whole: `"a b  c   d"` splits as `a | ·b | · | ·c | ·· | ·d`, and `"x\r\ny"` splits `\r` and `\n` into separate pieces. Checked against the reference before being written, not deduced afterwards.

#### The round trip, per class, with the two checks reported separately

48 records, seven required classes plus an embedded-NUL class. **Both checks:** `decode(encode(s))` byte-exact, **and** the id sequence element for element against the committed reference sequence. The second is the one that matters — a decode round trip alone passes under any bijection and would not catch a merge-order bug.

| Class | Records | decode round trip byte-exact | id sequence equals the reference |
|---|---|---|---|
| `utf8_multibyte` (5 of them outside the BMP) | 10 | **10 / 10** | **10 / 10** |
| `whitespace_runs` | 7 | **7 / 7** | **7 / 7** |
| `newlines_tabs` | 5 | **5 / 5** | **5 / 5** |
| `empty` | 1 | **1 / 1** | **1 / 1** — encodes to the empty sequence, decodes to zero bytes |
| `mid_merge_prefix` | 10 | **10 / 10** | **10 / 10** |
| `special_literal` | 5 | **5 / 5** | **5 / 5** |
| `embedded_nul` | 3 | **3 / 3** | **3 / 3** |
| `invalid_utf8` | 7 | **7 / 7** | **no reference sequence exists — see below** |

**410 ids compared in total**, every one below the vocabulary size 50257 read from the artifact. Fixtures: `tests/fixtures/tokenizer_roundtrip_corpus.tsv` (hex-encoded, so embedded NULs and non-UTF-8 bytes survive identically into both C and Python) and `tests/fixtures/tokenizer_expected_ids.tsv`, generated by `reference/stage1_oracle.py emit` and committed. Both are labelled **TEST FIXTURES** in their headers. **They are not the benchmark prompt set and imply nothing about it** — that is Stage 3's decision, §1 D3.

**The invalid-UTF-8 class has no reference id sequence, and that is the correct outcome, not a gap in the work.** The reference API takes `str`: raw `bytes` and a surrogate-escaped `str` are both rejected with `TypeError: TextInputSequence must be str`. The fixture records `NO_REFERENCE` with that reason — an empty field with a stated cause, never a plausible placeholder — and the C behaviour for such input is **documented rather than discovered**: a byte that begins no well-formed UTF-8 sequence is consumed as one character of class "other". The decode round trip is still asserted byte-exact for every one of those records, including one containing all 256 byte values.

**The alternative that existed and was not taken:** the raw bytes could have been mapped through the byte-level encoder into a string the reference API does accept, which would have produced an id sequence and a green check. That mapping is **part of the implementation under test**, so the comparison would have been partly against itself. A weaker oracle presented as a passing check is worse than a stated gap.

#### The no-libraries claim — three judgement calls, each resolved at build time

`TECHNICAL_SPEC.md` §2 places the tokenizer and the weight loader in C as part of the no-libraries claim, and Stage 1 is the first stage at which that claim could be broken. It was not: the safetensors header parsing, the tensor mapping, the BPE merge loop and the byte-level encoding and decoding are all in C. A library appears only on the far side of a comparison, in `reference/stage1_oracle.py`, which the engine never calls.

1. **The safetensors header is JSON — a minimal parser was written rather than a dependency taken.** The reason is not purity: a parser pulled in here is a parser in the inference path for the life of the project, and the document is fixed, small and machine-generated. **Accepted:** objects, arrays, strings, unsigned decimal integers, inter-token whitespace, the escapes `\" \\ \/ \b \f \n \r \t`, and unknown keys inside a tensor object (parsed and discarded, so a future key does not break the read). **Rejected, each with a named error:** `\uXXXX`, `true`/`false`/`null`, signs, decimal points, exponents, trailing commas, unterminated strings, raw control bytes inside strings, and any trailing byte after the top-level object.
2. **`tokenizer.json` gets a second, separate scanner** rather than sharing the first. The subsets genuinely differ — 14 KB flat and escape-free against 1.3 MB nested with a 50257-entry string table — and one general parser spanning both would be the very dependency-shaped object being avoided.
3. **The Unicode class table is generated, and this is the third judgement call.** The pre-tokenizer pattern uses `\p{L}`, `\p{N}` and `\s`, whose membership depends on the Unicode version of whichever engine evaluates them. Rather than copy a Unicode data file or link a regex engine, **every codepoint from U+0000 to U+10FFFF was put through the reference pre-tokenizer and classified by the piece it landed in**, and the result — **831 ranges** — was baked into `src/tokenizer.c` as a static table with a binary search. No library runs at inference time. **The dependency this creates must be stated:** the table encodes the behaviour of **tokenizers 0.23.2** specifically, and a different reference version could classify differently, in which case the C tokenizer would be matching a reference nobody is running. The confirmed oracle versions are recorded above under Conditions. **The evidence that copying a Unicode data file would have been wrong:** the derived table disagrees with **Python 3.12's own `unicodedata` at 5008 codepoints** — U+001C..U+001F, which Python calls whitespace and the reference treats as "other", and a block of codepoints Python's tables still call unassigned while the reference classifies them as letters.

One further check of the same kind: the byte-level alphabet was **verified against the reference rather than trusted**. 243 of the 256 byte values were observed in the reference's own mapped output and every one agreed, and the constructed 256-character alphabet set equals the reference's alphabet exactly. The 13 that could not be checked — 0xC0, 0xC1 and 0xF5..0xFF — are precisely the bytes that cannot appear in valid UTF-8, so the reference can never be made to emit them; they are exercised instead by the invalid-UTF-8 corpus records.

#### Where the artifacts contradicted the documents

1. **`TECHNICAL_SPEC.md` §4 did not describe the repository, and was amended in place on operator instruction.** It listed neither `tests/` nor `scripts/`, both of which exist and both of which appear in `CLAUDE.md`'s layout, and it had no home for a committed data artifact. Added to the existing tree, nothing regenerated and nothing reordered: `tests/`, `scripts/`, `models/` (gitignored, never committed), and `src/gpt2_tensor_inventory.json`. §1 was not touched and its warning against trusting its own table stands. The reason this was fixed rather than left as a reported discrepancy: §4 is the document every later stage prompt is written against for output placement, so a layout missing two top-level directories would mislead the Stage 2 prompt exactly as the dangling "see §7" pointer would have.
2. **`CLAUDE.md` carried a dangling filename.** The operator renamed `CC_PROMPT_FORMAT.md` to `CLAUDE_CODE_PROMPT_FORMAT.md` before the session; `CLAUDE.md` still named the old file in its document-rules section. Fixed in place, one line. The four remaining references inside `KICKOFF_TEMPLATE.md` and the renamed file's own first line were **left untouched** — they are inside the operator's authoring tools.
3. **No architecture value in any document was contradicted.** Q3 confirmed all five, and `TECHNICAL_SPEC.md` §1's "124M parameters" agrees with the 124,439,808 counted from the file.

#### What this taught

A weight loader has two failure modes and only one of them is detectable at this stage. Structural errors — a byte range off the end, a length disagreeing with the shape, a header that is not the document it claims to be — are all catchable from the file alone, and ten asserts catch them. Semantic errors are not: **orientation cannot be tested by comparing bytes**, because a transposed reading of the same bytes is byte-identical to a correct one. The honest output of this stage is therefore a loader that is *parsed and shape-checked*, plus an explicit statement of what remains unproven and where it gets proven.

The sharper lesson came from the tokenizer, and it is about what a reference is for. Three behaviours would each have been guessed wrong: that `<|endoftext|>` is literal text (it is not), that a whitespace run is one piece (a run followed by text gives up its last character), and that Unicode letter membership is whatever the nearest Unicode table says (the reference disagrees with Python's at 5008 codepoints). None of those is discoverable by reading the specification of byte-level BPE. Each was found by **probing the reference and matching what it does rather than what it ought to do** — and the third produced a generated artifact whose provenance is a probe rather than a citation. The corollary is the one to carry: where the reference cannot be probed at all, as with byte sequences it refuses to accept, the correct output is a documented behaviour and a stated gap, not an oracle bent until it answers.

## Stage 2 — Naive C forward pass (baseline)
*Predict the baseline itself from operation counts and measured machine throughput, before running it.*
**Status:** COMPLETE, 2026-09-19. Offline gate green (clean build, 20/20 tests). Seven timed configurations, 25 warmup and 30 samples each; **five VALID, two INVALID and named**. Correctness: the greedy token sequence matches the reference exactly and the divergence is measured and reported; **no tolerance was adopted and D2 remains Stage 3's**.

#### Prediction  (written 2026-09-18, before implementation)

**What is predicted.** Prefill latency in milliseconds for one forward pass over a prompt of L tokens, and decode latency in milliseconds per generated token at a context length c, with the timing bracket excluding weight loading and tokenization per `BENCHMARK_PROTOCOL.md` §2. There is no KV cache at this stage, so a decode step re-runs the full forward pass over the whole context and its cost is expected to grow with c.

**Assumptions, stated before measurement.**

1. The matmul uses the textbook `ijk` loop order with a dot-product inner loop. Loop interchange is itself a cache optimization and belongs to Stage 5.
2. The language-model head is computed at every position during prefill, and only at the last position during a decode step.
3. Scalar code throughout; the compiler does not auto-vectorize the matmul inner loop.

**Inputs and their sources.** Architecture values `n_embd` 768, `n_head` 12, `n_layer` 12, `n_ctx` 1024, `vocab_size` 50257, and the parameter count 124,439,808, from the Stage 1 `MEASUREMENTS.md` entry, which confirmed them against the config shipped with the weights. Measured scalar FP32 ceiling 8.565 GFLOP/s, measured L3-resident bandwidth 70.81 GB/s, cache capacities 32 KiB L1d / 256 KiB L2 / 8 MiB L3 with an effective single-thread edge at 4 MiB, and 64 B cache lines, all from `HARDWARE.md` §2, Stage 0b reference run 2.

**Figures deliberately excluded, with reasons.** The measured vectorised AVX2 peak of 48.411 GFLOP/s: this stage writes scalar code, and using the vectorised ceiling would also pre-empt Stage 6's entire framing (`PERSISTENT.md` §8, W8). The single-channel theoretical ceiling of 23.464 GB/s: it is derived from part numbers rather than measured, and a theoretical peak may not enter a prediction (`PERSISTENT.md` §8, W7). Measured CPU DRAM bandwidth: the `HARDWARE.md` §2 field is empty and Stage 0b left it empty with a better-evidenced reason (`PERSISTENT.md` §8, W1). The measured achievable GPU bandwidth of 170.882 GB/s and every cuBLAS throughput figure: both are GPU denominators and this is a CPU stage (`PERSISTENT.md` §8, W9). The 2496 MHz clock: `HARDWARE.md` §5.3 records it as a static nominal read rather than a live frequency. The 4.4 percent noise floor: it governs speedup claims and this stage claims none.

**Parameter partition, verified against the recorded total.** Per layer the matmul weights are `c_attn` 768×2304 = 1,769,472, `attn.c_proj` 768×768 = 589,824, `mlp.c_fc` 768×3072 = 2,359,296 and `mlp.c_proj` 3072×768 = 2,359,296, totalling 7,077,888. Adding the four biases (2304 + 768 + 3072 + 768 = 6,912) and the two layernorms (4 × 768 = 3,072) gives 7,087,872 per layer, and 85,054,464 across twelve layers. The top level contributes `wte` 38,597,376, `wpe` 786,432 and `ln_f` 1,536, totalling 39,385,344. The sum is 124,439,808, which equals the parameter count recorded in the Stage 1 entry exactly. The partition is therefore confirmed rather than assumed.

Matmul weight elements are 84,934,656 in the layers plus 38,597,376 in the tied head, totalling 123,532,032. The head is added explicitly: because it is tied, `wte` appears once in the parameter count but is used twice per forward pass, so a FLOP count built from the parameter count alone would undercount it.

**FLOPs per token.** Layers: 2 × 84,934,656 = 169,869,312. Head: 2 × 38,597,376 = 77,194,752. Total 247,064,064.

Attention adds, per token at context c, 2 × (12 heads × 64 head_dim × c) for the score matmul and the same again for the weighted sum of values, giving 3,072·c per layer and **36,864·c across twelve layers**. At c = 128 that is 4,718,592, or 1.9 percent of the weight matmuls; it is carried in full below but is not the decisive term at these lengths.

The elementwise work adds 12 × 3072 = 36,864 `gelu_new` tanh evaluations and 144·c softmax exponentials per token. At a scalar transcendental cost of order tens of nanoseconds this lands near 1 percent, and it is the weakest-sourced component of this prediction.

**Why prefill is not compute-bound here.** With `i` as the outermost loop, the whole weight matrix is re-walked for every token. `mlp.c_fc` at 9,437,184 bytes exceeds the 8 MiB L3, and the head at 154,389,504 bytes exceeds it eighteen-fold, so nothing survives between tokens. The prefill weight reuse that makes prefill compute-bound in `BENCHMARK_PROTOCOL.md` §1 is discarded by this loop order. Recovering it is exactly what Stage 5 does.

**Both routes evaluated, on the same unit — one full pass of the weight set for one token.** Memory route, as an optimistic bound: 4 × 123,532,032 = 494,128,128 bytes moved; no measured CPU DRAM figure exists and the theoretical one is barred, so the measured L3 bandwidth is used as a bound DRAM cannot beat, giving 494,128,128 / 70.81e9 = 6.98 ms. That is a floor, not an estimate. Compute route: 247,064,064 / 8.565e9 = 28.85 ms at full ceiling. The compute route binds, by a factor of four over an already-optimistic memory bound, so the prediction rests on the efficiency fraction.

**Efficiency fraction.** f = 0.15 of the measured scalar ceiling, giving an effective 1.28475 GFLOP/s. Three reductions apply against a ceiling measured on a register-resident loop with eight independent FMA chains and no memory traffic: the single accumulator in the inner loop exposes the FMA dependency those eight chains existed to hide; there are two loads per FMA where the ceiling had none; and the strided access to the weight matrix touches a fresh 64-byte line for every 4 bytes used, amortised sixteen-fold only across the middle loop. The plausible range is 0.08 to 0.30. As corroboration only, a naive `ijk` matmul on modern x86 commonly lands between 1 and 2 GFLOP/s, which brackets 1.285.

**Arithmetic for each configuration.** Prefill at L tokens is L × 247,064,064 for the weight matmuls plus 36,864 × L(L+1)/2 for attention. A decode step at context c runs the twelve layers over all c tokens (c × 169,869,312), attends over the same triangle (36,864 × c(c+1)/2), and computes the head once (77,194,752).

| Configuration | FLOPs | Predicted |
|---|---|---|
| Prefill, L = 32 | 7,925,514,240 (7.926 GFLOP) | **6,169 ms** |
| Prefill, L = 64 | 15,888,777,216 (15.889 GFLOP) | **12,367 ms** |
| Decode step, c = 32 | 5,532,476,928 (5.532 GFLOP) | **4,306 ms per token** |
| Decode step, c = 64 | 11,025,507,840 (11.026 GFLOP) | **8,582 ms per token** |
| Decode step, c = 128 | 22,124,815,872 (22.125 GFLOP) | **17,221 ms per token** |

**Decode-to-prefill ratio at length 64: 0.69.** The structural claim is that without a KV cache a decode step re-runs the full forward pass over the entire context, so it costs the same order as a prefill of that length rather than one-Lth of it. It falls below 1.0 only because prefill computes the head at all L positions while a decode step computes it once. Decode cost per token is predicted to rise very nearly linearly in c — the predicted values double from c = 32 to 64 to 128 — and flattening that curve is what Stage 4 exists to do.

**Falsified if:**

- The measured result is more than 3× faster than predicted and the generated machine code proves the matmul inner loop was vectorized. That falsifies the scalar assumption and would pre-empt Stage 6, and it must be caught at build time rather than diagnosed afterwards.
- The measured result is more than 3× slower than predicted, which would mean DRAM binds harder than the L3-derived floor permits and that the empty CPU DRAM bandwidth field is material at Stage 2 rather than at Stage 5.
- The decode-to-prefill ratio at equal length comes out near 1/64 rather than near 0.69, or decode cost per token is flat in c. Either would mean a KV cache was built, which is out of scope.
- A result within 2× in either direction does not falsify the reasoning; it indicates the efficiency fraction was imprecise, which is the expected outcome.

#### Conditions

| Condition | Value |
|---|---|
| Date, branch | 2026-09-19, branch `stage-2` cut from `main` at `ec0dd89` |
| Prediction commit | `b33cb35`, "stage 2: prediction committed", `MEASUREMENTS.md` only, committed **before any implementation file existed** and not consulted again until the Gap section below |
| Environment fingerprint | `machine_state.py verify`: **25 of 26 fields match, one differs** — `build_flags.BENCH_BUILD_TIMESTAMP`, stored `2026-09-18T06:56:02Z`, current `2026-09-19T09:38:41Z`. Every substantive field — driver 591.44, CUDA 13.1.80, compiler, all host and CUDA flags, power limit, Windows power plan — is unchanged, so `BENCHMARK_PROTOCOL.md` §4's condition is met and the comparison against prior stages is **not void**. **Stage 1 recorded `{"match": true, "differences": [], "fields_compared": 26}` against the same stored file after its own clean build.** Two stages ran one check on one file and got different answers; **this stage cannot establish why, and the cause is UNESTABLISHED.** Raised as `PERSISTENT.md` §8 **W12**, owned by Stage 3. The stored fingerprint file was NOT rewritten, regenerated or refreshed |
| Clock lock | `nvidia-smi -lgc 1365,1365` applied from this elevated session, then verified: `verify-lock --mhz 1365` → `locked: true`, **10 of 10 samples at 1365 MHz**, `off_target_samples: []`, exit 0. `-lmc` not attempted (§2 Q11: device-capability limit). The lock is a thermal control for a CPU measurement as much as a clock control for a GPU one — this laptop's CPU and GPU share one thermal solution |
| Compiler and flags | **Confirmed unchanged against `HARDWARE.md` §5.5.** MSVC 19.44.35229.0, toolset 14.44.35207, x64, VS 2022 Build Tools. Host flags `/DWIN32 /D_WINDOWS /EHsc /W3 /arch:AVX2 /fp:precise /MD /O2 /Ob2 /DNDEBUG`; CUDA `nvcc 13.1.80`, `-arch=sm_75`. Read from `build/generated/build_info.h`, which the build system writes from the same variables that reach every results file |
| AC power | **On AC.** `PowerOnline = True`, not on battery. A run on battery is INVALID outright and none was taken |
| Network | Wi-Fi ("Sahith5G 3"), `NetworkCostType = Unrestricted` — **not metered** |
| Session elevation | **Elevated** (`IsInRole(Administrator)` true, user `MSI\saket`), so the clock lock was applied without an operator round trip |
| Thread placement | Pinned to **logical CPU 2 of 8**, priority raised (ABOVE_NORMAL class, THREAD_PRIORITY_HIGHEST), applied outside every timed bracket by `bench_pin_current_thread` and **recorded as applied rather than assumed**: `pinned=1`, `priority_raised=1`. No deviation from the requested placement was observed |
| CPU timer | `QueryPerformanceCounter`, monotonic. Never wall clock |
| Reference oracle environment | The repository `.venv` — Python 3.14.2, torch 2.14.0+cu130, numpy 2.5.3. **CPU only**, `torch.set_num_threads(1)` applied and verified as taken (`torch.get_num_threads()` → 1). Recorded as conditions because thread count changes the reduction order and therefore the divergence figures Stage 3 inherits |
| Oracle weight-reading branch | **The second branch.** `import safetensors` **FAILS** in `.venv` (it lives only in `.venv-oracle`, which has no torch), and nothing was installed into either frozen environment. The oracle therefore reads the file with the standard library and numpy — 8-byte little-endian length prefix, JSON header, `numpy.frombuffer` on a fresh `bytes` read at the absolute file offset — and asserts its own parse against `src/gpt2_tensor_inventory.json` for **all 160 tensors** on name, dtype, shape, absolute file offset and byte length before using any of them. That cross-check is trustworthy rather than circular because the C loader that produced the inventory was proven byte-exact against the real safetensors library in Stage 1 |
| Ordering of correctness and timing | The correctness comparison ran **BEFORE** the timed runs and had exited before the first timed bracket. It holds the full weight set in numpy and again in torch, over a gigabyte resident on a machine with a single 8 GiB DIMM, and `BENCHMARK_PROTOCOL.md` §3 makes a run with another workload present INVALID. The oracle was **not running during any timed bracket** |
| CPU frequency telemetry | The committed `CpuTelemetrySampler` (`bench/machine_state.py`) was **reused** with the Stage 0b configuration: interval 0.02 s, exclusion set logical CPUs 0 and 2, resulting mask `0xf0` (logical CPUs 4–7, clear of CPU 2 where the measured thread runs and of its SMT sibling CPU 3), sampling from a separate process so no probe can land inside a timed bracket. Its per-probe cost and duty cycle were proven and recorded in Stage 0b, so this is reuse of a proven instrument, not the introduction of a new one |

**Process set, enumerated before the first timed run, named rather than summarised.**

*Holding a GPU context* (`nvidia-smi --query-compute-apps` and the process table, 16 entries): `dwm.exe`, `explorer.exe`, `ShellHost.exe`, `StartMenuExperienceHost.exe`, `SearchHost.exe`, `TextInputHost.exe`, `CrossDeviceResume.exe`, `LockApp.exe`, `ApplicationFrameHost.exe`, `SystemSettings.exe`, `PhoneExperienceHost.exe`, `OmApSvcBroker.exe` (MSI NBFoundation Service), `logioptionsplus_agent.exe`, `msedgewebview2.exe`, and **two processes of this session's own editor (`claude.exe`)**. GPU memory in use 486 MiB of 4096 MiB, utilisation 1%, temperature 49–55 °C, no throttle reason active (`sw_thermal_slowdown` Not Active, `hw_thermal_slowdown` Not Active).

*Significant CPU consumers* (accumulated CPU seconds at enumeration time): `MsMpEng` (Windows Defender real-time scanning) 2509 s, `System` 1848 s, `esrv_svc` (Intel Energy Server service) 798 s, `svchost` (pid 4824) 316 s, `WmiPrvSE` 274 s, **`claude.exe` ×4 (this session: 258 / 193 / 161 / 106 s)**, `dwm` 155 s, `logioptionsplus_agent` 122 s, `explorer` 57 s, `csrss` 50 s, `nvcontainer` 46 s, `SearchIndexer` 40 s, `conhost` 39 s, three further `svchost` instances 24–30 s, `SDXHelper` 22 s. 231 processes resident.

**How the set differs from Stage 0's, which is the set the 4.4% noise floor was measured against.**

- **Riot Vanguard (`vgc`, `vgtray`) — ABSENT.** Resident throughout Stage 0; absent in Stage 0b and absent here.
- **Nahimic — ABSENT.** Resident in Stage 0; closed in Stage 0b, absent here.
- **Intel DSA (`DSAService`, `DSATray`, `DSAUpdateService`) — ABSENT.** Present at this session's checkpoint and **closed by the operator before the first timed run**; verified absent by name afterwards. `esrv_svc`, the Intel Energy Server service, **remained running** and is recorded as present rather than assumed gone.
- **Logitech Options+ — PRESENT.** The operator attempted to close it; it is running under a new pid (2884, against 14976 at the checkpoint), so it respawned. **Recorded as present, not as closed** — the same handling Stage 0b gave it.
- **`msedgewebview2.exe` ×6 — PRESENT**, uncloseable, and equally resident during Stage 0, so a shared condition rather than a difference.
- **MSI service stack (`OmApSvcBroker`, `MSIService`) — PRESENT**, left running by the standing decision that stopping it risks silently changing the clock regime (D8, §2 Q9) with no way to detect it.
- **Windows Defender — PRESENT**, not excluded or modified.
- **NEW, and not a condition any prior stage had: this session's own editor, `claude.exe` ×9 processes**, four of them significant CPU consumers. Stage 0 and Stage 0b did not carry it.

Net: the set is **lighter than Stage 0's** on the two heaviest items (Vanguard and Nahimic both absent) and **carries one addition Stage 0 did not** (this session's editor). **The 4.4% noise floor is NOT re-derived, adjusted or restated here** — this stage did not re-run the full microbenchmark suite and cannot re-derive it (`PERSISTENT.md` §8 W5, whose Status is unchanged). It is recorded so that a stage which does re-derive it knows what this session ran under.

#### Architecture and config values, each with the artifact it came from

No architecture value in this entry came from `TECHNICAL_SPEC.md` §1, which warns against trusting its own table, and none was taken from the stage prompt or from memory. `src/model.c` reads every one of them at load time; nothing is compiled in except the tensor names.

| Value | Read | Artifact |
|---|---|---|
| `n_layer` | 12 | `models/gpt2/config.json` |
| `n_head` | 12 | `models/gpt2/config.json` |
| `n_embd` | 768 | `models/gpt2/config.json` |
| `head_dim` | 64 | derived as `n_embd / n_head`, and the division is checked to be exact at load |
| `n_ctx` | 1024 | `models/gpt2/config.json` |
| `vocab_size` | 50257 | `models/gpt2/config.json`, and **independently confirmed** by `tok_vocab_size()` reading `tokenizer.json` — asserted equal in `tests/test_model.c` and in the driver |
| `layer_norm_epsilon` | 1e-05 | `models/gpt2/config.json` |
| `activation_function` | `gelu_new` | `models/gpt2/config.json`. The loader **fails** if the config names anything else rather than substituting a nonlinearity |
| positional scheme | learned absolute | `wpe.weight` [1024, 768] exists in the file and no rotary or sinusoidal tensor does; `n_positions` = `n_ctx` = 1024 |
| every tensor's name, dtype, shape, absolute file offset, byte length | 160 tensors | `src/gpt2_tensor_inventory.json`, **cross-checked against the weight file's own header** through `st_find` before any byte is read — a disagreement on any of the five fails the load with `MODEL_ERR_INVENTORY` |
| the inventory's own `config_values_used` | `n_embd` 768, `n_layer` 12, `n_head` 12, `vocab_size` 50257 | asserted equal to the config being read now, so the two artifacts cannot describe different models unnoticed |

**Nothing disagreed.** Every value matches the table the Stage 1 entry recorded under "Config fields recorded for Stage 2", and the five architecture values match the Q3 closure.

#### Orientation reconciliation, as implemented

The reconciliation happens **at load time, in `src/model.c`, explicitly, and is recorded per tensor** in `model_tensor_record` — not as transposes scattered through the arithmetic. `gpt2_tool --records` prints the record; it is what the table below is taken from.

| Tensor (count) | Stored shape | Inventory orientation | What the loader did | What the forward pass consumes |
|---|---|---|---|---|
| `wte.weight` (1) | 50257 × 768 | `axis0_index_axis1_feature` | nothing | rows are tokens for the embedding lookup, and the **same storage** enters the tied head through the transposed-B form |
| `wpe.weight` (1) | 1024 × 768 | `axis0_index_axis1_feature` | nothing | rows are positions |
| `h.*.attn.c_attn.weight` (12) | 768 × 2304 | `axis0_input_axis1_output` | nothing | `[input, output]`, which is exactly `gemm_f32`'s B layout |
| `h.*.mlp.c_fc.weight` (12) | 768 × 3072 | `axis0_input_axis1_output` | nothing | `[input, output]` |
| `h.*.mlp.c_proj.weight` (12) | 3072 × 768 | `axis0_input_axis1_output` | nothing | `[input, output]` |
| `h.*.attn.c_proj.weight` (12) | 768 × 768 | **`unresolved_by_shape`** | **nothing, under the selected reading** — the alternative reading transposes the bytes once, at load | `[input, output]` |

**The square tensors were settled by measurement, not by inheriting Stage 1's inference.** Both readings are runnable (`model_cproj_reading`, `gpt2_tool --cproj as-stored|transposed`) and both were run against the same oracle at the same prompt:

| Reading | Max absolute divergence from the oracle | Top-1 agreement | Top-5 set agreement |
|---|---|---|---|
| **as-stored** `[input, output]` | **3.128e-04** | **8 of 8 positions** | **8 of 8** |
| transposed `[output, input]` | **121.468** | 2 of 8 | **0 of 8** |

Five orders of magnitude apart, and the transposed reading's *mean* absolute divergence (71.6) is larger than the entire dynamic range of a correct logit vector. `tests/test_model.c` exercises both readings and asserts they produce different logits, so the choice is decidable by observation; `tests/test_reference_impl.py` asserts the transposed reading diverges further. **The reading was selected by observation.**

**The limit on what the logit comparison establishes for orientation, stated plainly.** A reference that applied the same orientation decision as the C loader would agree with a wrong mapping. The oracle derives its own layer shapes from the config and from the shapes the file declares, and asserts each rectangular weight against the bias length that pins it — but for the square tensors it takes the reading as a parameter, exactly as the engine does. **So for orientation the discriminator is the both-readings comparison plus readable English text, not the logit match alone.** The generated text below is the part of that evidence which does not share an author with both implementations.

#### The language-model head is tied, and how that was verified

No tensor in the file is named `lm_head` and exactly one matrix carries the vocabulary size against the embedding width, so the head is `wte.weight` itself. **Verified by identity of storage, not by comparing values**: `model_head_weight(m) == model_token_embedding(m)` — the same pointer — asserted in `tests/test_model.c`. A transposed copy would pass a value comparison and fail this one. The oracle is checked the same way on its own side: `model.head.data_ptr() == model.wte.data_ptr()`.

Because the head is tied, `wte` is [vocab, n_embd] = `[output, input]`, which is the transpose of the layout the weight matmuls use. It is consumed through `gemm_naive_bt` rather than copied into a transposed buffer, because **tying the head means using that storage**; a transposed copy would also cost 154 MB of additional resident memory.

#### The causal mask was REGENERATED, not loaded

**Determination: regenerated.** The twelve `h.*.attn.bias` tensors are never read and no mask is materialized at all. Causality is the loop bound `j <= t` plus an explicit zero above the diagonal.

Reasons: those tensors are 50,331,648 bytes, 9.2% of the data segment, to express a triangular predicate that costs two integer comparisons per element; Stage 1 verified their contents are exactly lower-triangular 0/1 and recommended regeneration; and reading them would add 50 MB to both the load time and the resident set of a process that already holds 498 MB of parameters on a machine with one 8 GiB DIMM. **The equivalence is a test, not a comment**: `tests/test_model.c` changes the token at the last position and asserts that **no logit at any earlier position moves, bit for bit**, over 6 positions × 50,257 vocabulary entries — and separately asserts that the logits at the changed position *do* move, so the check cannot pass vacuously.

#### The matmul interface, the loop order, and the confirmation that it is scalar

**Interface** (`src/gemm/gemm.h`), which Stages 5 and 6 substitute against:

```c
void gemm_naive   (int M, int N, int K, const float *A, int lda,
                   const float *B,  int ldb,  float *C, int ldc);   /* B  is K x N */
void gemm_naive_bt(int M, int N, int K, const float *A, int lda,
                   const float *Bt, int ldbt, float *C, int ldc);   /* Bt is N x K */

typedef void (*gemm_fn)(int, int, int, const float *, int, const float *, int, float *, int);
typedef void (*gemm_bt_fn)(int, int, int, const float *, int, const float *, int, float *, int);
typedef struct { const char *name; gemm_fn mul; gemm_bt_fn mul_bt; } gemm_impl;
extern const gemm_impl gemm_impl_naive;
```

Row-major throughout, explicit leading dimensions so a caller may pass a sub-block of a wider buffer — attention does exactly that, one head at a time out of a `[tokens, n_embd]` activation matrix. `C` is written, not accumulated. No bias, no activation, no transpose of `A`: bias and the elementwise work stay in the forward pass, so what later stages measure is a matmul and not a fused kernel that changed shape between stages. A later implementation is substituted through `model_set_gemm`, which touches no arithmetic in the forward pass. **Two operations rather than one because the weight file forces both**: every 2-D weight is stored `[input, output]`, which is `gemm_f32`'s B layout, while the tied head is `[output, input]` and must be used in that storage.

**Loop order used: `ijk`** — `i` outermost, `j` middle, `k` innermost, dot product in the inner loop, **one accumulator**. Interchange to `ikj` is a cache optimization and belongs to Stage 5's measured gain; several accumulators would hide the dependency chain and belong to no stage that has claimed them. Neither was taken.

**Scalar codegen confirmed from the generated code, before any timed run.** The build emits an assembly listing for the `gemm_naive` translation unit compiled with **exactly the frozen flags** (`build/asm/gemm_naive.asm`, a `/FAs` compile beside the ordinary object — the same precedent as the two Stage 0 tests that read emitted PTX rather than trusting a comment). `tests/test_gemm_naive.c` reads it and asserts:

- **0 occurrences of `ymm`** anywhere in the listing;
- **0 occurrences of packed floating-point arithmetic** — `vfmadd132/213/231ps`, `vmulps`, `vaddps`, `vsubps`, `mulps`, `addps`, `vdpps`, `vhaddps`;
- **20 occurrences of scalar arithmetic** (10 × `vmulss`, 10 × `vaddss`), so the loop was compiled rather than eliminated.

**No flag was changed and no de-vectorizing pragma was needed** — MSVC did not vectorize the inner loop under `/O2 /arch:AVX2`, which is consistent with `/fp:precise` forbidding the reassociation a vectorized reduction requires. Note also that **no FMA was contracted**: the multiply and the add are separate instructions, again consistent with `/fp:precise`.

#### Inputs — PLACEHOLDERS, not the benchmark prompt set

`PERSISTENT.md` §1 **D3 is open and owned by Stage 3**; this stage did not touch it and does not propose these strings as the prompt set. It needed inputs to time anything, so it committed `tests/fixtures/stage2_placeholder_prompts.tsv` — ordinary English prose, labelled PLACEHOLDER in the fixture header, in this entry, and as structured fields in both results files (`prompt_set_status` / `prompt_set_open_decision` / `prompt_set_work_item` / `prompt_set_source` in the bench file, and a nested `"prompt_set"` object in the correctness file). The Stage 1 round-trip corpus was **not** a candidate: those strings exist to break a tokenizer and are labelled test fixtures in their own header.

The fixture's first row encodes to **180 tokens** with the Stage 1 tokenizer, and every configuration is a **prefix of that same token sequence**, truncated to the exact count it needs — so the token counts are exact rather than approximate. The consequence for the cross-stage waterfall is registered as **W11**, owned by Stage 3.

#### Warmup, and the ambiguity in §4.2

**25 warmup iterations, every configuration.** `BENCHMARK_PROTOCOL.md` §4.2 supports two readings — a workload-relative rule (raise warmup if stabilization takes longer than 5 iterations *of the workload being timed*) and a fixed adjusted value of 25 "used everywhere" since Stage 0. **The document is ambiguous on this point and 25 was chosen as the conservative reading**: extra warmup can only cost time, while shortening it can invalidate a measurement, and 25 preserves comparability of method with Stages 0 and 0b.

§4.2-style arithmetic for **this** stage's workload, which is what a later stage needs to disambiguate the document: the single-iteration probe at the shortest configuration measured **8.920 s**. 25 × 8.920 s = **223 s**; even the protocol default of 5 × 8.920 s = **44.6 s**. Both exceed the 0.075 s stabilization figure by three orders of magnitude, so **the rule's condition is not met at all** — by the workload-relative reading, 5 would have sufficed here. 25 was used anyway, at a cost of roughly 3.7 minutes of warmup per configuration.

#### Correctness — the divergence is MEASURED and REPORTED, and NO tolerance was adopted

**No tolerance was adopted and none was written into `BENCHMARK_PROTOCOL.md` §5. D2 remains open and owned by Stage 3.** A threshold invented by this stage and then passed by its own author would be circular and would prove nothing. What gates acceptance here are the two requirements that carry no tolerance — **greedy-decoded token sequence equality against the reference**, and **readable English output** — plus the structural property checks below.

Statistics computed in float32 on both sides over the **full logit vector at every position**, at two sequence lengths. Full structured record: `bench/results/stage2/stage2_correctness.json`.

| Statistic | Prefill L = 8 | Prefill L = 16 |
|---|---|---|
| max absolute difference | **3.1281e-04** | **3.1281e-04** |
| — at position, token id | position 0, token 23601 | position 0, token 23601 |
| max relative difference | **8.6958e-06** | **8.6958e-06** |
| — denominator | \|reference\| + 1e-6, elementwise | same |
| — elements below that floor | **0** | **0** |
| mean absolute difference | 6.4757e-05 | 5.5687e-05 |
| median absolute difference | 4.5776e-05 | 4.5776e-05 |
| RMS absolute difference | 9.1561e-05 | 7.6493e-05 |
| absolute p50 / p90 / p99 / p99.9 / max | 4.578e-05 / 1.945e-04 / 2.518e-04 / 2.785e-04 / 3.128e-04 | 4.578e-05 / 1.068e-04 / 2.441e-04 / 2.708e-04 / 3.128e-04 |
| relative p50 / p90 / p99 / p99.9 / max | 4.842e-07 / 4.934e-06 / 6.285e-06 / 6.967e-06 / 8.696e-06 | 4.307e-07 / 1.644e-06 / 6.026e-06 / 6.779e-06 / 8.696e-06 |
| **top-1 agreement** | **8 of 8 positions** | **16 of 16 positions** |
| **top-5 set agreement** | **8 of 8 positions** | **16 of 16 positions** |
| reference top-1/top-2 margin, min / median / max | **0.03231** / 0.26849 / 5.45984 | **0.01438** / 0.45702 / 5.45984 |
| **smallest margin ÷ largest divergence** | **103.3×** | **46.0×** |

**The margin is the figure Stage 3 should set D2 from**, and it is the most useful thing this stage hands forward: it says how much divergence greedy decoding can absorb before a token flips. At L = 16 the tightest decision in the sequence had 0.0144 of headroom against a worst-case divergence of 0.00031 — a factor of **46**. It is also the figure that degrades with length: the minimum margin more than halved from L = 8 to L = 16 while the divergence did not move, so **a tolerance chosen at short length need not hold at long length**, which is why two lengths were measured.

Tolerance-free checks, all passing:

- **greedy token sequence matches the reference exactly** — engine `[464, 3139, 1748, 286, 4881, 318, 262, 3139, 286]`, reference identical, on the placeholder prompt with 3 generated tokens;
- **attention weights sum to 1 along the attended axis** for every head, every position and every layer of a full forward pass — every row of 12 layers × 12 heads × the tested prefill length, the count asserted against that product, with the observed minimum and maximum row sum both within 1e-5 of 1;
- **layernorm output has zero mean and unit variance per row** before the affine transform (mean within 1e-5 of 0, variance within 1e-4 of 1);
- **causality** — changing the token at position *j* moves no logit at any position *i < j*, bit for bit; and the logits at *j* do move;
- **prefill/decode self-consistency** — a decode step at context *t*+1 reproduces the prefill logits at position *t* **bit for bit**, at every context length tested;
- **the head is tied**, verified as identity of storage;
- **`tok_vocab_size()` equals the config's `vocab_size`** (50257 both sides);
- **the weight reader agrees with the committed inventory on all 160 tensors** — name, dtype, shape, absolute file offset, byte length.

Correctness tests ran at **reduced token counts, stated**: 4–7 tokens in `tests/test_model.c`, 8 and 16 in `tests/test_reference_impl.py`, 3 generated tokens for the greedy comparison. **The reduction applies to the correctness tests only; no timed configuration was shortened.**

**What agreement between the engine and the oracle proves, and what it does not.** Both were written in one session from one reading of the architecture, so agreement is evidence that **the two implementations agree** — it is **not** evidence that either matches GPT-2. The external discriminator available is readable English generated from published weights, and its resolution must be stated with it: it catches a transposed weight, an untied head, a broken causal mask and a wrong positional scheme, because any of those produces garbage; it does **not** catch a wrong layernorm epsilon, a plain `gelu` substituted for `gelu_new`, or a mildly wrong attention scale. What *is* externally validated independently of this stage is the **input** side: the Stage 1 tokenizer was verified against the real `tokenizers` library across 48 records and 410 ids, and the Stage 1 loader is byte-exact against real safetensors across 160 of 160 tensors.

#### Generated text — the external discriminator, untimed

Both samples are greedy, 30 generated tokens, run **outside every timed bracket** and not counted against the time budget. Reported in full, unedited, including the repetition.

**Prompt A — "The capital city of France is"** (6 prompt tokens, 30 generated):

> The capital city of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French Republic.
>
> The French Republic is the largest

**Prompt B — "In the years before the railway arrived, the valley was reached only by a single road, and the people who lived there"** (24 prompt tokens, 30 generated):

> In the years before the railway arrived, the valley was reached only by a single road, and the people who lived there were not very good. The people who lived there were not very good. The people who lived there were not very good. The people who lived there

Both are readable, grammatical English with correct syntax, correct agreement, and a coherent continuation of the prompt's topic — which is what this check is for. Both then fall into a repetition loop, which is **ordinary behaviour for a 124M-parameter model under greedy decoding with no sampling, no repetition penalty and no beam search**, and is reported rather than tuned away. Prompt A additionally produces a correct paragraph break and capitalisation after the full stop. No prompt was selected to flatter the output; both come from the committed placeholder fixture.


#### Measurement

One run, 2026-09-19, results file `bench/results/stage2/stage2_forward.json` (stage id `stage-2`, 7 records, raw per-sample timings retained). Warmup **25** and **30 samples** for every configuration — the protocol's preferred count, not its minimum, and nothing was reduced. `QueryPerformanceCounter`, timing brackets computation only. Total wall time of the timed set: 3883.3 s.

**Time budget, as a build-time determination.** One untimed iteration at the shortest configuration measured **8.920 s** at the checkpoint and **8.296 s** as the run's own probe — both real measurements of a single iteration, and the first is what the budget was decided from. The budget stated before the run was **25 minutes per configuration, 90 minutes total**. Every one of the seven configurations was projected to fit at 30 samples, **nothing was dropped and no warmup was shortened**; the actual total came in at 64.7 minutes. The scheduling projections themselves are not measurements and appear nowhere in this entry as latencies.

**PREFILL** — the head is computed at every position; no KV cache.

| Configuration | Samples | Median | Min | Max | Std dev (% of median) | Verdict |
|---|---|---|---|---|---|---|
| Prefill, L = 32 | 30 | 7765.285 ms | 7589.682 ms | 9743.632 ms | **7.031%** | **INVALID** |
| Prefill, L = 64 | 30 | **15520.584 ms** | 15384.836 ms | 15911.225 ms | 0.840% | VALID |

**DECODE** — one decode step, sampled independently at each context; the head is computed once; no KV cache. Prefill and decode are never combined into one figure.

| Configuration | Samples | Median | Min | Max | Std dev (% of median) | Verdict |
|---|---|---|---|---|---|---|
| Decode step, c = 32 | 30 | **6561.243 ms** | 6500.116 ms | 6822.580 ms | 1.299% | VALID |
| Decode step, c = 64 | 30 | **13193.109 ms** | 12997.858 ms | 13650.530 ms | 1.217% | VALID |
| Decode step, c = 128 | 30 | **26842.550 ms** | 26373.518 ms | 31007.176 ms | 3.579% | VALID |

**ISOLATED GEMM** — the naive matmul alone, reported as its own configuration with its own verdict and never folded into prefill or decode. One shape family (M = 32, K = 768) at two working-set sizes in different cache tiers; only N moves, so the two records differ in the size of the streamed operand and in nothing else.

| Configuration | B operand | Samples | Median | Std dev | Throughput | Verdict |
|---|---|---|---|---|---|---|
| M=32 N=768 K=768 (`attn.c_proj` shape) | 2,359,296 B = **2.25 MiB**, inside the 8 MiB L3 | 30 | 24.056 ms | 1.317% | **1.5692 GFLOP/s** | VALID |
| M=32 N=3072 K=768 (`mlp.c_fc` shape) | 9,437,184 B = **9.00 MiB**, exceeds the 8 MiB L3 | 30 | 124.149 ms | **8.212%** | 1.2162 GFLOP/s | **INVALID** |

**INVALID configurations, named: prefill L = 32 (7.031%) and the isolated GEMM at N = 3072 (8.212%).** Both exceed the 5%-of-median limit and are reported as invalid. **Neither was retried, averaged away, or rescued by a robust statistic**, and no result below depends on either.

**Derived throughput** `[derived]`, arithmetic shown. Operation counts are of the work the code actually performs, which includes the full T × T score matrix rather than only its causal triangle — the naive implementation computes the whole square and then discards the upper part, and counting only the triangle would flatter it. Per token: layer weight matmuls 2 × 84,934,656 = **169,869,312**; tied head 2 × 38,597,376 = **77,194,752**; total **247,064,064**. Attention as implemented: 4·T²·head_dim per head per layer × 12 × 12 = **36,864·T²**.

| Configuration | FLOPs | Median | Achieved | % of the measured scalar ceiling (8.565 GFLOP/s) |
|---|---|---|---|---|
| Prefill, L = 64 | 64 × 247,064,064 + 36,864 × 64² = 15,963,095,040 | 15520.584 ms | **1.0285 GFLOP/s** | **12.01%** |
| Decode, c = 32 | 32 × 169,869,312 + 36,864 × 32² + 77,194,752 = 5,550,761,472 | 6561.243 ms | 0.8460 GFLOP/s | 9.88% |
| Decode, c = 64 | 64 × 169,869,312 + 36,864 × 64² + 77,194,752 = 11,099,825,664 | 13193.109 ms | 0.8413 GFLOP/s | 9.82% |
| Decode, c = 128 | 128 × 169,869,312 + 36,864 × 128² + 77,194,752 = 22,424,446,464 | 26842.550 ms | 0.8354 GFLOP/s | 9.75% |
| Isolated GEMM, N = 768 | 2 × 32 × 768 × 768 = 37,748,736 | 24.056 ms | **1.5692 GFLOP/s** | **18.32%** |
| Isolated GEMM, N = 3072 | 2 × 32 × 768 × 3072 = 150,994,944 | 124.149 ms | 1.2162 GFLOP/s | 14.20% — **INVALID, not a claim** |

Prefill at L = 32 is excluded from this table: its run is INVALID and no throughput is derived from it.

**Structural results.**

- **Decode cost per token is very nearly linear in context**, as it must be without a KV cache: 6.561 s at c = 32, 13.193 s at c = 64 (**2.011×**), 26.843 s at c = 128 (**2.035×**). Flattening that curve is what Stage 4 exists to do, and this is the curve it will be measured against.
- **Decode-to-prefill ratio at equal length 64: 0.850** (13193.109 / 15520.584). A decode step costs the same order as a prefill over the same context rather than one-64th of it, because without a cache it re-runs the entire forward pass; it sits below 1.0 because prefill computes the head at all 64 positions and a decode step computes it once.
- **No speedup is claimed by this stage**, so the 4.4% noise floor governs nothing here and was neither re-derived nor restated (`PERSISTENT.md` §8 W5, Status unchanged).
- **No headline number is quoted.** `BENCHMARK_PROTOCOL.md` §7 defines the decode headline against `HARDWARE.md` §1 (the GPU) and the prefill headline against cuBLAS (also GPU); neither `PROJECT.md` §6 headline is answerable by a CPU stage. No percent-of-DRAM-bandwidth figure is quoted for CPU decode (W1), no percent-of-cuBLAS figure for CPU prefill (W9), and the derived single-channel figure of 23.464 GB/s is used as a denominator nowhere (W7). The ceiling used throughout is the measured **scalar** 8.565 GFLOP/s, not the measured vectorised 48.411 (W8).

**Within-run drift, DIAGNOSTIC.** The 5%-of-median rule tests dispersion, not direction: a run can pass it while drifting. Median of the second half of the samples against the median of the first, signed, per configuration. These figures characterise the distribution; they never replace a reported median and they never convert an INVALID run into a valid one.

| Configuration | First half | Second half | Drift | IQR as % of median | Robust outliers (1.5 × IQR) | Share of squared deviation carried by them |
|---|---|---|---|---|---|---|
| Prefill L = 32 | 8313.175 ms | 7683.354 ms | **−7.576%** | 8.214% | 2 (9597, 9744 ms) | 67.0% |
| Prefill L = 64 | 15520.379 ms | 15520.789 ms | **+0.003%** | 0.841% | 2 (15871, 15911 ms) | 54.0% |
| Decode c = 32 | 6547.087 ms | 6618.839 ms | +1.096% | 1.541% | 1 (6823 ms) | 27.4% |
| Decode c = 64 | 13119.426 ms | 13225.087 ms | +0.805% | 1.953% | 0 | 0.0% |
| Decode c = 128 | 26765.542 ms | 26878.973 ms | +0.424% | 1.648% | 2 (29879, 31007 ms) | 92.3% |
| GEMM N = 768 | 23.962 ms | 24.346 ms | +1.601% | 2.348% | 0 | 0.0% |
| GEMM N = 3072 | 123.791 ms | 124.632 ms | +0.679% | 5.410% | 5 (141, 141, 143, 150, 167 ms) | 93.3% |

**What the drift figures show, and what they do not.** Over a 64.7-minute sustained single-core run, **no configuration's second half is systematically slower than its first by more than 1.7%**, and the four longest configurations sit between +0.003% and +1.10%. The one large drift is **negative** — prefill L = 32's first half was 7.6% *slower* than its second, which is the opposite sign from a thermal effect and is carried by two outliers of 9.6 and 9.7 s against a 7.77 s median that between them account for 67% of the squared deviation. That configuration is INVALID on the dispersion rule independently. **An absence of positive drift is evidence and is reported as such**, but it is evidence from timings alone: this machine has no live CPU package temperature source (`HARDWARE.md` §5.3), so a thermal contribution is **not measured** here, only bounded by what the timings would have shown.

**CPU frequency, recorded as a run condition.** The committed `CpuTelemetrySampler` ran alongside the whole timed set with the Stage 0b configuration — interval 0.02 s, affinity mask `0xf0` (logical CPUs 4–7), clear of logical CPU 2 where the measured thread ran and of its SMT sibling CPU 3, sampling from a separate process so no probe could land inside a timed bracket. **122,428 samples over 3883.3 s**, per-probe cost **49.6 µs**, sampler duty cycle **0.469% of one core** (Stage 0b measured 0.355% at the same interval; the figure is recorded as observed, not as inherited). Trace: `bench/results/stage2/stage2_forward_cpu_frequency_trace.json`, 18,025,343 bytes, raw samples retained.

| Source | n | Min | Median | Max |
|---|---|---|---|---|
| `\Processor Information(_Total)\% Processor Performance` | 122,428 | 115.97 | **165.92** | 228.08 |
| `\Processor Information(_Total)\% Performance Limit` | 122,428 | 100.0 | 100.0 | 100.0 |
| `\Processor Information(_Total)\Processor Frequency` | 122,428 | 2496.0 | 2496.0 | 2496.0 |

The live source moved across a range of **116% to 228% of nominal** during the session while the static nominal read stayed at 2496 MHz throughout — which is exactly why Stage 0b classified the third source as static and kept the first. `% Performance Limit` was **100.0 on every one of the 122,428 samples**, so no platform-imposed frequency ceiling was asserted at any point in the run. **What this does not establish:** the counter is a `_Total` across all logical processors, not the measured thread's own frequency, and no package temperature was available, so it bounds the session rather than attributing anything to a configuration.

#### SECOND MEASUREMENT — re-run at the D3 fixed prompt set, 2026-10-05 (W11)

*The original figures above are the PLACEHOLDER measurement and are left exactly as they were, with their labels intact. This block is a second, later measurement of the same binary on different inputs. The two are never merged and never averaged.*

**Stage 2 binary**, source at commit `20e4f4e`, re-parameterised for the fixed prompt set by Stage 3 with **identical timing code** — `bench_run`, the timed bodies, the bracket contents, the warmup and sample handling, the statistics and the 5%-of-median validity rule are unchanged; only the fixture loader, the configuration list, the results stem and the prompt-set labelling were parameterised. **Date** 2026-10-05. **Results file** `bench/results/stage3/stage3_harness.json`, which is simultaneously Stage 3's own measurement and this re-run — one timed set, one artifact, cited from both entries. **Conditions** are the Stage 3 entry's conditions block: fingerprint 25 of 25 clean, clock locked and verified at 1365 MHz, 25 warmup and 30 samples, thread pinned to logical CPU 2.

**PREFILL at the D3 prompt set.**

| Configuration | Row | Samples | Median | Min | Max | Std dev (% of median) | Verdict |
|---|---|---|---|---|---|---|---|
| Prefill, L = 16 | `d3_16` | 30 | **3537.051 ms** | 3514.392 ms | 3565.556 ms | 0.328% | VALID |
| Prefill, L = 32 | `d3_32` | 30 | **7058.383 ms** | 7032.262 ms | 7191.453 ms | 0.424% | VALID |
| Prefill, L = 64 | `d3_64` | 30 | **14121.470 ms** | 14095.015 ms | 14420.316 ms | 0.444% | VALID |
| Prefill, L = 128 | `d3_128` | 30 | 28420.739 ms | 28302.326 ms | 33181.064 ms | **5.723%** | **INVALID** |

**DECODE at the D3 prompt set.** Contexts match the placeholder run's exactly, so the Stage 4 comparison is like for like.

| Configuration | Row | Samples | Median | Min | Max | Std dev (% of median) | Verdict |
|---|---|---|---|---|---|---|---|
| Decode step, c = 32 | `d3_32` | 30 | **5925.150 ms** | 5906.286 ms | 6460.867 ms | 1.677% | VALID |
| Decode step, c = 64 | `d3_64` | 30 | 11858.971 ms | 11803.567 ms | 18595.912 ms | **15.936%** | **INVALID** |
| Decode step, c = 128 | `d3_128` | 30 | **23735.411 ms** | 23675.880 ms | 24120.160 ms | 0.462% | VALID |

**ISOLATED GEMM**, unchanged configuration, independent of the prompt set.

| Configuration | Samples | Median | Std dev (% of median) | Throughput | Verdict |
|---|---|---|---|---|---|
| M=32 N=768 K=768 | 30 | 21.323 ms | 3.490% | 1.770 GFLOP/s | VALID |
| M=32 N=3072 K=768 | 30 | 120.753 ms | 1.305% | 1.250 GFLOP/s | VALID |

**These re-run figures are the waterfall baseline from Stage 4 onward.** The placeholder figures above remain the historical record of what was measured on inputs no other stage uses, and are not compared against anything. **There is no baseline at prefill L = 128 or at decode c = 64**, because both are INVALID and an INVALID run is not a baseline. The regression comparison between the two runs was **REFUSED on every forward-pass configuration**, correctly, because the prompt sets differ — which is the whole reason W11 existed. Full detail in the Stage 3 entry.
#### The W3 observation, recorded without changing the item

`PERSISTENT.md` §8 **W3** predicts that decode-shaped timings will be INVALID roughly half the time under a 30-sample construction, from Stage 0's M = 1 GEMM configurations whose medians were **30–90 µs** with spike-carried variance. This stage produced the project's first decode timing, and it falls **far outside that band**: the decode-step medians are **6561, 13193 and 26843 ms** — five orders of magnitude above 30–90 µs — with dispersion of **1.299%, 1.217% and 3.579%** of median, all VALID.

Dispersion character, by the robust-outlier count against the interquartile range: c = 32 is **mixed** (IQR 1.541% of median, one outlier carrying 27.4% of squared deviation), c = 64 is **broad** (IQR 1.953%, zero outliers), c = 128 is **spike-carried** (IQR 1.648%, two outliers carrying 92.3%). W3's mechanism — a tight baseline punctured by single interruption events — is visible at c = 128 and absent at c = 64.

**Nothing about W3 is changed.** Its band was derived for microsecond-scale M = 1 GEMM configurations and this stage's decode workload is a whole forward pass at second scale; the two are not the same measurement, and a decode step timed after Stage 4 adds the KV cache will be a third thing again. Recorded so that Stage 3 and Stage 4 have this stage's numbers when they revisit it.

#### Gap — predicted against measured

The prediction was committed alone in `b33cb35` before any implementation file existed and was **not consulted again until this section**. It influenced no configuration, no sample count, no warmup value, no implementation choice and no decision to investigate anything.

| Configuration | Predicted | Measured | Measured / predicted |
|---|---|---|---|
| Prefill, L = 32 | 6,169 ms | 7765.285 ms **(INVALID)** | 1.259× |
| Prefill, L = 64 | 12,367 ms | **15520.584 ms** | **1.255×** |
| Decode step, c = 32 | 4,306 ms | **6561.243 ms** | **1.524×** |
| Decode step, c = 64 | 8,582 ms | **13193.109 ms** | **1.537×** |
| Decode step, c = 128 | 17,221 ms | **26842.550 ms** | **1.559×** |
| Decode-to-prefill ratio at length 64 | 0.69 | **0.850** | 1.232× |

**Every configuration landed within 2× of the prediction, all on the slow side, and the prediction's own criterion for that outcome is that the efficiency fraction was imprecise rather than the reasoning wrong.** The three falsification conditions it stated are all unmet:

- *More than 3× faster with vectorized machine code* — not met, in either half. The measurements are slower, not faster, and the generated code was proven scalar at build time: zero `ymm` registers and zero packed instructions in the emitted listing, checked by a unit test rather than diagnosed afterwards.
- *More than 3× slower* — not met; the worst case is 1.56×. The empty measured CPU DRAM bandwidth field is therefore **not** shown to be material at Stage 2.
- *Decode-to-prefill ratio near 1/64, or decode cost flat in c* — not met. The ratio is 0.850 and decode cost doubles with each doubling of context (2.011× and 2.035×). **No KV cache was built**, and the measurement confirms the structural claim the prediction rested on.

**The mechanism, quantified.** The prediction's single free parameter was the efficiency fraction f, taken as 0.15 of the measured scalar ceiling with a stated plausible range of 0.08 to 0.30. Measured:

| Workload | Achieved | f, as a fraction of the measured scalar ceiling |
|---|---|---|
| Prefill, L = 64 | 1.0285 GFLOP/s | **0.120** |
| Decode, c = 32 / 64 / 128 | 0.8460 / 0.8413 / 0.8354 GFLOP/s | **0.099 / 0.098 / 0.098** |
| Isolated GEMM, B inside L3 | 1.5692 GFLOP/s | **0.183** |

Both workloads land inside the stated range, and the whole of the gap is accounted for by f being 0.120 rather than 0.150 at prefill and 0.098 rather than 0.150 at decode: 0.150 / 0.120 = 1.25, which is the measured prefill ratio of 1.255 to three digits, and 0.150 / 0.098 = 1.53, which is the measured decode ratio. **The prediction's structure was right and its single estimated constant was 25% optimistic at prefill and 53% optimistic at decode.**

**The one structural thing the prediction did not anticipate is that f is not one number.** It assumed a single fraction for both workloads; the measurement shows prefill running at 0.120 and decode at 0.098, a 22% difference that is stable across all three contexts. The difference has a candidate mechanism that this stage can state but cannot prove: prefill computes the tied head at every position, and the head is the one matmul whose operands are both walked contiguously in the inner loop (`gemm_naive_bt`, because `wte` is stored `[output, input]`), while every layer weight matmul walks its B operand down a column, touching a fresh 64-byte line for every 4 bytes used. The head is 31.2% of prefill's per-token FLOPs and is amortised to a single application in a decode step, so prefill has a larger share of its work in the faster-access kernel. **The evidence available does not establish this**: it is consistent with the isolated GEMM figure below, and no measurement in this stage isolates the head.

**What the isolated GEMM establishes, and what it does not.** With the entire B operand resident in L3 (2.25 MiB against an 8 MiB shared L3 and a 4 MiB effective single-thread edge), the naive `ijk` loop reaches **1.5692 GFLOP/s, 18.3% of a ceiling measured on a register-resident loop with eight independent FMA chains and no memory traffic at all**. That is the cleanest evidence this stage has, and it says that **most of the shortfall is not the memory hierarchy**: even with no capacity misses to take, the single-accumulator dot product with two loads per multiply-add gives up more than five sixths of the ceiling. The three reductions the prediction named — the exposed FMA dependency chain, two loads per FMA where the ceiling had none, and 64-byte lines touched for 4 bytes used — are consistent with that, and the emitted assembly shows the further detail that **no FMA was contracted at all**: under `/fp:precise` the inner loop is a separate `vmulss` and `vaddss`, so the "FMA" in that reasoning is two dependent instructions rather than one.

**The configuration that would have separated the scalar issue-rate limit from the memory-hierarchy limit is INVALID, and the separation is therefore NOT established.** The second GEMM record — the same shape family with a 9.00 MiB B operand that exceeds the 8 MiB L3 — measured 1.2162 GFLOP/s, 22.5% below the L3-resident case, but its dispersion is 8.212% of median with five robust outliers carrying 93.3% of the squared deviation. **It is INVALID and no cache-tier conclusion may be drawn from it.** The honest statement is that the L3-resident case bounds the issue-rate contribution at roughly 18% of ceiling, and the incremental cost of leaving L3 is unmeasured at this stage.

**Hardware counter evidence: NONE was collected, and the cause is unestablished to that extent.** `PERSISTENT.md` §8 W4 records that a CPU PMU source exists on this machine (`xperf`, with `CacheMisses`, `LLCReference`, `LLCMisses`, `InstructionRetired` and more, at most 7 selectable simultaneously) and that it is deliberately unused. It was **not introduced here**: proving an instrument live, measuring its own cost and keeping it outside every timed bracket is a substantial addition to a baseline stage, and W4's own stated first use is confirming W1's dispersion, which is Stage 5's question. `BENCHMARK_PROTOCOL.md` §6's requirement that a CPU stage state which counter equivalent it used is satisfied by stating that **xperf is available and was deliberately not used** — the precedent Stage 0b set. Consequently: the gap explanation above is built from operation counts, the measured scalar ceiling and the isolated GEMM configuration, and **the counters that would distinguish an issue-rate limit from an L2/L3 miss-rate limit within the forward pass were not collected**. Where the evidence does not distinguish between candidate causes, that is written here rather than resolved by choosing one.

#### Files added outside this stage's declared outputs

**`.gitattributes`** — one file, four lines of comment and one rule, `*.tsv text eol=lf`.

The defect it fixes: `tests/test_stage1_oracle.py` regenerates `tests/fixtures/tokenizer_expected_ids.tsv` and asserts byte identity against the committed file. The generator writes LF (`reference/stage1_oracle.py`, `newline="\n"`) and the committed blobs are LF, but this machine has `core.autocrlf=true`, so **a checkout rewrites the working copy to CRLF and the test then fails on a file nobody edited**. It failed exactly that way during this stage's offline gate, on Stage 1 code this stage did not touch.

Evidence that the fix changed no committed content: the blob hashes are identical at `HEAD`, at the Stage 1 commit `55dbe2a`, and as `git hash-object` of the current working copies — `tokenizer_expected_ids.tsv` `7bf1d7f844883544bfc5f062151256affa060241`, `tokenizer_roundtrip_corpus.tsv` `fce51029bfac6ad4083473828598606bb3285fcb`. **Only the checkout representation moved; the Stage 1 artifacts are byte-identical.** The rule also governs this stage's new `tests/fixtures/stage2_placeholder_prompts.tsv`.

#### Where the artifacts contradicted the documents

1. **`HARDWARE.md` §2 carried the wrong dispersion for the Stage 0b scalar figure.** The cell read 2.773% for the scalar reference of the SIMD loop; `bench/results/stage0b/run2/cpu_simd_peak.json` records `stddev_pct_of_median` **1.8466** for the scalar record and **2.7734** for the vectorised one, so the vectorised figure had been transcribed into both cells. **Corrected in place, one cell**, with both values and the artifact stated. No measured value changed and no other cell was touched.
2. **`BENCHMARK_PROTOCOL.md` §2 named a timing function that does not exist on this platform.** It specified `clock_gettime(CLOCK_MONOTONIC)`, which is unavailable under MSVC, while every results file since Stage 0 and `bench_common.h` itself record `QueryPerformanceCounter`. The artifact wins: that one bullet now names `QueryPerformanceCounter` as this platform's monotonic source with `clock_gettime(CLOCK_MONOTONIC)` as the POSIX equivalent, and "Never wall clock" is preserved. Nothing else in §2 and nothing elsewhere in the file was altered — in particular **no tolerance was written into §5 and §4.2 was not touched**.
3. **`TECHNICAL_SPEC.md` §4 had no home for four things that now exist**: `src/gemm/gemm.h`, `bench/stage2_forward_bench.c`, `tests/fixtures/`, and `.gitattributes`. Added to the existing tree in the same bounded way Stage 1 amended it — nothing regenerated, nothing reordered, §1 untouched.
4. **The environment fingerprint disagreed with itself across stages**, which is recorded under Conditions above and registered as W12. Not resolved here.
5. **No architecture value in any document was contradicted.** Every value read from `config.json` and the inventory matches what the Stage 1 entry recorded.

#### What this stage established, in one place

A GPT-2 small forward pass in C, reading published weights through the Stage 1 loader and text through the Stage 1 tokenizer, producing readable English and matching a hand-written PyTorch oracle to a maximum absolute logit divergence of **3.128e-04** with **top-1 agreement at every position** and an exact greedy sequence match — at **15.52 s per 64-token prefill** and **26.84 s per decode step at a context of 128**, which is **12.0%** and **9.8%** of this machine's measured scalar FP32 ceiling. Both figures are baselines to be beaten, not results to be defended, and the stages that beat them are named in the code that produced them.


## Stage 3 — Benchmark harness and correctness gate
*Build the measurement infrastructure every later stage depends on, and close the two decisions that have been open since Stage 0.*
**Status:** COMPLETE, 2026-10-05. Offline gate green (clean build, **22/22 tests**). **Stage 3 is EXEMPT from the prediction gate** (Stage 0, 1, 3, 12 and the optional stages are exempt), so this entry carries no Prediction and no Gap section. Nine timed configurations in one set, 25 warmup and 30 samples each: **seven VALID, two INVALID and named**. **D2 RESOLVED at `6e-03`**, **D3 RESOLVED** at four rows of 16/32/64/128 tokens, **W12 CLOSED**, **W9 CLOSED**, **W11 CLOSED**, **W3 updated and still open for Stage 4**, **W13 raised**. Correctness gate: **PASS**. No Nsight counters and no headline ratio, both with reasons stated below.

#### Conditions

| Condition | Value |
|---|---|
| Date, branch | 2026-10-05, branch `stage-3` cut from `main` at `40622c5` |
| Prediction commit | **None — Stage 3 is exempt from the prediction gate.** No prediction was written, transcribed or committed |
| Environment fingerprint | `machine_state.py verify` after the clean build and again immediately before the first timed run: **25 of 25 compared fields clean, 0 differing, exit 0**. The comparison set is 25 rather than 26 because of this stage's W12 fix; `build_flags.BENCH_BUILD_TIMESTAMP` is reported in a provenance block as stored `2026-09-18T06:56:02Z` against current `2026-10-05T04:38:53Z`, labelled informational and **not compared**. **No substantive field differs**, so the comparison against prior stages stands. `bench/results/machine_fingerprint.json` was NOT rewritten and the `fingerprint` subcommand was never run |
| Clock lock | `nvidia-smi -lgc 1365,1365` applied from an elevated shell, then verified **by state**: `verify-lock --mhz 1365` → `locked: true`, **10 of 10 samples at 1365 MHz**, `off_target_samples: []`, exit 0. Re-verified after an interrupted first attempt and found still in effect — the lock is device state and survived. `-lmc` **not attempted** (§2 Q11) |
| Compiler and flags | **Confirmed unchanged against `HARDWARE.md` §5.5**, by the fingerprint comparison above and by the provenance the C layer compiled into the results file: MSVC 19.44.35229.0, toolset 14.44.35207. Host flags `/DWIN32 /D_WINDOWS /EHsc /W3 /arch:AVX2 /fp:precise /MD /O2 /Ob2 /DNDEBUG`; nvcc 13.1.80, `-arch=sm_75`. **No `-allow-unsupported-compiler`** |
| AC power | **On AC**, `ac_line_status 1`, battery 99%. A run on battery is INVALID outright and none was taken |
| Network | Wi-Fi (Intel AX201), `NetworkCostType = Unrestricted` — **not metered** |
| Session elevation | **Elevated**, `IsInRole(Administrator)` true, user `MSI\saket`, **checked live in this session** rather than carried forward. An earlier attempt in a non-elevated session returned exit 4 from `nvidia-smi -lgc` and changed nothing; the session was restarted elevated before any timed run |
| Thread placement | Pinned to **logical CPU 2 of 8**, priority raised (ABOVE_NORMAL class, THREAD_PRIORITY_HIGHEST), applied outside every timed bracket by `bench_pin_current_thread` and **recorded as applied** rather than assumed |
| CPU timer | `QueryPerformanceCounter`, monotonic. Never wall clock. The timed bracket contains `model_prefill` or `model_decode_step` only |
| Reference oracle environment | The repository `.venv` — Python 3.14.2, torch 2.14.0+cu130, numpy 2.5.3, confirmed **by import**. CPU only, `torch.set_num_threads(1)`, 160 of 160 tensors verified against the inventory |
| Tokenizer verification environment | `.venv-oracle` — Python 3.12.10, tokenizers 0.23.2, safetensors 0.8.0, confirmed **by import**. Nothing was installed into either frozen environment |
| Ordering of correctness against timing | The correctness gate ran **BEFORE** any timed run and the oracle process had **exited** before the first timed bracket. It holds the weight set in numpy and again in torch, over a gigabyte resident, so an overlap would have changed the thing being measured |
| CPU frequency telemetry | The committed `CpuTelemetrySampler` **REUSED** at the proven Stage 0b / Stage 2 configuration and not rebuilt: interval 0.02 s, requested exclusions logical CPUs 0 and 2, resolved mask **`0xf0`** (allowed 4–7, excluding CPU 2 and its SMT sibling), affinity application confirmed `ok: true` from a previous mask of `0xff`, running in a **separate process**, `sampled_inside_any_timed_bracket: false`. **179,010 samples over 5482.48 s**, 62.0 µs per probe per counter across three live counters, so `3 × 62.0 µs × 179,010 / 5482.48 s` = **0.607% duty of one core**, matching the sampler's own `sampler_duty_cycle_of_one_core` of 0.006073. The trace is a recorded run condition, not a measurement, and is not committed: this stage's declared outputs are three results files and the trace is not one of them |
| Run wall time | 5482.5 s = **91.4 min** for the probe pass and the timed set together |

**CPU frequency, from the live source only.** The static nominal is not usable and Stage 0b established why.

| Source | n | Min | Median | Max |
|---|---|---|---|---|
| `\Processor Information(_Total)\% Processor Performance` | 179,009 | 119.01 | **179.12** | 222.34 |
| `\Processor Information(_Total)\% Performance Limit` | 179,010 | 100.0 | 100.0 | 100.0 |
| `\Processor Information(_Total)\Processor Frequency` | 179,010 | 2496.0 | 2496.0 | 2496.0 |

The live source moved across **119% to 222% of nominal** while the static nominal read 2496 MHz throughout, reproducing Stage 0b's finding. `% Performance Limit` held at 100 for every one of the 179,010 samples, so **no CPU performance limit was asserted at any point during the timed set**.

#### Process set, enumerated immediately before the first timed run, named rather than summarised

*Holding a GPU context* (`nvidia-smi --query-compute-apps`, 14 distinct names): `dwm.exe`, `explorer.exe`, `ShellHost.exe`, `ShellExperienceHost.exe`, `StartMenuExperienceHost.exe`, `SearchHost.exe`, `TextInputHost.exe`, `CrossDeviceResume.exe`, `ApplicationFrameHost.exe`, `SystemSettings.exe`, `OmApSvcBroker.exe` (MSI NBFoundation Service), `logioptionsplus_agent.exe`, `msedgewebview2.exe`, and **this session's own editor (`claude.exe`)**.

*Significant CPU consumers* (accumulated CPU seconds, sampled during the run): `MsMpEng` (Windows Defender real-time scanning) 2318 s, `System` 648 s, **`stage2_forward_bench` 394 s** (the measured process itself), `dwm` 125 s, `svchost` (pid 4144) 96 s, `WmiPrvSE` 95 s, `svchost` (pid 8744) 85 s, `explorer` 75 s, `claude` 59 s, three further `svchost` instances 43–51 s, `logioptionsplus_agent` 51 s, `logioptionsplus_updater` 50 s. **222 processes resident** at the start of the timed set, against 236 at this session's checkpoint.

**How the set differs from Stage 2's, and from Stage 0's — which is the set the 4.4% noise floor was measured against.**

- **Riot Vanguard (`vgc`, `vgtray`) — ABSENT.** Resident throughout Stage 0; absent in Stage 0b, Stage 2 and here.
- **Nahimic — ABSENT.** Resident in Stage 0; closed in Stage 0b, absent in Stage 2. **Present at this session's checkpoint and closed by the operator before the first timed run**; verified absent by name afterwards.
- **Intel DSA (`DSAService`, `DSATray`, `DSAUpdateService`) — ABSENT.** Same handling as Stage 2: present at the checkpoint, **closed by the operator**, verified absent by name. `esrv`, the Intel Energy Server service, **remained running** and is recorded as present rather than assumed gone.
- **Logitech Options+ — ABSENT.** Present at the checkpoint under three processes (`logioptionsplus_agent`, `_appbroker`, `_updater`) and **closed by the operator**; verified absent by name before the run. It respawned during the run and appears in the CPU-consumer list above, so it is recorded as **present during part of the timed set** rather than as closed — the same honesty Stage 0b and Stage 2 applied to it.
- **`msedgewebview2.exe` ×6 — PRESENT**, uncloseable, and equally resident during Stage 0, so a shared condition rather than a difference.
- **MSI service stack — RUNNING**, by deliberate operator decision (§1 D8, §2 Q9).
- **Windows Defender — PRESENT**, not excluded or modified, and the largest single CPU consumer on the machine during the run.
- **This session's editor — PRESENT but reduced**, `claude.exe` ×1 against ×9 at the checkpoint and ×4 during Stage 2.

Net: the set is **lighter than Stage 0's**, lighter than Stage 2's on the editor count and on Logitech, and **heavier than none of them**. A floor measured under heavier load stays conservative under lighter load, so the 4.4% figure remains safe to use. **The 4.4% noise floor is NOT re-derived, adjusted or restated here** — this stage did not re-run the full microbenchmark suite and therefore cannot re-derive it (`PERSISTENT.md` §8 **W5**, whose Status this stage does not change).

#### Process set as an uncontrolled variable — a limitation this stage found in its own regression check

The regression comparison gates on the prompt set, on both sides being VALID, and on the substantive fingerprint fields. **The process set is none of those**, and the environment fingerprint has no process-set field. The one comparison that survived the gates this session — the isolated GEMM at N = 768 — therefore compared across a materially changed process set and reported an 11.363% improvement for which **no code change exists**. The check behaved exactly as specified; the specification does not cover this condition. Recorded here rather than papered over, and carried to Stage 4 in §7.

#### D2 — the numerical tolerance, RESOLVED

**`D2_MAX_ABS_LOGIT_DIFF = 6e-03`**, an elementwise ABSOLUTE difference in logit units. The full statement of form, application rule and arithmetic is in `BENCHMARK_PROTOCOL.md` §5, which is the authoritative copy; what follows is the measurement it rests on.

The gate figure is the maximum elementwise absolute difference over the **full logit vector at every position** of a prefill, float32 on both sides. `gpt2_tool --dump-logits` writes logits for **every** position (8-byte magic `TIE2LOGI`, int32 positions, int32 vocab_size, then `positions × vocab_size` float32 in row order), so both bounds below are measured over the whole prefill and not over a subset.

**Measured at the four D3 lengths**, results file `bench/results/stage3/stage3_correctness.json`, engine `gpt2_tool` with `--cproj as-stored`, oracle `reference/reference_impl.py`:

| D3 length | Max abs divergence | RMS abs | Max rel | Elements below the `|ref|+1e-6` floor | Min reference top-1/top-2 margin | Position | Median margin | Margin ÷ divergence | Top-1 agreement |
|---|---|---|---|---|---|---|---|---|---|
| L = 16 | 3.967285e-04 | 7.73904e-05 | 8.69579e-06 | 0 | 5.915833e-02 | 13 | 0.57100 | 149.12x | 16 of 16 |
| L = 32 | 4.272461e-04 | 6.86042e-05 | 8.69579e-06 | 0 | **7.812500e-03** | 11 | 0.58576 | 18.29x | 32 of 32 |
| L = 64 | 4.425049e-04 | 6.44972e-05 | 4.39644e-06 | 0 | 4.189301e-02 | 1 | 0.82385 | 94.67x | 64 of 64 |
| **L = 128** | **7.019043e-04** | 6.93404e-05 | 9.35623e-06 | 0 | 4.366302e-02 | 118 | 0.76766 | 62.21x | 128 of 128 |

**The arithmetic, both conditions evaluated at the longest D3 length.**

- **(a) Lower bound**, from observed divergence: `3 × 7.019043e-04` = **2.105713e-03**.
- **(b) Upper bound**, from the minimum decision margin: `4.366302e-02 / 3` = **1.455434e-02**.
- **(d) Window check**: `4.366302e-02 / 7.019043e-04` = **62.21x**, above the required factor of 9 end to end. The window opens; no stop.
- **(c) Value**: geometric mean `sqrt(2.105713e-03 × 1.455434e-02)` = **5.535997e-03**, rounded to one significant figure as **6e-03**, which keeps both ratios at or above 3.
- **Achieved safety factors: 8.55x** above the observed divergence (`6e-03 / 7.019043e-04`) and **7.28x** below the minimum margin (`4.366302e-02 / 6e-03`).

**The circularity that remains, and why it is survivable.** The lower bound is a property of **this engine's current divergence** and moves whenever the arithmetic is reassociated — Stage 5's blocking rewrite will produce its own figure. The upper bound is a property of the **reference alone**, and no change to the engine can move it. That asymmetry is why the upper bound is the one that makes the check meaningful: a threshold justified only against the thing being measured is justified by its own subject.

**A qualification that travels with the figure.** The minimum margin over the **whole** D3 set is **7.812500e-03 at L = 32**, smaller than at L = 128. The margin therefore does **not** degrade monotonically with length across this set, because the four D3 rows are **independent prose and not nested prefixes**. Stage 2's finding that the minimum margin more than halved between L = 8 and L = 16 was a nested-prefix effect — adding positions to one string can only lower a minimum — and it does not generalise here. Against that set-wide minimum, 6e-03 leaves only **1.30x**. Had the upper bound been taken set-wide the window would be `[2.105713e-03, 2.604167e-03]`, a span of 1.24x, and **no one-significant-figure value lies inside it** (2e-03 fails the lower bound at 2.85x, 3e-03 fails the upper at 2.60x); the set-wide margin-to-divergence factor of 11.13x does still clear condition (d). What protects the gate in practice is that **condition 2 is checked directly rather than inferred**: top-1 agreement was complete at all four lengths.

**Corroboration, not input.** Stage 2 measured 3.1281e-04 at L = 8 and at L = 16 on placeholder inputs. The D3 figures of 3.97e-04 to 7.02e-04 are the same order of magnitude. Unlike Stage 2's two nested lengths, where the maximum divergence did not move at all, divergence here **does** grow slowly with length. The Stage 2 figures entered neither bound.

**Gate result: PASS**, all three conditions. Maximum absolute difference 7.019043e-04 at or below 6e-03; top-1 agreement at every position of all four lengths; greedy sequence matched the reference on `d3_16` over 3 generated tokens (engine and reference both `464, 3329, 4512, 373, 2739, 11, 290, 262, 3859, 5901, 351, 661, 10627, 511, 16860, 13, 198, 198, 1`).

#### D3 — the prompt set, RESOLVED

Committed as `tests/fixtures/benchmark_prompts.tsv`, header labelling it FIXED PROMPT SET, naming D3, the date and both verifying tokenizer versions. Four rows of ordinary English prose, each standing alone and **not** a truncated prefix of another, which is what makes the per-shape M values exact rather than approximate.

| Row | Target | Reference tokenizer (`tokenizers` 0.23.2, `.venv-oracle`) | Committed C tokenizer (`src/tokenizer.c` via `gpt2_tool`) | Counts agree | **Full id sequences agree** |
|---|---|---|---|---|---|
| `d3_16` | 16 | 16 | 16 | yes | **yes** |
| `d3_32` | 32 | 32 | 32 | yes | **yes** |
| `d3_64` | 64 | 64 | 64 | yes | **yes** |
| `d3_128` | 128 | 128 | 128 | yes | **yes** |

Subjects, so a reader can tell the rows apart without opening the fixture: `d3_16` a late commuter train and a platform of people checking watches; `d3_32` a library reading room filled with students; `d3_64` a valley road, a river and a farmer's thirty-year crop rotation; `d3_128` a workshop of inherited hand tools on winter evenings. None is tuned to produce good model output and none is adversarial. The Stage 1 round-trip corpus was **not** a candidate — those strings exist to break a tokenizer.

The verification is not a one-off: the timing driver now **re-checks every row's declared count against what the C tokenizer produces on every run** and hard-fails on a mismatch rather than padding or truncating an inexact row into place. The run log above shows all four rows checked at the start of the timed pass.

**Configurations fixed for every later stage:** prefill at L = 16, 32, 64, 128; decode at context c = 32, 64, 128, one step each, using the row of that token count as the context. The decode contexts match Stage 2's exactly so the Stage 4 comparison against the no-cache curve is like for like. **M values produced: 16, 32, 64, 128** for the prefill denominator, plus **M = 1** decode-shaped.

`tests/fixtures/stage2_placeholder_prompts.tsv` is **not modified, not deleted and not relabelled**. Its PLACEHOLDER header is the record that W11 was handled deliberately rather than discovered later.

#### Measurement — the timed set

One run, 2026-10-05, results file `bench/results/stage3/stage3_harness.json` (stage id `stage-3`, 9 records, raw per-sample timings retained). Warmup **25** and **30 samples** for every configuration. **This single timed set is simultaneously the Stage 3 measurement and the W11 Stage 2 re-run** — see the W11 section below for why that is one artifact and not two.

**Time budget, from a measured probe and labelled as a projection.** The harness's own probe pass measured one untimed iteration of each configuration and projected `sum(probe) × (25 + 30)` = **5290 s = 88.2 min**. Actual wall time including the probe pass and the model load was **5482.5 s = 91.4 min**. The projection is a scheduling input, appears nowhere as a latency, and is recorded here only so the 3.6% shortfall against actual is on the record.

**PREFILL** — the head is computed at every position; no KV cache. Prefill and decode are never combined into one figure.

| Configuration | Row | Samples | Median | Min | Max | Std dev | Std dev (% of median) | Construction | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Prefill, L = 16 | `d3_16` | 30 | **3537.051 ms** | 3514.392 ms | 3565.556 ms | 11.615 ms | 0.328% | single-iteration, R=1, probe 3.771 s | VALID |
| Prefill, L = 32 | `d3_32` | 30 | **7058.383 ms** | 7032.262 ms | 7191.453 ms | 29.905 ms | 0.424% | single-iteration, R=1, probe 7.219 s | VALID |
| Prefill, L = 64 | `d3_64` | 30 | **14121.470 ms** | 14095.015 ms | 14420.316 ms | 62.757 ms | 0.444% | single-iteration, R=1, probe 14.319 s | VALID |
| Prefill, L = 128 | `d3_128` | 30 | 28420.739 ms | 28302.326 ms | 33181.064 ms | 1626.448 ms | **5.723%** | single-iteration, R=1, probe 28.881 s | **INVALID** |

**DECODE** — one decode step, sampled independently at each context, using the D3 row of that token count; the head is computed once; no KV cache.

| Configuration | Row | Samples | Median | Min | Max | Std dev | Std dev (% of median) | Construction | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Decode step, c = 32 | `d3_32` | 30 | **5925.150 ms** | 5906.286 ms | 6460.867 ms | 99.335 ms | 1.677% | single-iteration, R=1, probe 5.980 s | VALID |
| Decode step, c = 64 | `d3_64` | 30 | 11858.971 ms | 11803.567 ms | 18595.912 ms | 1889.803 ms | **15.936%** | single-iteration, R=1, probe 11.932 s | **INVALID** |
| Decode step, c = 128 | `d3_128` | 30 | **23735.411 ms** | 23675.880 ms | 24120.160 ms | 109.544 ms | 0.462% | single-iteration, R=1, probe 24.085 s | VALID |

**ISOLATED GEMM** — the naive matmul alone, carried through unchanged from the Stage 2 driver, reported as its own configuration with its own verdict and never folded into prefill or decode. Operand fill is outside the timed bracket. These records do not depend on the prompt set.

| Configuration | B operand | Samples | Median | Std dev (% of median) | Throughput | Verdict |
|---|---|---|---|---|---|---|
| M=32 N=768 K=768 (`attn.c_proj` shape) | 2.25 MiB, inside the 8 MiB L3 | 30 | 21.323 ms | 3.490% | **1.770 GFLOP/s** | VALID |
| M=32 N=3072 K=768 (`mlp.c_fc` shape) | 9.00 MiB, exceeds the 8 MiB L3 | 30 | 120.753 ms | 1.305% | **1.250 GFLOP/s** | VALID |

**The separation Stage 2 could not make is now measurable.** Stage 2 recorded the N = 3072 configuration as INVALID at 8.212%, and said explicitly that the configuration which would have separated a scalar issue-rate limit from a memory-hierarchy limit was therefore the one that failed. Here **both are VALID**, at 3.490% and 1.305%, and the L3-resident shape runs at 1.770 GFLOP/s against 1.250 GFLOP/s for the shape whose B operand exceeds L3 — a **29.4% throughput drop** for an operand change and nothing else, M and K held fixed. This is reported as a measurement, not as a change against Stage 2: the regression comparison **refused** that pair because the prior side is INVALID, and a refusal is not a licence to compare by eye. Stages 5 and 6 own the interpretation.

#### The two INVALID configurations, and why the expectation about them was wrong

**Both are reported as INVALID. Neither was retried, averaged away, or repaired by a robust statistic.** The harness has no retry path, so no configuration can be re-run into validity by it.

| Configuration | Std dev (% of median) | Samples more than 5% above the median | Largest sample | Excess over median |
|---|---|---|---|---|
| Prefill, L = 128 | 5.723% | 7 of 30 | 33181.064 ms | +4760.326 ms (**+16.7%**) |
| Decode step, c = 64 | 15.936% | 6 of 30 | 18595.912 ms | +6736.941 ms (**+56.8%**) |

**The expectation stated at this session's checkpoint was that the SHORT configurations would fail, and it was wrong.** The reasoning was that Stage 2's prefill at L = 32 was INVALID at 7.031% and that the D3 short configurations run the same implementation at similar and shorter durations. What actually happened is the reverse: L = 16 and L = 32 were the **tightest** configurations in the set at 0.328% and 0.424%, and the two **longest-running** configurations failed. Stating this plainly because an expectation that is quietly dropped after the fact is worse than one that was never written down.

**What the failures actually are.** Both are a small number of **large** events, not a broad widening: six or seven samples of thirty sit more than 5% above the median and the rest are tight. The excesses are **seconds-scale**, 4.8 s and 6.7 s. W3's model is the right *form* — at k = 7 of n = 30 with an event costing 0.167 of the baseline, `100 × 0.167 × sqrt(7×23/(30×29))` = 7.2% against the observed 5.7%; at k = 6 with an event costing 0.568, the same form gives 23.1% against the observed 15.9%, the overshoot in both cases because the events were not all one size. But the **magnitude regime is not W3's**: W3 characterised interruptions of order tens of microseconds against 30–90 µs baselines, and these are multi-second events against 12–28 s baselines. **Batching is not available as a remedy here**, because the per-sample bracket is already 10³ times the 10 ms floor and lengthening it further would multiply an already 91-minute run. The honest position is that these two configurations measured a machine that was doing something else for part of the run, and that the baseline simply does not exist at those two configurations.

**Consequence for the waterfall, stated so Stage 4 cannot miss it: there is no Stage 2 baseline at prefill L = 128 or at decode c = 64.** Later stages must say so rather than compare against a figure that failed its own validity rule.

#### Regression comparison against Stage 2

Compared against `bench/results/stage2/stage2_forward.json`. A REGRESSION is a slowdown of the median exceeding the **4.4%** noise floor (`BENCHMARK_PROTOCOL.md` §4.1, **used and not re-derived**), on a configuration VALID in BOTH files, at the SAME prompt set and the SAME configuration. **Substantive fingerprint differences between the two files: none.**

**Result: no regressions. One comparison performed, eight refused**, and the refusals are the point of the check rather than a failure of it.

| Configuration | Outcome |
|---|---|
| `isolated_gemm M=32,N=768,K=768` | **compared**: prior 24.056 ms → current 21.323 ms, **−11.363%**, recorded as an improvement — see the caveat below |
| `prefill L=16`, `prefill L=128` | REFUSED — the configuration does not appear in the prior file (Stage 2 ran only L = 32 and L = 64) |
| `prefill L=32` | REFUSED — the prompt sets differ, **and** the prior configuration is INVALID |
| `prefill L=64`, `decode c=32`, `decode c=128` | REFUSED — the prompt sets differ (prior PLACEHOLDER, current FIXED `d3_*`) |
| `decode c=64` | REFUSED — the prompt sets differ, **and** the current configuration is INVALID |
| `isolated_gemm M=32,N=3072,K=768` | REFUSED — the prior configuration is INVALID |

**The caveat on the one comparison that ran, and it is not a small one.** The isolated GEMM records carry no prompt-set annotation, correctly, because the operands are generated internally and the prompt set cannot affect them. The comparison therefore passed every gate the check applies and reported an 11.363% improvement — **for which no code change exists**: nothing in `gemm_naive`, in its flags, or in the matmul interface changed between Stage 2 and Stage 3. The plausible difference is the process set, which is materially lighter here. **No speedup is claimed from this figure and it is not a Stage 3 result.** It is recorded because it shows precisely what regression detection does and does not establish: it detects that two medians differ by more than the floor, and it establishes nothing whatsoever about why.

**What the check structurally cannot catch**, stated in the results file itself: any change smaller than the noise floor. On this machine a real 3% regression is indistinguishable from noise and is reported as *no measurable change* with the floor named.

#### W11 — Stage 2's placeholder baseline, CLOSED

**Route (a) taken**: the committed Stage 2 binary re-run at the fixed D3 prompt set under full protocol conditions. The route was available because the Stage 2 code is committed, the fingerprint is frozen, and the placeholder labelling keeps the two runs distinguishable.

**The one-artifact construction.** The harness's first run drives the Stage 2 timing driver at the D3 prompts and configurations. That single set of timings **is** the Stage 3 measurement and **is** the W11 re-run; it is written once, to `bench/results/stage3/stage3_harness.json`, and cited from both places. Running the same binary twice over the same inputs would have consumed another 91 minutes to produce a second set differing from the first only by noise. Nothing is counted twice: one set of timings, one file, two readers.

**The re-run figures become the waterfall baseline**; the Stage 2 placeholder figures remain in the Stage 2 entry, with their PLACEHOLDER labels untouched, as the historical record of what was measured on inputs no other stage uses. The dated second measurement row is appended inside the Stage 2 entry.

**The driver was re-parameterised, and the timing code was not touched.** `TECHNICAL_SPEC.md` §4 called `bench/stage2_forward_bench.c` "temporary; Stage 3's harness replaces it", which was ambiguous. Resolved as: **the harness replaces it as the top-level ORCHESTRATOR and it remains the timing driver.** It hardcoded the configuration set, read only the fixture's first row, and hardcoded the PLACEHOLDER labelling, so the parameterisation it needed was added: a multi-row fixture loader taking the prompt text from the last tab-separated column (so one parser serves both the Stage 2 and the D3 layouts); `--lengths` and `--contexts`, defaulting to the Stage 2 values so a bare invocation still reproduces Stage 2 exactly; `--out`; `--prompt-set-status` / `--prompt-set-decision` / `--prompt-set-work-item`, defaulting to the Stage 2 values, with `prompt_set_source` now reporting the fixture actually read; `--probe` extended to probe every requested configuration in a parseable form; per-configuration row selection with the exact-count hard fail; and a logits buffer sized from the longest requested configuration. **Not changed:** `bench_run`, the timed bodies, the bracket contents, warmup and sample handling, the statistics, the 5%-of-median validity rule, the drift diagnostic, and every output field that already existed. The W11 re-run therefore ran a **re-parameterised build of the Stage 2 timing driver with identical timing code**. Original source at commit `20e4f4e`; the re-parameterised source is committed with this stage.

#### W12 — the build timestamp inside the fingerprint, CLOSED

**Approach: EXCLUDE and REPORT.** `build_flags.BENCH_BUILD_TIMESTAMP` is removed from the comparison set and reported in a separate provenance block showing the stored and current values side by side, labelled informational with `compared: false` and a stated reason. The comparison set is **26 fields before, 25 after**. `verify` prints that count on every run together with the name of the excluded field, so nothing is silently dropped and a reader of the output can always see what was and was not compared.

**Three constraints held.** The fix is in the compare logic of `verify_fingerprint` only. `bench/results/machine_fingerprint.json` was **not** rewritten, regenerated or refreshed, and the `fingerprint` subcommand — which WRITES the stored file — was never run; confirmed by `git status` reporting the file unmodified. The timestamp is still reported and never silently discarded.

**Post-fix verification, after a clean build: 25 of 25 compared fields clean, 0 differing, exit 0.** Every difference named: none. This is the first clean verify any stage has produced after its own rebuild. `tests/test_machine_state.py` does **not** encode the comparison count of 26, so the one-assertion change contemplated for it **was not required** and the file was not touched.

**The Stage 1 / Stage 2 disagreement: the mechanism is now ESTABLISHED, with the evidence.** `CMakeLists.txt` line 79 evaluates `string(TIMESTAMP TIE_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)` and feeds it to the `configure_file` at line 102, so `build/generated/build_info.h` is regenerated at CMake **configure** time, not at compile time; `scripts/build.ps1 -Clean` deletes the build tree and forces a reconfigure. `verify` reads `build_flags` out of whichever build tree is live. A verify run taken **before** a reconfigure therefore sees the previous build's value and matches, and one taken **after** differs. That is sufficient to produce Stage 1's 26 of 26 and Stage 2's 25 of 26 from one check against one stored file, with no inconsistency in the check itself. **What remains unestablished** is *when in its session* Stage 1 ran verify relative to its clean build, which no artifact records; the fix removes the symptom and this session did not establish that specific ordering.

#### W3 — the timing construction, updated and still OPEN for Stage 4

The construction (a) through (f) is written into `bench/harness.py` as a comment block naming W3 and carrying the arithmetic, and every later stage inherits it. Its full text is in that file and in `PERSISTENT.md` §8 W3; the arithmetic is reproduced here because the entry is where a reader checks it.

**The dispersion model.** For n samples of a workload of uninterrupted time B, of which k are hit by one interruption costing 0.5·B, the sample standard deviation as a percentage of the median is `sd% = 50 · sqrt(k(n−k) / (n(n−1)))`. At n = 30, k = 1 that is `50 · sqrt(29/870)` = **9.13%**, roughly twice the 5% limit, agreeing with W3's own `0.5/sqrt(29)` = 9.29% to within the difference between the two ways of writing it. Solving for the 5% limit at k = 1 gives `sd% = 50/sqrt(n)`, so n > 100 — **but that is the wrong question**, because k is not fixed: interruptions arrive at a rate, so k grows with n. With a rate λ and a cost δ, over a bracket of duration T the event count has mean λT and standard deviation `sqrt(λT)`, so `relative sd = δ·sqrt(λT)/T = δ·sqrt(λ/T)`, **which depends on T and not on n**. More samples do not reduce it; n = 30, k = 1 gives 9.1287% against n = 60, k = 2 giving 9.0510%. Checking the form against Stage 0's observed case — B = 30–90 µs, one event in thirty samples costing 0.5·B — gives `0.5·sqrt(1/30)` = 9.13%, reproducing the figure. Batching R iterations into one bracket multiplies T by R and divides the dispersion by `sqrt(R)`: at p = 1/30, R = 10 gives **2.89%** and R = 30 gives **1.67%**. The closed form is executable in the harness as `predicted_dispersion_pct` and asserted in `tests/test_harness.py`, so the arithmetic in this entry is checked by a test rather than only written down.

**The construction in force from here.** 30 samples, never reduced and **never raised as a remedy for dispersion** — the arithmetic above is exactly why raising it is not a remedy. Construction chosen adaptively from one untimed **measured** probe per configuration: at or above a per-sample floor of **10 ms**, the repeated-single-iteration construction; below it, a batched construction with `R = ceil(10 ms / probe)`, reporting the per-iteration cost as the bracket divided by R. The 10 ms floor is chosen so an interruption of the absolute size Stage 0 observed — tens of microseconds — is a fraction of a percent of the bracket rather than half of it. The trade-off is stated wherever the figure appears: a batched sample is an **amortised mean**, so the per-iteration distribution is no longer observable and individual interruption events can no longer be counted or characterised. Under batching a decode step advances the context, so R consecutive steps are R different shapes; the harness records the context **range** and reports an amortised cost over [c, c+R), preferring cache-state restoration outside the bracket where the implementation permits R identical steps, and recording which of the two was done. **The 5%-of-median limit is never loosened**: it is not a parameter, the harness exposes no way to change it, and there is no retry path. Probe, construction, R and effective bracket duration are recorded per configuration.

**Every Stage 3 configuration took the single-iteration path.** Measured probes ran 3.771 s to 28.881 s, all three orders of magnitude above the 10 ms floor, so R = 1 everywhere. **The batched path is therefore BUILT and UNIT-TESTED this stage but NOT EXERCISED by any timed run here**, and the results file records `batched_path_exercised_by_a_timed_run: false`. One further gap is recorded rather than glossed: the harness emits `--repeat R` to the timing command when R > 1, and the Stage 2 timing driver **does not implement that flag**, because adding a batching loop inside its timed bracket would have been a change to timing code this stage was not permitted to make. **Stage 4 is the first stage whose decode steps are fast enough for R > 1 to be selected, and it owns adding the C side of the batched bracket.**

#### W9 — the cuBLAS prefill denominators at the D3 M values, CLOSED

`bench/microbench/cublas_sgemm_ref` re-run **unmodified** at warmup 25 and 30 samples, results file `bench/results/stage3/stage3_cublas_d3.json`. Its M sweep is compiled in as `1, 8, 16, 32, 64, 128, 256, 512, 1024` and cannot be driven from the command line or the environment — **but it already contains every D3 M value and M = 1**, so the binary needed no modification and received none, and the rows that matter are selected afterwards. Its command-line interface is `[warmup] [samples]` only.

K and N were verified against `src/gpt2_tensor_inventory.json` rather than taken from any prompt: `h.*.attn.c_attn.weight` [768, 2304], `h.*.attn.c_proj.weight` [768, 768], `h.*.mlp.c_fc.weight` [768, 3072], `h.*.mlp.c_proj.weight` [3072, 768]. All four matched.

**These are the per-shape prefill denominators later stages quote. W9 requires them per shape and never as one number.**

| Shape | Tensor | M | N | K | Throughput | Median | Std dev (% of median) | Verdict |
|---|---|---|---|---|---|---|---|---|
| qkv_projection | `c_attn` | 16 | 2304 | 768 | 391.56 GFLOP/s | 0.144608 ms | 3.609% | VALID |
| qkv_projection | `c_attn` | 32 | 2304 | 768 | 1113.05 GFLOP/s | 0.101744 ms | 5.580% | **INVALID** |
| qkv_projection | `c_attn` | 64 | 2304 | 768 | 1281.18 GFLOP/s | 0.176784 ms | 1.600% | VALID |
| qkv_projection | `c_attn` | 128 | 2304 | 768 | **1786.22 GFLOP/s** | 0.253600 ms | 1.055% | VALID |
| attn_output_projection | `attn.c_proj` | 16 | 768 | 768 | 377.73 GFLOP/s | 0.049968 ms | 1.499% | VALID |
| attn_output_projection | `attn.c_proj` | 32 | 768 | 768 | 633.71 GFLOP/s | 0.059568 ms | 14.986% | **INVALID** |
| attn_output_projection | `attn.c_proj` | 64 | 768 | 768 | 888.29 GFLOP/s | 0.084992 ms | 6.087% | **INVALID** |
| attn_output_projection | `attn.c_proj` | 128 | 768 | 768 | 1342.80 GFLOP/s | 0.112448 ms | 5.147% | **INVALID** |
| ffn_up | `mlp.c_fc` | 16 | 3072 | 768 | 424.26 GFLOP/s | 0.177952 ms | 2.484% | VALID |
| ffn_up | `mlp.c_fc` | 32 | 3072 | 768 | 1096.96 GFLOP/s | 0.137648 ms | 6.527% | **INVALID** |
| ffn_up | `mlp.c_fc` | 64 | 3072 | 768 | 1692.62 GFLOP/s | 0.178416 ms | 4.071% | VALID |
| ffn_up | `mlp.c_fc` | 128 | 3072 | 768 | **1944.81 GFLOP/s** | 0.310560 ms | 3.603% | VALID |
| ffn_down | `mlp.c_proj` | 16 | 768 | 3072 | 498.16 GFLOP/s | 0.151552 ms | 2.224% | VALID |
| ffn_down | `mlp.c_proj` | 32 | 768 | 3072 | 1089.37 GFLOP/s | 0.138608 ms | 5.297% | **INVALID** |
| ffn_down | `mlp.c_proj` | 64 | 768 | 3072 | 1417.85 GFLOP/s | 0.212992 ms | 2.219% | VALID |
| ffn_down | `mlp.c_proj` | 128 | 768 | 3072 | **1975.24 GFLOP/s** | 0.305776 ms | 0.799% | VALID |

**Six of the sixteen D3 rows are INVALID and are reported as INVALID**, concentrated at M = 32 (four of the four shapes) and across the whole of `attn_output_projection`, which is the smallest shape in the set. This is exactly the qualification `BENCHMARK_PROTOCOL.md` §4.1 attaches to the noise floor — small-M GEMM shapes are materially noisier than everything else, and the worst Stage 0 spreads came from the same place. **Where a denominator is INVALID, no denominator exists at that shape and M, and a later stage must say so rather than quote it.**

**The M = 1 decode-shaped rows, labelled as such and used for nothing.** They are not part of the prefill denominator. They are the first fresh evidence anyone has taken on W3's original failure mode since Stage 0.

| Shape | M | N | K | Throughput | Median | Std dev (% of median) | Verdict |
|---|---|---|---|---|---|---|---|
| qkv_projection | 1 | 2304 | 768 | 66.46 GFLOP/s | 0.053248 ms | 1.047% | VALID |
| attn_output_projection | 1 | 768 | 768 | 38.76 GFLOP/s | 0.030432 ms | 2.084% | VALID |
| ffn_up | 1 | 3072 | 768 | 72.16 GFLOP/s | 0.065392 ms | 4.220% | VALID |
| ffn_down | 1 | 768 | 3072 | 41.82 GFLOP/s | 0.112832 ms | **50.176%** | **INVALID** |

**Three of four are VALID, which is a weaker result for W3 than Stage 0 produced**, where all five M = 1 shapes were INVALID in at least one of two runs. One shape failed, and it failed hard at 50.176%. W3's own characterisation — a stochastic hit rate rather than a property of any shape — survives: the shape that failed here is not one that failed consistently before. The lighter process set is the obvious candidate difference and **this stage does not establish that it is the cause**. The rows corroborate W3's arithmetic on the GPU side at the sample count the protocol uses, and nothing more is claimed from them.

**Non-monotonicity, reported as an observation with no mechanism proposed.** Eight steps in the sweep go down as M goes up:

| Shape | M step | Throughput | Drop | Both sides VALID |
|---|---|---|---|---|
| qkv_projection | 8 → 16 | 415.37 → 391.56 GFLOP/s | 5.7% | no |
| qkv_projection | 128 → 256 | 1786.22 → 1535.79 GFLOP/s | 14.0% | **yes** |
| attn_output_projection | 128 → 256 | 1342.80 → 1289.67 GFLOP/s | 4.0% | no |
| attn_output_projection | 512 → 1024 | 1949.13 → 1556.26 GFLOP/s | 20.2% | **yes** |
| ffn_up | 8 → 16 | 432.18 → 424.26 GFLOP/s | 1.8% | **yes** |
| ffn_up | 128 → 256 | 1944.81 → 1561.25 GFLOP/s | 19.7% | **yes** |
| ffn_down | 128 → 256 | 1975.24 → 1470.17 GFLOP/s | 25.6% | **yes** |
| ffn_down | 512 → 1024 | 2359.52 → 1589.19 GFLOP/s | 32.6% | no |

The pattern Stage 0 recorded is reproduced and sharpened: **every one of the four shapes falls between M = 128 and M = 256**, and two fall again between 512 and 1024. Stage 0 observed a peak at M = 512 for `ffn_down` falling at M = 1024; that is reproduced here at 2359.52 → 1589.19 GFLOP/s. **The cause remains UNESTABLISHED.** No per-kernel profiling was done and none is proposed here; W9 is closed because recording the denominators per shape is what it asked for, not because the mechanism is now known. Stage 10 takes the explanation further.

#### No Nsight counters, and no headline ratio

**No counters were collected, and the results files say so with the reason.** `BENCHMARK_PROTOCOL.md` §6 requires Nsight Compute counters for GPU stages from **Stage 7 onward**. This stage writes no kernel; the engine it times runs on the CPU, and the cuBLAS binary it re-runs is Stage 0's, already profiled in its own stage. No profiler run was introduced.

**No headline ratio is quoted.** §7 defines the decode headline against measured GPU bandwidth and the prefill headline against cuBLAS at the engine's shapes. This stage's engine runs on the **CPU**, so the cuBLAS figures recorded above are **denominators for later stages and not a ratio for this one**. No speedup is claimed anywhere in this entry; the only cross-stage delta that survived its gates is the isolated GEMM improvement discussed above, which is explicitly not claimed as a result.

#### What this stage established, in one place

- **D2 is `6e-03` absolute**, justified from measurement at the longest D3 length with both conditions' arithmetic and both achieved factors on the record, and with the set-wide margin qualification attached to it permanently.
- **D3 is four rows at exactly 16, 32, 64 and 128 tokens**, twice verified on counts *and* full id sequences, and re-checked by the timing driver on every run.
- **The waterfall baseline exists at five of seven configurations.** Prefill 3537.051 / 7058.383 / 14121.470 ms at L = 16/32/64; decode 5925.150 and 23735.411 ms at c = 32 and c = 128. **It does not exist at prefill L = 128 or decode c = 64.**
- **The fingerprint can now report clean**, 25 of 25, and the one field that made that impossible is reported rather than compared.
- **The timing construction is fixed for the project**, and its batched half is built, tested and waiting for Stage 4 to supply the C-side bracket.
- **The per-shape cuBLAS denominators exist at the D3 M values**, with six of sixteen INVALID and named, and the non-monotonicity reproduced and still unexplained.
- **An expectation was written down before the run and falsified by it** — the short configurations were predicted to fail and were the tightest in the set.

#### What this taught

A tolerance is only as meaningful as the bound that does not depend on the thing being measured: the observed divergence sets a floor that moves with every rewrite, while the reference's own decision margin sets a ceiling that nothing in the implementation can touch. And dispersion is a property of how long each sample takes, not of how many samples are taken — which is why the remedy for a noisy fast measurement is a longer bracket, and why there is no remedy at all for a slow one.

## Stage 4 — KV cache
**Status:** COMPLETE, 2026-10-05. Offline gate green (clean build, **25/25 tests** — up from Stage 3's 22: `test_kv_cache`, `test_greedy32_cache` and `test_greedy32_nocache` added). Fifteen timed configurations, 25 warmup and 30 samples each: **ten VALID, five INVALID and named**. Correctness gate **PASS**, and all eleven DECISION E checks pass — the cached decode path is **bit-for-bit identical** to prefill (0 differing elements of 12,061,680) and to the no-cache step. **D2 RE-DERIVED to `2.3e-03`** under a corrected set-wide rule (DECISION A), at 3.28x above divergence and 3.40x below margin. **W3 updated and still open**, **W6 handed to Stage 5**, **W14 raised**, **W15 raised**. Decode's context-proportional term is gone: **2.0351x per doubling without the cache against 1.0362x with it**, for an in-session **27.69x / 54.66x / 106.80x** at c = 32 / 64 / 128. **Every cached decode latency is INVALID on dispersion and none is certified.** Prefill unchanged by the cache against an in-session control (−1.505 / −0.692 / −0.653 percent, all inside the 4.4 percent floor). No Nsight counters and no headline ratio, both with reasons stated. The timed set was measured in **eight chunks** after the host killed two single-invocation attempts; the reasons and the consequences are recorded below.

#### Prediction  (written 2026-10-05, before implementation)

```
Expected decode, per generated token, with the KV cache (decode step at context c = the step producing the logits at position c-1 of the D3 row with c tokens; positions 0..c-2 already cached):
  c = 32:  232 ms
  c = 64:  233 ms
  c = 128: 235 ms
  Decode curve: c = 128 over c = 32 = 1.014, i.e. 1.007 per doubling, against the measured no-cache 2.0014 per doubling. Flat to within the 4.4 percent noise floor; the residual growth is predicted not to be distinguishable from flat on this machine.

Expected prefill, with the cache written during prefill:
  L = 16:  3537 ms   L = 32:  7058 ms   L = 64:  14121 ms   (unchanged from the D3 baseline)
  L = 128: 28.5 s    (no baseline exists at this length)
  In-session, prefill with cache writes and prefill without them differ by less than the 4.4 percent noise floor at every VALID length.

Expected KV cache footprint: 73,728 bytes per position (12 layers x 2 x 768 x 4 B); 2.25 MiB at c = 32, 4.50 MiB at c = 64, 9.00 MiB at c = 128, 72.0 MiB at the full 1024-token context — 15.2 percent of the 497,759,232-byte parameter set.

Expected correctness: a cached decode step reproduces the prefill logits at the same position bit for bit, and the cached step matches the no-cache step bit for bit. Maximum prefill divergence against the oracle unchanged at 7.019043e-04, so D2 re-derived under the set-wide rule lands at 2.3e-03.

Reasoning:
- A cached decode step does one token's work: 169,869,312 FLOPs through the layer weights, 77,194,752 through the tied head, plus 36,864 x c for attention over c cached positions — 248,243,712 / 249,423,360 / 251,782,656 FLOPs at c = 32 / 64 / 128.
- The engine is still the scalar naive ijk matmul, so the step is bound by issue rate, not memory: one full read of the weights is 494,128,128 B, which takes about 7 ms even at the measured L3 bandwidth of 70.81 GB/s, about 33 times less than the compute time. Memory would only bind if sustained DRAM bandwidth were below about 2.2 GB/s.
- The efficiency fraction comes from the D3 baseline, which is the newest measurement of this binary: 0.132 of the measured 8.565 GFLOP/s scalar ceiling at prefill, 0.110 at no-cache decode. A cached decode token has the same operation mix as a prefill token — the head is 31 percent of both, against 1.4 percent of a no-cache decode step — and every GEMM row it runs is run identically by prefill, so the prefill fraction is the right starting point. One thing pulls it down: at M = 1 the 6.75 MiB c_attn weight (a quarter of the layer FLOPs) and the 2.25 MiB attn.c_proj weight can no longer be reused out of L3 across rows. Taken together: f = 0.125, giving 248,243,712 / (0.125 x 8.565e9) = 231.9 ms at c = 32, 233.0 ms at c = 64, 235.2 ms at c = 128.
- The Stage 2 and Stage 3 sessions measured the same binary about 10 percent apart, so these figures carry a condition uncertainty of that size on top of the fraction.
- Prefill gains only stores: 9,437,184 B of cache writes at L = 128, against at least 48.75 GB the pass already moves re-walking the MLP weights and the head every token — under 0.02 percent. So prefill does not move. L = 128 is 32,228,179,968 FLOPs at the prefill fraction 0.132: 28.5 s.
- 235 ms is more than twenty times the 10 ms per-sample floor, so every configuration takes the single-iteration construction (R = 1) and the batched path still does not fire.

Falsified if:
- The efficiency fraction was merely imprecise if every VALID cached decode median is within a factor of 2 of 232 ms (116 to 464 ms).
- The model of where the time goes is wrong if: the c = 128 over c = 32 cached decode ratio exceeds 1.10 (context-proportional work survives — a cache not consulted, or a per-step copy that grows with c); any VALID cached decode median exceeds 696 ms (three times the prediction: memory binds, or work is being recomputed); any VALID cached decode median is below 77 ms (a third of the prediction, with no vectorised code to explain it); or in-session paired prefill with and without cache writes differs by more than 4.4 percent at any VALID length.
```

#### Prediction derivation (full arithmetic behind the prediction)

```
  (a) What is predicted. Decode latency per generated token with the KV cache, at context c = 32, 64 and 128, where a decode step at context c is defined exactly as Stage 2 and Stage 3 defined it: the step that produces the logits at position c-1 of the D3 row with c tokens, the head computed once. With the cache, positions 0..c-2 are already cached (filled outside the timed bracket), and the timed step processes token c-1 alone and attends over c positions. Prefill latency at L = 16, 32, 64 and 128, where prefill now additionally writes every layer's keys and values into the cache. The KV cache's resident footprint as a function of context, layers and model width. The shape of the decode curve across c.

  (b) Inputs and their sources. Architecture values n_layer 12, n_head 12, n_embd 768, head_dim 64, n_ctx 1024, vocab 50257, from the Stage 1 and Stage 2 MEASUREMENTS.md entries, read from the config shipped with the weights. The Stage 2 FLOP partition: layer weight matmuls 169,869,312 FLOPs per token, tied head 77,194,752 FLOPs per application, attention 36,864 FLOPs per (query, key) pair summed over twelve layers (Stage 2 entry, derived-throughput section). Parameter count 124,439,808 (Stage 1 entry). The measured scalar FP32 ceiling 8.565 GFLOP/s (HARDWARE.md §2, Stage 0b run 2). The measured L3-resident bandwidth 70.81 GB/s (HARDWARE.md §2), used only as an optimistic floor for the memory route. The Stage 3 D3 baseline medians (MEASUREMENTS.md, Stage 2 entry second measurement and Stage 3 entry): prefill 3537.051 / 7058.383 / 14121.470 ms at L = 16 / 32 / 64, decode 5925.150 and 23735.411 ms at c = 32 and 128; no baseline exists at prefill L = 128 or decode c = 64, both INVALID. The Stage 2 efficiency fractions 0.120 at prefill and 0.098 at decode, measured on placeholder inputs in the Stage 2 session.
  Figures deliberately NOT used, with reasons: the measured vectorised AVX2 peak 48.411 GFLOP/s, because this stage runs scalar code and that ceiling is Stage 6's framing (PERSISTENT.md §8 W8); the single-channel theoretical ceiling 23.464 GB/s, because it is derived from part numbers and a theoretical figure may not enter a prediction (§8 W7); measured CPU DRAM bandwidth, because the HARDWARE.md §2 field is empty and its INVALID orientation medians are not measurements (§8 W1); every GPU figure — measured achievable bandwidth, cuBLAS throughputs, clocks — because this is a CPU stage; the 2496 MHz nominal clock, because it is a static read.

  (c) Derivation. Efficiency fractions recomputed from the D3 baseline, the newer measurement of the same binary, on the as-implemented FLOP basis (full T x T attention square): prefill L = 64, 64 x 247,064,064 + 36,864 x 64^2 = 15,963,095,040 FLOPs in 14.121470 s = 1.1304 GFLOP/s, f = 0.1320 of 8.565; L = 16 and 32 give 0.1308 and 0.1314. Decode c = 32, 32 x 169,869,312 + 36,864 x 32^2 + 77,194,752 = 5,550,761,472 FLOPs in 5.925150 s, f = 0.1094; c = 128, 22,424,446,464 FLOPs in 23.735411 s, f = 0.1103; mean 0.1098. The D3 fractions are about 10 percent above the Stage 2 session's 0.120 and 0.098 for the same binary and the same per-token work, which is the cross-session condition shift recorded as W14 below; the prediction therefore carries a condition uncertainty of that size in addition to its model uncertainty.
  FLOPs of a cached decode step at context c: one token through the layer weights, 169,869,312; the head once, 77,194,752; attention for one query row over c keys, 36,864 x c. Total 247,064,064 + 36,864c: c = 32, 248,243,712; c = 64, 249,423,360; c = 128, 251,782,656.
  Which fraction applies. The per-token FLOP mix of a cached decode step is the per-token mix of a prefill token, not of a no-cache decode step: the head is 77,194,752 / 248,243,712 = 31.1 percent of the cached step and 31.2 percent of a prefill token, against 1.4 percent of a no-cache decode step at c = 32. Stage 2 attributed its prefill-over-decode fraction difference, without establishing it, to the head being the one matmul whose operands are both walked contiguously. Every GEMM row the cached step executes is executed identically, in the same ijk order, by prefill. The one structural difference runs the other way: at M = 1 the L3-sized weights (c_attn 6.75 MiB, attn.c_proj 2.25 MiB) can no longer be re-used from L3 across rows, so each step streams them from beyond L3. Upper bracket: the prefill fraction, 0.1320. Lower bracket: the no-cache decode fraction, 0.1098. Central: 0.125, nearer the prefill fraction because the operation mix matches prefill, pulled down because c_attn (42,467,328 of the 169,869,312 layer FLOPs, 25 percent) and attn.c_proj lose cross-row L3 reuse at M = 1.
  Compute route. Central 0.125: c = 32, 248,243,712 / (0.125 x 8.565e9) = 231.9 ms; c = 64, 233.0 ms; c = 128, 235.2 ms. Upper bracket 0.1320: 219.6 / 220.6 / 222.7 ms. Lower bracket 0.1098: 263.9 / 265.1 / 267.6 ms.
  Memory route. Bytes per cached step: weights 4 x 123,532,032 = 494,128,128 B, plus the cache read 73,728 x c B and the one-token cache append 73,728 B; c = 128 gives 503,639,040 B. Against the L3-resident 70.81 GB/s as an optimistic floor, 7.1 ms. The compute route exceeds that floor by a factor of about 33, so the compute route binds. Memory would bind only if sustained DRAM read bandwidth were below 503,639,040 B / 0.2352 s = 2.14 GB/s; no measured DRAM figure exists and none is used, so this is stated as the condition under which the model is wrong rather than as an estimate.
  Decode curve shape. The only context-dependent term left is attention, 36,864c FLOPs and 73,728c cache bytes. Predicted c = 128 over c = 32: 251,782,656 / 248,243,712 = 1.0143, i.e. 1.0071 per doubling, against the measured no-cache 2.0014 per doubling. That residual is a third of the 4.4 percent noise floor and is predicted NOT to be distinguishable from flat on this machine. "Roughly constant in context" therefore predicts three medians within about 1.4 percent of each other.
  Prefill. The cache adds stores and no arithmetic: 73,728 B per token, 9,437,184 B at L = 128. The no-cache prefill already re-walks the two 9.0 MiB MLP weights in each of twelve layers and the 147.2 MiB head for every token, at least (2 x 9,437,184 x 12 + 154,389,504) x 128 = 48,752,885,760 B at L = 128, so the cache stores add at most 0.019 percent of the bytes the pass moves. Predicted: prefill unchanged within the 4.4 percent noise floor at every length. Predicted values: the D3 medians at L = 16 / 32 / 64; at L = 128, where no baseline exists, 32,228,179,968 FLOPs / (0.1320 x 8.565e9) = 28.51 s.
  KV footprint. Per token per layer, K and V each n_embd floats: 2 x 768 x 4 = 6,144 B; across 12 layers, 73,728 B per token. c = 32: 2,359,296 B (2.25 MiB); c = 64: 4.50 MiB; c = 128: 9,437,184 B (9.00 MiB, larger than the 8 MiB L3); full n_ctx 1024: 75,497,472 B (72.0 MiB), which is 15.2 percent of the 497,759,232 B parameter set. Comfortable on one 8 GiB DIMM.
  Correctness, structural. Because every row of every GEMM is computed by the same ijk inner loop in the same order, and attention for one row reduces over the same positions in the same order, a cached decode step is predicted to reproduce the prefill logits at the same position BIT FOR BIT, and the prefill divergence against the oracle is predicted unchanged at 7.019043e-04.
  The W3 check. The predicted cached step, 232 to 235 ms central and 220 to 268 ms across the brackets, is more than twenty times the 10 ms per-sample floor, so the batched construction is not selected at Stage 4 (R = 1 everywhere) and the C-side batching loop is not built speculatively.

  (d) Falsification condition. "The efficiency fraction was imprecise" if every VALID cached decode median lies within a factor of 2 of the central value. "The model of where the time goes is wrong" if any of: the c = 128 over c = 32 cached decode ratio exceeds 1.10, which would mean context-proportional work survives beyond the attention term (a cache not consulted, or an O(c) copy per step); any VALID cached decode median exceeds three times the central value, which would mean the memory route binds or work is being recomputed; any VALID cached decode median is below one third of the central value; or in-session paired prefill with and without cache writes differs by more than the 4.4 percent floor at any VALID length.

  (e) Not applicable before Stage 11.
```


#### Conditions

| Condition | Value |
|---|---|
| Date, branch | 2026-10-05, branch `stage-4` cut from `main` at `f8041ac` (the Stage 3 merge) |
| Prediction commit | **`b8543b464eaf972355165f89c885c23b8bd4a524`**, `stage 4: prediction committed`, `MEASUREMENTS.md` alone, committed before any implementation file existed. The prediction and its derivation were transcribed verbatim and were not consulted again until the Gap section below |
| Environment fingerprint | `machine_state.py verify` after the clean build and again before every one of the eight measured chunks: **25 of 25 compared fields clean, 0 differing, exit 0** every time. `build_flags.BENCH_BUILD_TIMESTAMP` is reported in the provenance block as stored `2026-09-18T06:56:02Z` against current `2026-10-05T16:56:04Z`, labelled informational and **not compared** (W12). **No substantive field differs.** `bench/results/machine_fingerprint.json` was NOT rewritten and the `fingerprint` subcommand was never run |
| Clock lock | `nvidia-smi -lgc 1365,1365` applied from this elevated session, then verified **by state**: `verify-lock --mhz 1365` → `locked: true`, `off_target_samples: []`, exit 0. Re-verified after the run and still in effect. `-lmc` **not attempted** (§2 Q11). The mitigation is **PARTIAL** and is stated as partial (§1 D8): whether Balanced is stock is unverified, the memory clock cannot be locked, and there is no live CPU package temperature source on this machine |
| GPU clock offsets | **UNVERIFIED — not obtainable by query (`PERSISTENT.md` §2 Q9).** No offset of zero is recorded here or in any results file |
| MSI Dragon Center scenario | **"Balanced"**, **operator-observed and confirmed at the checkpoint**. `machine_state.py verify` cannot detect the scenario (§2 Q9), so this is recorded as an operator observation and not as a queried value. No scenario change was requested, suggested or made |
| Compiler and flags | **Confirmed unchanged against `HARDWARE.md` §5.5**, read from `build/generated/build_info.h` after the clean build and carried into the results file by the C layer: MSVC 19.44.35229.0, toolset 14.44.35207. Host flags `/DWIN32 /D_WINDOWS /EHsc /W3 /arch:AVX2 /fp:precise /MD /O2 /Ob2 /DNDEBUG`; nvcc 13.1.80, `-arch=sm_75`. **No `-allow-unsupported-compiler`** |
| AC power | **On AC**, `PowerOnline = True`, battery 100%. A run on battery is INVALID outright and none was taken |
| Network | Wi-Fi (Intel AX201), `NetworkCostType = Unrestricted`, not roaming, not over data limit — **not metered** |
| Session elevation | **True**, checked live at the checkpoint and again immediately before applying the lock — not carried forward from any earlier record |
| Thread placement | **Applied and recorded, not assumed**: `pinned to logical cpu 2 of 8; priority raised (ABOVE_NORMAL class, THREAD_PRIORITY_HIGHEST)`, `pinned=1`, `logical_cpu=2`, `priority_raised=1`, applied outside every timed bracket |
| CPU timer | `QueryPerformanceCounter`, monotonic. Never wall clock. The bracket contains one model call and nothing else |
| Oracle environment | `.venv` Python 3.14.2, torch 2.14.0+cu130, numpy 2.5.3, `torch.set_num_threads(1)`, device cpu, 160 tensors verified against the inventory |
| Ordering of correctness against timing | **Every correctness run completed and exited before the first timed bracket.** The oracle was not resident during any timed run |
| Telemetry sampler | `CpuTelemetrySampler` at the proven Stage 0b / Stage 2 / Stage 3 configuration — interval **0.02 s**, exclusions logical CPUs 0 and 2, **resolved mask `0xf0`** (allowed 4,5,6,7; CPU 2's SMT sibling 3 also excluded), affinity application verified `ok: true` from a previous mask of `0xff`. One sampler per chunk, in a **separate process from the benchmark**, never inside a bracket. **Observed duty: 0.346% to 0.624% of one core**, per-probe 36.65–66.35 µs, 253,861 samples over the set |
| Run wall time | **8,217.7 s = 2.28 h** of measured chunks (464.2 + 912.8 + 1847.6 + 432.9 + 865.2 + 1756.7 + 15.8 + 1922.5 s), between `2026-10-05T20:51:29Z` and `2026-10-06T00:32:12Z`. Merge at `2026-10-06T00:32:20Z` |
| Results | `bench/results/stage4/stage4_harness.json` (merged from eight per-chunk part files under `bench/results/stage4/parts/`), `bench/results/stage4/stage4_correctness.json`. Raw per-sample timings retained in full for all fifteen configurations |

#### How the timed set was actually run, and the three interruptions that shaped it

This is recorded because an unrecorded deviation is a fabricated number, and because the shape of the run bears on the dispersion results below.

The set was specified as **one invocation over fifteen configurations in paired interleaved order**. It was measured instead as **eight sequential chunks, one harness invocation each**, after the host environment killed the single-invocation attempt twice:

1. **Attempt 1** — one invocation, all fifteen configurations. Killed about one hour in by the host's **memory-pressure reaper** (8,012 MB total RAM, roughly 1.3 GB held by five editor processes that could not be closed). **Nothing was written**: the timing driver emits its results only after the whole set finishes, so every configuration that had completed was lost.
2. **Attempt 2** — eight chunks. `p16`, `p32` and `p64` completed and were banked; the reaper took the run during `p128`.
3. **Attempt 3** — resumed with the decode chunks first. Killed **within about two minutes**, before the engine had warmed up, which established that the low-memory condition was **pre-existing and not caused by the run**.
4. **Attempt 4** — the run was launched as a **detached process** rather than as a managed background task, which the reaper does not track. It completed `d32`, `d64`, `d128`, then (after two bugs described below) `gemm` and `p128`.

**What the chunking preserved, and what it did not.** Every cache configuration and its no-cache control were measured **in the same chunk, adjacent in time**, which is what the paired ordering exists for — session drift falls on both members of a pair. The order within each workload class is unchanged. Each chunk re-verified the fingerprint and the clock lock before starting, which is stricter than verifying once before the first run. What the split does **not** preserve is a single uninterrupted session for the whole set: the chunks span `20:51Z` to `00:32Z` with gaps, and the decode chunks were measured roughly two hours after the prefill chunks. The order as measured was `p16, p32, p64` then `d32, d64, d128, gemm, p128`; the specified order placed `p128` before the decode chunks, and it was moved last when the remaining chunks were re-ordered to bank the stage's own claim ahead of the one configuration with no baseline of either kind.

**No partial or interrupted measurement entered any figure below.** A chunk writes its results only on completion, so each reap destroyed work rather than corrupting it, and every chunk reported here ran its full 25 warmup and 30 timed samples.

**Two bugs in the chunking path, found and fixed during the run**, both in `bench/harness.py` and neither touching a timed bracket or a statistic:

- A chunk containing only the isolated GEMM configurations produced **no probe lines**, because the driver probes prefill and decode only, and `run_probe` treated that as a driver failure. Fixed so the absence is accepted **only** when nothing probeable was requested, which is now unit-tested in both directions.
- With no probes, the selected-construction dictionary is empty, so the mixed-`R` guard saw `R values []` and refused the invocation. Fixed so an empty construction set means "nothing to batch" rather than "mixed batching".

The six chunks measured before these fixes never entered either code path. The fixes are Python-only and do not change the timing binary, so **all eight chunks were measured by one binary** — the merge asserts that, comparing `git_commit`, `build_timestamp`, device, both compilers, both flag sets and the CPU timer across every part file, and refuses to merge parts that disagree.

#### Process set, enumerated before the first timed run, named rather than summarised

**The operator was remote for this session and could close nothing**, so unlike Stages 0b, 2 and 3 the close-these list was presented, declined by circumstance, and every item on it is recorded as **PRESENT during the timed set**.

*Holding a GPU context* (`nvidia-smi --query-compute-apps`, 17 entries): `dwm.exe`, `ShellHost.exe`, `explorer.exe`, `CrossDeviceResume.exe`, `SearchHost.exe`, `StartMenuExperienceHost.exe`, `msedgewebview2.exe`, `LockApp.exe`, `TextInputHost.exe`, `ShellExperienceHost.exe`, `OmApSvcBroker.exe` (MSI NBFoundation Service), `claude.exe` ×2, `logioptionsplus_agent.exe`, `ApplicationFrameHost.exe`, `SystemSettings.exe`, `PhoneExperienceHost.exe`.

*Significant CPU consumers at the checkpoint* (accumulated CPU seconds): `System` 643.7 s, `MsMpEng` (Windows Defender real-time scanning) 510.6 s, `claude` (pid 8032) 259.3 s, `svchost` (pid 5640) 112.3 s, `dwm` 94.3 s, `WmiPrvSE` 91.3 s, `logioptionsplus_agent` 57.1 s, `explorer` 43.8 s, `conhost` 40.9 s, `csrss` 31.8 s, three further `claude` instances 25–31 s each, `nvcontainer` 19.4 s, `SearchIndexer` 17.3 s, `esrv_svc` 11.3 s, `OneDrive.Sync.Service` 11.1 s. **237 processes resident**, of which **ten were `claude.exe`**.

**NOT CLOSED, and recorded as present rather than assumed gone** — the close-these list as it stood: `logioptionsplus_agent` (18812), `logioptionsplus_appbroker` (17048), `logioptionsplus_updater` (17956), `OneDrive.Sync.Service` (15404), and **nine `claude.exe` instances beyond this session's own**. Re-enumerated immediately before the lock was applied and all were still present.

**How the set differs from Stage 3's, item by item.**

- **Riot Vanguard (`vgc`, `vgtray`) — ABSENT.** Same as Stage 3, Stage 2 and Stage 0b.
- **Nahimic — ABSENT.** Same as Stage 3 after its operator closed it.
- **Intel DSA (`DSAService`, `DSATray`, `DSAUpdateService`) — ABSENT.** Same as Stage 3. `esrv` and `esrv_svc` **remained running**, as in Stage 3, and are recorded as present.
- **Logitech Options+ — PRESENT, all three processes, for the whole set.** Stage 3 closed it and recorded it as respawning mid-run; here it was never closed. **This is heavier than Stage 3.**
- **OneDrive — PRESENT.** Stage 3 did not list it. **Heavier than Stage 3.**
- **`msedgewebview2.exe` ×6 — PRESENT**, uncloseable, and equally resident in Stage 0 and Stage 3: a shared condition, not a difference.
- **MSI service stack — RUNNING**, by standing operator decision (§1 D8, §2 Q9), as in Stage 3. The Dragon Center **UI** was not running.
- **Windows Defender — PRESENT**, not excluded or modified, and again among the largest consumers.
- **This session's editor — PRESENT and much heavier: `claude.exe` ×10 against Stage 3's ×1 during its timed set.** This is the largest single difference and it is also what made the machine reap the run three times.

**Net: this set is HEAVIER than Stage 3's, on three counts — the editor process count, Logitech, and OneDrive — and lighter than none of them.** That direction matters for two conclusions below. First, it is the leading candidate for the cross-session shift the comparison against Stage 3 reports. Second, and this is the honest consequence, **W5's argument does not rescue it**: W5 reasons that a noise floor measured under heavier load stays conservative under lighter load, which bounds claims **within** a session. It says nothing about a comparison whose conditions moved **between** sessions, and here they moved in the direction that makes the current session slower. **W5's Status is NOT changed by this stage**, and the 4.4% floor is **USED, not re-derived** — this stage did not re-run the Stage 0 microbenchmark suite and therefore cannot re-derive it.

#### Correctness — the DECISION E set, eleven checks, all PASS

Every check ran in the offline gate, **before** the checkpoint and before any timed bracket. Reduced token counts are stated where used; **no timed configuration is reduced**. Bit-for-bit means exact float32 equality of every element, reported as a count of differing elements and the maximum absolute difference, both zero for a pass.

| # | Check | Where | Result |
|---|---|---|---|
| **1** | The existing three-condition gate on the prefill path, run with the switch at **CACHE** and at **NO-CACHE** | `bench/correctness.py`, all four D3 rows | **PASS and PASS.** Max absolute divergence identical on both paths at every length — 3.967285e-04 / 4.272461e-04 / 4.425049e-04 / 7.019043e-04 at L = 16/32/64/128 — top-1 agreement complete (16/16, 32/32, 64/64, 128/128) on both. The two paths are additionally **BIT FOR BIT**: 0 differing elements |
| **2** | NEW — the **cached-decode path** against the oracle, logits at **every** position assembled from the cached path alone (a 1-token prefill, then one cached decode step per remaining position) | `gpt2_tool --dump-logits --via-decode`, all four rows | **PASS.** Max absolute divergence 3.967285e-04 / 4.272461e-04 / 4.425049e-04 / **7.019043e-04**; top-1 agreement at **every** position of all four rows; relative statistics reported and **not gating**, as in Stage 3. This divergence enters the D2 lower bound below |
| **3** | The cached-path matrix against the **prefill** matrix of the same row, position by position | same run | **BIT FOR BIT on every row. 0 differing elements of 12,061,680**, maximum absolute difference **0** |
| **4** | The cached step against the **no-cache step** at the same context | `tests/test_model.c` at 5 contexts; **and asserted inside the timing driver** before each cached decode configuration's warmup | **BIT FOR BIT.** 0 differing elements, max abs difference 0. The driver's pre-warmup assertion **PASSED at c = 32, 64 and 128** over all 50,257 logits — so no cached configuration was timed before it was proven to compute the same answer |
| **5** | Causality through the cached path, **non-vacuously** | `tests/test_model.c`, reduced to 7 tokens | **PASS.** Changing the token at position 6 moved **no** logit at any earlier position (0 differing over 6 positions × 50,257) **and** moved 50,257 logits at position 6, so the check cannot pass vacuously. **The unit test is the only coverage**: `gpt2_tool` cannot take an explicit token-id sequence as input, so the j = 24 on d3_32 run was not possible and was not done |
| **6** | **32** greedily generated tokens from `d3_16`, three producers | `tests/test_reference_impl.py --greedy32`, two CTest entries | **PASS. All three EXACTLY equal over all 48 ids** — engine cached path, engine no-cache path, and the oracle's own greedy generation. Reduced relative to n_ctx = 1024 and stated as reduced. 32 rather than Stage 3's 3 because a cache fault appearing only after several appends is invisible to a 3-token check |
| **7** | Cache state restoration, which the driver relies on | `tests/test_model.c` | **PASS.** A step, a length restoration, then the same step again give **bit-identical** logits (0 differing), **and** the cache contents at positions 0..c-2 are **BYTE-identical** before and after across all 24 buffers. The second half is the one that matters: a step corrupting an earlier cached position would still give the right answer once |
| **8** | Bounds | `tests/test_model.c`, `tests/test_kv_cache.c` | **PASS.** An append at capacity returns the error code and leaves **every byte unchanged** (compared against a copy); a decode at a context beyond capacity is refused **before any arithmetic**; a cache-mode prefill longer than capacity is refused; a restoration beyond capacity is refused; a cached step whose cache holds the wrong number of positions is refused rather than answered from the wrong context; cache mode with no cache allocated is refused rather than faulted |
| **9** | Footprint | `tests/test_kv_cache.c`, asserted and reported | **PASS.** Allocated bytes equal `n_layer × 2 × capacity × n_embd × 4` computed from the config at run time, at three capacities. At capacity 1024 that is exactly **75,497,472** |
| **10** | Attention weights sum to 1 within 1e-5 in the **cached** path, every head, layer and step | `tests/test_model.c`, reduced to 7 tokens | **PASS.** 1,008 rows checked (12 layers × 12 heads × 7 steps), min **0.999999937**, max **1.000000060** |
| **11** | Every existing test still passes unmodified | full suite | **PASS. 25 of 25**, including `test_gemm_naive`'s scalar-codegen assertions read from the generated listing: **0 ymm registers, 0 packed floating-point arithmetic, 40 scalar floating-point instructions**. `src/gemm/gemm_naive.c`, `src/gemm/gemm.h` and `tests/test_gemm_naive.c` are **byte-identical to `main`** and the compiler flags are unchanged |

**One carried-forward value disagrees with the re-run, and the re-run wins.** The stage prompt states the scalar-codegen assertion as "20 scalar arithmetic instructions". The live test asserts *presence* of scalar arithmetic rather than an exact count, and reports **40**. The disagreement is stated; nothing was changed to reconcile it.

**The pre-change comparison.** Before the first edit to `src/`, the main-branch build's prefill logits for the `test_model` probe prompt at 7 tokens were dumped outside the repository: `C:\Users\saket\tie_stage4_prechange\prechange_testmodel_T7.bin`, 1,407,212 bytes, **SHA-256 `138d48cab6a03262ed01d9c33f5941fa6671509c409f41ab0f5179cb1b10f583`**. `tests/test_model.c` asserts the no-cache path reproduces that dump's float payload bit for bit, by FNV-1a 64 = **`0x344664b8843fccf8`**, after first asserting the prompt still encodes to the same seven ids. **It does.** The no-cache path is the pre-change engine, not merely something that resembles it.

#### D2 — RE-DERIVED under the corrected rule (DECISION A)

**`D2_MAX_ABS_LOGIT_DIFF = 2.3e-03`**, superseding Stage 3's `6e-03`. The authoritative statement of the rule and the full arithmetic is `BENCHMARK_PROTOCOL.md` §5 under "Amended by Stage 4 on 2026-10-05"; what follows is the measurement it rests on and the outcome.

**Why the rule was corrected rather than the value adjusted.** Stage 3 evaluated both of its bounds **at the longest D3 length**, on the premise that the decision margin degrades with length. In the same section Stage 3 established that this premise is false for this prompt set — the four rows are independent prose, not nested prefixes, so each row's margin minimum is a property of its own text, and the set-wide minimum falls at **L = 32, position 11**, not at L = 128. Evaluating the upper bound at the longest length therefore used a margin **5.6×** larger than the one the gate must respect, and the committed `6e-03` failed its own condition (b) against the set-wide minimum at **1.30×** where 3× is required.

**The inputs, from this stage's own measurement, not Stage 3's.**

| Quantity | Value | Provenance |
|---|---|---|
| Set-wide maximum absolute divergence, over **both** engine paths | **7.019043e-04** (L = 128) | `[measured]` this stage, prefill path **and** cached-decode path, all positions of all four rows |
| Set-wide minimum reference top-1/top-2 margin | **7.812500e-03** at **d3_32 position 11** | `[measured]` this stage. **Reproduced exactly against Stage 3's figure**, which is what licenses its use — had the oracle not reproduced itself the derivation would have stopped |

Taking the divergence over both paths does **not** raise it: the two paths agree bit for bit (check 3), so the cached-decode path's divergence is identical to the prefill path's at every row. That is a result worth stating rather than a technicality — the lower bound is unchanged by the addition of a whole new engine path, because the new path computes the same floats.

**The arithmetic.** `[derived]`, shown inline:

- Lower bound: `3 × 7.019043e-04` = **2.105713e-03**
- Upper bound: `7.812500e-03 / 3` = **2.604167e-03**
- Window: `7.812500e-03 / 7.019043e-04` = **11.130×**, against the required minimum of 9. **The window is OPEN.** (The two stated conditions coincide at that minimum: `lower < upper` is `3d < m/3` is `m/d > 9`.)
- Geometric mean: `sqrt(2.105713e-03 × 2.604167e-03)` = **2.341715e-03**
- Rounding: **no one-significant-figure value lies inside** the window — 2e-03 is below the lower bound, 3e-03 above the upper. At two significant figures the candidates inside are **2.2e-03, 2.3e-03, 2.4e-03, 2.5e-03, 2.6e-03**. Nearest the geometric mean in **log** distance is **2.3e-03** (0.0180, against 0.0246 for 2.4e-03). The safety bound chose the precision; the rounding did not choose the safety bound.
- **Achieved factors: `2.3e-03 / 7.019043e-04` = 3.28× above the observed divergence, and `7.812500e-03 / 2.3e-03` = 3.40× below the minimum margin.** Both clear 3×, which `6e-03` did not on the upper side.

**Files amended in place:** `BENCHMARK_PROTOCOL.md` §5 — the single machine-readable line changed to `D2_MAX_ABS_LOGIT_DIFF = 2.3e-3`, with exactly one such line remaining in the document, and the Stage 3 derivation retained as provenance with each superseded statement marked; `PERSISTENT.md` §1 D2 — new value, new definition, pointer to §5 for the arithmetic.

**The gate was then re-run so that it READS the amended value, and it parses and gates on it: PASS.** Recorded against **both** values, from one measurement evaluated at two thresholds (the gate is a pure function of the statistics and the threshold, so no engine run was repeated):

| Threshold | prefill, no-cache | prefill, cache | cached-decode path |
|---|---|---|---|
| `6e-03` (superseded) | **PASS** | **PASS** | **PASS** |
| `2.3e-03` (in force) | **PASS** | **PASS** | **PASS** |

**The parser.** `bench/correctness.py` already accepted two significant figures; what it did **not** do was refuse a document carrying two machine-readable lines — it silently took the first. That is now an error naming the file, because a gate that runs against whichever threshold happens to appear first in a document is not reading the document. Unit tests assert `2.3e-3` and `6e-3` both parse, that two such lines are refused, and that the corrected rounding rule returns `2.3e-03` for the Stage 3 inputs, returns a one-significant-figure value where one lies inside the window, reports a closed window as closed and selects no value, and reports a window narrower than the required minimum.

**The dates.** The documents carried two dates for the Stage 3 D2/D3 work and both are correct, for different events. Determined from the artifacts: Stage 3 is **one** commit, `f924196`, authored `2026-10-05T00:49:14-07:00`; the correctness run the D2/D3 resolution rests on is stamped `2026-10-05T04:43:45Z` = **2026-10-04 21:43 local** at this machine's UTC-07:00; the clean build `04:38:53Z` = 2026-10-04 21:38 local; the harness timed set `2026-10-05T07:39:58Z` = **2026-10-05 00:39 local**; the cuBLAS run `07:40:52Z` = 00:40 local. **The session spanned local midnight.** So `2026-10-04` correctly dates the D2/D3 resolution in local time — which is what the fixture header, §5 and §1 D3 carry — and `2026-10-05` correctly dates the timed set and the session close, which is what the Stage 3 entry, the §6 log and "Last updated" carry. In UTC every one of these events falls on 2026-10-05. **No date was changed.** A clause naming the event and the time zone was added at §5 and at §1 D3. **The fixture needs no erratum** — its date is correct — and it is byte-identical to the blob Stage 3 committed, `37c5501cc3fa9604ca20688bfda18a604559d2b5`.

#### The KV cache footprint, as allocated and as derived

**Derived from shapes**, `[derived]`: per position per layer, K and V each `n_embd` float32 = `2 × 768 × 4` = **6,144 B**; across 12 layers, **73,728 B per position**. Footprint = `n_layer × 2 × capacity × n_embd × 4`.

| Context | Derived bytes | Allocated bytes, as reported by the engine | | 
|---|---|---|---|
| c = 32 | 2,359,296 (2.25 MiB) | **2,359,296** `[measured]` | from the `d32` records |
| c = 48 | 3,538,944 (3.375 MiB) | **3,538,944** `[measured]` | from `gpt2_tool --footprint` during the greedy-32 check |
| c = 64 | 4,718,592 (4.50 MiB) | **4,718,592** `[measured]` | from the `d64` records |
| c = 128 | 9,437,184 (9.00 MiB) | **9,437,184** `[measured]` | from the `d128` and `p128` records |
| c = 1024 (full n_ctx) | 75,497,472 (72.0 MiB) | **75,497,472** `[measured]` | asserted in `tests/test_kv_cache.c` from the arguments |

Derived and allocated agree exactly at every capacity. At the full context the cache is **15.2% of the 497,759,232-byte parameter set** (`75,497,472 / 497,759,232 = 0.15168`), and it outgrows the 8 MiB L3 between c = 64 and c = 128. **With the switch at no-cache the resident cache footprint is exactly 0 bytes** — the cache is allocated lazily by `model_kv_reserve`, so a model that never asks for one holds none, which `tests/test_model.c` asserts directly rather than inferring.

#### Measurement — PREFILL

25 warmup, 30 timed samples, every configuration. Every figure in this table is read from `bench/results/stage4/stage4_harness.json`. `[measured]`. Cache and no-cache members of each pair side by side; **prefill and decode are never combined into one figure anywhere in this entry.**

| L | path | samples | median (ms) | min | max | std dev (ms) | sd as % of median | construction | probe (ms) | R | repeat_applied | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | **cache** | 30 | **3951.321** | 3888.228 | 4050.282 | 45.97 | **1.163%** | single_iteration | 4592.5 | 1 | 1 | **VALID** |
| 16 | nocache | 30 | **4011.716** | 3945.895 | 4179.045 | 54.75 | **1.365%** | single_iteration | 4295.8 | 1 | 1 | **VALID** |
| 32 | **cache** | 30 | **7899.536** | 7723.028 | 8094.326 | 91.56 | **1.159%** | single_iteration | 8667.3 | 1 | 1 | **VALID** |
| 32 | nocache | 30 | **7954.613** | 7822.179 | 8244.753 | 102.49 | **1.288%** | single_iteration | 8483.2 | 1 | 1 | **VALID** |
| 64 | **cache** | 30 | **16081.705** | 15722.659 | 16405.540 | 169.95 | **1.057%** | single_iteration | 16745.8 | 1 | 1 | **VALID** |
| 64 | nocache | 30 | **16187.394** | 15762.177 | 16708.317 | 197.85 | **1.222%** | single_iteration | 15983.7 | 1 | 1 | **VALID** |
| 128 | **cache** | 30 | **33668.358** | 32898.820 | 35567.091 | 619.64 | **1.840%** | single_iteration | 36689.3 | 1 | 1 | **VALID** |
| 128 | nocache | - | - | - | - | - | - | - | - | - | - | **NOT RUN by decision** |

**All seven prefill configurations are VALID.** `L = 128` cache is the **first VALID observation this project has at that length** - Stage 3's attempt was INVALID at 5.723% - and it has **no baseline of either kind**: no Stage 3 VALID figure to compare against, and no in-session no-cache control, which was declined by decision because it would have added about 26 minutes to the longest-running configuration class on this machine.

**The prefill falsification test - the one DECISION B's in-session control exists to make possible:**

| L | cache (ms) | nocache (ms) | in-session difference | against the 4.4% floor |
|---|---|---|---|---|
| 16 | 3951.321 | 4011.716 | **-1.505%** | **INSIDE** |
| 32 | 7899.536 | 7954.613 | **-0.692%** | **INSIDE** |
| 64 | 16081.705 | 16187.394 | **-0.653%** | **INSIDE** |

**Every VALID length is inside the floor, so the prediction's prefill falsification condition is NOT met** and the correct statement is **"no measurable change"**, with the floor named: 4.4%, the p90 of the Stage 0 run-to-run spread, carrying its three standing qualifications - small-M and decode-shaped work is materially noisier, it was measured under irreducible background load, and it has not been re-derived since Stage 0 while the process set has changed (section 8 W5), which this session's heavier set makes more pointed rather than less.

**A diagnostic observation, labelled DIAGNOSTIC and not a claim.** The cache member is the faster of the pair at **all three** lengths, by 0.65% to 1.51%. Under pure noise the sign would be expected to vary. The cache only adds stores, so a genuine speedup from adding work is not a mechanism this stage can offer. There is a confound the design does not separate: **within every pair the cache configuration ran first**, so any within-chunk warming or load drift folds into the same sign. The paired design controls for drift **between** pairs, not for order **within** a pair. Three pairs is too few to separate the two, no mechanism is proposed, and **the reported conclusion remains "no measurable change" at every length.** An order-reversed repeat of each pair would separate them and is the next stage's to take if it wants the sign.

#### Measurement — DECODE

One step per timed bracket. For the cached configurations the cache was filled with the first c-1 tokens **outside every bracket**, and the cache length was restored to c-1 **outside the bracket** before each timed iteration, so every sample measures the same context (`cache_state_construction: "length restored outside bracket"`).

| c | path | samples | median (ms) | min | max | std dev (ms) | sd as % of median | construction | probe (ms) | R | repeat_applied | cached positions | cache bytes | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | **cache** | 30 | **242.571** | 237.887 | 414.381 | 34.38 | **14.173%** | single_iteration | 237.3 | 1 | 1 | 31 | 2,359,296 | **INVALID** |
| 32 | nocache | 30 | **6716.263** | 6562.532 | 7164.924 | 164.02 | **2.442%** | single_iteration | 6880.7 | 1 | 1 | - | 0 | **VALID** |
| 64 | **cache** | 30 | **251.123** | 248.854 | 440.343 | 47.88 | **19.065%** | single_iteration | 250.5 | 1 | 1 | 63 | 4,718,592 | **INVALID** |
| 64 | nocache | 30 | **13726.551** | 13506.509 | 14315.826 | 184.62 | **1.345%** | single_iteration | 14265.3 | 1 | 1 | - | 0 | **VALID** |
| 128 | **cache** | 30 | **260.445** | 253.897 | 444.943 | 37.71 | **14.477%** | single_iteration | 254.3 | 1 | 1 | 127 | 9,437,184 | **INVALID** |
| 128 | nocache | 30 | **27815.349** | 27572.212 | 28230.409 | 187.94 | **0.676%** | single_iteration | 27762.9 | 1 | 1 | - | 0 | **VALID** |

**All three CACHED decode configurations are INVALID. All three no-cache controls are VALID.** The cached medians are reported with their INVALID verdict attached everywhere they appear below, and **none was retried, averaged away, or rescued by a robust statistic.**

**The in-session change the cache produces** - computed from an **INVALID** cached median against a **VALID** no-cache median, and labelled so wherever it is used. It is reported as a self-relative ratio against the project's own baseline and as intermediate detail, **never as a headline figure**:

| c | cached (INVALID) | no-cache (VALID) | ratio |
|---|---|---|---|
| 32 | 242.571 ms | 6716.263 ms | **27.69x** |
| 64 | 251.123 ms | 13726.551 ms | **54.66x** |
| 128 | 260.445 ms | 27815.349 ms | **106.80x** |

The ratio roughly doubles with each doubling of context, which is the expected shape: the numerator is near-constant while the denominator grows with c.

#### Measurement — ISOLATED GEMM

Unchanged shapes from Stage 2 and Stage 3: `M = 32, K = 768`, only `N` moving, so the two records differ in the size of the streamed operand and in nothing else. These configurations are not probed by the driver - they are fixed shapes with no fixture row and no engine path - so they carry no construction record, and the driver reported `repeat_applied = 1` for both, which the harness checked.

| shape | N | samples | median (ms) | min | max | std dev (ms) | sd as % of median | repeat_applied | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `attn.c_proj`, B = 2.25 MiB, L3-resident | 768 | 30 | **23.370** | 23.142 | 30.734 | 1.39 | **5.951%** | 1 | **INVALID** |
| `mlp.c_fc`, B = 9.00 MiB, exceeds the 8 MiB L3 | 3072 | 30 | **130.986** | 124.536 | 154.812 | 10.24 | **7.815%** | 1 | **INVALID** |

**Both are INVALID**, where both were VALID in Stage 3 (21.323 ms at 0.272%, 120.753 ms at 0.389%). Same binary, same shapes, same construction; a heavier process set.

#### The decode curve — what the cache did to the shape

The claim Stage 4 exists to test, from `TECHNICAL_SPEC.md` section 3: *decode cost per token drops from growing with sequence length to roughly constant*. `[derived]`, from the medians above:

| path | c = 32 | c = 64 | c = 128 | c=128 / c=32 | per doubling | validity of the endpoints |
|---|---|---|---|---|---|---|
| **cache** | 242.571 | 251.123 | 260.445 | **1.0737** | **1.0362** | INVALID / INVALID / INVALID |
| nocache | 6716.263 | 13726.551 | 27815.349 | **4.1415** | **2.0351** | VALID / VALID / VALID |

**The shape changed, and by how much is measurable even though the cached endpoints are INVALID.** The no-cache path grows at **2.0351 per doubling** across VALID endpoints, which reproduces Stage 3's measured 2.0014 per doubling on the same configurations to within 1.7%. The cached path grows at **1.0362 per doubling** from INVALID endpoints. The cache therefore replaced a term that doubles with context by one that rises a few percent across a factor of four in context.

**What cannot be claimed from this.** The cached ratio **1.0737** rests on two INVALID medians, so it is **not** a VALID measurement of the curve and it is not reported as one. What the cached configurations do establish, because dispersion does not move a median by a factor of four, is that the context-proportional term is **gone**: a surviving O(c) term would have put c = 128 near four times c = 32, and the measured separation between those medians is 7.4%, not 314%.

#### Derived throughput, on NAMED FLOP bases, as fractions of the measured scalar ceiling

The ceiling is the **measured** scalar FP32 figure **8.565 GFLOP/s** (`HARDWARE.md` section 2, Stage 0b reference run 2). The vectorised AVX2 peak of 48.411 GFLOP/s does **not** appear as a denominator anywhere in this stage - this is scalar code and that ceiling is Stage 6's framing (section 8 W8) - and neither does the single-channel theoretical 23.464 GB/s, which is derived from part numbers (section 8 W7).

**The attention FLOP basis is named at both ends, per section 8 W13**, because the two paths count it differently and a comparison across them is not like for like unless the basis is stated: the **cached** step performs one query row over c keys, so there is no discarded upper triangle and its performed-FLOP count is **36,864 x c**; the **no-cache** path computes the full T x T square and masks it afterwards, so its count stays on Stage 2's **36,864 x c^2** basis. W13 is not changed by this stage.

| configuration | FLOP basis | FLOPs | GFLOP/s | fraction of 8.565 | verdict |
|---|---|---|---|---|---|
| prefill L = 16, cache | L x 247,064,064 + 36,864 x L^2 | 3,962,462,208 | 1.0028 | **0.1171** | VALID |
| prefill L = 16, nocache | L x 247,064,064 + 36,864 x L^2 | 3,962,462,208 | 0.9877 | **0.1153** | VALID |
| prefill L = 32, cache | L x 247,064,064 + 36,864 x L^2 | 7,943,798,784 | 1.0056 | **0.1174** | VALID |
| prefill L = 32, nocache | L x 247,064,064 + 36,864 x L^2 | 7,943,798,784 | 0.9986 | **0.1166** | VALID |
| prefill L = 64, cache | L x 247,064,064 + 36,864 x L^2 | 15,963,095,040 | 0.9926 | **0.1159** | VALID |
| prefill L = 64, nocache | L x 247,064,064 + 36,864 x L^2 | 15,963,095,040 | 0.9861 | **0.1151** | VALID |
| prefill L = 128, cache | L x 247,064,064 + 36,864 x L^2 | 32,228,179,968 | 0.9572 | **0.1118** | VALID |
| decode c = 32, **cache** | 247,064,064 + 36,864 x c (causal row) | 248,243,712 | 1.0234 | **0.1195** | INVALID |
| decode c = 64, **cache** | 247,064,064 + 36,864 x c (causal row) | 249,423,360 | 0.9932 | **0.1160** | INVALID |
| decode c = 128, **cache** | 247,064,064 + 36,864 x c (causal row) | 251,782,656 | 0.9667 | **0.1129** | INVALID |
| decode c = 32, nocache | c x 169,869,312 + 36,864 x c^2 + 77,194,752 (full square) | 5,550,761,472 | 0.8265 | **0.0965** | VALID |
| decode c = 64, nocache | c x 169,869,312 + 36,864 x c^2 + 77,194,752 (full square) | 11,099,825,664 | 0.8086 | **0.0944** | VALID |
| decode c = 128, nocache | c x 169,869,312 + 36,864 x c^2 + 77,194,752 (full square) | 22,424,446,464 | 0.8062 | **0.0941** | VALID |
| isolated GEMM N = 768 | 2 x M x N x K | 37,748,736 | 1.6153 | **0.1886** | INVALID |
| isolated GEMM N = 3072 | 2 x M x N x K | 150,994,944 | 1.1528 | **0.1346** | INVALID |

**This is the ceiling-referenced figure for this CPU stage**, and it is the only kind available: **no headline ratio from `BENCHMARK_PROTOCOL.md` section 7 is quoted, because section 7 defines the decode headline against measured GPU bandwidth and the prefill headline against cuBLAS at the engine's shapes, and this stage's engine runs on the CPU.** The results file records that absence with the same reason.

**The reading.** A cached decode step achieves **0.1129 to 0.1195** of the measured scalar ceiling, and a prefill token achieves **0.1118 to 0.1174** - the same band. That is the stage's substantive efficiency result: the cached step is not a different kind of work from a prefill token, it is the same work done once instead of c times. The no-cache decode path sits lower, at **0.0941 to 0.0965**, which is the Stage 2 and Stage 3 observation reproduced.

#### Comparison against the Stage 3 baseline, with the W14 shift stated alongside

`--compare bench/results/stage3/stage3_harness.json`. **Every figure in this table carries the section 8 W14 cross-session shift**, and the comparison does not control the process set because the environment fingerprint has no process-set field - which is W14's entire subject. Each comparison record in the results file carries `process_set_gated: false` and a pointer to W14 so the limitation travels inside the artifact.

| configuration | Stage 3 (ms) | Stage 4 (ms) | change | the check's verdict |
|---|---|---|---|---|
| prefill|L=16 | 3537.051 | 4011.716 | **+13.420%** | REGRESSION |
| prefill|L=32 | 7058.383 | 7954.613 | **+12.697%** | REGRESSION |
| prefill|L=64 | 14121.470 | 16187.394 | **+14.630%** | REGRESSION |
| decode|c=32 | 5925.150 | 6716.263 | **+13.352%** | REGRESSION |
| decode|c=128 | 23735.411 | 27815.349 | **+17.189%** | REGRESSION |

**Five configurations were flagged REGRESSION and not one of them is a code regression.** The no-cache path is the Stage 2 path unchanged, and this stage proved that rather than asserting it: its prefill logits are **bit-identical to the pre-change engine's dump** (FNV-1a 64 `0x344664b8843fccf8`, asserted in `tests/test_model.c`), and `src/gemm/gemm_naive.c`, `src/gemm/gemm.h` and the compiler flags are byte-identical to `main`. The engine did not get slower; **this session is slower**, by +12.7% to +17.2% on code that computes the same floats.

The direction is consistent with the process set: ten editor processes against Stage 3's one, Logitech Options+ never closed, OneDrive resident, on a machine with 8,012 MB of RAM that killed the run three times for want of memory. **That is a candidate, not an established cause** - no process-set field exists in the fingerprint to gate on and no CPU counters were collected, so the attribution is not established at the counter level.

**W14's own table, recomputed from the two results files** as the decision required, rather than copied from the prompt that carried it:

| configuration | Stage 2 (ms) | Stage 3 (ms) | change | note |
|---|---|---|---|---|
| prefill L = 32 | 7765.285 | 7058.383 | **-9.10%** | prior INVALID |
| prefill L = 64 | 15520.584 | 14121.470 | **-9.01%** |  |
| decode c = 32 | 6561.243 | 5925.150 | **-9.69%** |  |
| decode c = 64 | 13193.109 | 11858.971 | **-10.11%** | current INVALID |
| decode c = 128 | 26842.550 | 23735.411 | **-11.58%** |  |
| isolated GEMM N = 768 | 24.056 | 21.323 | **-11.36%** |  |
| isolated GEMM N = 3072 | 124.149 | 120.753 | **-2.74%** | prior INVALID |

**Two cells differ in the last digit from the figures the stage prompt carried** - prefill L = 64 at **-9.01%** against -9.02%, and decode c = 32 at **-9.69%** against -9.70%. The files win and the difference is stated. The remaining five agree exactly.

**Where no comparison was made, and why.** No comparison is reported at **prefill L = 128** or **decode c = 64**, because no Stage 3 baseline exists at either - both were INVALID there. The in-session no-cache decode at c = 64 measured here (13726.551 ms, VALID) is a **Stage 4 measurement and is labelled as such**; it does not retroactively become a Stage 3 baseline. Every cache-path configuration was refused by the comparison because it does not appear in the prior file, which is the correct refusal: **the cached-against-uncached change is this stage's RESULT, reported as that above and labelled, not as a regression against Stage 3.**

#### Every INVALID configuration, named, with what it was and what it was not

**Five of fifteen configurations are INVALID**, each because its standard deviation exceeded 5% of its median. Each is reported as INVALID, **none was retried, none was averaged away, and no diagnostic statistic was used to convert one**:

| configuration | median (ms) | sd as % of median | samples above 1.2x median | max / median | what it was |
|---|---|---|---|---|---|
| `decode|c=32|cache` | 242.571 | **14.173%** | 2 of 30 | 1.71 | a tight baseline punctured by one or more large events |
| `decode|c=64|cache` | 251.123 | **19.065%** | 2 of 30 | 1.75 | a tight baseline punctured by one or more large events |
| `decode|c=128|cache` | 260.445 | **14.477%** | 3 of 30 | 1.71 | a tight baseline punctured by one or more large events |
| `isolated_gemm|M=32,N=768,K=768` | 23.370 | **5.951%** | 1 of 30 | 1.32 | a tight baseline punctured by one or more large events |
| `isolated_gemm|M=32,N=3072,K=768` | 130.986 | **7.815%** | 0 of 30 | 1.18 | broad dispersion with no single dominant spike |

**The mechanism, quantified, and it is W3's.** Section 8 W3 derives that a single interruption of relative size `delta` hitting k of n samples produces `relative sd = delta x sqrt(k(n-k)/(n(n-1)))`. For the three cached decode configurations:

| configuration | largest sample / median | k (samples above 1.2x) | W3's predicted sd% | measured sd% |
|---|---|---|---|---|
| `decode|c=32|cache` | 1.71 | 2 | 18.0% | **14.173%** |
| `decode|c=64|cache` | 1.75 | 2 | 19.1% | **19.065%** |
| `decode|c=128|cache` | 1.71 | 3 | 21.6% | **14.477%** |

The closed form is evaluated here with `delta` taken from the LARGEST sample, which assumes all k interruptions are that size, so it is an upper estimate whenever k > 1. It matches almost exactly where k = 2 and both spikes are near the maximum (`decode|c=64|cache`: 19.1% predicted against 19.065% measured) and **overpredicts** where the k samples are spread below it (`decode|c=128|cache`: 21.6% against 14.477%). Read as an upper bound rather than a point estimate, it brackets every measured figure - which is the clearest evidence this project has that **W3 describes the real mechanism** rather than a worst case. What is new is the SCALE: W3 was derived from M = 1 GEMM configurations with medians of **30 to 90 microseconds**, and it is reproduced here at brackets of **242 to 260 milliseconds** - four orders of magnitude longer. The interruption on this machine is large in absolute terms (roughly 180 ms), not merely large relative to a microsecond kernel.

**W3's own remedy was not available, and that is the finding.** W3 prescribes a batched construction for configurations below the 10 ms per-sample floor. Every cached decode probe was **237 to 254 ms**, more than twenty times the floor, so the selector correctly chose `single_iteration` with `R = 1` - there is nothing to batch, and batching anyway would have amortised away the very per-iteration distribution the 5% rule tests. **A configuration can therefore be far above W3's floor and still fail the dispersion rule**, when the machine's absolute interruption magnitude is a large fraction of the bracket. That is a gap in W3's framing which Stage 4 found by measurement, and it is recorded here rather than papered over: the honest options are a quieter machine or a longer bracket, and this session could obtain neither - the operator was remote and could not close a single process.

**What the INVALID verdicts do and do not cost.** They cost the stage a VALID *latency* for the cached decode step, and with it any VALID figure for the decode curve. They do **not** cost the stage its correctness result, which is bit-exact and independent of timing, and they do not cost the *order of magnitude* of the change: dispersion of 14 to 19% cannot move a median by the factor of 27 to 107 that separates the cached path from its in-session control. The structural claim - that decode cost stopped growing with context - survives; the precise latency does not.

#### The regression comparison's own output, every refusal and its reason

| configuration | comparable | reason |
|---|---|---|
| `prefill|L=16` | yes | compared: +13.420%, REGRESSION |
| `prefill|L=32` | yes | compared: +12.697%, REGRESSION |
| `prefill|L=64` | yes | compared: +14.630%, REGRESSION |
| `decode|c=32` | yes | compared: +13.352%, REGRESSION |
| `decode|c=128` | yes | compared: +17.189%, REGRESSION |
| `prefill|L=16|cache` | **no** | the configuration does not appear in the prior file |
| `prefill|L=32|cache` | **no** | the configuration does not appear in the prior file |
| `prefill|L=64|cache` | **no** | the configuration does not appear in the prior file |
| `decode|c=32|cache` | **no** | the configuration does not appear in the prior file |
| `decode|c=64|cache` | **no** | the configuration does not appear in the prior file |
| `decode|c=64` | **no** | the prior configuration is INVALID |
| `decode|c=128|cache` | **no** | the configuration does not appear in the prior file |
| `isolated_gemm|M=32,N=768,K=768` | **no** | the current configuration is INVALID |
| `isolated_gemm|M=32,N=3072,K=768` | **no** | the current configuration is INVALID |
| `prefill|L=128|cache` | **no** | the configuration does not appear in the prior file |

Ten refusals and five comparisons. **The refusals are the check working, not failing**: seven cache-path configurations have no counterpart in the prior file because the cache did not exist then; `decode|c=64` is refused because the prior side is INVALID; and both isolated GEMMs are refused because the current side is INVALID. Every refusal and every comparison carries `process_set_gated: false` and the W14 pointer.

**The W3 confirmation gate reported `all_confirmed: true` over all fifteen configurations**, with `repeat_applied == 1` on every one. No bracket was divided by any R, the batched path was not exercised by a timed run (`batched_path_exercised_by_a_timed_run: false`), and the probe range across the set was **237.3 ms to 36,689.3 ms** - the whole set sits above the 10 ms floor by between 24x and 3,669x.

#### Within-run drift and dispersion character — DIAGNOSTIC

**DIAGNOSTIC ONLY.** These figures characterise the distribution. They never replace a reported median and they never convert an INVALID run into a valid one. Recomputed here from the retained raw per-sample arrays rather than read from the driver's metadata, because section 8 **W15** records that the driver's metadata arrays silently dropped one diagnostic counter at Stage 4's field count; the stored first- and second-half medians were verified against this recomputation and agree exactly.

| configuration | first-half median (ms) | second-half median (ms) | drift | verdict |
|---|---|---|---|---|
| `prefill|L=16|cache` | 3960.354 | 3950.972 | **-0.237%** | VALID |
| `prefill|L=16` | 4025.653 | 3995.623 | **-0.746%** | VALID |
| `prefill|L=32|cache` | 7872.365 | 8002.151 | **+1.649%** | VALID |
| `prefill|L=32` | 8002.688 | 7902.281 | **-1.255%** | VALID |
| `prefill|L=64|cache` | 16076.453 | 16094.358 | **+0.111%** | VALID |
| `prefill|L=64` | 16190.315 | 16184.473 | **-0.036%** | VALID |
| `prefill|L=128|cache` | 33647.820 | 33820.292 | **+0.513%** | VALID |
| `decode|c=32|cache` | 243.477 | 241.316 | **-0.888%** | INVALID |
| `decode|c=32` | 6636.554 | 6780.688 | **+2.172%** | VALID |
| `decode|c=64|cache` | 250.982 | 251.264 | **+0.113%** | INVALID |
| `decode|c=64` | 13672.716 | 13744.746 | **+0.527%** | VALID |
| `decode|c=128|cache` | 260.759 | 257.852 | **-1.115%** | INVALID |
| `decode|c=128` | 27870.310 | 27768.077 | **-0.367%** | VALID |
| `isolated_gemm|M=32,N=768,K=768` | 23.603 | 23.277 | **-1.381%** | INVALID |
| `isolated_gemm|M=32,N=3072,K=768` | 134.815 | 129.304 | **-4.087%** | INVALID |

**No configuration drifts in a way that suggests sustained thermal decay**: the largest magnitude is -4.087% on the isolated GEMM at N = 3072, which is also INVALID on dispersion, and every VALID configuration drifts by less than 2.2%. The dispersion failures are therefore **spikes, not trends** - which the per-sample counts above confirm directly - and the distinction matters because a trend would implicate the machine's thermal state while a spike implicates the process set.

#### CPU frequency during the set, from the live source only

The only live CPU frequency source on this machine is the PDH counter `\Processor Information(_Total)\% Processor Performance`, at about 12.5 microseconds per probe (section 8 W6, established by Stage 0b). The other two sources sampled are recorded as static and are **not** used as measurements: `% Performance Limit` and `\Processor Information(_Total)\Processor Frequency`. **No CPU package temperature source exists on this machine**, so the frequency axis is the only one available.

| chunk | samples | % Processor Performance: min | median | max | % Performance Limit (min/median/max) | Processor Frequency (static) |
|---|---|---|---|---|---|---|
| `p16` | 14631 | 131.84 | **165.32** | 200.04 | 100.00 / 100.00 / 100.00 | 2496 MHz |
| `p32` | 28795 | 114.95 | **160.57** | 195.71 | 100.00 / 100.00 / 100.00 | 2496 MHz |
| `p64` | 58287 | 124.50 | **159.65** | 188.51 | 100.00 / 100.00 / 100.00 | 2496 MHz |
| `d32` | 13620 | 118.79 | **163.73** | 189.41 | 100.00 / 100.00 / 100.00 | 2496 MHz |
| `d64` | 27232 | 124.93 | **158.35** | 199.34 | 100.00 / 100.00 / 100.00 | 2496 MHz |
| `d128` | 55270 | 103.02 | **156.04** | 203.27 | 100.00 / 100.00 / 100.00 | 2496 MHz |
| `gemm` | 494 | 122.15 | **161.37** | 199.06 | 100.00 / 100.00 / 100.00 | 2496 MHz |
| `p128` | 60531 | 111.44 | **156.75** | 200.75 | 100.00 / 100.00 / 100.00 | 2496 MHz |

**Two readings matter.** First, `% Processor Performance` is a percentage of the 2496 MHz nominal, so a median of roughly 156 to 165% corresponds to about **3.9 to 4.1 GHz** - the part was in turbo throughout and never near its nominal clock. Second, and more useful as a run condition: **`% Performance Limit` held at exactly 100.00 across all 253,861 samples of all eight chunks**, minimum and maximum alike. **No CPU performance limit was asserted at any point during 2.28 hours of sustained single-core load.** That is positive evidence against a CPU thermal or power throttle as an explanation for either the dispersion failures or the cross-session shift, and it is the strongest statement this machine's instrumentation can make on the subject - it is not evidence about the GPU, whose clock was locked, nor about package temperature, which cannot be read here at all.

The trace itself is a **run condition and is not committed**; it is written to `build/stage4_traces/` and the summary above is what the entry records.

#### Counters, and what is therefore not established

**No Nsight Compute counters.** `BENCHMARK_PROTOCOL.md` section 6 requires them for GPU stages from Stage 7 onward. This stage writes no kernel and its engine runs on the CPU.

**No CPU hardware counters. `xperf` is available on this machine and was deliberately not used**, following the Stage 2 and Stage 3 precedent (section 8 **W4**, whose Status this stage does not change). `xperf -pmcsources` enumerates the live Comet Lake PMU, so the capability exists. The consequence is stated rather than hidden: **the counters that would distinguish an issue-rate limit from a cache-miss limit in the cached decode step were not collected, so the decode gap is unestablished at the counter level.**

#### Gap

The prediction was opened for the first time at this point, having been sealed since commit `b8543b4`. It is addressed as written; the derivation block is used only where it explains the prediction's reasoning.

**Predicted against measured, every configuration the prediction names.**

| configuration | predicted | measured | ratio | verdict of the measurement |
|---|---|---|---|---|
| decode c = 32, cached | 232 ms | **242.571 ms** | **1.046** | INVALID |
| decode c = 64, cached | 233 ms | **251.123 ms** | **1.078** | INVALID |
| decode c = 128, cached | 235 ms | **260.445 ms** | **1.108** | INVALID |
| prefill L = 16, cache | 3537 ms | **3951.321 ms** | **1.117** | VALID |
| prefill L = 32, cache | 7058 ms | **7899.536 ms** | **1.119** | VALID |
| prefill L = 64, cache | 14121 ms | **16081.705 ms** | **1.139** | VALID |
| prefill L = 128, cache | 28.5 s | **33.668 s** | **1.181** | VALID |
| decode curve, c=128 / c=32 | 1.014 | **1.0737** | **1.059** | both endpoints INVALID |
| decode curve, per doubling | 1.007 | **1.0362** | **1.029** | both endpoints INVALID |
| KV footprint per position | 73,728 B | **73,728 B** | **1.000** | asserted |
| KV footprint at c = 1024 | 75,497,472 B | **75,497,472 B** | **1.000** | asserted |
| KV footprint as % of parameters | 15.2% | **15.168%** | **0.998** | derived |
| prefill divergence vs oracle | 7.019043e-04 | **7.019043e-04** | **1.000** | measured |
| D2 under the corrected rule | 2.3e-03 | **2.3e-03** | **1.000** | derived |
| cached vs prefill logits | bit for bit | **bit for bit**, 0 of 12,061,680 | exact | measured |
| cached vs no-cache step | bit for bit | **bit for bit**, 0 differing | exact | measured |

**The prediction's own falsification conditions, each answered.**

| condition the prediction set | met? | the measurement |
|---|---|---|
| *"The efficiency fraction was merely imprecise"* if every VALID cached decode median is within a factor of 2 of 232 ms, i.e. 116-464 ms | **cannot be evaluated as written** | **No cached decode median is VALID.** All three INVALID medians - 242.571, 251.123, 260.445 ms - do lie inside 116-464 ms, so the condition would have been satisfied had they been VALID, but the condition as written quantifies over VALID medians and there are none |
| The model is wrong if the c=128 / c=32 cached ratio **exceeds 1.10** | **NOT met** | measured **1.0737**, below 1.10 - though from INVALID endpoints. No context-proportional work survived: a cache that was not consulted, or an O(c) per-step copy, would have produced a ratio near 4 |
| The model is wrong if any VALID cached decode median **exceeds 696 ms** | **NOT met** | the largest cached median is 260.445 ms, and even the largest individual SAMPLE across all three configurations is 444.943 ms. The memory route did not bind and work was not being recomputed |
| The model is wrong if any VALID cached decode median is **below 77 ms** | **NOT met** | the smallest cached median is 242.571 ms and the smallest individual sample is 237.887 ms. Nothing was accidentally vectorised |
| The model is wrong if in-session paired prefill with and without cache writes differs by more than **4.4%** at any VALID length | **NOT met** | -1.505%, -0.692%, -0.653% at L = 16, 32, 64, all three VALID on both members. The largest is a third of the floor |

**So: no falsification condition was met, and the one that would have confirmed the fraction cannot be evaluated because the configurations it quantifies over are INVALID.** That is the stage's central honest result and it is stated before any of the agreement below.

**The mechanism, quantified where the measurements allow it.**

*1. The efficiency fraction the prediction chose was very nearly right, and for the reason it gave.* The prediction's central fraction was **f = 0.125**, argued from the claim that a cached decode token has the per-token operation MIX of a prefill token rather than of an uncached decode step — the tied head is 31 percent of both — pulled down slightly because at M = 1 the L3-sized `c_attn` and `attn.c_proj` weights lose cross-row reuse. Measured, from the INVALID medians: **f = 0.1195, 0.1160, 0.1129** at c = 32, 64, 128. The prediction's upper bracket was the prefill fraction 0.1320 and its lower was the uncached decode fraction 0.1098; the measured values sit inside that bracket, nearer the centre than either edge.

*2. The structural half of that argument is confirmed independently of the session's speed.* The prediction's reasoning rests on the cached step having prefill's operation mix. Measured in this session, a prefill token achieves **0.1159 to 0.1174** of the ceiling and a cached decode step achieves **0.1129 to 0.1195** — the same band — while the uncached decode path sits at **0.0941 to 0.0965**. The cached step behaves like a prefill token and not like the thing it replaced, which is what the prediction claimed and what justified reusing prefill's fraction rather than inventing one.

*3. The absolute latencies are high by a factor the prediction anticipated, and the prediction named the mechanism.* Every measured figure is 4.6 to 18.1 percent above its predicted value. The prediction's derivation states that the Stage 2 and Stage 3 sessions measured the same binary about 10 percent apart and that *"these figures carry a condition uncertainty of that size on top of the fraction"*. This session is **+12.7 to +17.2 percent** slower than Stage 3 on bit-identical no-cache code, so the predicted values — taken from Stage 3's medians for prefill, and derived against Stage 3's fraction for decode — are high by almost exactly the condition shift the prediction warned about. **The prediction's model was better than its absolute numbers, and it said so in advance.** Had the in-session control not been built (DECISION B), this would have been indistinguishable from the cache costing prefill 13 percent, which is precisely the failure the control was added to prevent.

*4. Prefill did not move, as predicted, and the bound the prediction used holds.* The prediction bounded the cache's added cost at prefill as **at most 0.019 percent of the bytes the pass already moves**, without using any DRAM bandwidth figure — 9,437,184 B of cache stores at L = 128 against at least 48,752,885,760 B the pass re-walks. Measured in-session: **-1.505, -0.692, -0.653 percent**, every one inside the 4.4 percent floor. A bound expressed against quantities that are counted survived a session whose absolute speed moved 13 percent.

*5. The residual-growth prediction was directionally right and quantitatively loose.* Predicted **1.014** total across c = 32 to 128, i.e. 1.007 per doubling, from the surviving attention term `36,864 x c` being 1.4 percent of the step. Measured **1.0737** total, **1.0362** per doubling — about five times the predicted residual, though still an order of magnitude below the uncached 2.0351 per doubling. The prediction's claim that the residual would be *"not distinguishable from flat on this machine"* is **borne out, though not for the reason it gave**: at 7.4 percent total the growth is nominally above the 4.4 percent floor and so would be marginally distinguishable, but both endpoints are INVALID at 14 to 19 percent dispersion, and a 14 percent dispersion cannot support a 7 percent claim. The honest reading is that the curve is flat to within what this session can resolve, and that resolving the residual exactly needs a quieter machine.

*6. Bit-exactness was predicted as a structural consequence and is confirmed exactly.* The prediction claimed a cached decode step would reproduce the prefill logits at the same position **bit for bit**, on the grounds that every GEMM row is computed by the same `ijk` inner loop over the same elements in the same order. Measured: **0 differing elements of 12,061,680**, maximum absolute difference **0**, across all four D3 rows; and the cached step against the no-cache step likewise 0 differing. The prefill divergence against the oracle is **unchanged at 7.019043e-04**, exactly as predicted, and therefore so is the D2 lower bound — which is why the re-derived value came out at the predicted **2.3e-03**. Predicting exactness in advance is what made these comparisons informative rather than a tolerance pass.

*7. The W3 check in the prediction was right about R and wrong about what that guaranteed.* The prediction stated that a 232 to 235 ms step is more than twenty times the 10 ms floor, so **R = 1 everywhere and the batched path does not fire**. Correct: R = 1 was selected for all fifteen configurations, probes ran 237.3 ms to 36,689.3 ms, and the C-side loop was not built. But the prediction — and W3 itself — treated being above the floor as sufficient for a valid measurement, and **it is not**: three configurations comfortably above the floor failed the dispersion rule anyway, because this machine's absolute interruption is roughly 180 ms against a 250 ms bracket. That is the one place the prediction's framing, rather than its arithmetic, was wrong, and it is recorded above as a gap in W3's framing.

**Where the evidence does not establish a cause.** Two things are unexplained and are left unexplained rather than attributed:

- **The cross-session shift of +12.7 to +17.2 percent** on bit-identical code. The process set is the leading candidate and this session's was measurably heavier, but the environment fingerprint has no process-set field, so nothing gates on it and nothing establishes it. Carried as section 8 **W14**, owned by Stage 5.

- **The cached decode step's cost structure.** The counters that would distinguish an issue-rate limit from a cache-miss limit in the cached decode step were not collected (section 8 W4). No choice is made between those candidate causes here. The prediction's own reasoning — that the step is issue-rate bound because a full weight read would take about 7 ms even at L3 bandwidth against a 235 ms compute time, so memory would bind only below about 2.14 GB/s of sustained DRAM bandwidth — **remains an argument and not a measurement**, because no measured DRAM figure exists on this machine (section 8 W1) and this stage did not produce one, did not estimate one, and quotes no bytes-per-second figure as a bandwidth result.

**The one-sentence verdict.** The cache works and is provably exact; it removed the context-proportional term from decode, turning a step that grew 2.0351x per doubling into one that grew 1.0362x per doubling, for a self-relative in-session improvement of 27.69x to 106.80x; and this session could not certify the resulting latency as VALID, because on a machine it could not quiet, a 250 ms bracket is too short to survive a 180 ms interruption.

#### Deferred-work items this stage touched

| Item | Owned by this stage? | What this stage did |
|---|---|---|
| **W3** — the timing construction | **Yes** | **Still OPEN.** Status updated. R = 1 was selected for all fifteen configurations (probe range 237.3 ms to 36,689.3 ms), so the C-side batched bracket was again not needed and was **not built speculatively**. What was built instead is the refusal: the driver accepts `--repeat 1` and records it, **refuses `--repeat N` above 1** with a non-zero exit naming W3, and refuses any unrecognised flag; every record carries `repeat_requested` and `repeat_applied`; the harness **hard-fails the whole run** before dividing any bracket by R unless the driver confirmed `repeat_applied == R`, checked even at R = 1; and a probe selecting R above 1 stops the run before the timed pass. All four are unit-tested against stubs that honour, ignore and omit the flag. **The hand-off is stated inside W3's Status: the C-side loop belongs to the first stage that actually selects R above 1** — Stage 5 if it times decode-shaped (M = 1) GEMMs, which at this machine's naive scalar throughput fall below the 10 ms floor. W3's Owning-stage cell was not edited. **Stage 4 also found a limitation in W3's framing** — see the INVALID section: being above the floor is necessary but not sufficient |
| **W14** — the cross-session shift | **Raised by this stage** | **NEW, OPEN.** Every percentage recomputed from the two results files rather than copied; two cells differ in the last digit from the figures carried into the stage and the files win. Owned by **Stage 5**. `bench/harness.py` now writes `process_set_gated: false` and a W14 pointer into **every** comparison and refusal record. Stage 4's own evidence makes the item sharper, not weaker: this session is +12.7 to +17.2 percent slower than Stage 3 on bit-identical code, which is a shift in the **opposite direction and larger** than the one W14 records |
| **W15** — silent metadata truncation | **Raised by this stage** | **NEW, OPEN.** The driver's results-metadata arrays drop entries past sixteen **without a warning**, and Stage 4's five new fields took the decode records to seventeen, losing `diagnostic_drift_n_second_half`. Verified that no measurement, statistic, verdict or raw sample array is affected and that the stored diagnostic values are correct. Owned by **Stage 5**; the fix is to raise the caps and, more importantly, to make the accessors fail loudly |
| **W1** — CPU DRAM bandwidth | No | Unchanged. The cached decode step is the project's first weight-streaming workload, and this stage **did not measure, estimate or fill** CPU DRAM bandwidth, and **quotes no bytes-per-second figure as a bandwidth result** |
| **W4** — CPU counters | No | Unchanged. `xperf` is **available and deliberately unused**, following the Stage 2 and Stage 3 precedent. Stated consequence: **the decode gap is unestablished at the counter level** |
| **W5** — the noise floor | No | **Status NOT changed.** This stage's evidence does bear on its basis and the entry says how: W5's "heavier load stays conservative under lighter load" argument bounds claims **within** a session and does **not** bound a cross-session comparison whose conditions moved — and here they moved toward **heavier**, which is the direction that makes this session slower and W5's reassurance inapplicable to the Stage 3 comparison. The 4.4 percent floor is **USED, never re-derived**; this stage did not re-run the Stage 0 suite and cannot re-derive it |
| **W6** — the Balanced-power-plan re-test | No | **Not run by this stage**, by design: Stage 4 measured a CPU workload against a frozen power-plan fingerprint and changing the plan mid-stage would have voided its own comparison. One dated sentence appended to W6's Status handing it to Stage 5. The power plan was not touched and the fingerprint stayed 25 of 25 clean |
| **W7, W8** — the theoretical bandwidth ceiling and the SIMD ceiling | No | Neither the 23.464 GB/s single-channel figure nor the 48.411 GFLOP/s AVX2 peak appears as a denominator anywhere in this stage |
| **W13** — the attention FLOP basis | No | Unchanged. The basis is **named next to every attention FLOP figure**: the cached step's causal row is `36,864 x c` per step, the no-cache path stays on Stage 2's full-square `36,864 x c^2` |

#### Decisions taken without the operator

The operator was remote for the whole timed phase and could not act on the machine. Four decisions were taken inside the stage's own authority and are recorded here rather than left implicit:

1. **The timed set was measured in eight chunks instead of one invocation**, after the host killed single-invocation attempts twice. Every pair stayed adjacent in time; the order within each workload class is unchanged; each chunk re-verified the fingerprint and the lock. A merge step was added to `bench/harness.py` which recomputes no statistic and refuses to combine parts from different builds or overlapping configuration sets.
2. **The run was moved to a detached process** once it was established that the low-memory condition was pre-existing and would kill any managed background task within minutes. This changed nothing about the measurement; it changed only what could terminate it.
3. **`prefill L = 128` was measured last** rather than before the decode chunks, so that the stage's own claim was banked ahead of the one configuration with no baseline of either kind. It is a singleton with no control, so its position affects no pairing.
4. **`bench/stage2_forward_bench.c` was NOT changed to fix W15 mid-run.** Changing it would have made the committed driver disagree with the binary that had already measured six chunks, and re-measuring the set to recover a dropped diagnostic counter would have cost about two hours for no measurement benefit. The defect is recorded as W15 with the evidence that nothing was corrupted.

#### What this stage established, in one place

- **A KV cache that is provably exact, not merely close.** The cached decode path reproduces the prefill logits **bit for bit** at every position of all four D3 rows — 0 differing elements of 12,061,680 — and the cached step matches the no-cache step bit for bit at every tested context. Exactness was **predicted in advance** as a structural consequence of the shared `ijk` reduction order, which is what makes the result a confirmation rather than a tolerance pass.
- **Decode's context-proportional term is gone.** The no-cache path grows **2.0351x per doubling** of context; the cached path grows **1.0362x per doubling**. In-session the cached step is **27.69x / 54.66x / 106.80x** faster at c = 32 / 64 / 128 — self-relative, against this project's own baseline, computed from an INVALID cached median against a VALID control.
- **Prefill is unchanged by the cache**, measured against an **in-session** control rather than a cross-session one: **−1.505, −0.692, −0.653 percent** at L = 16, 32, 64, every one inside the 4.4 percent floor, so **no measurable change**.
- **The cached step is a prefill token, not a decode step, in its cost structure.** It achieves 0.1129 to 0.1195 of the measured scalar ceiling against prefill's 0.1118 to 0.1174 and the uncached decode path's 0.0941 to 0.0965.
- **D2 re-derived under a corrected rule** to **2.3e-03**, at 3.28x above the set-wide divergence and 3.40x below the set-wide minimum margin, replacing a value that failed its own condition at 1.30x.
- **The first VALID prefill observation at L = 128** this project has: 33668.358 ms at 1.840 percent, where Stage 3's attempt was INVALID at 5.723 percent. It has no baseline of either kind and is labelled as such.
- **Three honest limits.** The cached decode latency is **INVALID at every context** and is not certified; the cross-session comparison shows **+12.7 to +17.2 percent** on bit-identical code and is **unexplained**; and with no CPU counters collected the step's cost structure is **unestablished at the counter level**.

#### What this taught

The stage's method lesson is about predicting a change to the *work* rather than to the *machine*. Recomputing the step's FLOPs under the new algorithm and dividing by the efficiency fraction the unchanged kernels already achieve produced f = 0.125 against a measured 0.1129 to 0.1195 — close, and close for the stated reason, because the fraction was taken from the workload whose operation mix matches rather than from the workload being replaced. The absolute latencies were wrong by 5 to 18 percent, and the prediction had already named why: a condition uncertainty of about 10 percent between sessions, which arrived as 13 to 17 percent.

The sharper lesson is that **a prediction's framing can fail where its arithmetic succeeds.** Every number in the W3 check was right — the step is above the floor, R = 1 is correct, the batched loop was not needed — and the conclusion drawn from them, that the measurement would therefore be valid, was wrong. Being above a per-sample floor bounds the *relative* cost of an interruption only if the interruption's absolute size is known, and on this machine it is roughly 180 ms, which a 250 ms bracket cannot absorb. A floor expressed in milliseconds is not a validity guarantee; it is a guarantee only against interruptions smaller than the floor was chosen for.

And the design lesson, which cost nothing and saved the stage: **keep the superseded path selectable and measure it interleaved.** The cross-session shift this session carried was 13 to 17 percent, three to four times the noise floor the prefill falsification test had to resolve. Against the Stage 3 baseline alone, prefill with cache writes would have looked 13 percent slower and the test would have failed on a condition change rather than on a cache. The in-session control answered the question in the only way that could not be confounded, and it existed because a decision was taken not to delete the old path when the new one worked.

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
