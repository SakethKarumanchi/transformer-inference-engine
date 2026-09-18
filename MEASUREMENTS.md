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
