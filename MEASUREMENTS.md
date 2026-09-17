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
| 3 | CPU cache ladder — L1-resident (8 KiB) | **127.38 GB/s** | 1.21% | VALID |
| 3 | CPU cache ladder — L2-resident (256 KiB) | **99.64 GB/s** | 1.62% | VALID |
| 3 | CPU cache ladder — L3-resident (4 MiB) | **62.31 GB/s** | 4.77% | VALID |
| 3 | CPU cache ladder — DRAM-resident | *(no value)* | 5.32%–16.45% | **INVALID in both runs — field left empty** |
| 4 | CPU SIMD peak — vectorised AVX2 | *(no value)* | 5.60% (run 1: 5.86%) | **INVALID in both runs — field left empty** |
| 4 | CPU SIMD peak — scalar reference of the same loop | 8.347 GFLOP/s | 2.51% | VALID |
| 5 | Host-to-device, pinned | **12.589 GB/s** | 1.97% | VALID |
| 5 | Device-to-host, pinned | **10.573 GB/s** | 4.56% | VALID |
| 5 | Host-to-device, pageable | **5.528 GB/s** | 4.37% | VALID |
| 5 | Device-to-host, pageable | **4.952 GB/s** | 4.42% | VALID (run 1 INVALID at 17.93%) |
| 6 | cuBLAS SGEMM — best VALID **prefill** figure observed | **2344.35 GFLOP/s** (ffn_down, M=512, N=768, K=3072) | 0.40% | VALID in run 1; the same configuration was INVALID in run 2 at 8.78%. Best prefill figure VALID in the reference run: **1943.01 GFLOP/s** (attn_output_projection, M=512, N=768, K=768, 4.51%) |
| 6 | cuBLAS SGEMM — **decode** (M=1), reported separately and never merged with prefill | ffn_up **72.64**, ffn_down **61.66**, lm_head **85.13** GFLOP/s | 3.40% / 1.86% / 1.66% | VALID. qkv_projection and attn_output_projection at M=1 were **INVALID in both runs** and have no value |
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
