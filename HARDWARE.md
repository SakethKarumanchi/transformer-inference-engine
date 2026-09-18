# HARDWARE.md

*Every prediction in this project cites this file, and the Stage 10 performance model consumes it directly as input. Nothing here may be filled from memory or a spec sheet. Vendor spec sheets state theoretical peaks real code never reaches; a model built on theoretical numbers is wrong by construction.*

**Status: POPULATED by Stage 0 on 2026-09-17.** Sections 1, 2, 4 and 5 are filled from runs that happened on this machine. Two fields are deliberately **empty** because their measurements came back INVALID under the protocol and no value may be substituted: CPU measured DRAM bandwidth (§2) and CPU measured peak FP32 SIMD (§2). Each states its reason in the cell. Section 6 remains unpopulated because no secondary device was used.

---

## 0. Tagging

- `[queried]` — returned by a runtime API (`cudaGetDeviceProperties`, `lscpu`)
- `[measured]` — produced by a microbenchmark in this repo; cite which one
- `[spec]` — vendor documentation ⚠️ theoretical, never used in a prediction or in the model
- `[derived]` — computed from the above; show the arithmetic

---

## 1. Primary measurement device

**All headline numbers come from one GPU.** Mixing devices inside a comparison invalidates it.

| Field | Value | Tag |
|---|---|---|
| Primary GPU | NVIDIA GeForce GTX 1650 Ti (TU117) | `[queried]` |
| Compute capability | **7.5 (sm_75)** — `cudaGetDeviceProperties` major.minor = 7.5 and `nvidia-smi --query-gpu=compute_cap` = 7.5; the two agree | `[queried]` |
| SM count | 16 (`multiProcessorCount`) | `[queried]` |
| CUDA cores | 1024 = 16 SM x 64 FP32 lanes/SM. CUDA exposes no core-count property; lanes/SM comes from the vendor compute-capability table `[spec]` | `[derived]` |
| Base / boost clock | rated clock 1485 MHz (`cudaDeviceGetAttribute(cudaDevAttrClockRate)` = 1485000 kHz); max SM clock 2100 MHz (`nvidia-smi`). Base clock is not exposed by any API queried here, so it is not recorded. **Every Stage 0 figure was taken at a locked 1365 MHz** — see 5.1 | `[queried]` |
| VRAM total | 4294639616 B = 4095.5 MiB (`totalGlobalMem`); `nvidia-smi` reports 4096 MiB | `[queried]` |
| Memory bus width | 128 bit | `[queried]` |
| Theoretical peak bandwidth | 192.032 GB/s = 2 transfers/clock x 6001 MHz (`cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate)` = 6001000 kHz) x 128 bit / 8 | `[derived]` from `[queried]` values ⚠️ never in a prediction |
| **Measured achievable bandwidth** | **170.882 GB/s**, std dev 2.54% of median, VALID. `gpu_bandwidth` (microbenchmark 1): float4 grid-stride device-to-device copy, 268435456 B per buffer, working set 536870912 B = 512x the queried 1048576 B L2, CUDA-event timed. Reference run 2; the identical configuration was INVALID in run 1 at 5.84% | `[measured]` |
| Bandwidth efficiency (measured ÷ theoretical) | **0.8899 (88.99%)** = 170.882 ÷ 192.032. Corroborated by Nsight Compute on the same kernel: `dram__bytes_read.sum.per_second` 87.152 GB/s + `dram__bytes_write.sum.per_second` 87.025 GB/s = 174.18 GB/s counter-side | `[derived]` |
| Theoretical peak FP32 | 3041.28 GFLOP/s at the 1485 MHz rated clock = 1024 x 1.485e9 x 2; 4300.80 GFLOP/s at the 2100 MHz max clock. Neither is the operating point: at the locked 1365 MHz the ceiling is 2795.52 GFLOP/s = 1024 x 1.365e9 x 2 | `[derived]` from `[queried]` clocks and `[spec]` lanes/SM ⚠️ |
| **Measured peak FP32** | **2786.49 GFLOP/s**, std dev 2.20% of median, VALID. `gpu_fp32_peak` (microbenchmark 2): 32 independent FMA chains held in registers, 32768 inner iterations, grid 64 x block 256, no global memory traffic in the inner loop. 2786.49 ÷ 2795.52 = **99.68%** of the locked-clock ceiling | `[measured]` |
| Tensor cores present | **Yes — a tensor pipe is present and executes HMMA. This contradicts the expectation recorded below.** `cudaDeviceProp` contains **no tensor-core field**, so this was settled empirically rather than inferred from the product name: a WMMA 16x16x16 kernel compiled `-arch=sm_75`, ran, and returned all 256 elements exactly 32.0; Nsight Compute on that kernel reported `sm__inst_executed_pipe_tensor_op_hmma.sum` = 4 and `sm__pipe_tensor_cycles_active.sum` = 512. **No throughput figure was measured and none is claimed** — 512 cycles for 4 instructions cannot distinguish full-rate silicon at very low occupancy from a reduced implementation | `[measured]`; no `[queried]` property exists |
| L2 cache size | 1048576 B (1 MiB) | `[queried]` |
| Shared memory per block | 49152 B (48 KiB); `sharedMemPerBlockOptin` 65536 B | `[queried]` |
| Shared memory per SM | 65536 B (64 KiB) | `[queried]` |
| Max threads per SM | 1024 | `[queried]` |
| Max blocks per SM | 16 | `[queried]` |
| Registers per SM | 65536 (`regsPerMultiprocessor`); `regsPerBlock` also 65536 | `[queried]` |
| Warp size | 32 | `[queried]` |
| CUDA toolkit / driver version | toolkit nvcc 13.1.80; driver 591.44; VBIOS 90.16.51.00.14 | `[queried]` |

### Tensor cores

The GTX 16-series (Turing TU117) is expected to ship without tensor cores or RT cores. Confirm via `cudaGetDeviceProperties` rather than assuming. If confirmed, Stage 11 is justified against cuBLAS SGEMM and any tensor-core comparison is a labelled secondary-device experiment.

**Stage 0 correction, measured 2026-09-17 — the expectation above is wrong for this die.** `cudaGetDeviceProperties` cannot answer the question at all: the struct has no tensor-core field, and CUDA exposes no API for one. Determined empirically instead. A WMMA 16x16x16 HMMA kernel compiled for `sm_75`, launched, and produced numerically exact results (all 256 elements exactly 32.0, zero wrong). Profiled, it reported `sm__inst_executed_pipe_tensor_op_hmma.sum` = 4 and `sm__pipe_tensor_cycles_active.sum` = 512. **A tensor pipe exists on this die and executes `hmma`.** What this does *not* establish: any throughput figure. 512 cycles for 4 instructions is 128 cycles per instruction, far off full-rate silicon, and these counters cannot separate full-rate hardware at very low occupancy from a reduced implementation. No tensor-core throughput number was measured and none may be quoted. Separately, `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)` returns SUCCESS on this card, so math mode is **not** a valid tensor-core test — recorded so it is not retried. **Consequence for Stage 11:** its justification against cuBLAS SGEMM was made conditional on the absence of tensor cores, and that condition does not hold. Stage 11 must revisit its framing; it is not re-framed here. The project prohibition on `mma`, `ldmatrix` and `cp.async` is unaffected and still absolute.

### Compute capability gate

Record compute capability explicitly. **sm_75 means `cp.async`, `ldmatrix`, and `mma` intrinsics are unavailable** — these require sm_80 or newer. Stage 9 flash attention uses plain shared-memory staging. Any prompt proposing those intrinsics is wrong and must be rejected.

**Stage 0 confirmation, 2026-09-17: compute capability is 7.5, agreed by two independent queries** — `cudaGetDeviceProperties` major.minor = 7.5 and `nvidia-smi --query-gpu=compute_cap` = 7.5. This is no longer an expectation. The unavailability of `cp.async`, `ldmatrix` and `mma` is now a project-wide hard constraint, recorded as such in `PERSISTENT.md` §3.

## 2. CPU

| Field | Value | Tag |
|---|---|---|
| Model | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz (Comet Lake-H, Family 6 Model 165 Stepping 2) | `[queried]` |
| Physical cores / threads | 4 physical / 8 logical | `[queried]` |
| Base / boost clock | base 2496 MHz (`Win32_Processor.MaxClockSpeed`, which reports the nominal clock). **Boost clock not recorded** — no API queried in this stage returns it, and the per-sample CPU frequency logged during 5.3 read a constant 2496 MHz across all 87 samples, so that series is a static nominal read rather than a live frequency and cannot establish a boost figure | `[queried]` |
| L1d per core | 32768 B (32 KiB), 8-way, 64 B line (`GetLogicalProcessorInformationEx`); L1i also 32768 B, 8-way | `[queried]` |
| L2 per core | 262144 B (256 KiB) unified, 4-way | `[queried]` |
| L3 total | 8388608 B (8 MiB) unified, 16-way, shared | `[queried]` |
| Cache line size | 64 B at every level | `[queried]` |
| **Effective cache sizes from the ladder** | Plateau edges detected at **32 KiB, 256 KiB, 4 MiB, 8 MiB and 16 MiB**. The 32 KiB, 256 KiB and 8 MiB edges coincide exactly with the OS-reported L1d, L2 and L3 sizes — **the measured ladder agrees with the OS report at all three levels, so no measured value supersedes an OS value here.** Two edges have no corresponding OS-reported level: 4 MiB (inside L3, consistent with the shared 8 MiB L3 being only partly available to one thread) and 16 MiB (beyond L3, inside the DRAM tier). `cpu_cache_ladder` (microbenchmark 3); the OS sizes are reported alongside the measured edges, never replaced by them. **STAGE 0b, 2026-09-18 — re-measured and UNCHANGED.** The edge detector is derived from the same run as the bandwidth figures, so it was re-run with them. Stage 0b reference run 2 detected edges at **32 KiB, 256 KiB, 4 MiB, 8 MiB and 16 MiB** — the identical five, from independently pinned code under different background load. The Stage 0 edge set is confirmed, not superseded, and the agreement is a check on the ladder itself: a harness artefact would not reproduce the same five boundaries across a code change | `[measured]` |
| Widest SIMD ISA available | **AVX2** (with FMA3). Determined by CPUID: leaf 7 EBX bit 5 set and `xgetbv(0)` XCR0 YMM state enabled; the AVX-512F bit (leaf 7 EBX bit 16) is **absent**, and ZMM state is not enabled. Compiled ISA is AVX2 via `/arch:AVX2` and the benchmark asserts compiled-ISA == widest-supported-ISA | `[queried]` |
| **Measured DRAM bandwidth** | *(empty — not obtained)* **Reason: every DRAM-tier working set (8 MiB, 16 MiB, 32 MiB, 64 MiB, 128 MiB) exceeded the 5% standard-deviation limit in BOTH suite runs** (run 1: 10.34 / 9.70 / 10.47 / 8.54 / 7.73%; run 2: 16.45 / 12.52 / 9.23 / 9.14 / 5.32%), so no valid DRAM figure exists and none is substituted. For orientation only, the INVALID medians spanned 18.26–21.40 GB/s in run 2, which sits **below** the single-channel theoretical ceiling of 23.464 GB/s = 2933 MT/s x 8 B x 1 channel `[derived]` — the earlier concern that the measured figure might exceed theoretical did **not** reproduce under protocol conditions  **STAGE 0b, 2026-09-18 — re-measured, STILL INVALID, field remains empty.** `cpu_cache_ladder` was re-run twice after being fixed (thread pinned to one logical CPU, priority raised) and under reduced background load. Every DRAM-tier working set exceeded the limit again in both runs: run 1 17.36 / 21.46 / 9.54 / 6.16 / 5.35%, run 2 (reference) 16.57 / 11.77 / 8.32 / 8.06 / 7.23%. The Stage 0b diagnosis explains why the fix could not help here and the earlier reason could not: the DRAM-tier dispersion is **broad, not spike-carried** — at 8, 16 and 32 MiB in run 2 the robust outlier test flags **zero** samples while the interquartile range is 19.88 / 20.75 / 10.70% of median. A distribution with no outliers and a wide middle is not an occasional interruption; it is the working set contending for the shared 8 MiB L3 and the single DRAM channel with every other process on the machine, which no change inside the benchmark can control. For orientation only, the INVALID run-2 medians were 22.05 / 18.75 / 18.54 / 18.23 GB/s at 16–128 MiB, still below the 23.464 GB/s single-channel theoretical ceiling; the 8 MiB point read 32.42 GB/s, **above** that ceiling, which is itself evidence that an 8 MiB working set is not purely DRAM-resident but partly served from the equally-8-MiB L3. Raw samples `bench/results/stage0b/run1/` and `run2/` |
| **Measured L1 / L2 / L3 bandwidth** | L1-resident **127.38 GB/s** (8 KiB working set, std dev 1.21%, VALID); L2-resident **99.64 GB/s** (256 KiB, 1.62%, VALID); L3-resident **62.31 GB/s** (4 MiB, 4.77%, VALID). All three from reference run 2 of `cpu_cache_ladder`, single-threaded streaming read, 268435456 B read per sample. L1-to-DRAM dynamic range across the ladder is roughly 7x. **SUPERSEDED BY STAGE 0b, 2026-09-18 — see below. The Stage 0 figures above are retained as provenance and are no longer the project's values.** <br><br>**STAGE 0b VALUES (current):** L1-resident **143.85 GB/s** (4 KiB working set, std dev 1.272%, VALID); L2-resident **113.85 GB/s** (256 KiB, 4.457%, VALID); L3-resident **70.81 GB/s** (4 MiB, 4.621%, VALID). All three from Stage 0b reference run 2 of `cpu_cache_ladder`, single-threaded streaming read, 268435456 B read per sample — the same byte count at every point. **Why all five ladder values were replaced rather than only the DRAM one:** `BENCHMARK_PROTOCOL.md` §4 voids a comparison across changed code, and Stage 0b changed `cpu_cache_ladder.c` (the measured thread is now pinned to one logical CPU and runs at raised priority); separately, Stage 0b reduced background load, which is itself a §4 fixed condition. Mixed-provenance values inside one row are therefore not permitted, and a partial re-run would have concealed the condition change rather than avoided it. **These figures are for a benchmark that outranks ordinary background work** (ABOVE_NORMAL priority class, THREAD_PRIORITY_HIGHEST, pinned to logical CPU 2). That is what a ceiling measurement should be, but it is **not the same measurement Stage 0 attempted**, which ran unpinned at normal priority — part of the rise from 127.38 to 143.85 GB/s is that change and part is the lighter load, and this stage cannot separate the two. Note the L1 tier is now flat at 143.80–143.85 GB/s across 4, 16 and 32 KiB, agreeing to within 0.04%; the 8 KiB point Stage 0 used was INVALID in Stage 0b run 2 (6.365%) and is therefore not the quoted L1 figure. L1-to-DRAM dynamic range is roughly 7.9x. Raw samples `bench/results/stage0b/run2/cpu_cache_ladder.json` | `[measured]` |
| **Measured peak FP32 (SIMD)** | *(empty — not obtained)* **Reason: the vectorised AVX2 configuration exceeded the 5% standard-deviation limit in BOTH suite runs** (run 1: 5.86%, run 2: 5.60%), so no valid figure exists and none is substituted. The scalar reference of the same loop WAS valid (8.347 GFLOP/s, 2.51%), and the vectorised-over-scalar ratio of 5.19x confirms the compiler really did vectorise rather than silently dropping it — but that ratio is not a peak figure. `cpu_simd_peak` (microbenchmark 4), single-threaded, one core, 8 lanes x 8 chains. <br><br>**STAGE 0b, 2026-09-18 — FIELD NOW POPULATED. Measured peak FP32 (SIMD) = 48.411 GFLOP/s**, std dev **2.773%**, **VALID**, Stage 0b reference run 2; run 1 gave 49.506 GFLOP/s at 1.075%, also VALID. **Valid in both runs**, where Stage 0 was invalid in both. Single-threaded, one core, AVX2 256-bit, 8 lanes x 8 independent FMA chains, 2000000 inner iterations, flops derived from the trip count as iterations x chains x lanes x 2. The scalar reference of the same loop measured 8.565 GFLOP/s (2.773% run 2) and the vectorised-over-scalar ratio is **5.65x** (run 1: 5.84x), so the evidence that the compiler genuinely vectorised survives the change; the two paths are additionally now proven to compute the same arithmetic — vector checksum 114111040 over 8 lanes = 14263880 per lane against scalar 14263877, a relative difference of 2.1e-7. **What changed:** the measured thread is pinned to one logical CPU and runs at raised priority. The trip count, the arithmetic and the flop derivation are unchanged from Stage 0 — deliberately, because lengthening a sample so it averages over excursions would have lowered the reported standard deviation by measuring something else. **This figure is for a benchmark that outranks ordinary background work and owns a physical core**, which is what a per-core ceiling should be, but is not the same measurement Stage 0 attempted. Raw samples `bench/results/stage0b/run2/cpu_simd_peak.json` | `[measured]` — VALID, Stage 0b |

## 3. Required Stage 0 microbenchmarks

Each writes a value above. All are reusable tested code in `bench/microbench/`, not throwaway scripts — later stages re-run them to confirm the machine has not drifted.

1. **GPU bandwidth** — large strided device-to-device copy sized beyond L2, CUDA-event timed. GB/s.
2. **GPU FP32 peak** — FMA loop held in registers, no memory traffic. GFLOP/s.
3. **CPU cache ladder** — streaming read across working-set sizes spanning L1 to DRAM. Plateau edges reveal effective cache sizes, which often differ from the OS report.
4. **CPU SIMD peak** — FMA loop at widest available vector width. GFLOP/s.
5. **Host-to-device transfer** — pinned and pageable, both directions. GB/s each.
6. **cuBLAS SGEMM reference** — at the model's real matrix shapes, **not square matrices**. This is the prefill denominator for the entire project.
7. **Kernel launch overhead** — empty kernel, measured. Required by the Stage 10 model as a fixed additive term; without it the model will systematically overpredict speed on small kernels.
8. **Shared memory bandwidth** — measured, not assumed. Required by the model for any tiled kernel.
9. **Occupancy sweep** — a simple kernel run at varying block sizes and register pressures, measuring achieved occupancy against theoretical. Feeds the model's occupancy term.

Each follows `BENCHMARK_PROTOCOL.md` §3: minimum 5 warmup iterations discarded, minimum 20 timed samples, median reported with min/max/std dev, run INVALID if std dev exceeds 5% of median.

## 4. Profiling toolchain

The Stage 10 model and every gap explanation depend on hardware counters. Stage 0 must confirm these work before the project proceeds.

| Field | Value | Tag |
|---|---|---|
| Nsight Compute available | Yes - `ncu` is on PATH (Nsight Compute 2025.4.0 install directory) | `[queried]` |
| Version | 2025.4.0.0 (build 36690805, public-release) | `[queried]` |
| Counter collection permitted (may need admin / driver flag on Windows) | **Yes.** Unblocked before this session by the operator via NVIDIA Control Panel -> Desktop -> Enable Developer Settings -> Developer -> Manage GPU Performance Counters -> allow access to all users, then a full restart. That step established collection works for a **non-elevated** user; it was verified by the operator pre-session and is **not re-tested here**, because this Stage 0 session deliberately ran **elevated** so that it could apply `nvidia-smi -lgc` itself | `[queried]` |
| Counters confirmed collectable | **All 32 required metric names resolved, and all 32 populated with values on a live kernel. Zero unavailable, zero unresolved.** Verified by profiling microbenchmark 1 (`gpu_bandwidth`, kernel `mb_bw_copy_kernel`): achieved occupancy 99.46%, theoretical occupancy 100.00%, block size 256, DRAM read 87.152 GB/s, DRAM write 87.025 GB/s, L2 hit rate 49.26%, shared-memory bank conflicts 0, instructions executed 5832704, duration 3080448 ns, and all 18 warp stall reasons populated (dominant: `long_scoreboard` 364.60, i.e. waiting on global memory - the expected signature for a bandwidth-bound copy). Metric names are resolved against the installed version at run time, never hardcoded. Full mapping and per-counter status in `bench/results/profile_gpu_bandwidth.json` | `[measured]` |

Minimum counters needed downstream: achieved occupancy, DRAM read/write throughput, L2 hit rate, shared memory bank conflicts, warp stall reasons, and instructions executed. **If counter collection is blocked, flag it in Stage 0 rather than discovering it at Stage 8.** On Windows this commonly requires enabling GPU performance counters for all users in the NVIDIA control panel, or running the profiler elevated.

**Stage 0 metric-name mapping, resolved against Nsight Compute 2025.4.0.0 on 2026-09-17.** Names were queried from the tool for this chip rather than hardcoded; `launch__*` and `sm__maximum_warps*` do not appear in the default `ncu --query-metrics` and require `--query-metrics-collection launch` / `occupancy`, which `bench/profile.py` handles.

| Required counter | Metric name used | Status on microbenchmark 1 |
|---|---|---|
| achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | populated, 99.46 % |
| theoretical occupancy | `sm__maximum_warps_per_active_cycle_pct` | populated, 100.00 % |
| theoretical occupancy (max warps) | `sm__maximum_warps_avg_per_active_cycle` | populated, 32.0 warp |
| occupancy limit, blocks | `launch__occupancy_limit_blocks` | populated, 16.0 block |
| occupancy limit, warps | `launch__occupancy_limit_warps` | populated, 4.0 block |
| occupancy limit, registers | `launch__occupancy_limit_registers` | populated, 16.0 block |
| occupancy limit, shared mem | `launch__occupancy_limit_shared_mem` | populated, 16.0 block |
| block size | `launch__block_size` | populated, 256 |
| DRAM read throughput | `dram__bytes_read.sum.per_second` | populated, 87152071387.02 byte/s |
| DRAM write throughput | `dram__bytes_write.sum.per_second` | populated, 87025253469.63 byte/s |
| L2 hit rate | `lts__t_sector_hit_rate.pct` | populated, 49.26 % |
| shared memory bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | populated, 0.0 - a real measured zero, not a blank |
| instructions executed | `smsp__inst_executed.sum` | populated, 5832704 inst |
| duration | `gpu__time_duration.sum` | populated, 3080448 ns |
| warp stall reasons (18) | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` for barrier, branch_resolving, dispatch_stall, drain, imc_miss, lg_throttle, long_scoreboard, math_pipe_throttle, membar, mio_throttle, misc, no_instruction, not_selected, selected, short_scoreboard, sleeping, tex_throttle, wait | all 18 populated |

**Unavailable counters: none.** `unresolved_counters` was empty and no counter was left blank or defaulted to zero.

The wrapper was additionally run against microbenchmark 9 (`occupancy_sweep`) to supply the achieved-occupancy term that the sweep deliberately leaves as `pending`, since the benchmark never copies theoretical into achieved:

| Registers/thread | Block size | Theoretical occupancy `[derived]` | Achieved occupancy `[measured]` |
|---|---|---|---|
| 12 | 1024 | 100.00 % | 98.93 % |
| 36 | 1024 | 100.00 % | 97.98 % |
| 64 | 1024 | 100.00 % | 97.93 % |
| 96 | 512 | 50.00 % (register-limited) | 48.82 % |

Theoretical occupancy is computed as `cudaOccupancyMaxActiveBlocksPerMultiprocessor * blockSize / maxThreadsPerMultiProcessor[queried] * 100`, with `maxThreadsPerMultiProcessor` = 1024 `[queried]`. The 96-register case is the register-limited one and the sweep names it as such: 96 regs/thread x 512 threads = 49152 registers against 65536 per SM permits 1 block per SM, hence 512 / 1024 = 50.00 % theoretical.

## 5. Machine state and stability

*The machine must be in a controlled, reproducible state before any measurement is trustworthy. A measurement taken on a machine whose clocks move is not a measurement of anything.*

### 5.1 Clock and power state

| Field | Value | Tag |
|---|---|---|
| GPU core clock offset vs stock | **NOT OBTAINABLE BY QUERY - UNVERIFIED, and explicitly not recorded as zero.** `nvidia-smi -q` contains no clock-offset field at all (grep for Offset / Locked / GpuLock returns zero matching lines); Applications Clocks and Default Applications Clocks both return "Requested functionality has been deprecated". Vendor offsets applied through MSI Dragon Center are invisible to `nvidia-smi`. This session additionally retried the vendor WMI path **with elevation** (previously Access denied): `root\WMI MSI_VGA` now reads, but returns 18 unlabeled integer instances with no documented schema and no clock-offset field; `MSI_ACPI` returns "Not supported". Q9 cannot be closed by query on this machine | `[queried]` - query returns nothing |
| GPU memory clock offset vs stock | **NOT OBTAINABLE BY QUERY - UNVERIFIED, not recorded as zero.** Same reason as the core offset above | `[queried]` - query returns nothing |
| Power limit (current / default / max) | 55.00 W / 55.00 W / 55.00 W; min 1.00 W; enforced 55.00 W. All three identical, so **no power headroom exists and power-limit tuning is not an available variable in this project**. Note `nvidia-smi --query-gpu=power.limit` returns `[N/A]` on this card; the figures come from `nvidia-smi -q` | `[queried]` |
| Temperature limit | shutdown 97 C, slowdown 92 C, max operating 102 C. Measured behaviour disagrees with the slowdown figure: the driver asserted SW thermal slowdown at **86 C**, well below 92 C - see 5.3 | `[queried]` |
| Overclocking utility present and running (MSI Dragon Center, Afterburner, etc.) | MSI Afterburner **installed** but **not running**; its stored profile for DEV_2192 has `CoreClkBoost=` and `MemClkBoost=` both empty and Profile2D/Profile3D = -1. MSI vendor services were resident during the runs (MSI Foundation / Central / Companion services, `OmApSvcBroker.exe`) | `[queried]` |
| Active performance / shift mode profile | MSI user scenario **"Balanced"** `[operator-observed]`, read from the vendor utility's own UI. Not obtainable programmatically: `root\WMI MSI_VGA` exposes no such field even with elevation | `[operator-observed]`, not `[queried]` |
| Fan profile | **Not applicable - no fan-profile control is exposed for the Balanced scenario**; those controls appear only under "Extreme Performance" | `[operator-observed]` |
| Windows power plan | Ultimate Performance, GUID `69f3d390-66d3-4992-be72-f285309743af`. **Frozen fingerprint value** - changed from Balanced before this project and not changed since | `[queried]` |
| Hardware-accelerated GPU scheduling enabled | Yes, `HwSchMode = 2`. **Frozen fingerprint value** | `[queried]` |
| Can clocks be locked (`nvidia-smi -lgc` / `-lmc`)? | **Graphics clock: YES, with elevation.** From a non-elevated shell `nvidia-smi -lgc` returns "The current user does not have permission to change clocks for GPU 00000000:01:00.0" with exit code 4 - a permission limit, not a device limit. **Memory clock: NO - not supported by this device.** `nvidia-smi -lmc 6001` returns "Setting locked Memory clocks is not supported for GPU 00000000:01:00.0." with **exit code 0** from a non-elevated shell; exit 0 plus that message is a device-capability limit, so elevation would not change it and it was not re-attempted in this stage | `[queried]` |
| Clocks locked for this project? | **Graphics clock: YES, locked at 1365 MHz** by this session (which ran elevated) with `nvidia-smi -lgc 1365,1365`; command returned "GPU clocks set to (gpuClkMin 1365, gpuClkMax 1365)" and exit 0. Verified in effect, not assumed: `machine_state.py verify-lock --mhz 1365` observed 1365 MHz on 10 of 10 samples under load, and the subsequent 5-minute sustained run recorded 1365 MHz on **122 of 122 samples**. **Memory clock: NO - cannot be pinned on this device.** The clock regime is therefore **partially, not fully, controlled**, and the 5.4 VRAM integrity check is the only evidence available on memory stability. **Clock locks do not survive a reboot**; the lock command is a per-session step for every later stage, recorded in 5.5 and in `PERSISTENT.md` | `[queried]` |

### 5.2 The overclock decision

**Any non-zero clock offset must be resolved before Stage 1.** Three risks, in increasing severity:

1. **Ceilings become those of an overclocked card.** Survivable if fixed and documented, since the project reports ratios against measured ceilings. Fatal if the overclock changes mid-project — a profile that does not persist across reboots means Stage 5 and Stage 9 were measured on different machines and the waterfall is void.
2. **Variance.** Aggressive clocks drift more under sustained thermal load, pushing runs past the 5%-std-dev invalidity threshold for reasons unrelated to the code.
3. **Silent numerical corruption.** Consumer GDDR6 has no ECC. An unstable memory overclock does not crash — it returns slightly wrong values. This surfaces as intermittent correctness-gate failures that vanish on re-run, and would be attributed to kernel logic rather than to the memory clock. For a project gated on numerical correctness against a reference, this is the worst available failure mode.

**Default recommendation: run stock for the project's duration.** Nothing is gained by the overclock here — the reported figures are ratios, not absolute performance — and an entire class of confusing failure is removed. If the overclock is kept, it must be verified stable per §5.4, documented here, and unchanged for the life of the project.

Record the decision and its date:

| Decision | Value |
|---|---|
| Running stock or overclocked | MSI Dragon Center user scenario **"Balanced"** for the life of the project, with the graphics clock locked. **Whether Balanced is truly stock is UNVERIFIED** - see the 5.1 core/memory offset rows; it cannot be established by any query available on this machine, and no offset of zero is recorded. Operator switches to "Extreme Performance" only outside this project. Decision D8, pre-decided by the operator and not reopened by this stage |
| If overclocked, exact offsets | Unknown and unobtainable. `nvidia-smi` exposes no offset field; the vendor WMI class exposes none either, even with elevation. Recorded as unverified rather than as zero |
| Stability verified (§5.4) | Yes - VRAM integrity bit-exact over 3 rounds x 4 patterns on a 2048 MiB buffer under thermal load, and compute determinism bit-identical. See 5.4 |
| Date fixed | 2026-09-17 (graphics clock locked at 1365 MHz this date; D8 scenario decision pre-dated it) |

### 5.3 Thermal and boost behavior under sustained load

A single short benchmark does not reveal throttling. Run a sustained load and log clock, temperature, and power over time.

| Field | Value | Tag |
|---|---|---|
| Idle clock / temp | **Unlocked baseline run:** 300 MHz SM / 405 MHz memory, 51 C, 3.86 W, GPU-idle throttle flag only. **Locked run (the project's operating regime):** 1365 MHz, 66 C, 13.89 W - `-lgc 1365,1365` pins minimum as well as maximum, so the card idles at the locked clock and there is no ramp | `[measured]` `machine_state.py sustained` |
| Clock after 30s sustained load | **Unlocked:** 1785 MHz, 73 C, 54.87 W, SW power cap active. **Locked:** 1365 MHz, 73 C, 31.85 W, no power-cap or thermal reason | `[measured]` |
| Clock after 5min sustained load | **Unlocked:** 1410 MHz, 86 C, 36.62 W, SW thermal slowdown active - a 22% clock loss from the 30-second point, with power draw falling from 54.87 W to 36.62 W, which shows the tail is **thermally** limited rather than power limited. **Locked:** 1365 MHz, 82 C, 33.66 W, no throttle reason - identical to the 30-second value | `[measured]` |
| Peak temperature reached | **Unlocked: 87 C. Locked: 82 C.** Neither reaches the 92 C slowdown limit that `nvidia-smi` reports, yet the driver asserted SW thermal slowdown at 86 C in the unlocked run - the effective software thermal target is below the reported hardware slowdown temperature | `[measured]` |
| Throttle reason reported, if any | **Unlocked run: yes, two.** `0x0000000000000004` SW power cap, first seen at t = 3.502 s at 1815 MHz and never cleared; `0x0000000000000020` SW thermal slowdown, first seen at t = 103.185 s at 86 C / 1695 MHz. Clocks decayed monotonically 1815 -> 1365 MHz across the window. **Locked run: none.** Only `0x0` and `0x0000000000000001` (GPU idle) were seen across 122 samples - no power cap, no thermal slowdown, no HW slowdown for the full five minutes. **Locking at the measured floor removed every throttle reason** | `[queried]` |
| **Time until clocks stabilize** | **Unlocked: 3.502 s** to settle at 1815 MHz by the criterion "5 consecutive samples within +/-30 MHz" - but that criterion fires on the initial boost step and is misleading here, because the clock then decays for the rest of the run and never truly stabilises. **Locked: 0.075 s**, and structurally zero: since `-lgc` pins the minimum as well as the maximum, the card is already at 1365 MHz before the first warmup iteration and no ramp exists to cover | `[measured]` |
| **Maximum safe continuous benchmark duration** | **Unlocked: 3.502 s**, derived as the time at which the first throttle reason appeared (SW power cap at t = 3.502 s). **Locked: at least 299.219 s**, the full logged window, with no throttle reason at any point. That is a lower bound set by the length of the run, not a measured ceiling - the run was stopped, not throttled | `[derived]` from the 5.3 series |

The stabilization time sets the warmup requirement. If clocks take 20 seconds to settle, five warmup iterations of a fast kernel are not enough, and `BENCHMARK_PROTOCOL.md` §3 must be adjusted upward for this machine. State the adjusted value.

**Adjusted warmup for this machine: 25 iterations** (protocol minimum is 5). Arithmetic: measured stabilization under the locked regime is 0.075 s. The reference timed workload is `gpu_bandwidth`, whose measured median iteration is 3.118 ms, so 5 warmup iterations cover 5 x 3.118 ms = 15.6 ms, which is **less** than 0.075 s and therefore insufficient by the rule above; 25 warmup iterations cover 25 x 3.118 ms = 77.9 ms, which exceeds 0.075 s. 25 was used for every benchmark in this stage. Note that under the lock this is a belt-and-braces figure rather than a strict requirement, because `-lgc 1365,1365` pins the idle clock to the operating clock and there is no ramp to warm through; it still serves to warm caches, TLBs and the driver's kernel-launch path.

**A separate finding the sustained-load run produced, which the 5.3 table above cannot express.** The unlocked card is not merely power-capped, it is thermally governed in the tail: over five minutes its clock fell 1815 -> 1365 MHz (-24.8%) while power draw fell 54.87 -> 36.62 W. A benchmark of any length taken on the unlocked card would therefore have been measuring a different machine at its start than at its end, and two benchmarks run at different points in a session would not be comparable. This is the single strongest justification for the clock lock in this project, and it was measured rather than assumed.

**CPU telemetry during 5.3 is not trustworthy and no conclusion is drawn from it.** The logger recorded CPU frequency and package temperature alongside the GPU series as required. Across all 87 samples of the unlocked run and all 122 samples of the locked run, CPU frequency read a constant 2496 MHz and CPU package temperature a constant 73.05 C, while CPU load varied between 4% and 36%. Two series that do not move while their driver does are static nominal reads, not live measurements. **Consequently the question of whether the Ultimate Performance power plan raises CPU heat enough to pull GPU boost down under sustained load CANNOT be answered from this data, in either direction.** What *is* established independently is that the GPU asserted a thermal throttle reason and lost 24.8% of its clock on the unlocked card, which satisfies the condition for flagging a re-test on the Balanced plan before Stage 5. That flag is recorded in `PERSISTENT.md`. The power plan was **not** changed by this stage - it is a frozen fingerprint value.

### 5.4 Stability and determinism checks

All four must pass before Stage 1. A failure here is a hard stop, not a warning.

1. **VRAM integrity.** Allocate a large buffer, write known patterns, read back, verify bit-exact. Repeat under thermal load. Any mismatch means the memory overclock is unstable — drop to stock and re-run.
2. **Compute determinism.** Run the same microbenchmark's *numerical output* twice and verify bit-identical results. Floating-point GPU kernels with a fixed reduction order should be deterministic; non-determinism here indicates a hardware or driver problem, not a code problem.
3. **Timing reproducibility.** Run the full microbenchmark suite twice in the session. Medians should agree within a few percent. Record the observed run-to-run spread — it is the noise floor for every later measurement, and no stage can claim a speedup smaller than it.
4. **Background load.** Enumerate processes using GPU or significant CPU. Record what was running. Browsers, Discord, and overlay software are the usual offenders.

**Stage 0 results, 2026-09-17. All four ran; none is a hard stop.**

| Check | Result | Tag |
|---|---|---|
| 1. VRAM integrity | **PASS.** 2048 MiB buffer, patterns `0x00`, `0xFF`, `0xAA`, `0x55`, 3 rounds each = 12 write/readback cycles, every one bit-exact, zero bad bytes, run immediately after the 5-minute thermal load with the GPU at 82-87 C. This is the **only** evidence available on memory stability, because the memory clock cannot be pinned on this device | `[measured]` |
| 2. Compute determinism | **PASS.** The same GEMM (M=768, N=2304, K=768) run twice produced bit-identical output buffers | `[measured]` |
| 3. Timing reproducibility | **Run-to-run spread over 49 configurations valid in both suite runs: median 0.317%, mean 1.407%, 90th percentile 4.401%, maximum 19.913%.** Over all 96 configurations regardless of validity: mean 2.059%, maximum 19.913%. The maximum is `cublas_sgemm_ref` prefill qkv_projection at M=16, one of the small-M shapes where launch and setup latency dominate the arithmetic | `[measured]` |
| 4. Background load | Recorded below. Nothing was closed by this session | `[queried]` |

**Adopted noise floor: 4.4%** (the 90th percentile of run-to-run spread across configurations valid in both runs). No stage may claim a speedup smaller than this. Two qualifications that must travel with the figure: (a) the median spread is far lower at 0.317%, so 4.4% is deliberately conservative; (b) small-M GEMM configurations are materially noisier than the rest and a claim about decode-shaped work needs a larger margin than a claim about prefill-shaped work.

**Background load during the timed runs** (the operator closed overlay and monitoring software before the session; this is what remained, re-enumerated rather than carried forward). Holding GPU contexts: `dwm.exe`, `explorer.exe`, `ShellHost.exe`, `StartMenuExperienceHost.exe`, `LockApp.exe`, `TextInputHost.exe`, `SearchHost.exe`, `CrossDeviceResume.exe`, `ApplicationFrameHost.exe`, `SystemSettings.exe`, `msedgewebview2.exe`, `vgtray.exe` (Riot Vanguard), `OmApSvcBroker.exe` (MSI NBFoundation Service), `NahimicSvc64.exe`, `Nahimic3.exe`, and Claude Code itself, which cannot be closed because it drives the runs. Top CPU consumers: `MsMpEng` (Windows Defender), `System`, `WmiPrvSE`, `dwm`, `svchost`, `Memory Compression`, `vgc` (Riot Vanguard service), `nvcontainer`, `SearchIndexer`, `SDXHelper`. Resident vendor services: MSI Foundation Service, MSI Central Service, MSI Companion Service, MSIAPService, MSIService, NahimicService.

**This is a finding, not a clean bill of health.** Riot Vanguard is a kernel-mode anti-cheat driver and Nahimic is an audio driver that hooks process creation; both were resident throughout. Windows Defender real-time scanning was the single largest CPU consumer. These are the most likely explanation for the pattern of invalid configurations recorded in `MEASUREMENTS.md`, in which the same configuration is valid in one suite run and invalid in the other with medians that agree to well under one percent - the signature of an occasional multi-millisecond interruption inflating the standard deviation, not of an unstable machine.

**Background load during the STAGE 0b timed runs, 2026-09-18.** Recorded as a separate dated entry; the Stage 0 record above is not overwritten and remains the condition its own figures were taken under.

*Closed by the operator before the Stage 0b runs and confirmed absent at re-enumeration:* `Nahimic3.exe`, `NahimicSvc64.exe`, `NahimicService` (the audio driver that hooks process creation), `DSAService.exe` (Intel Driver & Support Assistant).

*Absent for reasons other than the close list:* **Riot Vanguard (`vgc`, `vgtray`) was not running at all this session.** It was resident throughout Stage 0 and is named in the Stage 0 record above as a likely contributor. Its absence is a real and favourable condition difference, not an action taken by this stage. `LockApp.exe` was also absent.

*Closed by the operator but RESPAWNED before the timed runs — recorded, not assumed closed:* `logioptionsplus_agent.exe`, `logioptionsplus_appbroker.exe`, `logioptionsplus_updater.exe` (Logitech Options+). The agent holds a GPU context and was the 17th-largest CPU consumer at re-enumeration (13.5 s). Options+ restarts itself; the close did not take, and this is recorded rather than reported as a successful close.

*Could not be closed — respawn immediately on End Task:* `msedgewebview2.exe`, all 6 instances, holding GPU contexts. **This is a condition Stage 0b SHARES with Stage 0, not a difference between them** — §5.4's Stage 0 record lists `msedgewebview2.exe` among the processes holding GPU contexts.

*Left running deliberately by operator decision, and NOT to be closed in future stages:* the MSI service stack — `MSI.CentralServer`, `MSI_Central_Service`, `MSI_Companion_Service`, `MSIAPService`, `MSIService`, `OmApSvcBroker.exe`. Reason: decision D8 fixes the MSI user scenario at "Balanced" for the life of the project and `PERSISTENT.md` Q9 establishes that no query available on this machine can read that scenario back. Stopping the stack risks silently changing the clock regime every project measurement is taken under, with no way to detect it happened, in exchange for removing processes that do not appear among the top CPU consumers. Stage 0b times no GPU kernel, so "holds a GPU context" bears little on it; CPU contention is what matters here. Also left running deliberately because the kill-and-respawn churn is worse than the load: `ShellExperienceHost`, `SearchHost`, `CrossDeviceResume`, `ApplicationFrameHost`, `dwm`.

*Not excluded and not modified:* **Windows Defender (`MsMpEng`)**, the second-largest CPU consumer at 330.7 s. Resident for all of Stage 0 and remains a recorded condition.

*Uncloseable system consumers, recorded:* `System` (376.2 s), `MsMpEng` (330.7 s), `WmiPrvSE`, `SearchIndexer`, `nvcontainer`, `dwm` (100.8 s), `csrss`, `lsass`, `svchost`, `Memory Compression`, `RuntimeBroker`, and **Claude Code itself** across 4 processes in the top 25 (133.1 / 110.3 / 66.9 / 47.5 s), which cannot be closed because it drives the runs.

*Instrument added by this stage, recorded as a condition rather than assumed negligible:* a CPU frequency sampler ran in a separate process alongside every Stage 0b benchmark, at a 20 ms interval, measured per-probe cost 37.5–39.2 us, **duty cycle 0.355–0.372% of one core**. It was pinned to logical CPUs 4, 5, 6 and 7 (mask `0xf0`), clear of logical CPU 2 where the measured thread runs, clear of CPU 2's SMT sibling CPU 3, and clear of CPU 0 and its sibling CPU 1. The logical-to-physical mapping was queried with `GetLogicalProcessorInformationEx(RelationProcessorCore)`, not assumed.

*Network — a recorded run condition.* Stage 0b ran over a **mobile hotspot** ("iPhone 2", Intel Wi-Fi 6 AX201 160MHz, link 216→551 Mbps), not the connection present during Stage 0. **Windows did NOT mark it metered** (`NetworkCostType = Unrestricted`) and **Windows Update was NOT paused** (`PauseUpdatesExpiryTime` null; `wuauserv` Stopped, which is not a pause). Driver 591.44 is a frozen fingerprint value and an update installed mid-session would void every comparison against Stage 0 with no undo; the fingerprint was verified clean before the first timed run and no update occurred during the session. **One disconnection and reconnection was observed**, during the build-and-setup phase and **before any Stage 0b timed run began**; no timed run overlapped it and no run is invalidated by it.

*Power:* **on AC** for every timed run (`GetSystemPowerStatus.ACLineStatus` = 1, battery at 98%). `BENCHMARK_PROTOCOL.md` §3 makes a run on battery INVALID outright; this was checked programmatically before the first timed run, not assumed.

**The Stage 0b condition is strictly lighter than Stage 0's** — the close list removed load and Riot Vanguard was absent — except for the added frequency sampler, whose duty cycle is under 0.4% of one core on physical cores the measured thread never touches.

### 5.5 Environment fingerprint

Recorded here and re-verified at the start of every later stage. If any value has changed, the comparison against prior stages is void until re-established.

**Per-session step every later stage performs before its first timed run.** Clock locks do not survive a reboot.

```powershell
# Requires an elevated shell. Verify afterwards; never assume it took.
nvidia-smi -lgc 1365,1365
python bench/machine_state.py verify-lock --mhz 1365
```

`nvidia-smi -lmc` is **not** attempted: the memory clock is not lockable on this device (see 5.1). To release the lock outside the project, `nvidia-smi -rgc`.

**Python environment** (frozen, do not reinstall): the repository-root `.venv` on Python 3.14.2 with torch 2.14.0+cu130, `torch.version.cuda` = 13.0, `torch.cuda.is_available()` = **True**, device "NVIDIA GeForce GTX 1650 Ti", capability (7, 5), `arch_list` includes `sm_75`, numpy 2.5.3. **PyTorch with CUDA does install and work on Python 3.14 here, so the fallback to Python 3.12 is not needed.** The oracle is GPU-capable. One trap worth recording: a bare `python` on PATH resolves to a different system interpreter that has no torch at all, so all project Python must be invoked through `.venv/Scripts/python.exe`.

**This Stage 0 session ran ELEVATED**, deliberately and as an exception, so that it could apply `nvidia-smi -lgc` itself without an operator round-trip (`IsInRole(Administrator)` = True, user `MSI\saket`). The separate pre-session finding that Nsight Compute counter collection works for a **non-elevated** user was established by the operator before this session and was **not re-tested here**. Later stages do not require elevation except for the clock-lock command above.

| Field | Value |
|---|---|
| Driver version | 591.44 (VBIOS 90.16.51.00.14) |
| CUDA toolkit version | nvcc 13.1.80; target architecture `sm_75` |
| Compiler and version | MSVC 19.44.35229.0, toolset 14.44.35207, x64, from Build Tools for Visual Studio 2022. Host flags `/DWIN32 /D_WINDOWS /EHsc /W3 /arch:AVX2 /fp:precise /MD /O2 /Ob2 /DNDEBUG`. CUDA flags `-D_WINDOWS -Xcompiler=" /EHsc" -O3 --generate-line-info -Xptxas=-O3 -Xptxas=-v -Xcompiler=/arch:AVX2 -Xcompiler=/fp:precise -arch=sm_75`. **Toolset is pinned two ways** - `scripts/build.ps1` enters the VS2022 `vcvars64` environment and passes that `cl.exe` as `CMAKE_C/CXX/CUDA_HOST_COMPILER`, and `CMakeLists.txt` raises a `FATAL_ERROR` if `MSVC_VERSION >= 1950`. VS 2026 toolsets 14.51.36231 and 14.50.35717 are also installed and are the ones CUDA 13.1 rejects. `-allow-unsupported-compiler` is never used |
| Clock offsets | **UNVERIFIED - not obtainable by any query on this machine, and not recorded as zero.** See 5.1. What *is* fixed and re-verifiable: MSI user scenario "Balanced" `[operator-observed]`, and the graphics clock locked at 1365 MHz |
| Power limit | 55.00 W, with default and maximum also 55.00 W - no headroom |
| Windows power plan | Ultimate Performance, GUID `69f3d390-66d3-4992-be72-f285309743af` |
| OS build | Windows 11 Home 10.0.26200 |

## 6. Secondary devices

Populate only if used. Any figure from these is labelled with the device name everywhere it appears and never enters the main waterfall.

| Device | Access | Used for | Notes |
|---|---|---|---|
| | | | |

- **Free tiers with tensor cores** — Colab and Kaggle offer free T4 access (sm_75 as well, so still no `cp.async`; T4 *does* have tensor cores). Verify current quotas.
- **Cheap hourly rental** — vast.ai, RunPod, Lambda. Verify current pricing.
- **Borrowed RTX 5060** — Blackwell, has tensor cores, supports sm_80+ intrinsics, short sessions only. Best used for one scoped comparison.

**Rule:** the primary device stays fixed for the entire project. Faster hardware does not improve this project's conclusions, because the reported figures are ratios against that device's own measured ceilings.

## 7. Open hardware questions

| Question | Blocks | Status |
|---|---|---|
| Does the primary GPU report tensor cores? | Stage 11 framing | **CLOSED, with a divergence.** No property reports them - `cudaDeviceProp` has no such field. Measured empirically: a tensor pipe **is** present and executes HMMA (WMMA kernel numerically correct; `sm__inst_executed_pipe_tensor_op_hmma.sum` = 4). No throughput measured, none claimed. **This contradicts the expectation in §1 and flags Stage 11 to revisit its framing** |
| Compute capability — confirm sm_75 | Stage 9 implementation | **CLOSED. 7.5 confirmed**, agreed by `cudaGetDeviceProperties` and `nvidia-smi`. No `cp.async` / `ldmatrix` / `mma`, now a hard constraint |
| Widest SIMD ISA on this CPU | Stage 6 prediction | **CLOSED. AVX2 + FMA3**, by CPUID; AVX-512 absent and ZMM state not enabled. Compiled with `/arch:AVX2` |
| Can Nsight Compute collect counters here? | Stage 10 and all gap analysis | **CLOSED. Yes.** All 32 required metrics resolved AND populated on a live kernel; zero unavailable. See §4 |
| Measured versus theoretical bandwidth ratio | every GPU prediction | **CLOSED. 0.8899** = 170.882 / 192.032 GB/s |
| **Current clock offsets, and stock-or-overclocked decision** | **every measurement in the project** | **Decision CLOSED (D8: Balanced, graphics clock locked at 1365 MHz). Offsets remain OPEN and UNVERIFIED** - not obtainable by any query on this machine, including the vendor WMI path retried with elevation. Never recorded as zero |
| Does VRAM pass the integrity check at current clocks? | correctness gate reliability | **CLOSED. PASS** - bit-exact over 12 write/readback cycles on a 2048 MiB buffer under thermal load |
| Can clocks be locked on this card? | measurement variance | **CLOSED. Graphics clock yes** (needs elevation; locked at 1365 MHz and verified). **Memory clock no** - device-capability limit, exit 0 with "not supported". Clock regime is therefore only partially controlled |
| Time until clocks stabilize under load | warmup requirement | **CLOSED. 0.075 s locked** (structurally zero - the lock pins the idle clock too); 3.502 s unlocked, though the unlocked card then decays for the rest of the run and never truly stabilises. Adjusted warmup: **25 iterations** |
| Run-to-run noise floor | minimum claimable speedup | **CLOSED. Median 0.317%, mean 1.407%, p90 4.401%, max 19.913%** across 49 configurations valid in both suite runs. **Adopted floor 4.4%**; small-M GEMM shapes need a wider margin |
