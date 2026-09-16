# HARDWARE.md

*Every prediction in this project cites this file, and the Stage 10 performance model consumes it directly as input. Nothing here may be filled from memory or a spec sheet. Vendor spec sheets state theoretical peaks real code never reaches; a model built on theoretical numbers is wrong by construction.*

**Status: UNPOPULATED. Stage 0 populates this file. No optimization stage begins until it is complete.**

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
| Primary GPU | | |
| Compute capability | | `[queried]` |
| SM count | | `[queried]` |
| CUDA cores | | `[derived]` |
| Base / boost clock | | `[queried]` |
| VRAM total | | `[queried]` |
| Memory bus width | | `[queried]` |
| Theoretical peak bandwidth | | `[spec]` ⚠️ never in a prediction |
| **Measured achievable bandwidth** | | `[measured]` |
| Bandwidth efficiency (measured ÷ theoretical) | | `[derived]` |
| Theoretical peak FP32 | | `[spec]` ⚠️ |
| **Measured peak FP32** | | `[measured]` |
| Tensor cores present | | `[queried]` |
| L2 cache size | | `[queried]` |
| Shared memory per block | | `[queried]` |
| Shared memory per SM | | `[queried]` |
| Max threads per SM | | `[queried]` |
| Max blocks per SM | | `[queried]` |
| Registers per SM | | `[queried]` |
| Warp size | | `[queried]` |
| CUDA toolkit / driver version | | `[queried]` |

### Tensor cores

The GTX 16-series (Turing TU117) is expected to ship without tensor cores or RT cores. Confirm via `cudaGetDeviceProperties` rather than assuming. If confirmed, Stage 11 is justified against cuBLAS SGEMM and any tensor-core comparison is a labelled secondary-device experiment.

### Compute capability gate

Record compute capability explicitly. **sm_75 means `cp.async`, `ldmatrix`, and `mma` intrinsics are unavailable** — these require sm_80 or newer. Stage 9 flash attention uses plain shared-memory staging. Any prompt proposing those intrinsics is wrong and must be rejected.

## 2. CPU

| Field | Value | Tag |
|---|---|---|
| Model | | `[queried]` |
| Physical cores / threads | | `[queried]` |
| Base / boost clock | | `[queried]` |
| L1d per core | | `[queried]` |
| L2 per core | | `[queried]` |
| L3 total | | `[queried]` |
| Cache line size | | `[queried]` |
| **Effective cache sizes from the ladder** | | `[measured]` |
| Widest SIMD ISA available | | `[queried]` |
| **Measured DRAM bandwidth** | | `[measured]` |
| **Measured L1 / L2 / L3 bandwidth** | | `[measured]` |
| **Measured peak FP32 (SIMD)** | | `[measured]` |

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
| Nsight Compute available | | `[queried]` |
| Version | | `[queried]` |
| Counter collection permitted (may need admin / driver flag on Windows) | | `[queried]` |
| Counters confirmed collectable | | `[measured]` |

Minimum counters needed downstream: achieved occupancy, DRAM read/write throughput, L2 hit rate, shared memory bank conflicts, warp stall reasons, and instructions executed. **If counter collection is blocked, flag it in Stage 0 rather than discovering it at Stage 8.** On Windows this commonly requires enabling GPU performance counters for all users in the NVIDIA control panel, or running the profiler elevated.

## 5. Machine state and stability

*The machine must be in a controlled, reproducible state before any measurement is trustworthy. A measurement taken on a machine whose clocks move is not a measurement of anything.*

### 5.1 Clock and power state

| Field | Value | Tag |
|---|---|---|
| GPU core clock offset vs stock | | `[queried]` |
| GPU memory clock offset vs stock | | `[queried]` |
| Power limit (current / default / max) | | `[queried]` |
| Temperature limit | | `[queried]` |
| Overclocking utility present and running (MSI Dragon Center, Afterburner, etc.) | | `[queried]` |
| Active performance / shift mode profile | | `[queried]` |
| Fan profile | | `[queried]` |
| Windows power plan | | `[queried]` |
| Hardware-accelerated GPU scheduling enabled | | `[queried]` |
| Can clocks be locked (`nvidia-smi -lgc` / `-lmc`)? | | `[queried]` |
| Clocks locked for this project? | | `[queried]` |

### 5.2 The overclock decision

**Any non-zero clock offset must be resolved before Stage 1.** Three risks, in increasing severity:

1. **Ceilings become those of an overclocked card.** Survivable if fixed and documented, since the project reports ratios against measured ceilings. Fatal if the overclock changes mid-project — a profile that does not persist across reboots means Stage 5 and Stage 9 were measured on different machines and the waterfall is void.
2. **Variance.** Aggressive clocks drift more under sustained thermal load, pushing runs past the 5%-std-dev invalidity threshold for reasons unrelated to the code.
3. **Silent numerical corruption.** Consumer GDDR6 has no ECC. An unstable memory overclock does not crash — it returns slightly wrong values. This surfaces as intermittent correctness-gate failures that vanish on re-run, and would be attributed to kernel logic rather than to the memory clock. For a project gated on numerical correctness against a reference, this is the worst available failure mode.

**Default recommendation: run stock for the project's duration.** Nothing is gained by the overclock here — the reported figures are ratios, not absolute performance — and an entire class of confusing failure is removed. If the overclock is kept, it must be verified stable per §5.4, documented here, and unchanged for the life of the project.

Record the decision and its date:

| Decision | Value |
|---|---|
| Running stock or overclocked | |
| If overclocked, exact offsets | |
| Stability verified (§5.4) | |
| Date fixed | |

### 5.3 Thermal and boost behavior under sustained load

A single short benchmark does not reveal throttling. Run a sustained load and log clock, temperature, and power over time.

| Field | Value | Tag |
|---|---|---|
| Idle clock / temp | | `[measured]` |
| Clock after 30s sustained load | | `[measured]` |
| Clock after 5min sustained load | | `[measured]` |
| Peak temperature reached | | `[measured]` |
| Throttle reason reported, if any | | `[queried]` |
| **Time until clocks stabilize** | | `[measured]` |
| **Maximum safe continuous benchmark duration** | | `[derived]` |

The stabilization time sets the warmup requirement. If clocks take 20 seconds to settle, five warmup iterations of a fast kernel are not enough, and `BENCHMARK_PROTOCOL.md` §3 must be adjusted upward for this machine. State the adjusted value.

### 5.4 Stability and determinism checks

All four must pass before Stage 1. A failure here is a hard stop, not a warning.

1. **VRAM integrity.** Allocate a large buffer, write known patterns, read back, verify bit-exact. Repeat under thermal load. Any mismatch means the memory overclock is unstable — drop to stock and re-run.
2. **Compute determinism.** Run the same microbenchmark's *numerical output* twice and verify bit-identical results. Floating-point GPU kernels with a fixed reduction order should be deterministic; non-determinism here indicates a hardware or driver problem, not a code problem.
3. **Timing reproducibility.** Run the full microbenchmark suite twice in the session. Medians should agree within a few percent. Record the observed run-to-run spread — it is the noise floor for every later measurement, and no stage can claim a speedup smaller than it.
4. **Background load.** Enumerate processes using GPU or significant CPU. Record what was running. Browsers, Discord, and overlay software are the usual offenders.

### 5.5 Environment fingerprint

Recorded here and re-verified at the start of every later stage. If any value has changed, the comparison against prior stages is void until re-established.

| Field | Value |
|---|---|
| Driver version | |
| CUDA toolkit version | |
| Compiler and version | |
| Clock offsets | |
| Power limit | |
| Windows power plan | |
| OS build | |

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
| Does the primary GPU report tensor cores? | Stage 11 framing | open |
| Compute capability — confirm sm_75 | Stage 9 implementation | open |
| Widest SIMD ISA on this CPU | Stage 6 prediction | open |
| Can Nsight Compute collect counters here? | Stage 10 and all gap analysis | open |
| Measured versus theoretical bandwidth ratio | every GPU prediction | open |
| **Current clock offsets, and stock-or-overclocked decision** | **every measurement in the project** | open |
| Does VRAM pass the integrity check at current clocks? | correctness gate reliability | open |
| Can clocks be locked on this card? | measurement variance | open |
| Time until clocks stabilize under load | warmup requirement | open |
| Run-to-run noise floor | minimum claimable speedup | open |
