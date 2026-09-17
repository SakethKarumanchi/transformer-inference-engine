# transformer-inference-engine

A GPT-2 inference engine written from scratch in C and CUDA, optimized in measured stages with every speedup predicted before it was measured.

No ML library sits in the inference path. PyTorch appears only as a correctness
oracle. Prefill and decode are always measured and reported separately. The
primary measurement device never changes for the life of the project, and no
performance figure is estimated, projected, or rounded into existence: every
number comes from a run that happened, and where a run did not happen the field
is left empty with a stated reason.

## Stage 0: instrumenting the machine

Stage 0 contains no inference code. It builds the measurement foundation the
rest of the project stands on: a build system, a shared measurement harness,
nine reusable microbenchmarks, machine-state tooling, and an Nsight Compute
wrapper. Its output is a populated `HARDWARE.md` describing this specific
machine as measured, not as advertised.

## HARDWARE.md holds measured and queried values, not spec-sheet figures

Every value in `HARDWARE.md` carries a tag saying where it came from. The tags
are not decoration; the Stage 10 performance model is allowed to consume some
of them and forbidden from consuming others.

| Tag | Meaning |
|-----|---------|
| `[queried]` | Returned by a runtime API: `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, NVML, `nvidia-smi`, `lscpu`, CPUID. |
| `[measured]` | Produced by a microbenchmark in this repository. The entry names which one. |
| `[spec]` | Vendor documentation. Theoretical, flagged, and **never usable in a prediction or in the performance model**. It exists only as a ceiling to check a measurement against. |
| `[derived]` | Computed from the above, with the arithmetic shown inline. |

A figure quoted from a product page is a `[spec]` figure no matter how
plausible it looks, and it never becomes a `[measured]` one by being repeated.

## Repository layout

```
/
  PROJECT.md              what the project is and why
  HARDWARE.md             this machine, measured and queried, every value tagged
  TECHNICAL_SPEC.md       the engine's design
  BENCHMARK_PROTOCOL.md   how a measurement is taken and when it is invalid
  MEASUREMENTS.md         the per-stage record of predictions and results
  LEARNING.md             concepts encountered, marked read or unread
  PERSISTENT.md           open questions, decisions, and project-wide constraints
  README.md               this file
  CMakeLists.txt          the build; the flag set here is frozen for the project
  src/
    gemm/                 (Stage 2 onward)
    cuda/                 (Stage 2 onward)
  bench/
    microbench/           the nine microbenchmarks and the shared harness
    results/              structured JSON results; deliverables, not artifacts
    machine_state.py      HARDWARE.md sections 5.1-5.5 as reusable code
    profile.py            Nsight Compute wrapper
  model/results/          (Stage 10 performance model output)
  reference/              (PyTorch oracle scripts, Stage 2 onward)
  dashboard/              (Stage 11)
  scripts/                build driver and generated-header templates
  tests/                  the unit test suite
```

`bench/results/` is deliberately **not** in `.gitignore`. Those files are
deliverables: the Stage 10 performance model reads them directly.

## Building

The build path is native Windows with the NVIDIA CUDA Toolkit and MSVC.

Two Visual Studio toolsets are installed on the measurement machine. CUDA 13.1
accepts Visual Studio 2019 through 2022 and rejects the 2026 toolset
(MSVC 19.50+) in `host_config.h`. Auto-detection picks the newest, which is the
wrong one, so the toolset is pinned two ways:

1. `scripts/build.ps1` enters the VS 2022 Build Tools environment and passes
   that `cl.exe` explicitly as `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` and
   `CMAKE_CUDA_HOST_COMPILER`.
2. `CMakeLists.txt` fails the configure step outright if it still ends up with
   `MSVC_VERSION >= 1950`.

`-allow-unsupported-compiler` is never used. It permits incorrect runtime
execution, and this project is gated on numerical correctness against a
reference.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1            # configure + build
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Clean     # from scratch
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Test      # build then run the suite
```

The configure step prints the exact compiler and flag set it will use, and the
same strings are baked into `build_info.h` so every results file carries the
toolchain that produced it. **Those flags are frozen for the project.** Later
stages compare against Stage 0 and the comparison is void if they change; they
are recorded in `HARDWARE.md` section 5.5.

## Running the microbenchmark suite

Benchmarks must run on an idle machine: no other GPU or CPU workload, nothing
thermally throttling, not on battery, overlay and monitoring software closed.
Run the machine-state checks first.

```powershell
.venv\Scripts\python bench\machine_state.py query                    # 5.1 clock and power state
.venv\Scripts\python bench\machine_state.py sustained --duration 300 `
    --load build\gpu_fp32_peak.exe                                   # 5.3 thermal and boost log
.venv\Scripts\python bench\machine_state.py stability                # 5.4 the four stability checks
.venv\Scripts\python bench\machine_state.py fingerprint              # 5.5 capture
.venv\Scripts\python bench\machine_state.py verify                   # 5.5 compare against the capture
.venv\Scripts\python bench\machine_state.py verify-lock --mhz 1500 `
    --load build\gpu_fp32_peak.exe                                   # confirm a clock lock took
```

Then the nine:

```powershell
.venv\Scripts\python bench\microbench\run_all.py
.venv\Scripts\python bench\microbench\run_all.py --only gpu_bandwidth
.venv\Scripts\python bench\microbench\run_all.py --compare bench\results\stage0_consolidated.json
```

Each benchmark writes its own `bench/results/<name>.json`; the driver writes one
consolidated `bench/results/stage0_consolidated.json`. `--compare` re-runs the
suite and reports every value that moved against a stored Stage 0 file, which is
how a later stage confirms the machine has not drifted.

Exit codes: `0` every configuration valid, `2` ran but at least one
configuration exceeded the 5%-of-median standard deviation limit and is reported
INVALID, anything else a failure. **An invalid run is reported as invalid.** It
is never averaged away and never silently retried.

Profiling:

```powershell
.venv\Scripts\python bench\profile.py --list-metrics
.venv\Scripts\python bench\profile.py --exe build\gpu_bandwidth.exe
```

`profile.py` resolves metric names against the installed Nsight Compute rather
than hardcoding them, and reports each required counter as populated with its
value or unavailable with the reason. It never emits a blank that could be
mistaken for zero, and it fails loudly rather than returning an empty result set
when the profiler is blocked.

## The nine microbenchmarks

| # | Binary | Reports |
|---|--------|---------|
| 1 | `gpu_bandwidth` | Device-to-device bandwidth, working set sized beyond the queried L2, CUDA-event timed. GB/s |
| 2 | `gpu_fp32_peak` | Register-resident FMA chains, no memory traffic in the loop. GFLOP/s |
| 3 | `cpu_cache_ladder` | Streaming-read bandwidth across working-set sizes, full curve plus inferred plateau edges |
| 4 | `cpu_simd_peak` | FMA loop at the widest available vector width, with a scalar reference. GFLOP/s |
| 5 | `host_device_transfer` | Pinned and pageable, both directions, four separate figures. GB/s |
| 6 | `cublas_sgemm_ref` | cuBLAS SGEMM at GPT-2 small's real shapes, never square, prefill and decode separate. GFLOP/s |
| 7 | `kernel_launch_overhead` | Empty-kernel launch, synchronized-per-launch and back-to-back. us |
| 8 | `shared_mem_bandwidth` | Shared memory, conflict-free and 32-way conflicting. GB/s |
| 9 | `occupancy_sweep` | Achieved against theoretical occupancy across block sizes and register pressures |

## Measurement protocol

The rules live in one place, `bench/microbench/bench_common.h`, and nowhere
else:

- at least 5 warmup iterations, discarded;
- at least 20 timed samples per configuration;
- median reported, with min, max and sample standard deviation recorded
  alongside;
- a run is **INVALID** if the standard deviation exceeds 5% of the median, if
  another workload was running, if thermal throttling occurred, or if the
  machine was on battery.

CPU timing uses the highest-resolution monotonic counter the platform offers —
`QueryPerformanceCounter` on Windows, `clock_gettime(CLOCK_MONOTONIC)`
elsewhere — never wall clock. GPU timing uses CUDA events with an explicit
device synchronize before the timer is read, because kernel launches are
asynchronous and timing without synchronizing measures launch overhead rather
than execution. Timing brackets computation only.

## Tests

Structural tests: they assert that a benchmark is correctly constructed, not
that this machine is fast. They therefore run in the offline gate before any
timed measurement and do not require an idle machine. Everything runs from one
command:

```powershell
ctest --test-dir build --output-on-failure
```

| Test | What it asserts |
|------|-----------------|
| `test_bench_common` | Median, min, max and sample standard deviation against a hand-computed answer; the 5% rule on both sides of the threshold; warmup iterations excluded from the statistics; the emitted JSON parses and carries the raw per-sample array |
| `test_gpu_bandwidth` | Finite, positive, and not above the theoretical peak recomputed from the queried bus width and memory clock; working set beyond the queried L2; protocol minimums met |
| `test_gpu_fp32_peak` | No global memory traffic in the inner loop, asserted by scanning the generated PTX; not above the theoretical peak from queried core count and clock; FLOP count derived from the loop trip count |
| `test_cpu_cache_ladder` | The sweep brackets every OS-reported cache level; bandwidth non-increasing within the measured noise; at least one plateau edge detected; the OS figures carried through beside the measured edges |
| `test_cpu_simd_peak` | The compiled ISA matches the widest the CPU reports; the scalar reference is slower than the vector run, which is the evidence vectorisation happened |
| `test_host_device_transfer` | Four finite positive figures, never merged; transfers long enough that launch overhead is irrelevant |
| `test_cublas_sgemm_ref` | No shape is square; every N and K matches the architecture-derived values; the M sweep includes 1; the results file records the provenance string; the cuBLAS handle lifecycle and every status are checked |
| `test_kernel_launch_overhead` | The kernel body is genuinely empty, asserted from the PTX; the two cases reported separately; launch-to-completion costs more than a pipelined launch, which is the evidence the synchronize is inside the timed region |
| `test_shared_mem_bandwidth` | The working set fits the queried shared memory per block; the access pattern is stated; the conflicting variant is measured alongside and is slower |
| `test_occupancy_sweep` | Block sizes are multiples of the queried warp size and within the queried maximum; theoretical occupancy recomputed from queried properties; achieved occupancy left for the profiler rather than copied from theoretical; at least one configuration register-limited and named |
| `test_run_all` | All nine registered and run; a failed benchmark reported as failed with no stale or placeholder value in the consolidated file; drift mode flags a changed value against a synthetic prior file |
| `test_machine_state` | The fingerprint verify mode detects a changed field; the VRAM integrity check detects a deliberately corrupted buffer; the stabilization time is derived from the logged series rather than defaulted; the spread calculation is correct over two synthetic runs |
| `test_profile` | Every required counter appears in the mapping; an uncollected counter is marked unavailable with its error text rather than defaulted; the wrapper fails loudly when the profiler is blocked |

The shared-versus-global bandwidth comparison is deliberately *not* a unit
test. Unit tests run before any timing exists, so global bandwidth is
unavailable to them; `run_all.py` makes that comparison after both benchmarks
have run and reports it as a finding.
