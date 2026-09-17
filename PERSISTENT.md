# PERSISTENT.md

*Cross-chat state. Read before writing any Claude Code prompt. Updated at the end of every session.*

**Last updated:** 2026-09-17 (Stage 0 complete)
**Current stage:** 0 — complete; Stage 1 is next
**Primary device:** GTX 1650 Ti (D1 resolved)

---

## 1. Open decisions

| # | Decision | Blocks | Status |
|---|---|---|---|
| D1 | Primary measurement device | everything | **RESOLVED — GTX 1650 Ti, fixed for project life** |
| D2 | Numerical tolerance for the correctness gate | Stage 3 | open |
| D3 | Prompt set and token counts for benchmarking | Stage 3 | open — **Stage 0 flag:** the cuBLAS reference in `bench/microbench/cublas_sgemm_ref.cu` swept M across 1, 8, 16, 32, 64, 128, 256, 512, 1024 as a placeholder. Stage 3 fixes the real prompt set and token counts, and every later stage holds them constant |
| D4 | Sequence lengths for the flash attention memory sweep | Stage 9 | open |
| D5 | Performance model structure — which terms to include | Stage 10 | open, decide at Stage 10 |
| D6 | Which optional stages to build | after Stage 12 | open |
| D7 | Whether to attempt Stage 16 upstream contribution | stretch | open |
| D8 | Stock or overclocked, and the clock regime | every measurement | **RESOLVED 2026-09-17.** MSI Dragon Center user scenario **"Balanced"** for the life of the project, **graphics clock locked at 1365 MHz**. Memory clock cannot be locked on this device, and whether Balanced is stock is UNVERIFIED, so the mitigation is **PARTIAL**. Pre-decided by the operator; Stage 0 chose and applied the lock value. See §2.1 |

## 2. Open questions requiring verification

| # | Question | Tag | Blocks |
|---|---|---|---|
| Q1 | 1650 Ti SM count, clocks, bus width, achievable bandwidth | **RESOLVED 2026-09-17** `[queried]`/`[measured]` | **CLOSED.** 16 SM; rated clock 1485 MHz, max 2100 MHz, **operating clock locked at 1365 MHz**; 128-bit bus; memory clock 6001 MHz; theoretical bandwidth 192.032 GB/s; **measured achievable 170.882 GB/s, efficiency 0.8899**. `HARDWARE.md` §1 |
| Q2 | Widest SIMD ISA on this CPU | **RESOLVED 2026-09-17** `[queried]` | **CLOSED. AVX2 + FMA3**, by CPUID leaf 7 EBX bit 5 with YMM state enabled; AVX-512F bit absent, ZMM state not enabled. Compiled `/arch:AVX2`; the benchmark asserts compiled-ISA == widest-supported-ISA. **Caveat for Stage 6: the vectorised peak figure itself is INVALID** (std dev 5.86% / 5.60% across two runs) and `HARDWARE.md` §2 leaves it empty |
| Q3 | GPT-2 small config values — confirm against shipped config, never hardcode from memory | needs `[doc]` | Stage 1 — **still open, and now load-bearing.** Stage 0 read `n_embd` 768, `n_head` 12, `n_layer` 12, `n_ctx` 1024, `vocab_size` 50257 from the published `config.json` of `openai-community/gpt2` (fetched 2026-09-16) and built the entire cuBLAS prefill denominator on them. **Stage 1 must re-confirm these against the config shipped with the downloaded weights**; if any differs, microbenchmark 6 must be re-run because it is the denominator for the whole project |
| Q4 | **Can Nsight Compute collect counters on this machine?** | **RESOLVED 2026-09-17** `[measured]` | **CLOSED. Yes — fully.** ncu 2025.4.0.0. All 32 required metric names resolved against the installed version, and **all 32 populated with values on a live kernel** (microbenchmark 1); zero unavailable, zero unresolved, nothing blank or defaulted to zero. Mapping and per-counter status in `HARDWARE.md` §4 and `bench/results/profile_gpu_bandwidth.json`. Note: collection for a **non-elevated** user was established by the operator pre-session; the Stage 0 session itself ran elevated and did not re-test that |
| Q5 | Confirm compute capability is sm_75 | **RESOLVED 2026-09-17** `[queried]` | **CLOSED. 7.5 confirmed**, agreed independently by `cudaGetDeviceProperties` (major.minor = 7.5) and `nvidia-smi --query-gpu=compute_cap` (7.5). The no-`cp.async` / no-`ldmatrix` / no-`mma` rule is now a hard constraint in §3, not an expectation |
| Q6 | Does the primary GPU report tensor cores? | **RESOLVED 2026-09-17** `[measured]` — **the expectation was WRONG** | **CLOSED, with a divergence.** No property reports them: `cudaDeviceProp` has **no tensor-core field** and CUDA exposes no API, so this cannot be answered by query at all. Determined empirically: a WMMA 16x16x16 HMMA kernel compiled for `sm_75`, ran, and returned all 256 elements exactly 32.0; `sm__inst_executed_pipe_tensor_op_hmma.sum` = 4 and `sm__pipe_tensor_cycles_active.sum` = 512. **A tensor pipe exists on this die and executes `hmma`.** No throughput was measured and none may be claimed — 128 cycles/instruction cannot separate full-rate silicon at low occupancy from a reduced implementation. **`cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)` returns SUCCESS on this card and is NOT a valid tensor-core test** — recorded so it is not repeated. **Stage 11 must revisit its framing** (see §7) |
| Q7 | Free-tier GPU quotas (Colab, Kaggle) and rental pricing | `[training]` ⚠️ changes often | secondary comparisons |
| Q8 | Published roofline baseline ~34% average error on DL kernels (NeuSight, arXiv 2407.13853) | `[training]` ⚠️ verify before citing | Stage 10 positioning |
| Q9 | **Current GPU clock offsets.** MSI Dragon Center has been used to apply core and memory offsets. Exact values unknown | **STILL OPEN — UNVERIFIED, and NOT closable by query on this machine** | **Q9 CANNOT BE CLOSED.** `nvidia-smi -q` contains no clock-offset field at all (zero matching lines for Offset / Locked / GpuLock); Applications Clocks and Default Applications Clocks return "Requested functionality has been deprecated". Stage 0 additionally retried the vendor WMI path **with elevation** (previously Access denied): `root\WMI MSI_VGA` reads but returns 18 unlabeled integers with no schema and no offset field; `MSI_ACPI` returns "Not supported". Supporting evidence only: MSI user scenario **"Balanced"** `[operator-observed]`, fan profile not applicable (no control exposed for that scenario), MSI Afterburner installed but not running with `CoreClkBoost=` / `MemClkBoost=` both empty. **Whether Balanced is stock remains UNVERIFIED. No offset of zero is recorded anywhere, and none may be.** Mitigated, not resolved, by the D8 clock lock |
| Q10 | Does VRAM pass an integrity check at current clocks? | **RESOLVED 2026-09-17** `[measured]` | **CLOSED. PASS.** 2048 MiB buffer, patterns `0x00`/`0xFF`/`0xAA`/`0x55`, 3 rounds each = 12 write/readback cycles, every one bit-exact with zero bad bytes, run immediately after the five-minute thermal load at 82–87 C. Compute determinism also bit-identical. **This is the ONLY evidence available on memory stability in this project, because the memory clock cannot be pinned** — treat it as such and re-run it if intermittent correctness failures ever appear |
| Q11 | Can clocks be locked on this card (`nvidia-smi -lgc` / `-lmc`)? | **RESOLVED 2026-09-17** `[queried]` | **CLOSED, split answer. Graphics clock: YES, with elevation** — non-elevated `-lgc` returns "The current user does not have permission to change clocks" with exit code 4, a permission limit. **Locked at 1365 MHz and verified in effect.** **Memory clock: NO — device-capability limit.** `-lmc` returns "Setting locked Memory clocks is not supported for GPU 00000000:01:00.0." with **exit code 0**; exit 0 plus that message means elevation would not help. **Do not attempt `-lmc` again.** Consequence: the clock regime is **partially, not fully, controlled** |
| Q12 | Time until clocks stabilize under sustained load | **RESOLVED 2026-09-17** `[measured]` | **CLOSED. 0.075 s under the lock**, and structurally zero because `-lgc 1365,1365` pins the minimum as well as the maximum, so the card idles at its operating clock and no ramp exists. Unlocked it is 3.502 s by the stabilization criterion, but that figure is misleading: the unlocked card never stabilizes, it decays 1815 → 1365 MHz over five minutes. **Adjusted warmup: 25 iterations**, arithmetic in `BENCHMARK_PROTOCOL.md` §4.2 |
| Q13 | Run-to-run noise floor of the microbenchmark suite | **RESOLVED 2026-09-17** `[measured]` | **CLOSED. Adopted floor 4.4%** (90th percentile). Full suite run twice; across the 49 configurations VALID in both runs the spread was **median 0.317%, mean 1.407%, p90 4.401%, max 19.913%**. Two qualifications travel with it: **small-M GEMM shapes are materially noisier and need a wider margin**, and the floor was measured on a machine with irreducible background load |

## 2.1 D8 — STOCK OR OVERCLOCKED. Decide in Stage 0, before Stage 1.

MSI Dragon Center has been used to apply core clock and VRAM offsets. This must be resolved before any inference code exists. Three risks, increasing in severity:

1. **Ceilings become an overclocked card's ceilings.** Survivable if fixed and documented. Fatal if the profile changes mid-project — a Dragon Center profile that does not persist across reboots, or a different shift mode on battery, means different stages measured different machines and the waterfall is void.
2. **Variance.** Aggressive clocks drift more under thermal load, pushing runs past the 5%-std-dev invalidity threshold for reasons unrelated to the code.
3. **Silent numerical corruption.** Consumer GDDR6 has no ECC. An unstable memory overclock does not crash — it returns slightly wrong values. This would surface as intermittent correctness-gate failures that vanish on re-run, most likely around Stage 8, and would be attributed to kernel logic rather than to the memory clock. Days lost debugging the wrong thing.

**Recommendation: run stock for the project's duration.** Nothing is gained here — the reported figures are ratios against measured ceilings, not absolute performance — and an entire class of confusing failure disappears. If the overclock is kept, it must pass the `HARDWARE.md` §5.4 stability checks, be documented exactly, and stay unchanged for the life of the project.

**RESOLVED 2026-09-17. D8: MSI Dragon Center user scenario "Balanced" for the life of the project, with the graphics clock locked at 1365 MHz.** Pre-decided by the operator and not reopened by Stage 0; Stage 0's job was to choose the lock value from measurement and apply it.

**The mitigation is PARTIAL, and that must not be forgotten.** Three things remain true:

1. **Whether Balanced is stock is UNVERIFIED** and cannot be verified on this machine — see Q9. No offset of zero is recorded.
2. **The memory clock cannot be locked** on this device (Q11), so only the graphics clock is pinned. Risk 3 above — silent numerical corruption from an unstable memory clock — is therefore mitigated only by the §5.4 VRAM integrity check, which **passed** under thermal load. That check is the sole evidence available and should be re-run at the first sign of intermittent correctness failures.
3. **The lock value was chosen from measurement, and the specified criterion could not be met.** The criterion was the highest clock held for five minutes with no throttle reason appearing; on this card the SW power cap asserts at t = 3.5 s and never clears, so no such clock exists. The highest clock actually *sustained* across the window was taken instead: 1365 MHz. The choice is vindicated — at 1365 MHz the card ran five minutes with **no throttle reason of any kind**, 122 of 122 samples at exactly 1365 MHz, peak 82 C.

## 3. Hard constraints

- **sm_75 gate — CONFIRMED BY QUERY 2026-09-17, no longer an expectation.** Compute capability 7.5, agreed by `cudaGetDeviceProperties` and `nvidia-smi`. No `cp.async`, no `ldmatrix`, no `mma` intrinsics anywhere. These require sm_80+. Stage 9 flash attention uses plain shared-memory staging. Any prompt proposing them is wrong. **This holds even though a tensor pipe was measured to be present on this die (Q6) — the prohibition is a project constraint, not a hardware inference.**
- **Primary device never changes.** Headline figures are ratios against this device's own measured ceilings.
- **4GB VRAM.** Constrains model size and the flash attention sweep's maximum sequence length.
- **Environment fingerprint frozen after Stage 0.** Driver version, CUDA toolkit, compiler, clock offsets, power limit, and Windows power plan are fixed. Every stage re-verifies before its first timed run. Any change voids comparisons against prior stages until reverted or re-measured.
- **No stage claims a speedup smaller than the Stage 0 noise floor, measured at 4.4%.** A 3% gain on a machine with 4.4% run-to-run spread is noise. Small-M / decode-shaped work is noisier still and needs a wider margin.
- **The graphics clock must be locked before every timed run, in every later stage.** Clock locks do not survive a reboot. Run `nvidia-smi -lgc 1365,1365` from an **elevated** shell, then verify with `python bench/machine_state.py verify-lock --mhz 1365` — never assume it took. Do **not** attempt `-lmc`; it is unsupported on this device. Unlocked, this card decays from 1815 MHz to 1365 MHz over five minutes, so an unlocked measurement is a measurement of a different machine at its start than at its end.
- **All project Python runs through `.venv/Scripts/python.exe`.** A bare `python` on PATH resolves to a different system interpreter with no torch installed.

## 4. Known traps

- **Porting only the matmul to CUDA** makes Stage 7 a measurement of PCIe bandwidth. Weights stay resident; the whole forward pass moves.
- **Stage 11 may be slower than Stage 8.** Expected. Do not tune until it wins; do not build the dashboard or resume bullet assuming a monotonically improving waterfall.
- **Do not claim tensor cores are weight-stationary systolic arrays.** They are documented as units performing small matrix multiply-accumulates. The systolic framing is a TPU analogy.
- **Self-relative speedups are not headline numbers.** A slow baseline inflates them.
- **Naive online softmax loses precision at long sequences.** Stage 9 must verify specifically at the longest tested length, not just at short ones.
- **Do not fit the performance model to Stage 11.** Stage 11 is its prospective test. Any term added after seeing Stage 11 data must be stated, and Stage 11 then stops counting as prospective.
- **Stage 2 is slow by design.** A long runtime there is not a hang. Use reduced token counts for correctness tests and state that you did.
- **Nsight counter access may be blocked on Windows.** Discover this in Stage 0, not at Stage 8.
- **An unstable memory overclock looks like a kernel bug.** Intermittent correctness failures that vanish on re-run are the signature. If they appear at any stage, check clocks before debugging logic.
- **Overlay and monitoring software consumes GPU.** Dragon Center, Afterburner overlays, Discord, and browsers are the usual offenders. Enumerated in Stage 0, closed before every timed run.
- **Some background load on this machine is irreducible and it shows up as INVALID runs, not as slow runs.** Riot Vanguard (a kernel-mode anti-cheat driver), Nahimic (an audio driver that hooks process creation), MSI vendor services and Windows Defender real-time scanning were all resident during Stage 0 and could not be closed. The signature is a configuration that is VALID in one suite run and INVALID in the next **with medians agreeing to well under one percent** — an occasional multi-millisecond interruption inflating the standard deviation. Expect it, report it as invalid, and do not chase it as a code defect.
- **`cudaDeviceProp::clockRate` and `::memoryClockRate` were REMOVED in CUDA 13.** Read them via `cudaDeviceGetAttribute(cudaDevAttrClockRate / cudaDevAttrMemoryClockRate)` instead. `bench/microbench/device_info.cu` already handles this; a naive re-query is the single most likely thing to break.
- **`nvidia-smi --query-gpu=power.limit` returns `[N/A]` on this card.** The power-limit figures come from `nvidia-smi -q`.
- **The reported 92 C slowdown temperature is not where this card starts losing clock.** The driver asserted SW thermal slowdown at **86 C**. Use the measured figure when reasoning about thermal headroom.
- **cuBLAS SGEMM throughput is non-monotonic in M on this device** — it peaks at intermediate M and falls at M=1024 for the same shape. The prefill denominator is therefore **shape-dependent and must be quoted per shape, never as a single number.** Stage 0's counters do not establish the cause; confirming it needs per-kernel profiling across the M sweep.
- **Background load is a recorded run condition, not a checklist item.** Defender (MsMpEng), Riot Vanguard (`vgc`, `vgtray`), WmiPrvSE, SearchIndexer, nvcontainer, dwm and Claude Code itself cannot be closed and were present for every Stage 0 measurement. The 4.4% p90 noise floor was measured with them running, and 17 of 96 configurations were invalid in both suite runs with 30 more invalid in one. That process population is part of the measurement condition and is recorded in each stage's entry per `BENCHMARK_PROTOCOL.md` §4. If a later stage's set differs from Stage 0's, the floor may no longer apply and the difference must be stated.
- **A zero exit code does not mean a push succeeded.** Stage 0 saw `git push` hang on an interactive credential prompt while the wrapper reported exit 0 and the log read `fatal: could not read Username` — Git Credential Manager resolved the unqualified remote host to a stored account that was not the repository owner. Every push is confirmed with `git ls-remote` against the branch, comparing the remote ref to local HEAD, never by exit status.

## 5. Corrections to the original capstone spec

Errors in `capstone_project_spec.md`, superseded by these foundation documents. Recorded so they are not reintroduced.

1. Treated matmul as uniformly compute-bound and prescribed cache blocking as the fix. Decode is memory-bound; blocking barely helps it.
2. Contained no KV cache.
3. Described porting only the matmul to CUDA.
4. Asserted the CUDA systolic dataflow is "how tensor cores actually work." Not supported.
5. Cited "Marvell AI Infrastructure" as a top-ranked reachable role justifying the project. **No such role exists in the targets tracker — zero rows.** Marvell's actual openings are firmware, DV, DFT, validation, reliability. The project is correct for other reasons; this justification is unsupported and must not be re-cited.
6. Assumed the project was unsaturated. It is a known genre with a public course (`tiny-vllm`) teaching it, and Georgia Tech CS 8803 sets nearly this exact capstone. Hence the differentiators in `PROJECT.md` §5.
7. Contained no flash attention. Its absence in a serious inference engine is conspicuous.
8. Framed the prediction discipline as the project's differentiator. It is a learning mechanism, not a resume feature. The differentiator is the Stage 10 performance model — the same intellectual content, built as a program instead of a journal.

## 6. Session log

| Session | Stage | Date | Commit | Outcome | Flags raised |
|---|---|---|---|---|---|
| 1 | 0 | 2026-09-16 | none | Recon; authored the Stage 0 code tree on branch `stage-0`. Nothing committed | — |
| 2 | 0 | 2026-09-16 | none | Re-ran the offline gate; fixed `test_cpu_cache_ladder` (replaced an unsound point-to-point monotonicity assertion with a dynamic-range check). Stopped at the hard checkpoint on operator instruction | — |
| 3 | 0 | 2026-09-17 | see PR | **Stage 0 COMPLETE, run unattended and elevated.** Offline gate green (53/53 build, 13/13 tests). Clock locked at 1365 MHz and verified. Nine microbenchmarks run twice. Profiler verified per counter. `HARDWARE.md`, `BENCHMARK_PROTOCOL.md`, `MEASUREMENTS.md`, `PERSISTENT.md`, `LEARNING.md` populated | Q6 tensor-core divergence → **Stage 11 must re-frame**; Q3 shape re-confirmation → **Stage 1**; D3 prompt set → **Stage 3**; single-channel RAM → **Stages 5 and 6**; Balanced-plan re-test → **before Stage 5**; Q9 remains open and unclosable |

## 7. Flags for the next session

*Cleared and rewritten each session.*

**Raised by Stage 0, 2026-09-17.**

- **STAGE 1 — re-confirm the GEMM shapes.** Stage 0 built the cuBLAS prefill denominator on `n_embd` 768, `n_head` 12, `n_layer` 12, `n_ctx` 1024, `vocab_size` 50257, read from the published `config.json` of `openai-community/gpt2` on 2026-09-16. **Verify these against the config file shipped with the downloaded weights.** If any value differs, microbenchmark 6 must be re-run, because it is the prefill denominator for the entire project.
- **STAGE 3 — fix the prompt set and token counts (D3).** Stage 0's M sweep of 1, 8, 16, 32, 64, 128, 256, 512, 1024 is a placeholder. Once Stage 3 fixes the real values, every later stage holds them constant.
- **STAGE 11 — re-frame the tensor-core justification.** Its case against cuBLAS SGEMM was made conditional on this card having no tensor cores. **That condition is false**: a tensor pipe is present and executes HMMA (Q6). Stage 0 deliberately did not re-frame Stage 11. Note that no tensor-core *throughput* was measured, so the re-framing cannot assume one, and the project prohibition on `mma` / `ldmatrix` / `cp.async` is unaffected.
- **STAGES 5 and 6 — the CPU memory system is half as wide as the part numbers suggest.** One 8 GiB DIMM, **single channel**, rated 3200 MT/s and running at 2933, giving a theoretical ceiling of 23.464 GB/s rather than ~46.9 GB/s. Cache blocking and SIMD are being evaluated against that. Also note `HARDWARE.md` §2 leaves **measured DRAM bandwidth** and **measured SIMD peak** EMPTY because both came back INVALID in both suite runs — if Stage 5 or 6 needs either figure, it must measure it, not borrow one.
- **BEFORE STAGE 5 — a Balanced-power-plan re-test is worth doing.** On the unlocked card the GPU asserted a thermal throttle reason and lost 24.8% of its clock under sustained load. Stage 0's CPU telemetry was **static and unusable** (constant 2496 MHz and 73.05 C across 209 samples), so whether the Ultimate Performance plan contributes CPU heat **cannot be answered from Stage 0 data in either direction**. Any such re-test needs a working CPU telemetry source first. **Do not change the power plan casually — it is a frozen fingerprint value and changing it mid-project voids comparisons against prior stages.**
- **EVERY STAGE — lock the graphics clock before the first timed run.** `nvidia-smi -lgc 1365,1365` from an elevated shell, then `python bench/machine_state.py verify-lock --mhz 1365`. Locks do not survive a reboot. Never attempt `-lmc`.
- **EVERY STAGE — re-verify the environment fingerprint** with `python bench/machine_state.py verify` before the first timed run.
- **Q9 stays open and is not closable on this machine.** Do not spend session time re-attempting it; the elevated WMI path was already tried and exposes no offset field. Never record an offset of zero.
- **Expect INVALID runs and report them as invalid.** Of 96 configurations, 17 were invalid in both Stage 0 runs and 30 in one of the two, on a machine whose background load cannot be removed. Never average an invalid run away and never retry until a number looks acceptable.
