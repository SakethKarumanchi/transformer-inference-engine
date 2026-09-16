# PERSISTENT.md

*Cross-chat state. Read before writing any Claude Code prompt. Updated at the end of every session.*

**Last updated:** project initialization
**Current stage:** 0 — not started
**Primary device:** GTX 1650 Ti (D1 resolved)

---

## 1. Open decisions

| # | Decision | Blocks | Status |
|---|---|---|---|
| D1 | Primary measurement device | everything | **RESOLVED — GTX 1650 Ti, fixed for project life** |
| D2 | Numerical tolerance for the correctness gate | Stage 3 | open |
| D3 | Prompt set and token counts for benchmarking | Stage 3 | open |
| D4 | Sequence lengths for the flash attention memory sweep | Stage 9 | open |
| D5 | Performance model structure — which terms to include | Stage 10 | open, decide at Stage 10 |
| D6 | Which optional stages to build | after Stage 12 | open |
| D7 | Whether to attempt Stage 16 upstream contribution | stretch | open |

## 2. Open questions requiring verification

| # | Question | Tag | Blocks |
|---|---|---|---|
| Q1 | 1650 Ti SM count, clocks, bus width, achievable bandwidth | `[training]` → needs `[queried]`/`[measured]` | every GPU prediction |
| Q2 | Widest SIMD ISA on this CPU | needs `[queried]` | Stage 6 |
| Q3 | GPT-2 small config values — confirm against shipped config, never hardcode from memory | needs `[doc]` | Stage 1 |
| Q4 | **Can Nsight Compute collect counters on this machine?** Windows commonly requires enabling GPU performance counters for all users, or elevation | needs `[queried]` | Stage 10 and every gap analysis |
| Q5 | Confirm compute capability is sm_75 | needs `[queried]` | Stage 9 implementation |
| Q6 | Does the primary GPU report tensor cores? Expected no (Turing TU117) | needs `[queried]` | Stage 11 framing |
| Q7 | Free-tier GPU quotas (Colab, Kaggle) and rental pricing | `[training]` ⚠️ changes often | secondary comparisons |
| Q8 | Published roofline baseline ~34% average error on DL kernels (NeuSight, arXiv 2407.13853) | `[training]` ⚠️ verify before citing | Stage 10 positioning |
| Q9 | **Current GPU clock offsets.** MSI Dragon Center has been used to apply core and memory offsets. Exact values unknown | needs `[queried]` | **every measurement in the project** |
| Q10 | Does VRAM pass an integrity check at current clocks? | needs `[measured]` | correctness gate reliability |
| Q11 | Can clocks be locked on this card (`nvidia-smi -lgc` / `-lmc`)? | needs `[queried]` | measurement variance |
| Q12 | Time until clocks stabilize under sustained load | needs `[measured]` | warmup requirement |
| Q13 | Run-to-run noise floor of the microbenchmark suite | needs `[measured]` | minimum claimable speedup |

## 2.1 D8 — STOCK OR OVERCLOCKED. Decide in Stage 0, before Stage 1.

MSI Dragon Center has been used to apply core clock and VRAM offsets. This must be resolved before any inference code exists. Three risks, increasing in severity:

1. **Ceilings become an overclocked card's ceilings.** Survivable if fixed and documented. Fatal if the profile changes mid-project — a Dragon Center profile that does not persist across reboots, or a different shift mode on battery, means different stages measured different machines and the waterfall is void.
2. **Variance.** Aggressive clocks drift more under thermal load, pushing runs past the 5%-std-dev invalidity threshold for reasons unrelated to the code.
3. **Silent numerical corruption.** Consumer GDDR6 has no ECC. An unstable memory overclock does not crash — it returns slightly wrong values. This would surface as intermittent correctness-gate failures that vanish on re-run, most likely around Stage 8, and would be attributed to kernel logic rather than to the memory clock. Days lost debugging the wrong thing.

**Recommendation: run stock for the project's duration.** Nothing is gained here — the reported figures are ratios against measured ceilings, not absolute performance — and an entire class of confusing failure disappears. If the overclock is kept, it must pass the `HARDWARE.md` §5.4 stability checks, be documented exactly, and stay unchanged for the life of the project.

## 3. Hard constraints

- **sm_75 gate.** No `cp.async`, no `ldmatrix`, no `mma` intrinsics anywhere. These require sm_80+. Stage 9 flash attention uses plain shared-memory staging. Any prompt proposing them is wrong.
- **Primary device never changes.** Headline figures are ratios against this device's own measured ceilings.
- **4GB VRAM.** Constrains model size and the flash attention sweep's maximum sequence length.
- **Environment fingerprint frozen after Stage 0.** Driver version, CUDA toolkit, compiler, clock offsets, power limit, and Windows power plan are fixed. Every stage re-verifies before its first timed run. Any change voids comparisons against prior stages until reverted or re-measured.
- **No stage claims a speedup smaller than the Stage 0 noise floor.** A 3% gain on a machine with 5% run-to-run spread is noise.

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
| | | | | | |

## 7. Flags for the next session

*Cleared and rewritten each session.*

- Stage 0 produces no inference code. Resist scope creep into Stage 1.
- Stage 0 must resolve Q1, Q2, Q4, Q5, Q6, Q9, Q10, Q11, Q12, Q13 and decision D8.
- **Check Q4 (profiler counter access) and Q9 (clock offsets) EARLY in the session, not last.** Both can invalidate work done after them. A blocked profiler changes the toolchain choice; an unresolved overclock contaminates every measurement taken before it is settled.
- D8 must be decided before Stage 1. Recommendation is stock.
- The Stage 0 prompt must not pre-state any hardware figure. If it tells Claude Code what bandwidth to expect, the measurement is contaminated.
