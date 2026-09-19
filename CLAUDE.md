# CLAUDE.md

Guidance for Claude Code working in this repository. Holds only what is true in **every**
stage. Each stage prompt is authoritative for its own scope and overrides defaults here.

## Project

**Transformer inference engine in C/CUDA.** GPT-2 small (124M) from published weights, no ML
library in the inference path, optimized through measured stages, with an analytical
performance model (Stage 10) as the primary deliverable.

Authoritative sources, in precedence order:
`PROJECT.md` (scope / done) → `TECHNICAL_SPEC.md` (what gets built, stage sequence) →
`BENCHMARK_PROTOCOL.md` (what counts as a real number) → `HARDWARE.md` (measured machine) →
`PERSISTENT.md` (cross-session state, open questions, hard constraints).

`MEASUREMENTS.md` is the per-stage record. `LEARNING.md` tracks concepts.

## Commands

```powershell
# Build (pins the VS2022 toolset; always build through this, never bare cmake)
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Clean

# Full test suite — the offline gate
ctest --test-dir build --output-on-failure --timeout 300

# Build + test in one
.\scripts\build.ps1 -Test

# Timed runs (operator-gated — never before the checkpoint)
python bench/microbench/run_all.py
python bench/machine_state.py
python bench/profile.py
```

Python is the repo-root `.venv` (3.14.2, torch 2.14.0+cu130, CUDA-capable). Do not reinstall it.

## Toolchain — frozen for the project

| | |
|---|---|
| CUDA | nvcc 13.1.80, `-arch=sm_75` |
| Host compiler | MSVC **19.44** (VS 2022 Build Tools, 14.44.35207) |
| CPU flags | `/arch:AVX2` — AVX2 + FMA3 is the widest ISA here, **no AVX-512** |

VS 2026 (MSVC 19.50/19.51) is also installed and is **rejected by CUDA 13.1**. Auto-detection
picks it — that is why `scripts/build.ps1` enters vcvars64 and `CMakeLists.txt` hard-fails on
`MSVC_VERSION >= 1950`. **Never** use `-allow-unsupported-compiler`; it permits incorrect runtime
execution and this project is gated on numerical correctness.

Compiler and flags are recorded in `HARDWARE.md` §5.5 and frozen. Changing them voids every
comparison against prior stages.

## Layout

```
*.md        seven foundation documents (root) — authoritative, in-place edits only
src/        gemm/ (C) cuda/ (CUDA C++) — engine, Stages 1-11
bench/      microbench/ (Stage 0, reusable + tested) harness.py correctness.py profile.py results/
model/      perf_model.py validate.py results/ — Stage 10
reference/  reference_impl.py — PyTorch oracle only
tests/      one test per source file, registered with CTest
scripts/    build.ps1 and CMake helpers
dashboard/  React, Stage 12
```

## Hard constraints — never violated, any stage

- **No sm_80+ intrinsics.** No `cp.async`, no `ldmatrix`, no `mma`. Device is Turing sm_75. Flash
  attention uses plain shared-memory staging. Any instruction proposing otherwise is wrong.
- **No ML library in the inference path.** PyTorch is a correctness oracle, never a component.
- **Primary device never changes** — GTX 1650 Ti, 16 SMs, 4 GB, 55 W hard power limit. Headline
  figures are ratios against this device's own measured ceilings.
- **Environment fingerprint frozen after Stage 0** (driver, CUDA, compiler, clock offsets, power
  limit, Windows power plan). Every stage re-verifies it before its first timed run.
- **Prefill and decode are separate workloads.** Never combined into one figure. Prefill is
  compute-bound, decode is memory-bandwidth-bound; that asymmetry is the project's central lesson.
- **No stage accepted without passing the correctness gate.** A faster wrong answer is a regression.

## Measurement discipline

Every value written into a document carries a tag: `[queried]` (runtime API) · `[measured]`
(a named benchmark in this repo) · `[spec]` (vendor doc — never usable in a prediction or the
Stage 10 model) · `[derived]` (show the arithmetic inline).

- Minimum 5 warmup iterations discarded, minimum 20 timed samples. Report median with min, max,
  std dev alongside.
- A run is **INVALID** if std dev exceeds 5% of median, another workload was running, thermal
  throttling occurred, or the machine was on battery. Report it as invalid — never average it
  away, never silently retry.
- CPU timing: monotonic counter only, never wall clock. GPU timing: CUDA events with an explicit
  device synchronize before reading. Timing brackets computation only.
- **No number is estimated, projected, or rounded into existence.** If a run did not happen, the
  field stays empty with a stated reason. Never a plausible placeholder.
- No stage claims a speedup smaller than the Stage 0 noise floor.
- GPU stages from Stage 7 collect Nsight counters; every gap explanation names the counter that
  supports it, or says the counters do not establish the cause.
- Results JSON in `bench/results/` is a contract the Stage 10 model reads: stage id, git commit,
  timestamp, device, compiler flags, **raw per-sample timings**, statistics, counters.

## Document rules

The seven foundation documents are the project's authoritative sources. **In-place edits only** —
fill blanks in existing tables, append to existing sections, preserve surrounding text exactly.
Never regenerate, restructure, or reorder one. Where a row cannot be filled, write the reason in
the cell. If a foundation document is missing, **STOP and report** — do not author a replacement.

`CLAUDE_CODE_PROMPT_FORMAT.md` and `KICKOFF_TEMPLATE.md` are the operator's authoring tools, not project
outputs. Do not edit, move, delete, or reference them.

Claude Code never writes, revises, improves, corrects, or comments on a prediction. The
prediction arrives in the stage prompt, is transcribed verbatim -- mistakes included -- and
is committed alone before any implementation code exists. After that commit it is sealed:
it is not consulted again until the Gap section, and it never influences a configuration, a
sample count, an implementation choice, or the decision to investigate an anomaly. A
prediction produced by the thing being measured is worthless.

## Session protocol

One stage per session, one commit-complete unit. Work on branch `stage-N` cut from `main`; the
session ends in a PR into `main`. Never commit directly to `main`.

Default posture is **recon-then-build**: verify the machine and the tree first, report anything
blocking before writing code against it, then build. Plan mode only when the prompt opts in.

Then, in order: self-run the offline gate (clean build + full test suite, 5-minute per-test cap,
fix-driven reruns only, no blind looping) → **STOP** and hand the operator a HARD CHECKPOINT →
wait for "proceed" → timed runs under protocol conditions → populate documents → commit, push,
open the PR. **No timed run, no commit, and no push before the operator says proceed.**

Commands that change device state (`nvidia-smi -lgc`, `-lmc`) need elevation this session does not
have. Choose the value, hand over the exact command, wait, then verify it actually took.

## Reporting

`STATE`, `DETERMINE`, and `CONFIRM` in a prompt are imperatives — each one is answered explicitly
in the final report, not implied by the work. Where something could not be obtained, say so and
say why. Where a re-run disagrees with a value carried forward from a prior session, the re-run
wins and the disagreement is stated.
