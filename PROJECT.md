# PROJECT.md

*Foundation document. Defines what this project is, what it is not, and what "done" means. Every other document is downstream of this one. If a proposed change does not serve something stated here, it is scope creep.*

---

## 1. What this is

**Transformer Inference Engine in C/CUDA.**

A from-scratch inference engine that loads GPT-2's published weights and generates text with no ML libraries in the inference path, optimized through measured stages, including a hand-written flash attention kernel and an **analytical performance model that predicts each kernel's runtime before it is measured**.

PyTorch appears only as a correctness oracle, never as a component.

## 2. What the deliverable actually is

Three things, in order of weight:

1. **The performance model and its validation data** — a program that predicts kernel runtime from hardware parameters, validated across every kernel in the project, with the error distribution reported.
2. **`MEASUREMENTS.md`** — per-stage predictions, measurements, and explained gaps, each gap backed by Nsight Compute hardware counters rather than a plausible story.
3. **The engine itself** — evidence that 1 and 2 describe something real.

A generated repository is worthless in an interview. A validated performance model is not, because it either predicts correctly or it does not, and either result is a finding.

## 3. Why this project exists

**It sits on the compute bottleneck.** Inference, not training, consumes most AI compute in production. At batch size 1, inference is limited by how fast weights move out of memory, not by arithmetic throughput. That single fact is the compute bottleneck in miniature, and it is what inference and compiler teams work on daily.

**It is the honest on-ramp to the second internship.** NVIDIA's TensorRT performance-software internship asks for C/C++ and Python, CPU and GPU architecture knowledge, deep learning familiarity, and experience with performance modelling, profiling, and code optimization. Interns develop optimized GPU kernels for deep learning inference. This project produces exactly that evidence, and "performance modelling" is not a stretched interpretation — Stage 10 is literally that. *(Verify the posting's wording against the live req before citing it in an application.)*

**It is a graduate-level scope.** Georgia Tech's CS 8803 (GPU Hardware and Software) sets a capstone covering numerically-stable softmax, tiled GEMM kernels, the FlashAttention forward pass in CUDA, a KV cache for autoregressive decoding, and end-to-end GPT-2 inference with custom kernels. That is roughly Stages 2 through 9 here. Stages 10 onward exceed it.

## 4. Who it targets

- **Internship #1 (this cycle):** Marvell, Intel Foundry, Analog Devices, Draper, GE Vernova, The Nuclear Company, Micron. These want C, computer architecture fluency, Python test automation, and measured performance work.
- **Internship #2 (next cycle):** NVIDIA, AMD, Cerebras — inference, compiler, CUDA tooling, performance software.

Deliberately **not** aimed at ASIC or RTL verification roles.

## 5. What actually differentiates this build

The plain from-scratch CUDA inference engine is a known genre with public implementations (`yalm`, `bw24`, `Annotated-LLM-Runtime`, `verbum.cpp`) and at least one full course teaching it (`tiny-vllm`). Flash attention is going the same way — a Deep-ML course, a DEV course, multiple standalone repos, a GPU MODE lecture.

So neither the engine nor flash attention is the differentiator. These are:

**D1 — The analytical performance model (Stage 10). The primary differentiator.**
A program that takes a kernel's parameters and the machine's measured characteristics and predicts runtime before execution. Validated retroactively against every kernel built in Stages 5 through 9, then prospectively on Stage 11. Reported as an error distribution, not a single number.

This is an active research area — quantitative roofline models built from microbenchmarks, and more recent learned approaches. That literature supplies a **baseline for comparison**: published work reports roofline analysis averaging roughly **34% prediction error** on deep learning kernels. Your model's error has a citable reference point, which converts a personal metric into a positioned one. *(Figure from the NeuSight paper, arXiv 2407.13853; verify before citing.)*

Almost nobody builds one of these at portfolio level. It is the artifact that says "understands hardware" rather than "tuned until it got faster."

**D2 — Counter-evidenced gap analysis.**
Every prediction-versus-measurement gap is explained with Nsight Compute hardware counters — achieved occupancy, memory throughput, warp stall reasons, shared memory bank conflicts — not with a plausible narrative. Where the counters do not establish a cause, the document says so. This is what makes the gap analysis credible instead of decorative.

**D3 — Upstream contribution (Stage 16, stretch).**
A merged kernel in `llama.cpp` / `ggml` is external validation no personal repo provides. Not required for done.

**Supporting, not differentiating:** flash attention, the systolic dataflow, the two headline ratios. These are expected content for a serious engine. Their absence would be conspicuous; their presence is not remarkable.

## 6. The headline numbers

Raw speedup against a self-chosen baseline is inflatable and will be discounted. The reported figures all have ceilings outside your control:

- **Performance model error distribution** across all kernels, against the ~34% published roofline baseline.
- **Percent of measured achievable memory bandwidth** on decode.
- **Percent of cuBLAS** on prefill, at the model's real matrix shapes.
- **Flash attention memory scaling** — peak memory versus sequence length, demonstrating the quadratic-to-linear change.

## 7. Definition of done

1. Engine generates coherent text from published weights, matching the PyTorch reference within a documented tolerance.
2. Prefill and decode benchmarked separately at every stage.
3. Flash attention implemented, correct, and its memory scaling measured.
4. Performance model built and validated across all kernels, error distribution reported.
5. Every gap in `MEASUREMENTS.md` backed by Nsight counters or explicitly marked as unestablished.
6. All four headline numbers measured and reproducible.

## 8. Explicitly out of scope

- Training or fine-tuning.
- Multi-GPU, distributed inference, tensor parallelism.
- Serving infrastructure, request batching, HTTP APIs.
- Models above roughly 1.5B parameters.
- RTL, SystemVerilog, FPGA.
- `cp.async` / `ldmatrix` / `mma` intrinsics — these require sm_80 or newer and the primary device is sm_75 (Turing). Flash attention uses plain shared memory staging.
- Any optimization that cannot be measured and explained.

## 9. Resume bullet shape

Do not write until the numbers exist.

> Built a transformer inference engine from scratch in C/CUDA with hand-written flash attention and a custom tiled GEMM; developed an analytical performance model predicting kernel runtime within **[X]%** across all kernels (published roofline baselines average ~34%), reaching **[Y]%** of measured achievable bandwidth on decode and **[Z]%** of cuBLAS on prefill, with Nsight-counter-backed analysis of every prediction gap.
