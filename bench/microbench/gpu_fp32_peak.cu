/* Microbenchmark 2 -- GPU FP32 peak throughput.
 *
 * MB_FP32_ACC independent FMA chains held entirely in registers. The chains do
 * not depend on one another, so the measurement is issue-rate limited rather
 * than latency limited, and the inner loop performs no memory traffic of any
 * kind: the single global store exists only to defeat dead-code elimination and
 * sits outside the loop behind a condition the compiler cannot fold away.
 *
 * tests/test_gpu_fp32_peak.cu asserts the no-global-traffic property by
 * scanning the generated PTX, not by taking this comment's word for it.
 */
#include "microbench.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MB_FP32_ACC   32
#define MB_FP32_BLOCK 256

__global__ void mb_fp32_peak_kernel(float *sink, int iters, float a, float b)
{
    float acc[MB_FP32_ACC];
#pragma unroll
    for (int i = 0; i < MB_FP32_ACC; ++i) acc[i] = a + (float)i;

    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < MB_FP32_ACC; ++i) acc[i] = fmaf(acc[i], b, a);
    }

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < MB_FP32_ACC; ++i) s += acc[i];

    /* Unreachable in practice; the compiler cannot prove it, so the chains
     * survive. The store is outside the timed inner loop by construction. */
    if (s == -1.0e30f) sink[threadIdx.x] = s;
}

typedef struct {
    float *sink;
    int    iters, grid, block;
    bench_cuda_timer *timer;
} fp32_ctx;

static double fp32_body(void *vctx, int iteration)
{
    (void)iteration;
    fp32_ctx *c = (fp32_ctx *)vctx;
    bench_cuda_timer_start(c->timer);
    mb_fp32_peak_kernel<<<c->grid, c->block>>>(c->sink, c->iters, 1.0f, 1.0000001f);
    return bench_cuda_timer_stop_ms(c->timer);
}

extern "C" int mb_gpu_fp32_peak_run(int warmup, int samples, mb_fp32_peak_result *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) return 1;

    fp32_ctx c;
    memset(&c, 0, sizeof c);
    c.block = MB_FP32_BLOCK;
    /* Fill every SM to its resident-thread limit. */
    c.grid  = dev.sm_count * (dev.max_threads_per_sm / MB_FP32_BLOCK);
    c.iters = 32768;

    if (cudaMalloc(&c.sink, (size_t)MB_FP32_BLOCK * sizeof(float)) != cudaSuccess) return 1;
    cudaMemset(c.sink, 0, (size_t)MB_FP32_BLOCK * sizeof(float));

    c.timer = bench_cuda_timer_create();
    if (!c.timer) { cudaFree(c.sink); return 1; }

    int n_alloc = samples > BENCH_MIN_SAMPLES ? samples : BENCH_MIN_SAMPLES;
    double *raw = (double *)malloc((size_t)n_alloc * sizeof(double));
    if (!raw) { bench_cuda_timer_destroy(c.timer); cudaFree(c.sink); return 1; }

    int eff_w = 0, eff_s = 0;
    bench_stats st = bench_run(fp32_body, &c, warmup, samples, raw, &eff_w, &eff_s);

    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        fprintf(stderr, "gpu_fp32_peak: kernel error %s\n", cudaGetErrorString(last));
        free(raw); bench_cuda_timer_destroy(c.timer); cudaFree(c.sink);
        return 1;
    }

    int threads = c.grid * c.block;
    /* FLOP count derived from the loop trip count, never hardcoded:
     * threads * iterations * accumulators * 2 flops per FMA. */
    double flops = (double)threads * (double)c.iters * (double)MB_FP32_ACC * 2.0;

    out->stats               = st;
    out->warmup              = eff_w;
    out->samples             = eff_s;
    out->inner_iterations    = c.iters;
    out->accumulators        = MB_FP32_ACC;
    out->threads             = threads;
    out->flops_per_iteration = flops;
    out->theoretical_gflops  = dev.theoretical_fp32_gflops_at_max_clock;
    out->gflops = (st.median > 0.0) ? flops / (st.median * 1.0e-3) / 1.0e9 : 0.0;

    {
        char cfg[BENCH_LABEL_LEN];
        snprintf(cfg, sizeof cfg, "%d FMA chains, %d iters, grid=%d block=%d",
                 MB_FP32_ACC, c.iters, c.grid, c.block);
        bench_kv_num nums[] = {
            { "threads",                       (double)threads },
            { "inner_iterations",              (double)c.iters },
            { "accumulators",                  (double)MB_FP32_ACC },
            { "flops_per_iteration",           flops },
            { "theoretical_gflops_at_rated_boost", dev.theoretical_fp32_gflops },
            { "theoretical_gflops_at_max_clock",   dev.theoretical_fp32_gflops_at_max_clock },
            { "sm_count_queried",              (double)dev.sm_count },
            { "cores_per_sm_spec",             (double)dev.cores_per_sm_spec },
        };
        bench_kv_str strs[] = {
            { "flop_count_derivation",
              "threads * inner_iterations * accumulators * 2 flops per FMA" },
            { "cores_per_sm_provenance",
              "vendor compute-capability table [spec]; CUDA exposes no core-count property" },
            { "tag", "measured" },
        };
        bench_record rec;
        memset(&rec, 0, sizeof rec);
        rec.benchmark = "gpu_fp32_peak";
        rec.configuration = cfg;
        rec.units = "GFLOP/s";
        rec.value = out->gflops;
        rec.warmup = eff_w;
        rec.samples_requested = eff_s;
        rec.samples_ms = raw;
        rec.n_samples = eff_s;
        rec.stats = st;
        rec.meta_num = nums; rec.n_meta_num = (int)(sizeof nums / sizeof nums[0]);
        rec.meta_str = strs; rec.n_meta_str = (int)(sizeof strs / sizeof strs[0]);
        bench_write_results(NULL, "gpu_fp32_peak", &rec, 1);
    }

    free(raw);
    bench_cuda_timer_destroy(c.timer);
    cudaFree(c.sink);
    return 0;
}

#ifndef BENCH_NO_MAIN
int main(int argc, char **argv)
{
    int warmup  = (argc > 1) ? atoi(argv[1]) : BENCH_MIN_WARMUP;
    int samples = (argc > 2) ? atoi(argv[2]) : BENCH_MIN_SAMPLES;

    mb_fp32_peak_result r;
    if (mb_gpu_fp32_peak_run(warmup, samples, &r) != 0) {
        fprintf(stderr, "gpu_fp32_peak: FAILED\n");
        return 1;
    }
    printf("gpu_fp32_peak  %.3f GFLOP/s  median %.4f ms  stddev %.3f%% of median  %s\n",
           r.gflops, r.stats.median, r.stats.stddev_pct_of_median,
           r.stats.valid ? "VALID" : "INVALID");
    if (!r.stats.valid) printf("  INVALID: %s\n", r.stats.invalid_reason);
    printf("  threads %d, iters %d, acc %d, theoretical ceiling %.3f GFLOP/s\n",
           r.threads, r.inner_iterations, r.accumulators, r.theoretical_gflops);
    return r.stats.valid ? 0 : 2;
}
#endif
