/* Microbenchmark 8 -- shared memory bandwidth, measured rather than assumed.
 *
 * The Stage 10 model needs this for tiled kernels. Two access patterns run:
 *
 *   conflict-free  -- consecutive threads read consecutive 4-byte words, so a
 *                     warp's 32 lanes land on 32 distinct banks. Bank-conflict
 *                     free by construction, not by measurement.
 *   conflicting    -- each thread reads at a stride of 32 words, so every lane
 *                     in a warp lands on bank 0: a 32-way conflict. Measured
 *                     alongside for contrast.
 *
 * The working set is sized to fit inside the QUERIED shared memory per block.
 * The shared-versus-global comparison is deliberately NOT made here: global
 * bandwidth is a protocol measurement that does not exist at unit-test time.
 * bench/microbench/run_all.py makes that comparison after both have run.
 */
#include "microbench.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MB_SHMEM_FLOATS 8192          /* 32 KiB, checked against the queried limit */
#define MB_SHMEM_MASK   (MB_SHMEM_FLOATS - 1)
#define MB_SHMEM_BLOCK  256
#define MB_SHMEM_ACC    4
#define MB_SHMEM_ITERS  4096
#define MB_SHMEM_CONFLICT_STRIDE 32   /* words; 32 banks -> every lane on bank 0 */

__global__ void mb_shmem_conflict_free_kernel(float *sink, int iters)
{
    __shared__ float s[MB_SHMEM_FLOATS];
    for (int i = threadIdx.x; i < MB_SHMEM_FLOATS; i += blockDim.x) s[i] = (float)i;
    __syncthreads();

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int base = threadIdx.x;
    for (int it = 0; it < iters; ++it) {
        int o = it * MB_SHMEM_BLOCK;
        a0 += s[(base + o)                        & MB_SHMEM_MASK];
        a1 += s[(base + o + MB_SHMEM_BLOCK)       & MB_SHMEM_MASK];
        a2 += s[(base + o + 2 * MB_SHMEM_BLOCK)   & MB_SHMEM_MASK];
        a3 += s[(base + o + 3 * MB_SHMEM_BLOCK)   & MB_SHMEM_MASK];
    }
    float t = a0 + a1 + a2 + a3;
    if (t == -1.0e30f) sink[threadIdx.x] = t;
}

__global__ void mb_shmem_conflicting_kernel(float *sink, int iters)
{
    __shared__ float s[MB_SHMEM_FLOATS];
    for (int i = threadIdx.x; i < MB_SHMEM_FLOATS; i += blockDim.x) s[i] = (float)i;
    __syncthreads();

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int base = threadIdx.x * MB_SHMEM_CONFLICT_STRIDE;
    for (int it = 0; it < iters; ++it) {
        a0 += s[(base + it)     & MB_SHMEM_MASK];
        a1 += s[(base + it + 1) & MB_SHMEM_MASK];
        a2 += s[(base + it + 2) & MB_SHMEM_MASK];
        a3 += s[(base + it + 3) & MB_SHMEM_MASK];
    }
    float t = a0 + a1 + a2 + a3;
    if (t == -1.0e30f) sink[threadIdx.x] = t;
}

typedef struct {
    float *sink;
    int    grid, iters, conflicting;
    bench_cuda_timer *timer;
} shmem_ctx;

static double shmem_body(void *vctx, int iteration)
{
    (void)iteration;
    shmem_ctx *c = (shmem_ctx *)vctx;
    bench_cuda_timer_start(c->timer);
    if (c->conflicting)
        mb_shmem_conflicting_kernel<<<c->grid, MB_SHMEM_BLOCK>>>(c->sink, c->iters);
    else
        mb_shmem_conflict_free_kernel<<<c->grid, MB_SHMEM_BLOCK>>>(c->sink, c->iters);
    return bench_cuda_timer_stop_ms(c->timer);
}

extern "C" int mb_shared_mem_bandwidth_run(int warmup, int samples, mb_shmem_result *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) return 1;

    size_t working_set = (size_t)MB_SHMEM_FLOATS * sizeof(float);
    out->shared_bytes_per_block      = working_set;
    out->queried_shared_per_block    = dev.shared_per_block;
    out->conflict_stride             = MB_SHMEM_CONFLICT_STRIDE;
    snprintf(out->access_pattern, sizeof out->access_pattern,
             "conflict-free: word index = tid + k*%d (32 lanes -> 32 banks); "
             "conflicting: word index = tid*%d + k (32 lanes -> bank 0)",
             MB_SHMEM_BLOCK, MB_SHMEM_CONFLICT_STRIDE);

    if (working_set > dev.shared_per_block) {
        fprintf(stderr, "shared_mem_bandwidth: working set %zu exceeds queried "
                        "shared memory per block %zu\n", working_set, dev.shared_per_block);
        return 1;
    }

    shmem_ctx c;
    memset(&c, 0, sizeof c);
    c.grid  = dev.sm_count * (dev.max_threads_per_sm / MB_SHMEM_BLOCK);
    c.iters = MB_SHMEM_ITERS;
    if (cudaMalloc(&c.sink, MB_SHMEM_BLOCK * sizeof(float)) != cudaSuccess) return 1;
    cudaMemset(c.sink, 0, MB_SHMEM_BLOCK * sizeof(float));

    c.timer = bench_cuda_timer_create();
    if (!c.timer) { cudaFree(c.sink); return 1; }

    int n_alloc = samples > BENCH_MIN_SAMPLES ? samples : BENCH_MIN_SAMPLES;
    double *raw_cf = (double *)malloc((size_t)n_alloc * sizeof(double));
    double *raw_cx = (double *)malloc((size_t)n_alloc * sizeof(double));
    if (!raw_cf || !raw_cx) {
        free(raw_cf); free(raw_cx); bench_cuda_timer_destroy(c.timer); cudaFree(c.sink);
        return 1;
    }

    int eff_w = 0, eff_s = 0;
    c.conflicting = 0;
    out->conflict_free_stats = bench_run(shmem_body, &c, warmup, samples, raw_cf, &eff_w, &eff_s);
    c.conflicting = 1;
    out->conflicting_stats   = bench_run(shmem_body, &c, warmup, samples, raw_cx, &eff_w, &eff_s);

    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        fprintf(stderr, "shared_mem_bandwidth: %s\n", cudaGetErrorString(last));
        free(raw_cf); free(raw_cx); bench_cuda_timer_destroy(c.timer); cudaFree(c.sink);
        return 1;
    }

    /* bytes = threads * iterations * loads-per-iteration * 4 */
    double bytes = (double)c.grid * MB_SHMEM_BLOCK *
                   (double)c.iters * MB_SHMEM_ACC * sizeof(float);
    out->conflict_free_gb_s = (out->conflict_free_stats.median > 0.0)
        ? bytes / (out->conflict_free_stats.median * 1.0e-3) / 1.0e9 : 0.0;
    out->conflicting_gb_s   = (out->conflicting_stats.median > 0.0)
        ? bytes / (out->conflicting_stats.median * 1.0e-3) / 1.0e9 : 0.0;
    out->warmup  = eff_w;
    out->samples = eff_s;

    {
        bench_kv_num nums[] = {
            { "shared_bytes_per_block_working_set", (double)working_set },
            { "queried_shared_per_block",           (double)dev.shared_per_block },
            { "conflict_stride_words",              (double)MB_SHMEM_CONFLICT_STRIDE },
            { "bytes_read_per_iteration",           bytes },
            { "conflicting_gb_per_s",               out->conflicting_gb_s },
        };
        bench_kv_str strs[] = {
            { "access_pattern", out->access_pattern },
            { "tag",            "measured" },
        };
        bench_record recs[2];
        memset(recs, 0, sizeof recs);
        recs[0].benchmark = "shared_mem_bandwidth";
        recs[0].configuration = "conflict-free";
        recs[0].units = "GB/s";
        recs[0].value = out->conflict_free_gb_s;
        recs[0].warmup = eff_w; recs[0].samples_requested = eff_s;
        recs[0].samples_ms = raw_cf; recs[0].n_samples = eff_s;
        recs[0].stats = out->conflict_free_stats;
        recs[0].meta_num = nums; recs[0].n_meta_num = 5;
        recs[0].meta_str = strs; recs[0].n_meta_str = 2;

        recs[1] = recs[0];
        recs[1].configuration = "32-way bank conflict";
        recs[1].value = out->conflicting_gb_s;
        recs[1].samples_ms = raw_cx;
        recs[1].stats = out->conflicting_stats;

        bench_write_results(NULL, "shared_mem_bandwidth", recs, 2);
    }

    free(raw_cf);
    free(raw_cx);
    bench_cuda_timer_destroy(c.timer);
    cudaFree(c.sink);
    return 0;
}

#ifndef BENCH_NO_MAIN
int main(int argc, char **argv)
{
    int warmup  = (argc > 1) ? atoi(argv[1]) : BENCH_MIN_WARMUP;
    int samples = (argc > 2) ? atoi(argv[2]) : BENCH_MIN_SAMPLES;

    mb_shmem_result r;
    if (mb_shared_mem_bandwidth_run(warmup, samples, &r) != 0) {
        fprintf(stderr, "shared_mem_bandwidth: FAILED\n");
        return 1;
    }
    printf("shared_mem_bandwidth  working set %zu B (queried limit %zu B)\n",
           r.shared_bytes_per_block, r.queried_shared_per_block);
    printf("  pattern: %s\n", r.access_pattern);
    printf("  conflict-free  %10.2f GB/s  stddev %6.3f%%  %s\n",
           r.conflict_free_gb_s, r.conflict_free_stats.stddev_pct_of_median,
           r.conflict_free_stats.valid ? "VALID" : "INVALID");
    printf("  %d-way conflict %9.2f GB/s  stddev %6.3f%%  %s\n",
           r.conflict_stride, r.conflicting_gb_s, r.conflicting_stats.stddev_pct_of_median,
           r.conflicting_stats.valid ? "VALID" : "INVALID");
    return (r.conflict_free_stats.valid && r.conflicting_stats.valid) ? 0 : 2;
}
#endif
