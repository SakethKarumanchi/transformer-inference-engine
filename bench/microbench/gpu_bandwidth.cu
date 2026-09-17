/* Microbenchmark 1 -- GPU device-to-device bandwidth.
 *
 * A grid-strided copy over a working set deliberately sized far beyond the
 * queried L2 so that every access misses L2 and the figure is DRAM bandwidth
 * rather than cache bandwidth. Timed with CUDA events around the kernel only,
 * with an explicit device synchronize before the timer is read.
 */
#include "microbench.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {            \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                             \
                cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

/* 16 bytes per thread per step keeps the load/store units saturated. */
__global__ void mb_bw_copy_kernel(float4 *__restrict__ dst,
                                  const float4 *__restrict__ src,
                                  size_t n)
{
    size_t i      = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

typedef struct {
    float4 *src, *dst;
    size_t  n;
    int     grid, block;
    bench_cuda_timer *timer;
} bw_ctx;

static double bw_body(void *vctx, int iteration)
{
    (void)iteration;
    bw_ctx *c = (bw_ctx *)vctx;
    bench_cuda_timer_start(c->timer);
    mb_bw_copy_kernel<<<c->grid, c->block>>>(c->dst, c->src, c->n);
    return bench_cuda_timer_stop_ms(c->timer);
}

extern "C" int mb_gpu_bandwidth_run(int warmup, int samples,
                                    size_t bytes_per_buffer,
                                    mb_bandwidth_result *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) return 1;

    if (bytes_per_buffer == 0) bytes_per_buffer = 256u * 1024u * 1024u;
    /* Round down to whole float4 elements. */
    size_t n = bytes_per_buffer / sizeof(float4);
    bytes_per_buffer = n * sizeof(float4);

    /* The working set is src + dst. The protocol requires it to exceed L2;
     * refuse to run rather than report a cache-resident figure as DRAM. */
    size_t working_set = 2 * bytes_per_buffer;
    if (working_set <= dev.l2_bytes) {
        fprintf(stderr, "gpu_bandwidth: working set %zu does not exceed L2 %zu\n",
                working_set, dev.l2_bytes);
        return 1;
    }
    if (working_set > (size_t)((double)dev.total_global_bytes * 0.70)) {
        fprintf(stderr, "gpu_bandwidth: working set %zu exceeds 70%% of VRAM %zu\n",
                working_set, dev.total_global_bytes);
        return 1;
    }

    bw_ctx c;
    memset(&c, 0, sizeof c);
    c.n     = n;
    c.block = 256;
    c.grid  = dev.sm_count * 32;

    CHECK(cudaMalloc(&c.src, bytes_per_buffer));
    CHECK(cudaMalloc(&c.dst, bytes_per_buffer));
    CHECK(cudaMemset(c.src, 0x3c, bytes_per_buffer));
    CHECK(cudaMemset(c.dst, 0x00, bytes_per_buffer));

    c.timer = bench_cuda_timer_create();
    if (!c.timer) { cudaFree(c.src); cudaFree(c.dst); return 1; }

    int n_alloc = samples > BENCH_MIN_SAMPLES ? samples : BENCH_MIN_SAMPLES;
    double *raw = (double *)malloc((size_t)n_alloc * sizeof(double));
    if (!raw) { bench_cuda_timer_destroy(c.timer); cudaFree(c.src); cudaFree(c.dst); return 1; }

    int eff_w = 0, eff_s = 0;
    bench_stats st = bench_run(bw_body, &c, warmup, samples, raw, &eff_w, &eff_s);

    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        fprintf(stderr, "gpu_bandwidth: kernel error %s\n", cudaGetErrorString(last));
        free(raw); bench_cuda_timer_destroy(c.timer); cudaFree(c.src); cudaFree(c.dst);
        return 1;
    }

    out->stats                    = st;
    out->warmup                   = eff_w;
    out->samples                  = eff_s;
    out->working_set_bytes        = working_set;
    out->l2_bytes                 = dev.l2_bytes;
    out->bytes_moved_per_iter     = 2 * bytes_per_buffer;   /* one read + one write */
    out->stride_elements          = c.grid * c.block;       /* grid stride, in float4 */
    out->theoretical_peak_gb_per_s= dev.theoretical_dram_gb_s;
    out->gb_per_s = (st.median > 0.0)
                  ? (double)out->bytes_moved_per_iter / (st.median * 1.0e-3) / 1.0e9
                  : 0.0;

    /* Emit the results file here so the caller of the library API gets it too. */
    {
        char cfg[BENCH_LABEL_LEN];
        snprintf(cfg, sizeof cfg, "float4 grid-stride copy, %zu B/buffer, grid=%d block=%d",
                 bytes_per_buffer, c.grid, c.block);
        bench_kv_num nums[] = {
            { "working_set_bytes",         (double)working_set },
            { "l2_bytes_queried",          (double)dev.l2_bytes },
            { "bytes_moved_per_iteration", (double)out->bytes_moved_per_iter },
            { "theoretical_peak_gb_per_s", out->theoretical_peak_gb_per_s },
            { "efficiency_measured_over_theoretical",
              out->theoretical_peak_gb_per_s > 0.0
                  ? out->gb_per_s / out->theoretical_peak_gb_per_s : 0.0 },
        };
        bench_kv_str strs[] = {
            { "theoretical_peak_derivation",
              "2 transfers/clock * memoryClockRate[queried] * busWidth[queried]/8" },
            { "tag", "measured" },
        };
        bench_record rec;
        memset(&rec, 0, sizeof rec);
        rec.benchmark = "gpu_bandwidth";
        rec.configuration = cfg;
        rec.units = "GB/s";
        rec.value = out->gb_per_s;
        rec.warmup = eff_w;
        rec.samples_requested = eff_s;
        rec.samples_ms = raw;
        rec.n_samples = eff_s;
        rec.stats = st;
        rec.meta_num = nums; rec.n_meta_num = (int)(sizeof nums / sizeof nums[0]);
        rec.meta_str = strs; rec.n_meta_str = (int)(sizeof strs / sizeof strs[0]);
        bench_write_results(NULL, "gpu_bandwidth", &rec, 1);
    }

    free(raw);
    bench_cuda_timer_destroy(c.timer);
    cudaFree(c.src);
    cudaFree(c.dst);
    return 0;
}

#ifndef BENCH_NO_MAIN
int main(int argc, char **argv)
{
    int warmup  = (argc > 1) ? atoi(argv[1]) : BENCH_MIN_WARMUP;
    int samples = (argc > 2) ? atoi(argv[2]) : BENCH_MIN_SAMPLES;

    mb_bandwidth_result r;
    if (mb_gpu_bandwidth_run(warmup, samples, 0, &r) != 0) {
        fprintf(stderr, "gpu_bandwidth: FAILED\n");
        return 1;
    }
    printf("gpu_bandwidth  %.3f GB/s  median %.4f ms  stddev %.3f%% of median  %s\n",
           r.gb_per_s, r.stats.median, r.stats.stddev_pct_of_median,
           r.stats.valid ? "VALID" : "INVALID");
    if (!r.stats.valid) printf("  INVALID: %s\n", r.stats.invalid_reason);
    printf("  working set %zu B, L2 %zu B, theoretical peak %.3f GB/s, efficiency %.4f\n",
           r.working_set_bytes, r.l2_bytes, r.theoretical_peak_gb_per_s,
           r.theoretical_peak_gb_per_s > 0 ? r.gb_per_s / r.theoretical_peak_gb_per_s : 0.0);
    return r.stats.valid ? 0 : 2;
}
#endif
