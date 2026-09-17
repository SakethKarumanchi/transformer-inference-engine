/* Microbenchmark 7 -- kernel launch overhead.
 *
 * The Stage 10 performance model needs this as a fixed additive term. Without
 * it the model systematically overpredicts speed on small kernels, because it
 * charges them only for the work they do and nothing for being launched.
 *
 * Two figures, reported separately because they answer different questions:
 *   sync-per-launch  -- launch to completion, one device synchronize per launch.
 *                       This is the additive term for a dependent sequence.
 *   back-to-back     -- a batch of launches with one synchronize at the end,
 *                       divided by the batch size. This is the amortised cost
 *                       when launches pipeline.
 *
 * Timed with the monotonic CPU counter rather than CUDA events: at this scale
 * the event pair's own cost is a significant fraction of the quantity being
 * measured. Both variants include an explicit device synchronize, so neither
 * measures enqueue time.
 */
#include "microbench.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Genuinely empty. tests/test_kernel_launch_overhead.cu asserts this by
 * scanning the generated PTX for the body, not by trusting this comment. */
__global__ void mb_empty_kernel(void) { }

#define MB_LAUNCH_BATCH 1000

typedef struct { int batch; } launch_ctx;

static double launch_sync_body(void *vctx, int iteration)
{
    (void)iteration;
    launch_ctx *c = (launch_ctx *)vctx;
    double t0 = bench_cpu_time_seconds();
    for (int i = 0; i < c->batch; ++i) {
        mb_empty_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();     /* launch-to-completion, every launch */
    }
    double t1 = bench_cpu_time_seconds();
    return (t1 - t0) * 1.0e3;
}

static double launch_btb_body(void *vctx, int iteration)
{
    (void)iteration;
    launch_ctx *c = (launch_ctx *)vctx;
    double t0 = bench_cpu_time_seconds();
    for (int i = 0; i < c->batch; ++i) mb_empty_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();         /* one synchronize closes the batch */
    double t1 = bench_cpu_time_seconds();
    return (t1 - t0) * 1.0e3;
}

extern "C" int mb_kernel_launch_overhead_run(int warmup, int samples,
                                             mb_launch_result *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) return 1;

    launch_ctx c;
    c.batch = MB_LAUNCH_BATCH;

    int n_alloc = samples > BENCH_MIN_SAMPLES ? samples : BENCH_MIN_SAMPLES;
    double *raw_sync = (double *)malloc((size_t)n_alloc * sizeof(double));
    double *raw_btb  = (double *)malloc((size_t)n_alloc * sizeof(double));
    if (!raw_sync || !raw_btb) { free(raw_sync); free(raw_btb); return 1; }

    int eff_w = 0, eff_s = 0;
    out->sync_stats          = bench_run(launch_sync_body, &c, warmup, samples,
                                         raw_sync, &eff_w, &eff_s);
    out->back_to_back_stats  = bench_run(launch_btb_body,  &c, warmup, samples,
                                         raw_btb,  &eff_w, &eff_s);

    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        fprintf(stderr, "kernel_launch_overhead: %s\n", cudaGetErrorString(last));
        free(raw_sync); free(raw_btb);
        return 1;
    }

    out->batch_size = c.batch;
    out->warmup     = eff_w;
    out->samples    = eff_s;
    /* median is milliseconds for the whole batch; per launch in microseconds. */
    out->sync_per_launch_us = out->sync_stats.median * 1.0e3 / (double)c.batch;
    out->back_to_back_us    = out->back_to_back_stats.median * 1.0e3 / (double)c.batch;

    {
        bench_kv_num nums[] = {
            { "batch_size",          (double)c.batch },
            { "sync_per_launch_us",  out->sync_per_launch_us },
            { "back_to_back_us",     out->back_to_back_us },
        };
        bench_kv_str strs[] = {
            { "timer",  bench_cpu_timer_name() },
            { "note",   "both variants include an explicit cudaDeviceSynchronize" },
            { "tag",    "measured" },
        };
        bench_record recs[2];
        memset(recs, 0, sizeof recs);
        recs[0].benchmark = "kernel_launch_overhead";
        recs[0].configuration = "sync per launch (launch to completion)";
        recs[0].units = "us";
        recs[0].value = out->sync_per_launch_us;
        recs[0].warmup = eff_w; recs[0].samples_requested = eff_s;
        recs[0].samples_ms = raw_sync; recs[0].n_samples = eff_s;
        recs[0].stats = out->sync_stats;
        recs[0].meta_num = nums; recs[0].n_meta_num = 3;
        recs[0].meta_str = strs; recs[0].n_meta_str = 3;

        recs[1] = recs[0];
        recs[1].configuration = "back to back, one sync per batch";
        recs[1].value = out->back_to_back_us;
        recs[1].samples_ms = raw_btb;
        recs[1].stats = out->back_to_back_stats;

        bench_write_results(NULL, "kernel_launch_overhead", recs, 2);
    }

    free(raw_sync);
    free(raw_btb);
    return 0;
}

#ifndef BENCH_NO_MAIN
int main(int argc, char **argv)
{
    int warmup  = (argc > 1) ? atoi(argv[1]) : BENCH_MIN_WARMUP;
    int samples = (argc > 2) ? atoi(argv[2]) : BENCH_MIN_SAMPLES;

    mb_launch_result r;
    if (mb_kernel_launch_overhead_run(warmup, samples, &r) != 0) {
        fprintf(stderr, "kernel_launch_overhead: FAILED\n");
        return 1;
    }
    printf("kernel_launch_overhead  batch %d, timer %s\n",
           r.batch_size, bench_cpu_timer_name());
    printf("  sync per launch  %8.4f us  stddev %6.3f%%  %s\n",
           r.sync_per_launch_us, r.sync_stats.stddev_pct_of_median,
           r.sync_stats.valid ? "VALID" : "INVALID");
    printf("  back to back     %8.4f us  stddev %6.3f%%  %s\n",
           r.back_to_back_us, r.back_to_back_stats.stddev_pct_of_median,
           r.back_to_back_stats.valid ? "VALID" : "INVALID");
    if (!r.sync_stats.valid) printf("  INVALID(sync): %s\n", r.sync_stats.invalid_reason);
    if (!r.back_to_back_stats.valid)
        printf("  INVALID(btb): %s\n", r.back_to_back_stats.invalid_reason);
    return (r.sync_stats.valid && r.back_to_back_stats.valid) ? 0 : 2;
}
#endif
