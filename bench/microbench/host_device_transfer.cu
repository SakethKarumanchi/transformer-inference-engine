/* Microbenchmark 5 -- host/device transfer bandwidth.
 *
 * Four separate figures: pinned and pageable, host-to-device and
 * device-to-host. They are never merged into one number -- pinned and pageable
 * differ by enough that an average of the two describes no transfer that ever
 * happens.
 */
#include "microbench.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {            \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                             \
                cudaGetErrorString(e_), __FILE__, __LINE__); goto fail; } } while (0)

typedef struct {
    void  *host, *dev;
    size_t bytes;
    cudaMemcpyKind kind;
    bench_cuda_timer *timer;
} xfer_ctx;

static double xfer_body(void *vctx, int iteration)
{
    (void)iteration;
    xfer_ctx *c = (xfer_ctx *)vctx;
    bench_cuda_timer_start(c->timer);
    if (c->kind == cudaMemcpyHostToDevice) cudaMemcpy(c->dev, c->host, c->bytes, c->kind);
    else                                   cudaMemcpy(c->host, c->dev, c->bytes, c->kind);
    return bench_cuda_timer_stop_ms(c->timer);
}

static double run_one(xfer_ctx *c, int warmup, int samples,
                      double *raw, int *eff_w, int *eff_s, bench_stats *st)
{
    *st = bench_run(xfer_body, c, warmup, samples, raw, eff_w, eff_s);
    return (st->median > 0.0) ? (double)c->bytes / (st->median * 1.0e-3) / 1.0e9 : 0.0;
}

extern "C" int mb_host_device_transfer_run(int warmup, int samples,
                                           size_t transfer_bytes,
                                           mb_transfer_result *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);
    if (transfer_bytes == 0) transfer_bytes = 256u * 1024u * 1024u;

    void  *pinned = NULL, *pageable = NULL, *dev = NULL;
    double *raw_ph = NULL, *raw_pd = NULL, *raw_gh = NULL, *raw_gd = NULL;
    int rc = 1;

    CHECK(cudaMalloc(&dev, transfer_bytes));
    CHECK(cudaHostAlloc(&pinned, transfer_bytes, cudaHostAllocDefault));
    pageable = malloc(transfer_bytes);
    if (!pageable) { fprintf(stderr, "host_device_transfer: malloc failed\n"); goto fail; }
    memset(pinned,   0x5a, transfer_bytes);
    memset(pageable, 0x5a, transfer_bytes);
    CHECK(cudaMemset(dev, 0x5a, transfer_bytes));

    {
        int n_alloc = samples > BENCH_MIN_SAMPLES ? samples : BENCH_MIN_SAMPLES;
        raw_ph = (double *)malloc((size_t)n_alloc * sizeof(double));
        raw_pd = (double *)malloc((size_t)n_alloc * sizeof(double));
        raw_gh = (double *)malloc((size_t)n_alloc * sizeof(double));
        raw_gd = (double *)malloc((size_t)n_alloc * sizeof(double));
        if (!raw_ph || !raw_pd || !raw_gh || !raw_gd) goto fail;

        bench_cuda_timer *timer = bench_cuda_timer_create();
        if (!timer) goto fail;

        xfer_ctx c;
        c.bytes = transfer_bytes;
        c.dev   = dev;
        c.timer = timer;
        int w = 0, s = 0;

        c.host = pinned;   c.kind = cudaMemcpyHostToDevice;
        out->pinned_h2d_gb_s   = run_one(&c, warmup, samples, raw_ph, &w, &s, &out->pinned_h2d);
        c.host = pinned;   c.kind = cudaMemcpyDeviceToHost;
        out->pinned_d2h_gb_s   = run_one(&c, warmup, samples, raw_pd, &w, &s, &out->pinned_d2h);
        c.host = pageable; c.kind = cudaMemcpyHostToDevice;
        out->pageable_h2d_gb_s = run_one(&c, warmup, samples, raw_gh, &w, &s, &out->pageable_h2d);
        c.host = pageable; c.kind = cudaMemcpyDeviceToHost;
        out->pageable_d2h_gb_s = run_one(&c, warmup, samples, raw_gd, &w, &s, &out->pageable_d2h);

        bench_cuda_timer_destroy(timer);
        out->warmup          = w;
        out->samples         = s;
        out->transfer_bytes  = transfer_bytes;

        double m = out->pinned_h2d.median;
        if (out->pinned_d2h.median   < m) m = out->pinned_d2h.median;
        if (out->pageable_h2d.median < m) m = out->pageable_h2d.median;
        if (out->pageable_d2h.median < m) m = out->pageable_d2h.median;
        out->measured_median_ms_min = m;

        {
            bench_record recs[4];
            memset(recs, 0, sizeof recs);
            const char *cfg[4]   = { "pinned host-to-device", "pinned device-to-host",
                                     "pageable host-to-device", "pageable device-to-host" };
            double      val[4]   = { out->pinned_h2d_gb_s, out->pinned_d2h_gb_s,
                                     out->pageable_h2d_gb_s, out->pageable_d2h_gb_s };
            double     *rawp[4]  = { raw_ph, raw_pd, raw_gh, raw_gd };
            bench_stats sts[4]   = { out->pinned_h2d, out->pinned_d2h,
                                     out->pageable_h2d, out->pageable_d2h };
            bench_kv_num nums[]  = { { "transfer_bytes", (double)transfer_bytes } };
            bench_kv_str strs[]  = { { "tag", "measured" },
                                     { "note", "four figures reported separately; never merged" } };
            for (int i = 0; i < 4; ++i) {
                recs[i].benchmark = "host_device_transfer";
                recs[i].configuration = cfg[i];
                recs[i].units = "GB/s";
                recs[i].value = val[i];
                recs[i].warmup = w; recs[i].samples_requested = s;
                recs[i].samples_ms = rawp[i]; recs[i].n_samples = s;
                recs[i].stats = sts[i];
                recs[i].meta_num = nums; recs[i].n_meta_num = 1;
                recs[i].meta_str = strs; recs[i].n_meta_str = 2;
            }
            bench_write_results(NULL, "host_device_transfer", recs, 4);
        }
        rc = 0;
    }

fail:
    free(raw_ph); free(raw_pd); free(raw_gh); free(raw_gd);
    if (pageable) free(pageable);
    if (pinned)   cudaFreeHost(pinned);
    if (dev)      cudaFree(dev);
    return rc;
}

#ifndef BENCH_NO_MAIN
int main(int argc, char **argv)
{
    int warmup  = (argc > 1) ? atoi(argv[1]) : BENCH_MIN_WARMUP;
    int samples = (argc > 2) ? atoi(argv[2]) : BENCH_MIN_SAMPLES;

    mb_transfer_result r;
    if (mb_host_device_transfer_run(warmup, samples, 0, &r) != 0) {
        fprintf(stderr, "host_device_transfer: FAILED\n");
        return 1;
    }
    printf("host_device_transfer  %zu B per transfer\n", r.transfer_bytes);
    printf("  pinned   H2D %8.3f GB/s  stddev %6.3f%%  %s\n",
           r.pinned_h2d_gb_s, r.pinned_h2d.stddev_pct_of_median,
           r.pinned_h2d.valid ? "VALID" : "INVALID");
    printf("  pinned   D2H %8.3f GB/s  stddev %6.3f%%  %s\n",
           r.pinned_d2h_gb_s, r.pinned_d2h.stddev_pct_of_median,
           r.pinned_d2h.valid ? "VALID" : "INVALID");
    printf("  pageable H2D %8.3f GB/s  stddev %6.3f%%  %s\n",
           r.pageable_h2d_gb_s, r.pageable_h2d.stddev_pct_of_median,
           r.pageable_h2d.valid ? "VALID" : "INVALID");
    printf("  pageable D2H %8.3f GB/s  stddev %6.3f%%  %s\n",
           r.pageable_d2h_gb_s, r.pageable_d2h.stddev_pct_of_median,
           r.pageable_d2h.valid ? "VALID" : "INVALID");
    int ok = r.pinned_h2d.valid && r.pinned_d2h.valid &&
             r.pageable_h2d.valid && r.pageable_d2h.valid;
    return ok ? 0 : 2;
}
#endif
